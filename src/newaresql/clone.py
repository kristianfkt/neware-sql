import concurrent.futures
import pathlib
import random
from _thread import LockType
from threading import Event, Lock
from typing import Callable, Generator, Literal

import polars as pl
import tqdm.auto as tqdm

import newaresql.connectors
import newaresql.utils as utils

DEFAULT_CHUNKSIZE = 100_000


def _exit(event: Event | None) -> bool:
    return (event is not None) and event.is_set()


def _tests_antijoin(remote: pl.DataFrame, local: pl.DataFrame) -> pl.DataFrame:
    """
    Perform an anti-join between the remote and local tests dataframes to identify
    tests that are present in the remote but not in the local.

    """
    if local.is_empty():
        test = remote
    else:
        test = remote.join(
            local.filter(pl.col("end_time").is_not_null()).select(
                *["dev_uid", "unit_id", "chl_id", "test_id"]
            ),
            on=["dev_uid", "unit_id", "chl_id", "test_id"],
            how="anti",
        )
    return test


def _stream_chunks(
    test: dict,
    remote: newaresql.connectors.BTSConnector,
    local: newaresql.connectors.FileConnector | newaresql.connectors.SQLiteConnector,
    chunksize: int = DEFAULT_CHUNKSIZE,
    event: Event | None = None,
) -> Generator[pl.DataFrame, None, None]:

    seq_id = local.get_stats(test)["seq_id"] or 0
    i = seq_id + 1
    j = i + chunksize - 1
    while not (
        chunk := remote.get_data(test, where={"seq_id": (i, j)}, extend=False)
    ).is_empty():
        yield chunk
        i = j + 1
        j = i + chunksize - 1
        if _exit(event):
            break
    return


def test_to_files(
    test: dict,
    remote: newaresql.connectors.BTSConnector,
    local: newaresql.connectors.FileConnector,
    chunksize: int = DEFAULT_CHUNKSIZE,
    event: Event | None = None,
    lock: LockType | None = None,
) -> None:
    name = utils.test_name(test)
    for chunk in _stream_chunks(test, remote, local, chunksize=chunksize, event=event):
        local.write_table(chunk, name, append=True)
    if not _exit(event):
        utils.dump_json(test, local.path.joinpath(f"{name}/test.json"))
    return


def test_to_sqlite(
    test: dict,
    remote: newaresql.connectors.BTSConnector,
    local: newaresql.connectors.SQLiteConnector,
    chunksize: int = DEFAULT_CHUNKSIZE,
    event: Event | None = None,
    lock: LockType | None = None,
) -> None:
    name = utils.test_name(test)
    for chunk in _stream_chunks(test, remote, local, chunksize=chunksize, event=event):
        local.write_table(chunk, name, append=True)

    if not _exit(event) and lock is None:
        row = pl.DataFrame([test])
        mask = (
            (pl.col("dev_uid") != test["dev_uid"])
            | (pl.col("unit_id") != test["unit_id"])
            | (pl.col("chl_id") != test["chl_id"])
            | (pl.col("test_id") != test["test_id"])
        )
        new = pl.concat([local.get_tests().filter(mask), row], how="diagonal_relaxed")
        local.write_table(new, "tests", append=False)

    elif not _exit(event) and lock is not None:
        with lock:
            row = pl.DataFrame([test])
            mask = (
                (pl.col("dev_uid") != test["dev_uid"])
                | (pl.col("unit_id") != test["unit_id"])
                | (pl.col("chl_id") != test["chl_id"])
                | (pl.col("test_id") != test["test_id"])
            )
            new = pl.concat(
                [local.get_tests().filter(mask), row], how="diagonal_relaxed"
            )
            local.write_table(new, "tests", append=False)
    return


def _serial(
    tests: list[dict],
    remote: newaresql.connectors.BTSConnector,
    local: newaresql.connectors.FileConnector | newaresql.connectors.SQLiteConnector,
    chunksize: int = DEFAULT_CHUNKSIZE,
    event: Event | None = None,
    callback: Callable[[dict], bool] | None = None,
    lock: LockType | None = None,
) -> None:

    for test in tqdm.tqdm(tests, desc="Cloning tests", unit="test"):
        if _exit(event):
            break
        if (callback is not None) and (not callback(test)):
            continue

        if isinstance(local, newaresql.local.FileConnector):
            test_to_files(
                test, remote, local, chunksize=chunksize, event=event, lock=lock
            )
        elif isinstance(local, newaresql.local.SQLiteConnector):
            test_to_sqlite(
                test, remote, local, chunksize=chunksize, event=event, lock=lock
            )
    return


def _parallel(
    tests: list[dict],
    remote: newaresql.bts.BTSConnector,
    local: newaresql.local.FileConnector | newaresql.local.SQLiteConnector,
    chunksize: int = DEFAULT_CHUNKSIZE,
    event: Event | None = None,
    callback: Callable[[dict], bool] | None = None,
    lock: LockType | None = None,
    workers: int = -1,
) -> None:

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers if workers > 0 else -1
    ) as executor:
        futures = []
        for test in tests:
            if _exit(event):
                break
            if (callback is not None) and (not callback(test)):
                continue
            if isinstance(local, newaresql.local.FileConnector):
                futures.append(
                    executor.submit(
                        test_to_files,
                        test,
                        local=local,
                        remote=remote,
                        chunksize=chunksize,
                        event=event,
                        lock=lock,
                    )
                )
            elif isinstance(local, newaresql.local.SQLiteConnector):
                futures.append(
                    executor.submit(
                        test_to_sqlite,
                        test,
                        local=local,
                        remote=remote,
                        chunksize=chunksize,
                        event=event,
                        lock=lock,
                    )
                )

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Cloning tests",
            unit="test",
        ):
            try:
                _ = future.result()
            except Exception as e:
                print(f"Error processing test {utils.test_name(test)}: {e}")


def clone(
    
    path: str | pathlib.Path | None = None,
    format: str | None = None,
    host: str | None = None,
    port: str | int | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    version: str | None = None,
    target: Literal["files", "sqlite"] = "files",
    chunksize: int = DEFAULT_CHUNKSIZE,
    workers: int | None = None,
    event: Event | None = None,
    callback: Callable[[dict], bool] | None = None,
    lock: LockType | None = None,
):

    if lock is None:
        lock = Lock()

    with (
        newaresql.bts.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            version=version,
        ) as remote,
        newaresql.local.connect(
            path=path,
            format=format,
            target=target,
        ) as local,
    ):
        tests = _tests_antijoin(remote.get_tests(), local.get_tests()).to_dicts()
        random.shuffle(tests)  # Shuffle the tests to distribute the load more evenly
        if workers is None:
            _serial(
                tests,
                remote,
                local,
                chunksize=chunksize,
                event=event,
                callback=callback,
                lock=lock,
            )
        elif (workers == -1) or (workers > 1):
            _parallel(
                tests,
                remote,
                local,
                chunksize=chunksize,
                event=event,
                callback=callback,
                lock=lock,
                workers=workers,
            )
        else:
            raise ValueError("workers must be None, -1, or greater than 1")
