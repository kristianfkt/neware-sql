import concurrent.futures
import pathlib
import uuid
from typing import Callable, Literal

import polars as pl

import newaresql.local.connector
import newaresql.sql
import newaresql.sql.connector
import newaresql.utils as utils


def _callback(*args, **kwargs) -> bool:
    return True


def _check_local_finished(
    test: dict, local: newaresql.local.connector.Connector
) -> bool:

    root = local.root.joinpath(utils.test_name(test))
    path = root.joinpath("metadata.json")
    meta = utils.load_dict(path) if path.exists() else {}

    return meta.get("end_time") is not None


def _clone_test(
    test: dict,
    remote: newaresql.sql.connector.Connector,
    local: newaresql.local.connector.Connector,
):

    path = local.root.joinpath(utils.test_name(test))
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    if any(path.glob(f"*.{local.fmt}")):
        max_seq_id = (
            local.scan_data(test)
            .select("seq_id")
            .max()
            .collect(engine="streaming")
            .item()
        )
    else:
        max_seq_id = 0
    i = max_seq_id + 1
    j = i + remote.chunksize - 1

    while (
        chunk := newaresql.sql.get_data(
            test, connector=remote, where={"seq_id": (i, j)}
        )
    ).height > 0:
        utils.dump_frame(chunk, path.joinpath(str(uuid.uuid4())), fmt=local.fmt)
        i = i + remote.chunksize
        j = i + remote.chunksize - 1

    # Finally
    utils.dump_dict(test, path.joinpath("meta.json"))

    return


def clone(
    root: str | pathlib.Path | None = None,
    host: str | None = None,
    port: int | str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    fmt: Literal["parquet", "csv", "feather", "ipc"] | None = None,
    chunksize: int | None = None,
    callback: Callable[..., bool] | None = None,
):

    if callback is None:
        callback = _callback
    with (
        newaresql.sql.connector.Connector(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            chunksize=chunksize,
        ) as remote,
        newaresql.local.connector.Connector(
            root=root,
            fmt=fmt,
        ) as local,
        concurrent.futures.ThreadPoolExecutor() as executor,
    ):
        futures = {}

        tests_local = pl.DataFrame(local.list_tests())
        test_remote = pl.DataFrame(remote.list_tests())
        tests = test_remote.join(
            tests_local.filter(pl.col("end_time").is_not_null()).select(
                "dev_uid", "unit_id", "chl_id", "test_id"
            ),
            on=["dev_uid", "unit_id", "chl_id", "test_id"],
            how="anti",
        )

        for test in tests.to_dicts():
            if not callback(test):
                continue

            if _check_local_finished(test, local):
                continue

            futures[executor.submit(_clone_test, test, remote, local)] = test
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                test = futures[future]
                print(f"Error cloning test {utils.test_name(test)}: {e}")
