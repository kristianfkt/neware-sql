import concurrent.futures
import threading

import polars as pl
import tqdm.auto as tqdm

from newaresql import defaults
from newaresql.export.sink import FileSink
from newaresql.export.source import BTSSource
from newaresql.types import Test

Source = BTSSource
Sink = FileSink


def check_event(event: threading.Event | None) -> bool:
    return event.is_set() if event is not None else False


def get_tests_to_update(source: Source, sink: Sink) -> list[Test]:
    #
    #  We want to remove all tests from source where sink has end-time
    tests_source = source.get_tests()
    tests_sink = sink.get_tests()
    if (tests_sink is None) or tests_sink.is_empty():
        tests = tests_source.to_dicts()
    else:
        tests = tests_source.join(
            tests_sink.filter(pl.col("end_time").is_not_null()),
            on=["dev_uid", "unit_id", "chl_id", "test_id"],
            how="anti",
        ).to_dicts()
    return tests


def export_test(
    test: Test,
    source: Source,
    sink: Sink,
    chunk_size: int = defaults.CHUNK_SIZE,
    event: threading.Event | None = None,
) -> None:
    """
    Export a single test from the source to the sink.

    Parameters:
    - test: The test to export.
    - source: The source from which to export the test.
    - sink: The sink to which to export the test.
    - chunk_size: The size of chunks to use when exporting data.
    - event: An optional threading.Event to allow early termination.
    """
    # Check early exit
    if check_event(event):
        return

    if sink.contains_test(test) and (seq_id := sink.get_max_seq_id(test)) is not None:
        seq_id = sink.get_max_seq_id(test)
        if isinstance(seq_id, int):
            where = {"seq_id": (seq_id + 1, None)}
        else:
            where = None
    else:
        where = None

    for chunk in source.chunk_data(test, where=where, chunk_size=chunk_size):
        sink.update_data(test, chunk)
        if check_event(event):
            return

    sink.update_meta(test)
    return


def export_threaded(
    source: Source,
    sink: Sink,
    workers: int = -1,
    chunk_size: int = defaults.CHUNK_SIZE,
    progress: bool = True,
    event: threading.Event | None = None,
):
    if workers == 0:
        raise ValueError("Number of workers cannot be zero.")
    if workers < -1:
        raise ValueError("Number of workers cannot be less than -1.")

    tests = get_tests_to_update(source, sink)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=None if workers == -1 else workers
    ) as executor:
        futures = [
            executor.submit(
                export_test, test, source, sink, chunk_size=chunk_size, event=event
            )
            for test in tests
        ]
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            disable=not progress,
        ):
            _ = future.result()
        return


def export_serially(
    source: Source,
    sink: Sink,
    progress: bool = True,
    chunk_size: int = defaults.CHUNK_SIZE,
    event: threading.Event | None = None,
):
    tests = get_tests_to_update(source, sink)
    for test in tqdm.tqdm(tests, disable=not progress):
        export_test(test, source, sink, chunk_size=chunk_size, event=event)
        if check_event(event):
            return


def export(
    source: Source,
    sink: Sink,
    workers: int | None = None,
    progress: bool = True,
    chunk_size: int = defaults.CHUNK_SIZE,
    event: threading.Event | None = None,
):

    if workers is None:
        export_serially(
            source,
            sink,
            progress=progress,
            chunk_size=chunk_size,
            event=event,
        )
    else:
        export_threaded(
            source=source,
            sink=sink,
            workers=workers,
            progress=progress,
            chunk_size=chunk_size,
            event=event,
        )
    return


if __name__ == "__main__":
    # We pill some configuration from the command line and execute export accordingly.
    pass
