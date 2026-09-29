import concurrent.futures
import threading

import tqdm.auto as tqdm

from newaresql.clone.sink import FileSink
from newaresql.clone.source import BTSSource
from newaresql.clone.tables import TableType, get_type
from newaresql.protocols import CloneSink, CloneSource

SINKS = {"file": FileSink}
SOURCES = {"bts": BTSSource}
Sink = CloneSink | FileSink
Source = CloneSource | BTSSource


def check_event(event: threading.Event | None) -> bool:
    return event.is_set() if event is not None else False


def clone_combo(
    table: str,
    combo: dict | None,
    source: Source,
    sink: Sink,
    event: threading.Event | None = None,
):
    source_index = source.get_max_seq_id(table, combo=combo)
    if source_index is None:
        return

    sink_index = sink.get_max_seq_id(table, combo=combo)
    if sink_index is None:
        sink_index = 0

    if source_index < sink_index:
        raise ValueError("Source index is less than sink index")
    if source_index == sink_index:
        return

    if combo is None:
        combo = {}

    for chunk in source.chunk_table(
        table, where={**combo, "seq_id": (sink_index + 1, None)}
    ):
        sink.update_data(table, chunk)
        if check_event(event):
            return
    return


def clone_data(
    table: str, source: Source, sink: Sink, event: threading.Event | None = None
):
    combos = source.get_combinations(table)
    if combos is None:
        combos = [None]
    for combo in combos:
        clone_combo(table, combo, source, sink, event=event)
        if check_event(event):
            return
    return


def clone_meta(table: str, source: Source, sink: Sink):
    meta = source.get_table(table)
    sink.update_meta(table, meta)
    return


def clone_table(
    table: str, source: Source, sink: Sink, event: threading.Event | None = None
):
    """
    Clone a single table from the source to the sink.

    Parameters:
    - table: The name of the table to clone.
    - source: The source from which to clone the table.
    - sink: The sink to which to clone the table.
    """
    table_type = get_type(table)
    if table_type == TableType.META:
        clone_meta(table, source, sink)
    elif table_type == TableType.OTHER:
        pass
    else:
        clone_data(table, source, sink, event=event)
    return


def clone_serially(
    source: Source,
    sink: Sink,
    progress: bool = True,
    event: threading.Event | None = None,
):
    """
    Clone all tables from the source to the sink serially.

    Parameters:
    - source: The source from which to clone tables.
    - sink: The sink to which to clone tables.
    - progress: Whether to display a progress bar. Defaults to True.
    - event: An optional threading.Event to allow early termination.
    """
    for table in tqdm.tqdm(source.list_tables(), disable=not progress):
        clone_table(table, source, sink, event=event)
        if check_event(event):
            return


def clone_threaded(
    source: Source,
    sink: Sink,
    workers: int = -1,
    progress: bool = True,
    event: threading.Event | None = None,
):
    """
    Clone all tables from the source to the sink using multiple threads.

    Parameters:
    - source: The source from which to clone tables.
    - sink: The sink to which to clone tables.
    - workers: The number of worker threads to use. If -1, uses the default number of threads.
    - progress: Whether to display a progress bar. Defaults to True.
    - event: An optional threading.Event to allow early termination. Now it supports stopping the cloning process early if the event is set.
    """
    if workers == 0:
        raise ValueError("Number of workers cannot be zero.")
    if workers < -1:
        raise ValueError("Number of workers cannot be less than -1.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(clone_table, table, source, sink, event=event)
            for table in source.list_tables()
        ]
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            disable=not progress,
        ):
            _ = future.result()
        return


def clone(
    *,
    sink: str | Sink,
    source: str | Source,
    workers: int | None = None,
    progress: bool = True,
    event: threading.Event | None = None,
):

    if isinstance(sink, str):
        sink = SINKS[sink]()
    if isinstance(source, str):
        source = SOURCES[source]()

    if workers is None:
        clone_serially(source, sink, progress=progress, event=event)
    else:
        clone_threaded(source, sink, workers=workers, progress=progress, event=event)
    return


if __name__ == "__main__":
    # Actually - we should pull some stuff from config
    # and execute clone as a script from the command line.
    pass
