from pathlib import Path

import polars as pl

from newaresql.defaults import FILE_FORMAT
from newaresql.types import FileFormat

WRITE = {
    "parquet": pl.DataFrame.write_parquet,
    "csv": pl.DataFrame.write_csv,
    "feather": pl.DataFrame.write_ipc,
    "ipc": pl.DataFrame.write_ipc,
}

SCAN = {
    "parquet": pl.scan_parquet,
    "csv": pl.scan_csv,
    "feather": pl.scan_ipc,
    "ipc": pl.scan_ipc,
}


def scan(path: str | Path, file_format: FileFormat = FILE_FORMAT, **kwargs):
    return SCAN[file_format](path, **kwargs)


def write(
    data: pl.DataFrame,
    path: str | Path,
    file_format: FileFormat = FILE_FORMAT,
    **kwargs,
):
    return WRITE[file_format](data, path, **kwargs)
