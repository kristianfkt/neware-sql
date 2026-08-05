import json
import os
import pathlib
from typing import Any

import polars as pl

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


def test_name(test: dict):
    """
    Generate a unique name for a test based on its identifying attributes.
    """

    keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
    return "-".join([f"{test[k]}" for k in keys])


def load_json(path: pathlib.Path) -> dict:
    """
    Load a JSON file and return its contents as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(data: dict, path: pathlib.Path) -> None:
    """
    Dump a dictionary to a JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return


def filter_lazyframe(
    frame: pl.LazyFrame,
    columns: str | list[str] | None = None,
    where: dict[str, tuple | list] | None = None,
) -> pl.LazyFrame:
    if isinstance(columns, str):
        columns = [columns]
    if columns is not None:
        frame = frame.select(*columns)
    if where is None:
        where = {}
    for col, val in where.items():
        if isinstance(val, list):
            frame = frame.filter(pl.col(col).is_in(val))
        elif isinstance(val, tuple) & (len(val) == 2):
            lo, hi = val
            if (lo is not None) and (hi is not None):
                frame = frame.filter(pl.col(col).is_between(lo, hi))
            elif hi is not None:
                frame = frame.filter(pl.col(col) <= hi)
            elif lo is not None:
                frame = frame.filter(pl.col(col) >= lo)
        else:
            frame = frame.filter(pl.col(col) == val)
    return frame


def get_config(
    key: str, value: Any | None = None, default: Any | None = None
) -> Any | None:
    """
    Get a configuration value from the environment or a default.
    """
    if value is None:
        value = os.getenv(f"NEWARE_{key.upper()}", default)
    if value is None:
        raise ValueError(
            f"Configuration value for {key} is not set and no default is provided."
        )
    return value
