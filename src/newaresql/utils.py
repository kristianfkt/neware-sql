import json
import os
import pathlib
from typing import Any, Literal

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


class MissingConfigError(Exception):
    pass


def ensure_path(path: str | pathlib.Path, suffix: str | None = None) -> pathlib.Path:
    if isinstance(path, str):
        path = pathlib.Path(path)
    if (suffix is not None) and (not path.suffix == suffix):
        path = path.with_suffix(suffix)
    return path


def dump_dict(data: dict, path: str | pathlib.Path):
    """
    Dump a dictionary to a JSON file.
    """
    path = ensure_path(path, suffix=".json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return


def load_dict(path: str | pathlib.Path) -> dict:
    """
    Load a dictionary from a JSON file.
    """
    path = ensure_path(path, suffix=".json")
    with open(path, "r") as f:
        return json.load(f)


def dump_frame(
    data: pl.DataFrame,
    path: str | pathlib.Path,
    fmt: Literal["parquet", "csv", "feather", "ipc"] = "parquet",
):
    """
    Dump a polars DataFrame to a file in the specified format.
    """
    if fmt not in WRITE:
        raise ValueError(
            f"Unsupported format: {fmt}. Supported formats are: {list(WRITE.keys())}"
        )
    path = ensure_path(path, suffix=f".{fmt}")
    WRITE[fmt](data, path)
    return


def scan_frame(
    path: str | pathlib.Path,
    fmt: Literal["parquet", "csv", "feather", "ipc"] = "parquet",
) -> pl.LazyFrame:
    """
    Scan a polars DataFrame from a file in the specified format.
    """
    if fmt not in SCAN:
        raise ValueError(
            f"Unsupported format: {fmt}. Supported formats are: {list(SCAN.keys())}"
        )
    path = ensure_path(path).joinpath(f"*.{fmt}")
    return SCAN[fmt](path)


def collect_frame(lazy: pl.LazyFrame) -> pl.DataFrame:
    """
    Collect a lazy polars DataFrame into a DataFrame.
    """
    return lazy.collect(engine="streaming")


def load_frame(
    path: str | pathlib.Path,
    fmt: Literal["parquet", "csv", "feather", "ipc"] = "parquet",
) -> pl.DataFrame:
    """
    Load a polars DataFrame from a file in the specified format.
    """
    lazy = scan_frame(path, fmt=fmt)
    return collect_frame(lazy)


def get_config(key: str, value: Any | None = None, default: Any | None = None) -> Any:
    """
    Fetch a config value from the provided value, dictionary, or environment variable.
    """
    if value is None:
        value = os.getenv(f"NEWARE_{key.upper()}", default)
    if value is None:
        raise MissingConfigError(f"Missing config for key: {key}")

    return value


def test_name(test: dict):
    """
    Generate a unique name for a test based on its identifying attributes.
    """

    keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
    return "-".join([f"{test[k]}" for k in keys])
