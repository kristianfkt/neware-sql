import pathlib
from typing import Literal, overload

import polars as pl

import newaresql.utils as utils


@overload
def get_config(
    key: Literal["root"],
    value: str | pathlib.Path | None = None,
    default: None = None,
) -> pathlib.Path: ...


@overload
def get_config(
    key: Literal["fmt"],
    value: Literal["parquet", "csv", "feather", "ipc"] | None = None,
    default: Literal["parquet"] = "parquet",
) -> Literal["parquet", "csv", "feather", "ipc"]: ...


def get_config(
    key: Literal["root", "fmt"],
    value: str
    | pathlib.Path
    | Literal["parquet", "csv", "feather", "ipc"]
    | None = None,
    default: str
    | pathlib.Path
    | Literal["parquet", "csv", "feather", "ipc"]
    | None = None,
) -> pathlib.Path | Literal["parquet", "csv", "feather", "ipc"]:

    value = utils.get_config(key, value=value, default=default)
    if key == "root":
        value = pathlib.Path(value)
    return value


class Connector:
    def __init__(
        self,
        root: str | pathlib.Path | None = None,
        fmt: Literal["parquet", "csv", "feather", "ipc"] | None = None,
    ):
        self._root = get_config("root", value=root)
        self._fmt = get_config("fmt", value=fmt, default="parquet")
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    @property
    def root(self) -> pathlib.Path:
        return self._root

    @property
    def fmt(self) -> Literal["parquet", "csv", "feather", "ipc"]:
        return self._fmt

    def list_tests(self) -> list[dict]:
        files = list(self.root.rglob("*.json"))
        return [utils.load_dict(f) for f in files]

    def get_data(self, test: dict) -> pl.LazyFrame | pl.DataFrame:
        name = utils.test_name(test)
        path = self.root.joinpath(name)
        lazy = utils.scan_frame(path, fmt=self.fmt)
        return utils.collect_frame(lazy)

    def scan_data(self, test: dict) -> pl.LazyFrame:
        name = utils.test_name(test)
        path = self.root.joinpath(name)
        return utils.scan_frame(path, fmt=self.fmt)
