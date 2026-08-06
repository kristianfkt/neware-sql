from __future__ import annotations

import pathlib
import uuid
from typing import Any, Generator

import polars as pl
import sqlalchemy as sa

import newaresql.core.utils as utils
from newaresql.core.connector import BaseConnector, SQLConnector


class SQLiteConnector(SQLConnector):
    def __init__(
        self,
        path: str | pathlib.Path | None = None,
    ):
        path = utils.get_config("sqlite_path", value=path)
        super().__init__(url=f"sqlite+pysqlite:///{path}")
        return

    def __enter__(self) -> SQLiteConnector:
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().__exit__(exc_type, exc_value, traceback)
        return

    def delete_table(self, table: str) -> None:
        sa.Table(table, sa.MetaData()).drop(self._engine, checkfirst=True)
        return

    def write_table(self, data: pl.DataFrame, table: str, append: bool = True) -> None:
        with self._engine.connect() as conn:
            data.write_database(
                table,
                conn,
                if_table_exists="append" if append else "replace",
            )
        return

    def get_tests(self) -> pl.DataFrame:
        if "tests" not in self.list_tables():
            tests = pl.DataFrame()
        else:
            tests = self.read_table("tests")
        return tests

    def list_tests(self) -> list[dict]:
        return self.get_tests().to_dicts()

    def get_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.DataFrame:
        name = utils.test_name(test)
        return self.read_table(name, columns=columns, where=where)

    def stream_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        chunksize: int = 100_000,
    ) -> Generator[pl.DataFrame, None, None]:
        name = utils.test_name(test)
        yield from self.stream_table(
            name, columns=columns, where=where, chunksize=chunksize
        )

    def scan_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.LazyFrame:
        name = utils.test_name(test)
        return self.scan_table(name, columns=columns, where=where)

    def get_stats(self, test: dict) -> dict:
        stats = {}
        name = utils.test_name(test)
        if name in self.list_tables():
            stats["seq_id"] = (
                self.get_query(
                    f"SELECT MAX(Record Count / 1) as max_seq_id FROM {name}"
                )
                .select("max_seq_id")
                .to_series()
                .item()
            )
        else:
            stats["seq_id"] = None
        return stats


class FileConnector(BaseConnector):
    def __init__(
        self,
        path: str | pathlib.Path | None = None,
        format: str | None = None,
    ):
        if path is None:
            path = utils.get_config("file_path", value=path)

        if format is None:
            format = utils.get_config("file_format", value=format, default="parquet")
        if not (path and format):
            raise ValueError("Both 'path' and 'format' must be specified")
        if format not in utils.WRITE:
            raise ValueError(f"Unsupported format: {format}")

        super().__init__(config={"path": pathlib.Path(path), "format": format})
        return

    def __enter__(self) -> FileConnector:
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().__exit__(exc_type, exc_value, traceback)
        return

    @property
    def path(self) -> pathlib.Path:
        return self.config["path"]

    @property
    def format(self) -> str:
        return self.config["format"]

    def delete_table(self, table: str) -> None:
        path = self.path.joinpath(table)
        for f in path.glob(f"*.{self.format}"):
            f.unlink()
        if not any(path.iterdir()):
            path.rmdir()
        return

    def write_table(self, data: pl.DataFrame, table: str, append: bool = True) -> None:
        path = self.path.joinpath(table)
        if not append:
            self.delete_table(table)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        utils.WRITE[self.format](data, path.joinpath(f"{uuid.uuid4()}.{self.format}"))
        return

    def scan_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.LazyFrame:
        path = self.path.joinpath(table)
        lazy = utils.SCAN[self.format](path.joinpath(f"*.{self.format}"))
        return utils.filter_lazyframe(lazy, columns=columns, where=where)

    def read_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.DataFrame:
        return self.scan_table(table, columns=columns, where=where).collect(
            engine="streaming"
        )

    def stream_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        chunksize: int = 100_000,
    ) -> Generator[pl.DataFrame, None, None]:
        lazy = self.scan_table(table, columns=columns, where=where)
        yield from lazy.collect_batches(chunk_size=chunksize, maintain_order=True)

    def list_tests(self) -> list[dict]:
        files = self.path.rglob("*test.json")
        return [utils.load_json(f) for f in files]

    def get_tests(self) -> pl.DataFrame:
        return pl.DataFrame(self.list_tests())

    def get_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.DataFrame:
        table = utils.test_name(test)
        return self.read_table(table, columns=columns, where=where)

    def scan_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.LazyFrame:
        table = utils.test_name(test)
        return self.scan_table(table, columns=columns, where=where)

    def stream_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        chunksize: int = 100_000,
    ) -> Generator[pl.DataFrame, None, None]:
        lazy = self.scan_data(test, columns=columns, where=where)
        yield from lazy.collect_batches(chunk_size=chunksize, maintain_order=True)

    def get_stats(self, test: dict) -> dict:
        stats = {}

        name = utils.test_name(test)
        path = self.path.joinpath(name)
        if (path.exists()) and any(path.glob(f"*.{self.format}")):
            stats["seq_id"] = (
                self.scan_data(test)
                .select("Record Count / 1")
                .max()
                .collect(engine="streaming")
                .select("Record Count / 1")
                .to_series()
                .item()
            )
        else:
            stats["seq_id"] = None
        return stats
