from __future__ import annotations

from typing import Any, Generator

import polars as pl
import sqlalchemy as sa

from newaresql.core.orm import make_select_query


class BaseConnector:
    def __init__(
        self,
        config: dict | None = None,
    ):
        self._config = config or {}
        return

    @property
    def config(self) -> dict:
        return self._config

    def __enter__(self) -> BaseConnector:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return

    def get_query(
        self,
        query: str,
        schema: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError("get_query() must be implemented in subclasses")

    def stream_query(
        self,
        query: str,
        chunksize: int = 100_000,
        schema: dict[str, Any] | None = None,
    ) -> Generator[pl.DataFrame, None, None]:
        raise NotImplementedError("stream_query() must be implemented in subclasses")

    def delete_table(self, table: str) -> None:
        raise NotImplementedError("delete_table() must be implemented in subclasses")

    def write_table(
        self,
        data: pl.DataFrame,
        table: str,
        schema: dict[str, Any] | None = None,
        append: bool = True,
    ) -> None:
        raise NotImplementedError("write_table() must be implemented in subclasses")

    def read_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.DataFrame:
        raise NotImplementedError("read_table() must be implemented in subclasses")

    def scan_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.LazyFrame:
        raise NotImplementedError("scan_table() must be implemented in subclasses")

    def stream_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        chunksize: int = 100_000,
    ) -> Generator[pl.DataFrame, None, None]:
        raise NotImplementedError("stream_table() must be implemented in subclasses")

    def get_tests(self) -> pl.DataFrame:
        raise NotImplementedError("get_tests() must be implemented in subclasses")

    def list_tests(self) -> list[dict]:
        raise NotImplementedError("list_tests() must be implemented in subclasses")

    def get_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        extend: bool = True,
    ) -> pl.DataFrame:
        raise NotImplementedError("get_data() must be implemented in subclasses")

    def scan_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        extend: bool = True,
    ) -> pl.LazyFrame:
        raise NotImplementedError("scan_data() must be implemented in subclasses")

    def stream_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        chunksize: int = 100_000,
        extend: bool = True,
    ) -> Generator[pl.DataFrame, None, None]:
        raise NotImplementedError("stream_data() must be implemented in subclasses")

    def get_stats(self, test: dict) -> dict:
        raise NotImplementedError("get_stats() must be implemented in subclasses")


class SQLConnector(BaseConnector):
    def __init__(
        self,
        url: str,
    ):

        super().__init__(config={"url": url})
        self._engine = sa.create_engine(url)
        return

    @property
    def url(self) -> str:
        return self.config["url"]

    @property
    def engine(self) -> sa.engine.Engine:
        return self._engine

    def __enter__(self) -> SQLConnector:
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._engine.dispose()
        super().__exit__(exc_type, exc_value, traceback)
        return

    def list_tables(self) -> list[str]:
        with self._engine.connect() as conn:
            return sa.inspect(conn).get_table_names()

    def get_query(
        self,
        query: str,
        schema: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        schema = pl.Schema(schema) if schema is not None else None
        with self._engine.connect() as conn:
            return pl.read_database(query, conn, schema_overrides=schema)

    def stream_query(
        self,
        query: str,
        chunksize: int = 100_000,
        schema: dict[str, Any] | None = None,
    ) -> Generator[pl.DataFrame, None, None]:
        schema = pl.Schema(schema) if schema is not None else None
        with self._engine.connect().execution_options(
            stream_results=True, yield_per=chunksize
        ) as conn:
            yield from pl.read_database(
                query,
                conn,
                iter_batches=True,
                batch_size=chunksize,
                schema_overrides=schema,
            )

    def read_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        return self.get_query(
            make_select_query(table, columns=columns, where=where), schema=schema
        )

    def scan_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> pl.LazyFrame:
        return self.read_table(
            table, columns=columns, where=where, schema=schema
        ).lazy()

    def stream_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        chunksize: int = 100_000,
        schema: dict[str, Any] | None = None,
    ) -> Generator[pl.DataFrame, None, None]:
        query = make_select_query(table, columns=columns, where=where)
        yield from self.stream_query(query, chunksize=chunksize, schema=schema)
