import logging
from typing import Iterator, Self

import polars as pl
import sqlalchemy as sa

import newaresql.defaults as defaults
from newaresql.types import Where
from newaresql.utils.sql import make_select_stmt, wrap_query

logger = logging.getLogger(__name__)


class SQLConnector:
    """
    Generic SQL connector class - wrapping everything supported by SQLAlchemy. All SQL connectors should inherit from this class.

    """

    def __init__(
        self,
        *,
        url: str | sa.URL,
    ):
        """
        url is non-optional
        """
        if isinstance(url, str):
            url = sa.make_url(url)

        self._engine = sa.create_engine(url)
        return

    @property
    def url(self) -> sa.URL:
        return self._engine.url

    @property
    def engine(self) -> sa.engine.Engine:
        return self._engine

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._engine.dispose()
        return

    def list_tables(self) -> list[str]:
        """
        Inspect the database and return a list of table names.
        """
        with self._engine.connect() as conn:
            return sa.inspect(conn).get_table_names()

    def get_query(
        self,
        query: str | sa.Selectable | sa.TextClause,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Passes the query and kwargs as-is to polars.read_database.
        """
        if isinstance(query, str):
            query = wrap_query(query)
        with self._engine.connect() as conn:
            return pl.read_database(query, conn, **kwargs)

    def chunk_query(
        self,
        query: str | sa.Selectable | sa.TextClause,
        chunk_size: int = defaults.CHUNK_SIZE,
        **kwargs,
    ) -> Iterator[pl.DataFrame]:
        """
        Passes the query and kwargs as-is to polars.read_database
        iter_batches and batch_size is explicitly set to True and chunk_size respectively in kwargs.
        """
        if isinstance(query, str):
            query = wrap_query(query)

        # Some explicit options
        if "iter_batches" in kwargs:
            logger.warning("Overriding option 'iter_batches' in chunk_query")
        if "batch_size" in kwargs:
            logger.warning("Overriding option 'batch_size' in chunk_query")
        kwargs["iter_batches"] = True
        kwargs["batch_size"] = chunk_size

        with self._engine.connect().execution_options(
            stream_results=True, yield_per=chunk_size
        ) as conn:
            yield from pl.read_database(query, conn, **kwargs)

    def get_table(
        self,
        table: str | sa.Table,
        where: Where | None = None,
        columns: str | list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Retrieve the specified table from the database as a Polars DataFrame.
        The `where` parameter can be used to filter the data based on specific conditions.
        The `columns` parameter allows selecting specific columns to retrieve.
        """
        stmt = make_select_stmt(table, self._engine, columns=columns, where=where)
        return self.get_query(stmt)

    def chunk_table(
        self,
        table: str | sa.Table,
        where: Where | None = None,
        columns: str | list[str] | None = None,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]:
        """
        Retrieve the specified table from the database in chunks as an iterator of Polars DataFrames.
        The `where` parameter can be used to filter the data based on specific conditions.
        The `columns` parameter allows selecting specific columns to retrieve.
        The `chunk_size` parameter specifies the number of rows per chunk.
        """
        stmt = make_select_stmt(table, self._engine, columns=columns, where=where)
        yield from self.chunk_query(stmt, chunk_size=chunk_size)

    def get_distinct(
        self,
        table: str | sa.Table,
        columns: list[str],
        where: Where | None = None,
    ) -> list[dict]:
        """
        Get distinct combinations of certain columns
        """
        stmt = make_select_stmt(
            table, self._engine, columns=columns, where=where
        ).distinct()
        return self.get_query(stmt).to_dicts()
