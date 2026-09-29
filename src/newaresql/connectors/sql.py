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


#     def get_tests(self) -> pl.DataFrame:
#         """
#         Retrieve the "tests" table from the database as a Polars DataFrame.
#         """
#         return self.get_table("tests")

#     def list_tests(self) -> list[Test]:
#         """
#         Inspect the database and return a list of tests as dictionaries.
#         """
#         return self.get_tests().to_dicts()

#     def get_stats(self, test: Test) -> dict:
#         """
#         Retrieve statistics for the specified test from the database as a dictionary.
#         """
#         keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
#         table = "_".join(["stats"] + [str(test[key]) for key in keys])
#         query = f"""
#         SELECT
#             MAX(seq_id) AS max_seq_id
#         FROM {table}
#         """
#         return self.get_query(query).to_dicts()[0]

#     def get_main_data(
#         self,
#         test: Test,
#         *,
#         where: Where | None = None,
#         columns: Columns | None = None,
#         naming: Naming = defaults.NAMING,
#         extend: bool = True,
#     ) -> pl.DataFrame:
#         """
#         Retrieve the main data for the specified test from the database as a Polars DataFrame.
#         The `where` parameter can be used to filter the data based on specific conditions.
#         The `columns` parameter allows selecting specific columns to retrieve.
#         The `naming` parameter specifies the naming convention for the columns.
#         The `extend` parameter determines whether to extend the data with additional computed columns.
#         """

#         keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
#         table = "_".join(["main"] + [str(test[key]) for key in keys])

#         data = convert(
#             self.get_table(table, where=where, columns=columns), "bts", naming
#         )
#         if extend:
#             data = extend_data(data)
#         return data

#     def get_aux_data(
#         self,
#         test: Test,
#         *,
#         where: Where | None = None,
#         columns: Columns | None = None,
#         naming: Naming = defaults.NAMING,
#         extend: bool = True,
#     ) -> pl.DataFrame:
#         """
#         Retrieve the auxiliary data for the specified test from the database as a Polars DataFrame.
#         The `where` parameter can be used to filter the data based on specific conditions.
#         The `columns` parameter allows selecting specific columns to retrieve.
#         The `naming` parameter specifies the naming convention for the columns.
#         The `extend` parameter determines whether to extend the data with additional computed columns.
#         """

#         keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
#         table = "_".join(["aux"] + [str(test[key]) for key in keys])

#         data = convert(
#             self.get_table(table, where=where, columns=columns), "bts", naming
#         )
#         if extend:
#             data = extend_data(data)
#         return data

#     def get_data(
#         self,
#         test: Test,
#         *,
#         where: Where | None = None,
#         naming: Naming = defaults.NAMING,
#         extend: bool = True,
#     ) -> pl.DataFrame:
#         """
#         Retrieve the combined main and auxiliary data for the specified test from the database as a Polars DataFrame.
#         The `where` parameter can be used to filter the data based on specific conditions.
#         The `naming` parameter specifies the naming convention for the columns.
#         The `extend` parameter determines whether to extend the data with additional computed columns.
#         """

#         main = self.get_main_data(test, where=where, naming="bts", extend=False)
#         aux = self.get_aux_data(test, where=where, naming="bts", extend=False)
#         data = convert(main.join(aux, on="seq_id", how="left"), "bts", naming)
#         if extend:
#             data = extend_data(data)
#         return data

#     def stream_main_data(
#         self,
#         test: Test,
#         *,
#         where: Where | None = None,
#         columns: Columns | None = None,
#         naming: Naming = defaults.NAMING,
#         chunk_size: int = defaults.CHUNK_SIZE,
#         extend: bool = True,
#     ) -> Iterator[pl.DataFrame]:
#         """
#         Stream the main data for the specified test from the database as an iterator of Polars DataFrames.
#         The `where` parameter can be used to filter the data based on specific conditions.
#         The `columns` parameter allows selecting specific columns to retrieve.
#         The `naming` parameter specifies the naming convention for the columns.
#         The `chunk_size` parameter determines the number of rows per chunk.
#         The `extend` parameter determines whether to extend the data with additional computed columns.
#         """
#         keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
#         table = "_".join(["main"] + [str(test[key]) for key in keys])
#         for chunk in self.stream_table(
#             table, where=where, columns=columns, chunk_size=chunk_size
#         ):
#             data = convert(chunk, "bts", naming)
#             if extend:
#                 data = extend_data(data)
#             yield data

#     def stream_aux_data(
#         self,
#         test: Test,
#         *,
#         where: Where | None = None,
#         columns: Columns | None = None,
#         naming: Naming = defaults.NAMING,
#         extend: bool = True,
#         chunk_size: int = defaults.CHUNK_SIZE,
#     ) -> Iterator[pl.DataFrame]:
#         """
#         Stream the auxiliary data for the specified test from the database as an iterator of Polars DataFrames.
#         The `where` parameter can be used to filter the data based on specific conditions.
#         The `columns` parameter allows selecting specific columns to retrieve.
#         The `naming` parameter specifies the naming convention for the columns.
#         The `chunk_size` parameter determines the number of rows per chunk.
#         The `extend` parameter determines whether to extend the data with additional computed columns.
#         """

#         keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
#         table = "_".join(["aux"] + [str(test[key]) for key in keys])

#         for chunk in self.stream_table(
#             table, where=where, columns=columns, chunk_size=chunk_size
#         ):
#             data = convert(chunk, "bts", naming)
#             if extend:
#                 data = extend_data(data)
#             yield data

#     def stream_data(
#         self,
#         test: Test,
#         *,
#         where: Where | None = None,
#         naming: Naming = defaults.NAMING,
#         extend: bool = True,
#         chunk_size: int = defaults.CHUNK_SIZE,
#     ) -> Iterator[pl.DataFrame]:
#         """
#         Stream the combined main and auxiliary data for the specified test from the database as an iterator of Polars DataFrames.
#         Implemented for compatability reasons; it streams the data in chunks based on the `seq_id` column.
#         The `where` parameter can be used to filter the data based on specific conditions.
#         The `naming` parameter specifies the naming convention for the columns.
#         The `chunk_size` parameter determines the number of rows per chunk.
#         The `extend` parameter determines whether to extend the data with additional computed columns.
#         """
#         # Need to change up the seq_id crap here

#         if where is None:
#             where = {}

#         if "seq_id" not in where:
#             N = 1
#             M = self.get_stats(test)["max_seq_id"]
#         elif ("seq_id" in where) and isinstance(where["seq_id"], tuple):
#             lo, hi = where["seq_id"]
#             if (not isinstance(lo, int)) and (lo is not None):
#                 raise ValueError("Start of seq_id must be an integer or None")
#             if (not isinstance(hi, int)) and (hi is not None):
#                 raise ValueError("End of seq_id must be an integer or None")

#             N = 1 if lo is None else lo
#             M = self.get_stats(test)["max_seq_id"] if hi is None else hi

#         I = list(range(N, M + 1, chunk_size))
#         J = [min(i_ + chunk_size - 1, M) for i_ in I]
#         for i, j in zip(I, J):
#             where["seq_id"] = (i, j)
#             chunk = self.get_data(test, where=where, naming=naming, extend=extend)
#             if chunk.height == 0:
#                 return
#             yield chunk
#         return


# class SQLiteConnector(SQLConnector):
#     """
#     SQLite connector for the Neware SQL database.
#     Uses SQLite as the underlying database engine.
#     """

#     def __init__(self, path: str | pathlib.Path | None = None):
#         path = get_config("NEWARE_SQLITE_PATH", value=path)
#         if isinstance(path, str):
#             path = pathlib.Path(path)

#         url = sa.URL.create("sqlite+pysqlite", database=str(path))
#         super().__init__(url=url)




