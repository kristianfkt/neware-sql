import datetime
import pathlib
from typing import Iterator, Self

import polars as pl
import sqlalchemy as sa

import newaresql.defaults as defaults
from newaresql.bdf import convert
from newaresql.transformations import extend_data
from newaresql.types import Columns, Naming, Test, Where
from newaresql.utils import get_config

PYTYPES: dict[type[sa.types.TypeEngine], type] = {
    sa.types.Integer: int,
    sa.types.Float: float,
    sa.types.Numeric: float,
    sa.types.String: str,
    sa.types.Text: str,
    sa.types.DateTime: datetime.datetime,
    sa.types.Date: datetime.date,
    sa.types.Boolean: bool,
}


def to_pytype(sqltype: sa.types.TypeEngine) -> type:
    """
    Convert a SQLAlchemy type to a Python type.
    """

    affinity = sqltype._type_affinity
    if affinity is None:
        return object
    return PYTYPES.get(affinity, object)


def wrap_table(
    table: str,
    engine: sa.engine.Engine,
) -> sa.Table:
    """
    Wrap a table name as a SQLAlchemy Table object.
    """
    return sa.Table(table, sa.MetaData(), autoload_with=engine)


def select_table(
    table: sa.Table,
    columns: Columns | None = None,
) -> sa.Select:
    """
    Select a table with optionally specified columns.
    """

    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        stmt = sa.select(table)
    else:
        stmt = sa.select(*[table.c[col] for col in columns])
    return stmt


def build_predicate(
    table: sa.Table,
    where: Where,
) -> sa.ColumnElement:
    predicates = []
    for c, p in where.items():
        if isinstance(p, list):
            predicates.append(table.c[c].in_(p))
        elif isinstance(p, tuple) and len(p) == 2:
            lo, hi = p
            if (lo is not None) and (hi is not None):
                predicates.append(table.c[c].between(lo, hi))
            elif hi is not None:
                predicates.append(table.c[c] <= hi)
            elif lo is not None:
                predicates.append(table.c[c] >= lo)
        else:
            predicates.append(table.c[c] == p)
    return sa.and_(*predicates)


def wrap_query(query: str) -> sa.TextClause:
    """
    Wrap a query string as a SQLAlchemy TextClause object.
    """
    return sa.text(query)


def compile_statement(stmt: sa.Select, engine: sa.engine.Engine) -> str:
    """
    Compile a SQLAlchemy statement to a SQL string.
    """
    return str(stmt.compile(engine, compile_kwargs={"literal_binds": True}))


def make_select_stmt(
    table: str | sa.Table,
    engine: sa.engine.Engine,
    where: Where | None = None,
    columns: Columns | None = None,
) -> sa.Select:

    if isinstance(table, str):
        table = wrap_table(table, engine)
    stmt = select_table(table, columns=columns)
    if where is not None:
        stmt = stmt.where(build_predicate(table, where))
    return stmt


def get_table_schema(
    table: str | sa.Table,
    engine: sa.engine.Engine,
) -> dict[str, type]:
    """
    Fetch the schema of a table as a dictionary mapping column names to Python types.
    """
    if isinstance(table, str):
        table = wrap_table(table, engine)
    return {col.name: to_pytype(col.type) for col in table.columns}


def union_statements(
    *stmts: sa.Select,
) -> sa.CompoundSelect:
    """
    Union multiple SQLAlchemy statements into a single statement.
    """
    if not stmts:
        raise ValueError("At least one statement is required for union.")
    return sa.union_all(*stmts)


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
            url = sa.URL.create(url)

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

    def stream_query(
        self,
        query: str | sa.Selectable | sa.TextClause,
        chunk_size: int = defaults.CHUNK_SIZE,
        **kwargs,
    ) -> Iterator[pl.DataFrame]:
        """
        Passes the query and kwargs as-is to polars.read_database
        iter_batches and batch_size is explicitly set to True and chunk_size respectively in kwargs.
        """

        # Some explicit options
        kwargs["iter_batches"] = True
        kwargs["batch_size"] = chunk_size

        if isinstance(query, str):
            query = wrap_query(query)

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
        stmt = make_select_stmt(table, self._engine, columns=columns, where=where)
        return self.get_query(stmt)

    def stream_table(
        self,
        table: str | sa.Table,
        where: Where | None = None,
        columns: str | list[str] | None = None,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]:

        stmt = make_select_stmt(table, self._engine, columns=columns, where=where)
        yield from self.stream_query(stmt, chunk_size=chunk_size)

    def list_tables(self) -> list[str]:
        """
        Inspect the database and return a list of table names.
        """
        with self._engine.connect() as conn:
            return sa.inspect(conn).get_table_names()

    def get_tests(self) -> pl.DataFrame:
        """
        Retrieve tests from the database.
        """
        return self.get_table("tests")

    def list_tests(self) -> list[Test]:
        """
        Inspect the database and return a list of test names.
        """
        return self.get_tests().to_dicts()

    def get_stats(self, test: Test) -> dict:
        keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
        table = "_".join(["stats"] + [str(test[key]) for key in keys])
        query = f"""
        SELECT 
            MAX(seq_id) AS max_seq_id
        FROM {table} 
        """
        return self.get_query(query).to_dicts()[0]

    def get_main_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:

        keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
        table = "_".join(["main"] + [str(test[key]) for key in keys])

        data = convert(
            self.get_table(table, where=where, columns=columns), "bts", naming
        )
        if extend:
            data = extend_data(data)
        return data

    def get_aux_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:

        keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
        table = "_".join(["aux"] + [str(test[key]) for key in keys])

        data = convert(
            self.get_table(table, where=where, columns=columns), "bts", naming
        )
        if extend:
            data = extend_data(data)
        return data

    def get_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:
        main = self.get_main_data(test, where=where, naming="bts", extend=False)
        aux = self.get_aux_data(test, where=where, naming="bts", extend=False)
        data = convert(main.join(aux, on="seq_id", how="left"), "bts", naming)
        if extend:
            data = extend_data(data)
        return data

    def stream_main_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        chunk_size: int = defaults.CHUNK_SIZE,
        extend: bool = True,
    ) -> Iterator[pl.DataFrame]:

        keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
        table = "_".join(["main"] + [str(test[key]) for key in keys])
        for chunk in self.stream_table(
            table, where=where, columns=columns, chunk_size=chunk_size
        ):
            data = convert(chunk, "bts", naming)
            if extend:
                data = extend_data(data)
            yield data

    def stream_aux_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]:

        keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
        table = "_".join(["aux"] + [str(test[key]) for key in keys])

        for chunk in self.stream_table(
            table, where=where, columns=columns, chunk_size=chunk_size
        ):
            data = convert(chunk, "bts", naming)
            if extend:
                data = extend_data(data)
            yield data

    def stream_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]:
        # Need to change up the seq_id crap here

        if where is None:
            where = {}

        if "seq_id" not in where:
            N = 1
            M = self.get_stats(test)["max_seq_id"]
        elif ("seq_id" in where) and isinstance(where["seq_id"], tuple):
            lo, hi = where["seq_id"]
            if (not isinstance(lo, int)) and (lo is not None):
                raise ValueError("Start of seq_id must be an integer or None")
            if (not isinstance(hi, int)) and (hi is not None):
                raise ValueError("End of seq_id must be an integer or None")

            N = 1 if lo is None else lo
            M = self.get_stats(test)["max_seq_id"] if hi is None else hi

        I = list(range(N, M + 1, chunk_size))
        J = [min(i_ + chunk_size - 1, M) for i_ in I]
        for i, j in zip(I, J):
            where["seq_id"] = (i, j)
            chunk = self.get_data(test, where=where, naming=naming, extend=extend)
            if chunk.height == 0:
                return
            yield chunk
        return


class SQLiteConnector(SQLConnector):
    def __init__(self, path: str | pathlib.Path | None = None):
        path = get_config("NEWARE_SQLITE_PATH", value=path)
        if isinstance(path, str):
            path = pathlib.Path(path)

        url = sa.URL.create("sqlite+pysqlite", database=str(path))
        super().__init__(url=url)
