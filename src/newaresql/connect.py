import datetime
import logging
from typing import Generator, Sequence

import polars as pl
import sqlalchemy as sa

from newaresql.schemas import get_data_schema

logger = logging.getLogger(__name__)


class Connector:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ):

        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._url = sa.URL.create(
            drivername="mysql+pymysql",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        self._engine = sa.create_engine(self._url)
        return

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def user(self) -> str:
        return self._user

    @property
    def database(self) -> str:
        return self._database

    @property
    def engine(self) -> sa.engine.Engine:
        return self._engine

    @property
    def url(self) -> sa.engine.URL:
        return self._url

    @property
    def tables(self) -> list[str]:
        with self._engine.connect() as conn:
            return sa.inspect(conn).get_table_names()

    @property
    def version(self) -> str:
        versions = (
            self.query("SELECT DISTINCT version FROM db_ver")
            .select("version")
            .to_series()
            .unique()
            .to_list()
        )
        if len(versions) != 1:
            raise ValueError(
                f"Expected one version, got {len(versions)} versions: {versions}"
            )
        return str(versions[0])

    @property
    def tests(self) -> list[dict]:
        return self.get_tests().to_dicts()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.dispose()
        return

    def compile_statement(self, stmt: sa.Selectable) -> str:
        """
        Compile a SQLAlchemy statement to a string.
        """
        return str(
            stmt.compile(
                dialect=self._engine.dialect,
                compile_kwargs={"literal_binds": True},
            )
        )

    def wrap_table(self, table: str) -> sa.Table:
        """
        Wrap a table name from the database in a SQLAlchemy Table object.
        """
        return sa.Table(table, sa.MetaData(), autoload_with=self._engine)

    def get_table_schema(self, table: str) -> dict[str, type]:
        """
        Get the schema of a table from the database as a dictionary mapping of column names to Python types.
        """
        _t = self.wrap_table(table)
        pytypes: dict[type[sa.types.TypeEngine], type] = {
            sa.types.Integer: int,
            sa.types.Float: float,
            sa.types.Numeric: float,
            sa.types.String: str,
            sa.types.Text: str,
            sa.types.DateTime: datetime.datetime,
            sa.types.Date: datetime.date,
            sa.types.Boolean: bool,
        }

        def aspytype(sqltype: sa.types.TypeEngine) -> type:
            return pytypes.get(sqltype._type_affinity, object)

        return {col.name: aspytype(col.type) for col in _t.columns}

    def select_table(
        self,
        table: str | sa.Table,
        columns: str | Sequence[str] | None = None,
        where: dict | None = None,
    ) -> sa.Selectable:
        """
        Select a table from the database, optionally filtering by columns and where conditions.

        where [str, condition]

            - equality: {col: value}
            - between: {col: (min, max)} inclusive
            - bigger than: {col: (min, None)} inclusive
            - smaller than: {col: (None, max)} inclusive
            - in list: {col: [value1, value2, ...]}
        """

        if isinstance(table, str):
            table = self.wrap_table(table)

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, Sequence):
            columns = list(columns)

        if columns is None:
            stmt = sa.select(table)
        else:
            stmt = sa.select(*(table.c[col] for col in columns))

        if where is not None:
            masks = []
            for col, pred in where.items():
                if isinstance(pred, tuple):
                    lo, hi = pred
                    if hi is None:
                        masks.append(table.c[col] >= lo)
                    elif lo is None:
                        masks.append(table.c[col] <= hi)
                    else:
                        masks.append(table.c[col].between(lo, hi))
                elif isinstance(pred, list):
                    masks.append(table.c[col].in_(pred))
                else:
                    masks.append(table.c[col] == pred)
            stmt = stmt.where(sa.and_(*masks))

        return stmt

    def select_union(
        self,
        *tables: str | sa.Table,
        columns: str | Sequence[str] | None = None,
        wheres: Sequence[dict | None] | dict | None = None,
    ) -> sa.Selectable:
        """
        A lot of tables share the same schema, and may be queried together.
        Wraps _select_table in a union_all.
        Union select will only include common columns across all tables, unless columns is specified, sorted alphabetically.
        """
        _t = [self.wrap_table(t) if isinstance(t, str) else t for t in tables]

        if wheres is None:
            wheres = [None] * len(_t)
        if isinstance(wheres, dict):
            wheres = [wheres] * len(_t)

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, Sequence):
            columns = list(columns)

        if len(_t) != len(wheres):
            raise ValueError(
                f"Number of tables ({len(_t)}) and wheres ({len(wheres)}) must match"
            )

        if len(_t) == 1:
            return self.select_table(_t[0], columns=columns, where=wheres[0])

        else:
            if columns is None:
                cols = set(_t[0].columns.keys())
                for wrap in _t[1:]:
                    cols = cols.intersection(set(wrap.columns.keys()))
                columns = list(sorted(cols))
            return sa.union_all(
                *[
                    self.select_table(wrap, columns=columns, where=where)
                    for wrap, where in zip(_t, wheres)
                ]
            )

    def make_main_statement(
        self,
        test: dict,
        where: dict | None = None,
        columns: str | Sequence[str] | None = None,
    ) -> sa.Selectable:
        """
        Make a SQLAlchemy statement to select main data for a test.
        """
        if test.get("main_second_table") is not None:
            stmt = self.select_union(
                test["main_first_table"],
                test["main_second_table"],
                columns=columns,
                wheres=[
                    {
                        **(where or {}),
                        "unit_id": test["unit_id"],
                        "chl_id": test["chl_id"],
                        "test_id": test["test_id"],
                    },
                    where,
                ],
            )
        else:
            stmt = self.select_table(
                test["main_first_table"],
                columns=columns,
                where={
                    **(where or {}),
                    "unit_id": test["unit_id"],
                    "chl_id": test["chl_id"],
                    "test_id": test["test_id"],
                },
            )
        return stmt

    def make_aux_statement(
        self,
        test: dict,
        where: dict | None = None,
        columns: str | Sequence[str] | None = None,
    ) -> sa.Selectable | None:
        """
        Make a SQLAlchemy statement to select auxiliary data for a test.
        """
        if (test.get("aux_first_table") is None) and (
            test.get("aux_second_table") is None
        ):
            stmt = None

        elif (test.get("aux_second_table") is not None) and (
            test.get("aux_first_table") is not None
        ):
            stmt = self.select_union(
                test["aux_first_table"],
                test["aux_second_table"],
                columns=columns,
                wheres=[
                    {
                        **(where or {}),
                        "unit_id": test["unit_id"],
                        "chl_id": test["chl_id"],
                        "test_id": test["test_id"],
                    },
                    where,
                ],
            )
        elif (test.get("aux_first_table") is not None) and (
            test.get("aux_second_table") is None
        ):
            stmt = self.select_table(
                test["aux_first_table"],
                columns=columns,
                where={
                    **(where or {}),
                    "unit_id": test["unit_id"],
                    "chl_id": test["chl_id"],
                    "test_id": test["test_id"],
                },
            )

        elif (test.get("aux_first_table") is None) and (
            test.get("aux_second_table") is not None
        ):
            stmt = self.select_table(
                test["aux_second_table"],
                columns=columns,
                where={
                    **(where or {}),
                    "unit_id": test["unit_id"],
                    "chl_id": test["chl_id"],
                    "test_id": test["test_id"],
                },
            )
        return stmt

    def make_main_query(
        self,
        test: dict,
        where: dict | None = None,
        columns: str | Sequence[str] | None = None,
    ) -> str:
        """
        Make a SQLAlchemy statement to select main data for a test.
        """
        stmt = self.make_main_statement(test, where=where, columns=columns)
        return self.compile_statement(stmt)

    def make_aux_query(
        self,
        test: dict,
        where: dict | None = None,
        columns: str | Sequence[str] | None = None,
    ) -> str | None:
        """
        Make a SQLAlchemy statement to select auxiliary data for a test.
        """
        stmt = self.make_aux_statement(test, where=where, columns=columns)
        if stmt is not None:
            return self.compile_statement(stmt)
        else:
            return None

    def query(
        self,
        query: str | sa.TextClause | sa.Selectable,
        schema: dict | None = None,
    ) -> pl.DataFrame:
        """
        Execute a query and return the results as a Polars DataFrame.
        explicit schema may be provided to override the inferred schema
        implements pl.read_database
        """

        with self._engine.connect() as conn:
            return pl.read_database(query, conn, schema_overrides=schema)

    def stream(
        self,
        query: str | sa.TextClause | sa.Selectable,
        schema: dict | None = None,
        chunksize: int = 100000,
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Execute a query and stream the results as Polars DataFrames in chunks.
        explicit schema may be provided to override the inferred schema
        implements pl.read_database

        """
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

    def get_table(
        self,
        table: str,
        columns: str | Sequence[str] | None = None,
        where: dict | None = None,
    ) -> pl.DataFrame:
        """
        Get a table from the database as a Polars DataFrame.
        """
        stmt = self.select_table(table, columns=columns, where=where)
        return self.query(stmt)

    def stream_table(
        self,
        table: str,
        columns: str | Sequence[str] | None = None,
        where: dict | None = None,
        chunksize: int = 100000,
    ) -> Generator[pl.DataFrame, None, None]:
        """
        Stream a table from the database as a Polars DataFrame.
        """
        stmt = self.select_table(table, columns=columns, where=where)
        yield from self.stream(stmt, chunksize=chunksize)

    def get_main_data(
        self,
        test: dict,
        where: dict | None = None,
        columns: str | Sequence[str] | None = None,
    ) -> pl.DataFrame:

        stmt = self.make_main_statement(test, where=where, columns=columns)

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, Sequence):
            columns = list(columns)
        schema = get_data_schema(self.version, test["dev_uid"])["main"]
        if columns is not None:
            schema = {k: v for k, v in schema.items() if k in columns}
        data = self.query(stmt, schema=schema)
        return data

    def get_aux_data(
        self,
        test: dict,
        where: dict | None = None,
        columns: str | Sequence[str] | None = None,
    ) -> pl.DataFrame | None:

        stmt = self.make_aux_statement(test, where=where, columns=columns)
        if stmt is None:
            return None

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, Sequence):
            columns = list(columns)
        schema = get_data_schema(self.version, test["dev_uid"])["aux"]
        if columns is not None:
            schema = {k: v for k, v in schema.items() if k in columns}
        data = self.query(stmt, schema=schema)
        return data

    def stream_main_data(
        self,
        test: dict,
        where: dict | None = None,
        columns: str | Sequence[str] | None = None,
        chunksize: int = 100000,
    ) -> Generator[pl.DataFrame, None, None]:

        stmt = self.make_main_statement(test, where=where, columns=columns)
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, Sequence):
            columns = list(columns)
        schema = get_data_schema(self.version, test["dev_uid"])["main"]
        if columns is not None:
            schema = {k: v for k, v in schema.items() if k in columns}
        yield from self.stream(stmt, chunksize=chunksize, schema=schema)

    def stream_aux_data(
        self,
        test: dict,
        where: dict | None = None,
        columns: str | Sequence[str] | None = None,
        chunksize: int = 100000,
    ) -> Generator[pl.DataFrame, None, None]:
        stmt = self.make_aux_statement(test, where=where, columns=columns)
        if stmt is None:
            return

        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, Sequence):
            columns = list(columns)
        schema = get_data_schema(self.version, test["dev_uid"])["aux"]
        if columns is not None:
            schema = {k: v for k, v in schema.items() if k in columns}
        yield from self.stream(stmt, chunksize=chunksize, schema=schema)

    def get_tests(self) -> pl.DataFrame:
        raise NotImplementedError("get_tests() must be implemented in subclasses")

    def dispose(self):
        self._engine.dispose()
        return


class Version0760Connector(Connector):
    def get_tests(self) -> pl.DataFrame:
        test = self.get_table("test")
        h_test = self.get_table("h_test")
        test_note = self.get_table("test_note")
        tests = (
            pl.concat([test, h_test], how="diagonal_relaxed")
            .join(test_note, on=["dev_uid", "unit_id", "chl_id", "test_id"], how="left")
            .sort("dev_uid", "unit_id", "chl_id", "test_id")
        )

        return tests


class Version0800Connector(Connector):
    def get_tests(self) -> pl.DataFrame:
        tables = [
            "test",
            *sorted(t for t in self.tables if t.startswith("h_test")),
        ]
        tests = pl.concat(
            [self.get_table(table) for table in tables],
            how="diagonal_relaxed",
        ).sort("dev_uid", "unit_id", "chl_id", "test_id")

        return tests


CONNECTORS = {
    "0760": Version0760Connector,
    "0800": Version0800Connector,
}


def connect(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> Connector:

    with Connector(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    ) as conn:
        version = conn.version
        if not version:
            raise ValueError("Failed to determine BTS version")
        if version not in CONNECTORS:
            raise ValueError(f"Unsupported BTS version: {version}")

    return CONNECTORS[version](
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )
