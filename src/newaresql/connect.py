from contextlib import contextmanager
from typing import Generator

import polars as pl
import sqlalchemy as sa

from newaresql.transform import transform


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
        self._engine = sa.create_engine(
            f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        )
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
    def tables(self) -> list[str]:
        with self._engine.connect() as conn:
            return sa.inspect(conn).get_table_names()

    @property
    def version(self) -> str:
        version = (
            self.query("SELECT DISTINCT version FROM db_ver")
            .select("version")
            .to_series()
        )
        if version.n_unique() != 1:
            raise ValueError(
                f"Expected one version, got {version.n_unique()} versions: {version.unique()}"
            )
        return str(version.first())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._engine.dispose()
        return

    def _wrap_table(self, table: str) -> sa.Table:
        return sa.Table(table, sa.MetaData(), autoload_with=self._engine)

    def _select_table(
        self,
        table: str | sa.Table,
        columns: list[str] | None = None,
        where: dict | None = None,
    ) -> sa.Selectable:

        if isinstance(table, str):
            table = self._wrap_table(table)

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
                else:
                    masks.append(table.c[col] == pred)
            stmt = stmt.where(sa.and_(*masks))

        return stmt

    def _select_union(
        self,
        *tables: str | sa.Table | None,
        columns: list[str] | None = None,
        where: dict | None = None,
    ) -> sa.Selectable | None:
        _t = [
            self._wrap_table(t) if isinstance(t, str) else t
            for t in tables
            if t is not None
        ]
        if len(_t) == 0:
            return None

        if len(_t) == 1:
            return self._select_table(_t[0], columns=columns, where=where)
        else:
            if columns is None:
                cols = set(_t[0].columns.keys())
                for wrap in _t[1:]:
                    cols = cols.intersection(set(wrap.columns.keys()))
                columns = list(sorted(cols))
            return sa.union_all(
                *[self._select_table(wrap, columns=columns, where=where) for wrap in _t]
            )

    def query(self, query: str | sa.TextClause | sa.Selectable) -> pl.DataFrame:
        with self._engine.connect() as conn:
            return pl.read_database(query, conn)

    def stream(
        self, query: str | sa.TextClause | sa.Selectable, chunksize: int = 1
    ) -> Generator[pl.DataFrame, None, None]:
        with self._engine.connect() as conn:
            yield from pl.read_database(
                query,
                conn.execution_options(stream_results=True),
                iter_batches=True,
                batch_size=chunksize,
            )

    def get_tests(self) -> pl.DataFrame:
        raise NotImplementedError("get_tests() must be implemented in subclasses")

    def get_data(self, test: dict, where: dict | None = None) -> pl.DataFrame:

        where = {
            **(where or {}),
            "unit_id": test["unit_id"],
            "chl_id": test["chl_id"],
            "test_id": test["test_id"],
        }

        main = self._select_union(
            test.get("main_first_table"),
            test.get("main_second_table"),
            where=where,
        )
        aux = self._select_union(
            test.get("aux_first_table"),
            test.get("aux_second_table"),
            where=where,
            columns=[
                "unit_id",
                "chl_id",
                "test_id",
                "seq_id",
                "test_tmp",
                "auxchl_id",
            ],
        )

        if main is None:
            raise ValueError("Expected at least one main table")
        if aux is None:
            stmt = main
        else:
            _main = main.subquery("main")  # ty:ignore[unresolved-attribute]
            _aux = aux.subquery("aux")  # ty:ignore[unresolved-attribute]

            on = (
                (_main.c["unit_id"] == _aux.c["unit_id"])
                & (_main.c["chl_id"] == _aux.c["chl_id"])
                & (_main.c["test_id"] == _aux.c["test_id"])
                & (_main.c["seq_id"] == _aux.c["seq_id"])
            )

            aux_cols = ["test_tmp", "auxchl_id"]
            main_cols = [c for c in _main.c.keys() if c not in aux_cols]
            stmt = sa.select(
                *[_main.c[c] for c in main_cols], *[_aux.c[c] for c in aux_cols]
            ).select_from(_main.outerjoin(_aux, on))

        return transform(self.query(stmt), test)


class Version0760Connector(Connector):
    def get_tests(self) -> pl.DataFrame:
        test = self.query("SELECT * FROM test")
        h_test = self.query("SELECT * FROM h_test")
        test_note = self.query("SELECT * FROM test_note")
        return (
            pl.concat([test, h_test], how="diagonal_relaxed")
            .join(test_note, on=["dev_uid", "unit_id", "chl_id", "test_id"], how="left")
            .sort("dev_uid", "unit_id", "chl_id", "test_id")
        )


class Version0800Connector(Connector):
    def get_tests(self) -> pl.DataFrame:
        tables = list(filter(lambda t: t.startswith("h_test"), self.tables))
        tables.insert(0, "test")
        return pl.concat(
            [self.query(f"SELECT * FROM {table}") for table in tables],
            how="diagonal_relaxed",
        ).sort("dev_uid", "unit_id", "chl_id", "test_id")


CONNECTOR_MAPPING = {
    "0760": Version0760Connector,
    "0800": Version0800Connector,
}


def make(
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
        if version not in CONNECTOR_MAPPING:
            raise ValueError(f"Unsupported BTS version: {version}")

    return CONNECTOR_MAPPING[version](
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


@contextmanager
def connect(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> Generator[Connector, None, None]:
    connector = make(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )
    try:
        yield connector
    finally:
        connector._engine.dispose()
