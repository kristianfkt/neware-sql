from functools import contextmanager
from typing import Generator

import polars as pl
import sqlalchemy as sa

from newaresql.bts import BTSVersion


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
    def version(self) -> BTSVersion:
        version = self.query("SELECT version from btsver").to_series().first()
        return BTSVersion(version)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._engine.dispose()
        return

    def query(self, query: str | sa.TextClause | sa.Selectable) -> pl.DataFrame:
        with self._engine.connect() as conn:
            return pl.read_database(query, conn)

    def stream(
        self, query: str | sa.TextClause | sa.Selectable, chunsize: int = 1
    ) -> Generator[pl.DataFrame, None, None]:
        with self._engine.connect() as conn:
            yield from pl.read_database(
                query,
                conn,
                iter_batches=True,
                batch_size=chunsize,
                execute_options={"stream_results": True},
            )


class BTS63Connector(Connector):
    pass


class BTS84Connector(Connector):
    pass


CONNECTOR_MAPPING = {
    BTSVersion.BTS63: BTS63Connector,
    BTSVersion.BTS84: BTS84Connector,
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
    return CONNECTOR_MAPPING[BTSVersion(version)](
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


@contextmanager
def connector(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> Generator[Connector, None, None]:

    with make(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    ) as conn:
        yield conn
