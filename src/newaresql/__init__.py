import polars as pl

import newaresql.bdf
import newaresql.types
from newaresql.connector import Connector
from newaresql.data import TestData


def connect(
    host: str | None = None,
    port: int | str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> Connector:
    return Connector(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
    )


def list_tests(
    connector: Connector | None = None,
    host: str | None = None,
    port: int | str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> list[newaresql.types.Test]:
    if connector is None:
        with connect(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        ) as c:
            return list_tests(connector=c)
    return connector.list_tests()


def get_tests(
    connector: Connector | None = None,
    host: str | None = None,
    port: int | str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> pl.DataFrame:
    if connector is None:
        with connect(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        ) as c:
            return get_tests(
                connector=c,
            )
    return connector.get_tests()


def get_data(
    test: newaresql.types.Test,
    conditions: newaresql.types.Conditions | None = None,
    naming: newaresql.bdf.Naming = newaresql.bdf.DEFAULT,
    threaded: bool = False,
    #
    connector: Connector | None = None,
    host: str | None = None,
    port: int | str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> TestData:
    """
    Connector OR credentials must be provided to fetch the test data.
    A supplied connector takes precedence over the individual credentials.
    """
    if connector is None:
        with connect(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        ) as c:
            return get_data(
                test=test,
                conditions=conditions,
                naming=naming,
                threaded=threaded,
                connector=c,
            )
    return connector.get_data(
        test=test,
        conditions=conditions,
        naming=naming,
        threaded=threaded,
    )


__all__ = [
    "connect",
    "list_tests",
    "get_tests",
    "get_data",
    "Connector",
]
