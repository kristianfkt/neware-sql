import polars as pl

from newaresql.connector import Connector
from newaresql.defaults import JOIN, NAMING, TRANSFORM
from newaresql.types import Conditions, Join, Naming, Test


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
) -> list[Test]:
    if connector is None:
        with connect(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        ) as connector:
            return list_tests(
                connector=connector,
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
            )
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
        ) as connector:
            return get_tests(
                connector=connector,
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
            )
    return connector.get_tests()


def get_data(
    test: Test,
    connector: Connector | None = None,
    host: str | None = None,
    port: int | str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
    conditions: Conditions | None = None,
    transform: bool = TRANSFORM,
    naming: Naming = NAMING,
    join: Join = JOIN,
) -> pl.DataFrame:
    if connector is None:
        with connect(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        ) as connector:
            return get_data(
                test=test,
                conditions=conditions,
                transform=transform,
                naming=naming,
                join=join,
                connector=connector,
                host=host,
                port=port,
                username=username,
                password=password,
                database=database,
            )
    return connector.get_data(
        test, conditions=conditions, transform=transform, naming=naming, join=join
    )


__all__ = [
    "connect",
    "list_tests",
    "get_tests",
    "get_data",
    "Connector",
    "Conditions",
    "Join",
    "Naming",
    "Test",
]
