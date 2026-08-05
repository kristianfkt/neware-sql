from typing import Any

from newaresql.bts.connector import CONNECTORS, BTSConnector


def connect(
    host: str | None = None,
    port: int | str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    version: str | None = None,
):

    if version is None:
        with BTSConnector(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        ) as conn:
            version = conn.get_version()
    if version not in CONNECTORS:
        raise ValueError(f"Unsupported BTS version: {version}")
    return CONNECTORS[version](
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )


def list_tests(
    connector: BTSConnector | None = None,
    host: str | None = None,
    port: int | str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    version: str | None = None,
):

    if connector is None:
        with connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            version=version,
        ) as conn:
            return list_tests(
                connector=conn,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                version=version,
            )
    return connector.list_tests()


def get_tests(
    connector: BTSConnector | None = None,
    host: str | None = None,
    port: int | str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    version: str | None = None,
):
    if connector is None:
        with connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            version=version,
        ) as conn:
            return get_tests(
                connector=conn,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                version=version,
            )
    return connector.get_tests()


def get_data(
    test: dict,
    columns: str | list[str] | None = None,
    where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    connector: BTSConnector | None = None,
    host: str | None = None,
    port: int | str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    version: str | None = None,
):
    if connector is None:
        with connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            version=version,
        ) as conn:
            return get_data(
                test=test,
                columns=columns,
                where=where,
                connector=conn,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                version=version,
            )
    return connector.get_data(test, columns=columns, where=where)
