from typing import Literal, overload

import polars as pl

from newaresql.sql.bdf import MAPPINGS, convert
from newaresql.sql.connector import CONNECTORS, Connector
from newaresql.sql.schemas import get_data_schema
from newaresql.sql.transform import extend_data, transform_aux, transform_main


def _list_tests(connector: Connector) -> list[dict]:
    return connector.list_tests()


def _get_data(
    test: dict,
    connector: Connector,
    where: dict | None = None,
    main_columns: list[str] | None = None,
    aux_columns: list[str] | None = None,
) -> pl.DataFrame:

    version = connector.get_version()
    dev_uid = test["dev_uid"]

    if aux_columns is None:
        aux_columns = ["auxchl_id", "seq_id", "test_tmp"]
    if main_columns is None:
        main_columns = list(get_data_schema(version, dev_uid)["main"].keys())
        main_columns.remove("test_tmp")

    main = connector.get_main_data(test, where=where, columns=main_columns)
    aux = connector.get_aux_data(test, where=where, columns=aux_columns)

    main = transform_main(main, version, dev_uid)
    if aux is not None:
        aux = transform_aux(aux, version, dev_uid)
    else:
        aux = None

    if aux is not None:
        data = main.join(aux, on="seq_id", how="left")
    else:
        data = main.with_columns(auxchl_id=pl.lit(None), test_tmp=pl.lit(None))
    data = extend_data(data)

    columns = MAPPINGS.get(("bts", "label"))
    if columns is None:
        raise ValueError("Invalid mapping from 'bts' to 'label'")

    return convert(data, src="bts", dst="label").select(columns.values())


def _get_version(
    host: str | None = None,
    port: int | str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
):
    with Connector(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    ) as conn:
        version = conn.get_version()
        if version not in CONNECTORS:
            raise ValueError(f"Unsupported BTS version: {version}")
    return version


def connect(
    host: str | None = None,
    port: int | str | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    version: str | None = None,
) -> Connector:
    """
    Main access point to connect to Neware's MySQL BTS database. Returns a Connector object for the specified version.
    with newaresql.sql.connect(...) as conn:
        pass
    """

    if version is None:
        version = _get_version(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
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
    connector: Connector | None = None,
    credentials: dict[
        Literal["host", "port", "user", "password", "database"], str | int | None
    ]
    | None = None,
) -> list[dict]:
    """
    List all availalbe tests as dictionaries
    """
    if connector is None:
        with connect(**(credentials or {})) as conn:  # ty:ignore[invalid-argument-type]
            return _list_tests(connector=conn)

    return _list_tests(connector=connector)


def get_data(
    test: dict,
    connector: Connector | None = None,
    credentials: dict[
        Literal["host", "port", "user", "password", "database"], str | int | None
    ]
    | None = None,
    where: dict | None = None,
    main_columns: list[str] | None = None,
    aux_columns: list[str] | None = None,
):
    """

    Get data for a given test as a polars dataframe
    """

    if connector is None:
        with connect(**(credentials or {})) as conn:  # ty:ignore[invalid-argument-type]
            return _get_data(
                test,
                connector=conn,
                where=where,
                main_columns=main_columns,
                aux_columns=aux_columns,
            )
    return _get_data(
        test,
        connector=connector,
        where=where,
        main_columns=main_columns,
        aux_columns=aux_columns,
    )


__all__ = ["connect", "list_tests", "get_data", "stream_data"]
