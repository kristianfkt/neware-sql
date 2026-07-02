import polars as pl

from newaresql.bdf import MAPPINGS, convert
from newaresql.connect import Connector, connect
from newaresql.schemas import get_data_schema
from newaresql.transform import extend_data, transform_aux, transform_main


def _list_tests(connector: Connector) -> list[dict]:
    return connector.get_tests().to_dicts()


def _get_data(
    test: dict,
    connector: Connector,
    where: dict | None = None,
    main_columns: list[str] | None = None,
    aux_columns: list[str] | None = None,
):

    if aux_columns is None:
        aux_columns = ["auxchl_id", "seq_id", "test_tmp"]
    if main_columns is None:
        main_columns = list(
            get_data_schema(connector.version, test["dev_uid"])["main"].keys()
        )
        main_columns.remove("test_tmp")

    main = connector.get_main_data(test, where=where, columns=main_columns)
    aux = connector.get_aux_data(test, where=where, columns=aux_columns)

    main = transform_main(main, connector.version, test["dev_uid"])
    if aux is not None:
        aux = transform_aux(aux, connector.version, test["dev_uid"])
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


def list_tests(
    connector: Connector | None = None,
    credentials: dict[str, str | int | None] | None = None,
) -> list[dict]:
    """
    List all availalbe tests as dictionaries
    """
    if connector is None:
        cred = credentials or {}
        with connect(**cred) as conn:  # ty:ignore[invalid-argument-type]
            return _list_tests(connector=conn)

    return _list_tests(connector=connector)


def get_data(
    test: dict,
    connector: Connector | None = None,
    credentials: dict[str, str | int | None] | None = None,
    where: dict | None = None,
    main_columns: list[str] | None = None,
    aux_columns: list[str] | None = None,
):
    """

    Get data for a given test as a polars dataframe
    """

    if connector is None:
        cred = credentials or {}
        with connect(**cred) as conn:  # ty:ignore[invalid-argument-type]
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


__all__ = ["connect", "list_tests", "get_data"]
