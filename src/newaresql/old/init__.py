import pathlib

import polars as pl

from newaresql.local.local import Connector


def list_tests(
    connector: Connector | None = None,
    credentials: dict | None = None,
) -> list[dict]:
    """
    List all tests available in the local storage.
    """
    if credentials is None:
        credentials = {}
    if connector is None:
        connector = Connector(
            root=credentials.get("root"),
            fmt=credentials.get("fmt", "parquet"),
            lazy=credentials.get("lazy", False),
        )
    return connector.list_tests()


def get_data(
    test: dict,
    connector: Connector | None = None,
    credentials: dict | None = None,
) -> pl.LazyFrame | pl.DataFrame:
    if credentials is None:
        credentials = {}
    if connector is None:
        connector = Connector(
            root=credentials.get("root"),
            fmt=credentials.get("fmt", "parquet"),
            lazy=credentials.get("lazy", False),
        )
    return connector.get_data(test)
