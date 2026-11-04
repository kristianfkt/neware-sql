"""Public convenience API for accessing Neware BTS test data."""

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
    """Create a database connector.

    Args:
        host: Database host, or ``None`` to use the environment variable.
        port: Database port, or ``None`` to use the environment variable.
        username: Database username, or ``None`` to use the environment variable.
        password: Database password, or ``None`` to use the environment variable.
        database: Database name, or ``None`` to use the environment variable.

    Returns:
        A connector that can be used as a context manager.
    """
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
    """List available tests.

    Args:
        connector: Existing connector to use. If omitted, one is created and
            closed for this call.
        host: Database host used when creating a connector.
        port: Database port used when creating a connector.
        username: Database username used when creating a connector.
        password: Database password used when creating a connector.
        database: Database name used when creating a connector.

    Returns:
        Test metadata dictionaries.
    """
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
    """Fetch the available-test overview as a Polars DataFrame.

    Args:
        connector: Existing connector to use. If omitted, one is created and
            closed for this call.
        host: Database host used when creating a connector.
        port: Database port used when creating a connector.
        username: Database username used when creating a connector.
        password: Database password used when creating a connector.
        database: Database name used when creating a connector.

    Returns:
        A DataFrame containing metadata for available tests.
    """
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
    """Fetch and transform data for a test.

    Args:
        test: Metadata for the test to fetch.
        conditions: Optional filters applied to the main and auxiliary data.
        naming: Output column naming convention.
        threaded: Whether to fetch data components concurrently.
        connector: Existing connector to use. When supplied, it takes
            precedence over the individual credentials.
        host: Database host used when creating a connector.
        port: Database port used when creating a connector.
        username: Database username used when creating a connector.
        password: Database password used when creating a connector.
        database: Database name used when creating a connector.

    Returns:
        A ``TestData`` object containing the fetched data.
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
