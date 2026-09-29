from __future__ import annotations

from functools import cache
from typing import Iterator

import polars as pl
import sqlalchemy as sa

import newaresql.bdf as bdf
import newaresql.defaults as defaults
import newaresql.transformations as transformations
from newaresql.connectors.sql import SQLConnector
from newaresql.schemas import get_data_schema
from newaresql.types import Columns, Naming, Test, Where
from newaresql.utils.bts import make_aux_statement, make_main_statement
from newaresql.utils.config import get_config


def get_tests_0760(connector: BTSConnector) -> pl.DataFrame:
    test = connector.get_table("test")
    h_test = connector.get_table("h_test")
    test_note = connector.get_table("test_note")
    tests = pl.concat([test, h_test], how="diagonal_relaxed").join(
        test_note, on=["dev_uid", "unit_id", "chl_id", "test_id"], how="left"
    )
    return tests


def get_tests_0800(connector: BTSConnector) -> pl.DataFrame:
    tables = ["test"] + [
        table for table in connector.list_tables() if table.startswith("h_test")
    ]
    frames = [connector.get_table(table) for table in tables]
    return pl.concat(frames, how="diagonal_relaxed")


_GET_TESTS = {
    "0760": get_tests_0760,
    "0800": get_tests_0800,
}


class BTSConnector(SQLConnector):
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):

        host = get_config("NEWARE_BTS_HOST", value=host)
        port = get_config("NEWARE_BTS_PORT", value=port)
        database = get_config("NEWARE_BTS_DATABASE", value=database)
        username = get_config("NEWARE_BTS_USERNAME", value=username)
        password = get_config("NEWARE_BTS_PASSWORD", value=password)
        if not all([host, port, database, username, password]):
            raise ValueError("Missing required BTS connection parameters")

        url = sa.URL.create(
            "mysql+pymysql",
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        super().__init__(url=url)
        return

    @cache
    def get_version(self) -> str:
        query = "SELECT DISTINCT version FROM db_ver"
        versions = (
            self.get_query(query).select("version").to_series().unique().to_list()
        )
        if len(versions) != 1:
            raise ValueError(
                f"Expected one version, got {len(versions)} versions: {versions}"
            )
        return str(versions[0])

    def list_tests(self) -> list[dict]:
        """
        Returns a list of available tests as dictionaries.
        """
        return self.get_tests().to_dicts()

    def get_tests(self) -> pl.DataFrame:
        """
        Returns a DataFrame containing all available tests in the database.
        """
        version = self.get_version()
        if version not in _GET_TESTS:
            raise ValueError(f"Unsupported BTS version: {version}")
        return _GET_TESTS[version](self)

    def get_main_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:
        """
        Fetches main data for the specified test from the database.

        Args:
            test: The test for which to fetch main data.
            where: Optional filtering conditions for the query.
            columns: Optional list of columns to retrieve.
            naming: Naming convention for the returned DataFrame.
            extend: Whether to extend the data using transformations beyond row-by-row.

        Returns:
            A DataFrame containing the main data for the test.
        """
        query = make_main_statement(
            test=test,
            engine=self.engine,
            where=where,
            columns=columns,
        )
        schema = get_data_schema(self.get_version(), test["dev_uid"])["main"]

        data = transformations.transform(
            self.get_query(query, schema_overrides=pl.Schema(schema)),
            self.get_version(),
            test["dev_uid"],
        )
        if extend:
            data = transformations.extend(data)
        return bdf.convert(data, "bts", naming)

    def get_aux_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:
        query = make_aux_statement(
            test,
            self.engine,
            where=where,
            columns=columns,
        )
        schema = get_data_schema(self.get_version(), test["dev_uid"])["aux"]
        if query is None:
            data = pl.DataFrame(schema=schema)
        else:
            data = transformations.transform(
                self.get_query(query, schema_overrides=pl.Schema(schema)),
                self.get_version(),
                test["dev_uid"],
            )
        if extend:
            data = transformations.extend(data)
        return bdf.convert(data, "bts", naming)

    def get_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:
        """
        Fetches and combines the main and auxiliary data for a given test.
        main- and auxillary columns are automatically included based on the test schema.

        Args:
            test: The test for which to fetch data.
            where: Optional filtering conditions for the query.
            naming: Naming convention for the returned DataFrame.
            extend: Whether to extend the data using transformations beyond row-by-row.

        Returns:
            A DataFrame containing the combined main and auxiliary data for the test.
        """
        # Remove test_tmp
        main_columns = get_data_schema(self.get_version(), test["dev_uid"])[
            "main"
        ].keys()
        main_columns = [col for col in main_columns if col != "test_tmp"]
        aux_columns = ["auxchl_id", "seq_id", "test_tmp"]

        main_data = self.get_main_data(
            test,
            where=where,
            columns=main_columns,
            naming="bts",
            extend=False,
        )
        aux_data = self.get_aux_data(
            test,
            where=where,
            columns=aux_columns,
            naming="bts",
            extend=False,
        )
        data = main_data.join(aux_data, on="seq_id", how="left")
        if extend:
            data = transformations.extend(data)
        return bdf.convert(data, "bts", naming)

    def chunk_main_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]:
        """
        Fetches the main data for a given test in chunks.

        Args:
            test: The test for which to fetch main data.
            where: Optional filtering conditions for the query.
            columns: Specific columns to fetch from the main data.
            naming: Naming convention for the returned DataFrame.
            extend: Whether to extend the data using transformations beyond row-by-row.
            chunk_size: The number of rows per chunk.

        Yields:
            DataFrames containing chunks of the main data for the test.
        """
        query = make_main_statement(
            test=test,
            engine=self.engine,
            where=where,
            columns=columns,
        )
        schema = get_data_schema(self.get_version(), test["dev_uid"])["main"]

        for chunk in self.chunk_query(
            query, schema_overrides=pl.Schema(schema), chunk_size=chunk_size
        ):
            chunk = transformations.transform(
                chunk, self.get_version(), test["dev_uid"]
            )
            if extend:
                chunk = transformations.extend(chunk)

            yield bdf.convert(chunk, "bts", naming)
        return

    def chunk_aux_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]:
        """
        Fetches the auxiliary data for a given test in chunks.

        Args:
            test: The test for which to fetch auxiliary data.
            where: Optional filtering conditions for the query.
            columns: Specific columns to fetch from the auxiliary data.
            naming: Naming convention for the returned DataFrame.
            extend: Whether to extend the data using transformations beyond row-by-row.
            chunk_size: The number of rows per chunk.

        Yields:
            DataFrames containing chunks of the auxiliary data for the test.
        """
        query = make_aux_statement(
            test=test,
            engine=self.engine,
            where=where,
            columns=columns,
        )
        schema = get_data_schema(self.get_version(), test["dev_uid"])["aux"]
        if query is None:
            return

        for chunk in self.chunk_query(
            query, schema_overrides=pl.Schema(schema), chunk_size=chunk_size
        ):
            chunk = transformations.transform(
                chunk, self.get_version(), test["dev_uid"]
            )
            if extend:
                chunk = transformations.extend(chunk)

            yield bdf.convert(chunk, "bts", naming)
        return

    def chunk_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]:
        """
        Fetches both main and auxiliary data for a given test in chunks.

        Args:
            test: The test for which to fetch data.
            where: Optional filtering conditions for the query.
            naming: Naming convention for the returned DataFrame.
            extend: Whether to extend the data using transformations beyond row-by-row.
            chunk_size: The number of rows per chunk.

        Yields:
            DataFrames containing chunks of the combined main and auxiliary data for the test.
        """
        i = 1
        j = i + chunk_size
        while True:
            chunk = self.get_data(
                test, where={"seq_id": (i, j)}, naming=naming, extend=extend
            )
            if chunk.is_empty():
                break
            yield chunk
            i += chunk_size
            j += chunk_size
        return
