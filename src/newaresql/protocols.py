from __future__ import annotations

from typing import Iterator, Protocol, Self

import polars as pl

import newaresql.defaults as defaults
from newaresql.types import Columns, Naming, Test, Where


class Connector(Protocol):
    """
    Defines the interface of which a connector must adhere to for interacting with the data source.
    """

    def __enter__(self) -> Self:
        """Enter the connector's context and return the connector."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the connector's context and release any associated resources."""
        ...

    def get_tests(self) -> pl.DataFrame:
        """Return a DataFrame with an overview of all available tests."""
        ...

    def list_tests(self) -> list[Test]:
        """Return all available tests as dictionaries."""
        ...

    def get_stats(self, test: Test) -> dict:
        """Return statistics for the specified test as a dictionary.

        Args:
            test: Test whose statistics to retrieve.

        Returns:
            A dictionary containing the statistics for the specified test.
            cycle_count: The number of cycles in the test.
            record_count: The number of records in the test.
            first_timestamp: The timestamp of the first record in the test.
            last_timestamp: The timestamp of the last record in the test.
            duration: The total duration of the test.
        """
        ...

    def get_main_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:
        """Return a test's main data as a DataFrame.

        Args:
            test: Test whose main data to retrieve.
            where: Optional conditions to filter the rows.
            columns: Optional columns to include in the result.
            naming: Naming convention to use for the columns.
            extend: Whether to add computed columns.
        """
        ...

    def get_aux_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:
        """Return a test's auxiliary data as a DataFrame.

        Args:
            test: Test whose auxiliary data to retrieve.
            where: Optional conditions to filter the rows.
            columns: Optional columns to include in the result.
            naming: Naming convention to use for the columns.
            extend: Whether to add computed columns.
        """
        ...

    def get_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:
        """Return the combined main and auxiliary data for a test.

        Main and auxiliary columns are selected automatically. Use
        ``get_main_data`` or ``get_aux_data`` to select columns separately.

        Args:
            test: Test whose data to retrieve.
            where: Optional conditions that must be valid for both data sources.
            naming: Naming convention to use for the columns.
            extend: Whether to add computed columns.
        """
        ...

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
        """Yield a test's main data as DataFrame chunks.

        Args:
            test: Test whose main data to retrieve.
            where: Optional conditions to filter the rows.
            columns: Optional columns to include in each chunk.
            naming: Naming convention to use for the columns.
            extend: Whether to add computed columns.
            chunk_size: Maximum number of rows per chunk.
        """
        ...

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
        """Yield a test's auxiliary data as DataFrame chunks.

        Args:
            test: Test whose auxiliary data to retrieve.
            where: Optional conditions to filter the rows.
            columns: Optional columns to include in each chunk.
            naming: Naming convention to use for the columns.
            extend: Whether to add computed columns.
            chunk_size: Maximum number of rows per chunk.
        """
        ...
