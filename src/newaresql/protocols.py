from __future__ import annotations

from typing import Any, Iterator, Protocol, Self

import polars as pl

import newaresql.defaults as defaults
from newaresql.types import Columns, Naming, Test, Where


class Connector(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    def get_tests(self) -> pl.DataFrame: ...

    """
    Returns a Polars DataFrame containing an overview of all available tests.
    """

    def list_tests(self) -> list[Test]: ...

    """
    Returns a list of all available tests as dictionaries.
    """

    def get_stats(self, test: Test) -> dict: ...

    """
    Returns a dictionary containing statistics for the specified test.
    """

    def get_main_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame: ...

    """
    Retrieve the main data for a given test as a Polars DataFrame.
    The `where` parameter can be used to filter the data based on specific conditions.
    The `columns` parameter allows selecting specific columns to retrieve.
    The `naming` parameter specifies the naming convention for the columns.
    The `extend` parameter determines whether to extend the data with additional computed columns.
    """

    def get_aux_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame: ...

    """
    Retrieve the auxiliary data for a given test as a Polars DataFrame.
    The `where` parameter can be used to filter the data based on specific conditions.
    The `columns` parameter allows selecting specific columns to retrieve.
    The `naming` parameter specifies the naming convention for the columns.
    The `extend` parameter determines whether to extend the data with additional computed columns.
    """

    def get_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame: ...

    """
    Returns a Polars DataFrame containing the combined main and auxiliary data for the specified test.
    Main- and auxiliary columns are automatically selected.
    Use `get_main_data()` and `get_aux_data()` for more fine control.
    The `where` parameter can be used to filter the data based on specific conditions. Must be valid for both main- and auxiliary data. 
    The `naming` parameter specifies the naming convention for the columns.
    The `extend` parameter determines whether to extend the data with additional computed columns.
    """

    def stream_main_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]: ...

    """
    Retrieve the main data for a given test in chunks as an iterator of Polars DataFrames.
    The `where` parameter can be used to filter the data based on specific conditions.
    The `naming` parameter specifies the naming convention for the columns.
    The `extend` parameter determines whether to extend the data with additional computed columns.
    The `chunk_size` parameter specifies the number of rows per chunk.
    """

    def stream_aux_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        columns: Columns | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]: ...

    """
    Retrieve the auxiliary data for a given test in chunks as an iterator of Polars DataFrames.
    The `where` parameter can be used to filter the data based on specific conditions.
    The `columns` parameter allows selecting specific columns to retrieve.
    The `naming` parameter specifies the naming convention for the columns.
    The `extend` parameter determines whether to extend the data with additional computed columns.
    The `chunk_size` parameter specifies the number of rows per chunk.
    """

    def stream_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]: ...

    """
    Retrieve the combined main and auxiliary data for a given test in chunks as an iterator of Polars DataFrames.
    The `where` parameter can be used to filter the data based on specific conditions. Must be valid for both main- and auxiliary data. 
    The `naming` parameter specifies the naming convention for the columns.
    The `extend` parameter determines whether to extend the data with additional computed columns.
    The `chunk_size` parameter specifies the number of rows per chunk.
    """


class Sink(Protocol):
    """
    Protocol for a sink that can receive and store data for a given test.
    """

    def contains_test(self, test: Test) -> bool: ...

    """
    Check if the sink already contains data for the given test.

    Returns:
        bool: True if the sink contains data for the test, False otherwise.
    """

    def contains_table(self, table: str) -> bool: ...

    """
    Check if the sink already contains data for the given table.

    Returns:
        bool: True if the sink contains data for the table, False otherwise.
    """

    def write_table(
        self, table: str, data: pl.DataFrame, append: bool = True
    ) -> None: ...

    """
    Write the given data to the specified table in the sink.

    Parameters:
        table (str): The name of the table to write the data to.
        data (pl.DataFrame): The data to be written to the table.
        append (bool): Whether to append the data to the existing table. Defaults to True.
    """

    def write_data(
        self,
        test: Test,
        data: pl.DataFrame,
    ) -> None: ...

    """
    Write the given data for the specified test to the sink.

    Parameters:
        test (Test): The test for which the data is being written.
        data (pl.DataFrame): The data to be written for the test.
    """

    def get_stats(self, test: Test) -> dict[str, Any]: ...

    """
    Retrieve statistics for the specified test from the sink.

    Parameters:
        test (Test): The test for which to retrieve statistics.

    Returns:
        dict[str, Any]: A dictionary containing the statistics for the test.
    """


class CloneSource(Protocol):
    def list_tables(self) -> list[str]: ...
    def get_table(self, table: str) -> pl.DataFrame: ...
    def get_combinations(self, table: str) -> list[dict] | None: ...
    def chunk_table(
        self,
        table: str,
        where: Where | None = None,
        chunk_size: int = defaults.CHUNK_SIZE,
    ) -> Iterator[pl.DataFrame]: ...

    def get_distinct(self, table: str) -> dict: ...

    def get_max_seq_id(self, table: str, combo: dict | None = None) -> int: ...


class CloneSink(Protocol):
    def update_data(self, table: str, data: pl.DataFrame) -> None: ...
    def update_meta(self, table: str, data: pl.DataFrame) -> None: ...
    def get_max_seq_id(self, table: str, combo: dict | None = None) -> int: ...


class ExportSource(Protocol):
    def get_tests(self) -> list[Test]: ...

    def list_tests(self) -> list[Test]: ...

    def get_main_data(self, test: Test) -> pl.DataFrame: ...
    def get_aux_data(self, test: Test) -> pl.DataFrame: ...
    def get_data(self, test: Test) -> pl.DataFrame: ...
    def chunk_main_data(
        self,
        test: Test,
        chunk_size: int = ...,
    ) -> Iterator[pl.DataFrame]: ...
    def chunk_aux_data(
        self,
        test: Test,
        chunk_size: int = ...,
    ) -> Iterator[pl.DataFrame]: ...
    def chunk_data(
        self,
        test: Test,
        chunk_size: int = ...,
    ) -> Iterator[pl.DataFrame]: ...


class ExportSink(Protocol): ...
