from __future__ import annotations

from typing import Iterator, Protocol, Self

import polars as pl

import newaresql.defaults as defaults
from newaresql.types import Columns, Naming, Test, Where


class Connector(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    def get_tests(self) -> pl.DataFrame: ...

    """
    Retrieve all tests as a Polars DataFrame.
    """

    def list_tests(self) -> list[Test]: ...

    """
    Retrieve a list of all available tests.
    Usually wraps gets_tests().to_dicts()
    """

    def get_stats(self, test: Test) -> dict: ...

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
    Retrieve the combined main and auxiliary data for a given test as a Polars DataFrame.
    Main- and auxiliary columns are automatically selected.
    Use get_main_data() and get_aux_data() for more fine control.
    """

    def stream_main_data(
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
    Retrieve the main data for a given test in chunks as an iterator of Polars DataFrames.
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
    """
