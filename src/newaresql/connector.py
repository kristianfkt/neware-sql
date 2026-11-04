"""Database access and query construction for Neware BTS data."""

from __future__ import annotations

import concurrent.futures
import datetime
import os
from functools import cache
from typing import Self

import polars as pl
import sqlalchemy as sa

import newaresql.bdf as bdf
from newaresql.data import TestData
from newaresql.expressions import transform_data
from newaresql.schemas import get_data_schema
from newaresql.types import Columns, Conditions, Test


def conditions_to_bts(conditions: Conditions) -> Conditions:
    """Convert BDF code names and labels in condition keys to BTS names.

    Matching is case-sensitive. For example, ``cycle_count`` and
    ``Cycle Count / 1`` both map to ``cycle``. Unknown keys pass through.
    Values are unchanged: this does not convert units or derived fields.

    Args:
        conditions: Filters keyed by BTS names, BDF codes, or BDF labels.

    Returns:
        A new dictionary without modifying the input. If multiple keys map
        to one BTS name, the last entry wins.
    """
    bts_conditions = {}
    codes = {field["code"]: field["bts"] for field in bdf.COLUMNS}
    labels = {field["label"]: field["bts"] for field in bdf.COLUMNS}
    for col, cond in conditions.items():
        if col in codes:
            bts_conditions[codes[col]] = cond
        elif col in labels:
            bts_conditions[labels[col]] = cond
        else:
            bts_conditions[col] = cond
    return bts_conditions


def build_conditions(conditions: Conditions) -> sa.ColumnElement[bool]:
    """Build AND-combined SQL predicates after normalizing condition names.

    Args:
        conditions: Filters using BTS names, BDF codes, or exact BDF labels.
            Scalars specify equality, lists specify membership, and two-item
            tuples specify inclusive bounds. A None endpoint omits that bound.
            Values must use raw database units.

    Returns:
        A SQLAlchemy condition expression; a tuple with two None endpoints
        contributes no predicate.
    """
    clauses = []

    for col, cond in conditions_to_bts(conditions).items():
        if isinstance(cond, list):
            clauses.append(sa.column(col).in_(cond))
        elif isinstance(cond, tuple) and len(cond) == 2:
            lo, hi = cond
            if (lo is not None) and (hi is not None):
                clauses.append(sa.column(col).between(lo, hi))
            elif lo is not None:
                clauses.append(sa.column(col) >= lo)
            elif hi is not None:
                clauses.append(sa.column(col) <= hi)
        else:
            clauses.append(sa.column(col) == cond)
    return sa.and_(*clauses)


def get_tests_0760(connector: Connector) -> pl.DataFrame:
    """Fetch and join test metadata for BTS build 0760.

    Args:
        connector: Open connector used to fetch the source tables.

    Returns:
        Test metadata joined with test notes.
    """
    test = connector.get_tables("test", "h_test")
    note = connector.get_table("test_note")
    return test.join(note, on=["dev_uid", "unit_id", "chl_id", "test_id"], how="left")


def get_tests_0800(connector: Connector) -> pl.DataFrame:
    """Fetch test metadata from the BTS 0800 test tables.

    Args:
        connector: Open connector used to discover and fetch the tables.

    Returns:
        Rows from the ``test`` table and all matching history tables.
    """
    tables = ["test"]
    tables.extend(
        table for table in connector.list_tables() if table.startswith("h_test")
    )
    return connector.get_tables(*tables)


GET_TESTS = {
    "0760": get_tests_0760,
    "0800": get_tests_0800,
}


class Connector:
    """Manage a connection to a Neware BTS MySQL database."""

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        """Create a connector and its SQLAlchemy engine.

        Args:
            host: Database host, falling back to ``NEWARE_BTS_HOST``.
            port: Database port, falling back to ``NEWARE_BTS_PORT``.
            username: Database username, falling back to
                ``NEWARE_BTS_USERNAME``.
            password: Database password, falling back to
                ``NEWARE_BTS_PASSWORD``.
            database: Database name, falling back to ``NEWARE_BTS_DATABASE``.
        """
        self._host = host or os.getenv("NEWARE_BTS_HOST")
        self._port = port or os.getenv("NEWARE_BTS_PORT")
        self._username = username or os.getenv("NEWARE_BTS_USERNAME")
        self._password = password or os.getenv("NEWARE_BTS_PASSWORD")
        self._database = database or os.getenv("NEWARE_BTS_DATABASE")
        self._url = sa.make_url(
            f"mysql+pymysql://{self._username}:{self._password}@{self._host}:{self._port}/{self._database}"
        )
        self._engine = sa.create_engine(self._url)
        return

    def __enter__(self) -> Self:
        """Return this connector for use in a ``with`` statement.

        Returns:
            This connector.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Dispose of the database engine when leaving a context manager.

        Args:
            exc_type: Exception type raised in the context, if any.
            exc_val: Exception raised in the context, if any.
            exc_tb: Traceback for the context exception, if any.
        """
        self._engine.dispose()
        return

    @cache
    def get_version(self) -> str:
        """Return the unique BTS build version reported by the database.

        Raises:
            ValueError: If the database reports zero or multiple versions.
        """
        query = "SELECT DISTINCT version FROM db_ver"
        versions = self.query(query).select("version").to_series().unique().to_list()
        if len(versions) != 1:
            raise ValueError(
                f"Expected one version, got {len(versions)} versions: {versions}"
            )
        return str(versions[0])

    def list_tables(self) -> list[str]:
        """Return the names of tables available in the database.

        Returns:
            Database table names.
        """
        with self._engine.connect() as connection:
            return sa.inspect(connection).get_table_names()

    def get_tests(self) -> pl.DataFrame:
        """Fetch the test overview for the connected BTS build.

        Returns:
            A Polars DataFrame containing test metadata.

        Raises:
            ValueError: If the database version is unsupported.
        """
        version = self.get_version()
        if version not in GET_TESTS:
            raise ValueError(f"No get_tests function for version {version}")
        return GET_TESTS[version](self)

    def list_tests(self) -> list[Test]:
        """Return each row of the test overview as a dictionary.

        Returns:
            Test metadata dictionaries.
        """
        return self.get_tests().to_dicts()

    def query(
        self,
        query: str | sa.Selectable | sa.TextClause,
        **options,
    ):
        """Execute a SQL query and return its results as a Polars DataFrame.

        Args:
            query: SQL string or SQLAlchemy selectable to execute.
            **options: Additional keyword arguments forwarded to
                ``polars.read_database``.

        Returns:
            Query results as a DataFrame.
        """
        if isinstance(query, str):
            query = sa.text(query)
        with self._engine.connect() as connection:
            return pl.read_database(query, connection=connection, **options)

    def select_table(
        self,
        table: str,
        columns: Columns | None = None,
        conditions: Conditions | None = None,
    ) -> sa.Select:
        """Build a SELECT statement for one table.

        Args:
            table: Name of the database table.
            columns: Optional column name or list of names to select. If
                omitted, all table columns are selected.
            conditions: Optional filters combined with AND.

        Returns:
            An SQLAlchemy ``Select`` statement.
        """
        t = sa.Table(table, sa.MetaData(), autoload_with=self._engine)
        if isinstance(columns, str):
            columns = [columns]
        if columns is None:
            stmt = sa.select(t)
        else:
            stmt = sa.select(*[t.c[col] for col in columns])
        if conditions is not None:
            stmt = stmt.where(build_conditions(conditions))
        return stmt

    def select_tables(
        self,
        *tables,
        columns: Columns | None = None,
        conditions: Conditions | list[Conditions | None] | None = None,
    ) -> sa.Select | sa.CompoundSelect:
        """Build a SELECT or UNION ALL statement for one or more tables.

        Args:
            *tables: Names of the database tables to select.
            columns: Optional column name or list of names to select.
            conditions: Optional filters shared by all tables, or one optional
                filter mapping per table.

        Returns:
            A ``Select`` for one table or a ``CompoundSelect`` for multiple
            tables.

        Raises:
            ValueError: If no table names are provided.
        """
        if len(tables) == 0:
            raise ValueError("At least one table must be specified")
        elif len(tables) == 1:
            if isinstance(conditions, list):
                conditions = conditions[0]

            return self.select_table(tables[0], columns=columns, conditions=conditions)
        else:
            if not isinstance(conditions, list):
                conditions = [
                    None if conditions is None else dict(conditions)
                    for _ in range(len(tables))
                ]
            stmts = [
                self.select_table(table, columns=columns, conditions=cond)
                for table, cond in zip(tables, conditions)
            ]
            return sa.union_all(*stmts)

    def get_table(
        self,
        table: str,
        columns: Columns | None = None,
        conditions: Conditions | None = None,
        **options,
    ) -> pl.DataFrame:
        """Fetch a table as a Polars DataFrame.

        Args:
            table: Name of the database table.
            columns: Optional column name or list of names to fetch.
            conditions: Optional filters combined with AND.
            **options: Additional keyword arguments forwarded to
                ``polars.read_database``.

        Returns:
            The selected rows as a DataFrame.
        """
        stmt = self.select_table(table, columns=columns, conditions=conditions)
        return self.query(stmt, **options)

    def get_tables(
        self,
        *tables,
        columns: Columns | None = None,
        conditions: Conditions | list[Conditions | None] | None = None,
        **options,
    ) -> pl.DataFrame:
        """Fetch and combine one or more tables as a DataFrame.

        Args:
            *tables: Names of the database tables to fetch.
            columns: Optional column name or list of names to fetch.
            conditions: Optional filters shared by all tables, or one optional
                filter mapping per table.
            **options: Additional keyword arguments forwarded to
                ``polars.read_database``.

        Returns:
            The selected rows as a DataFrame.
        """
        stmt = self.select_tables(*tables, columns=columns, conditions=conditions)
        return self.query(stmt, **options)

    def select_main_data(
        self,
        test: Test,
        conditions: Conditions | None = None,
    ) -> tuple[sa.Select | sa.CompoundSelect, dict[str, type]]:
        """Build a query and schema for a test's main measurement data.

        Args:
            test: Metadata identifying the test tables and database keys.
            conditions: Optional filters applied to the selected rows.

        Returns:
            The SQL statement and its Polars schema overrides.
        """
        schema = get_data_schema(self.get_version(), test["dev_uid"])["main"]
        columns = list(schema.keys())

        if test["main_second_table"] is None:
            stmt = self.select_table(
                test["main_first_table"],
                columns=columns,
                conditions={
                    **(conditions or {}),
                    "unit_id": test["unit_id"],
                    "chl_id": test["chl_id"],
                    "test_id": test["test_id"],
                },
            )
        else:
            stmt = self.select_tables(
                test["main_first_table"],
                test["main_second_table"],
                columns=columns,
                conditions=[
                    {
                        **(conditions or {}),
                        "unit_id": test["unit_id"],
                        "chl_id": test["chl_id"],
                        "test_id": test["test_id"],
                    },
                    conditions,
                ],
            )
        return stmt, schema

    def select_aux_data(
        self,
        test: Test,
        conditions: Conditions | None = None,
    ) -> tuple[sa.Select | sa.CompoundSelect | None, dict[str, type]]:
        """Build a query and schema for a test's auxiliary measurement data.

        Args:
            test: Metadata identifying the test tables and database keys.
            conditions: Optional filters applied to the selected rows.

        Returns:
            The SQL statement, or ``None`` if the test has no auxiliary table,
            and the corresponding schema overrides.
        """
        schema = get_data_schema(self.get_version(), test["dev_uid"])["aux"]
        columns = list(schema.keys())

        if (test["aux_first_table"] is None) and (test["aux_second_table"] is None):
            stmt = None

        elif (test["aux_first_table"] is not None) and (
            test["aux_second_table"] is None
        ):
            stmt = self.select_table(
                test["aux_first_table"],
                columns=columns,
                conditions={
                    **(conditions or {}),
                    "unit_id": test["unit_id"],
                    "chl_id": test["chl_id"],
                    "test_id": test["test_id"],
                },
            )
        elif (test["aux_first_table"] is None) and (
            test["aux_second_table"] is not None
        ):
            stmt = self.select_table(
                test["aux_second_table"],
                columns=columns,
                conditions=conditions or {},
            )
        else:
            stmt = self.select_tables(
                test["aux_first_table"],
                test["aux_second_table"],
                columns=columns,
                conditions=[
                    {
                        **(conditions or {}),
                        "unit_id": test["unit_id"],
                        "chl_id": test["chl_id"],
                        "test_id": test["test_id"],
                    },
                    conditions,
                ],
            )
        return stmt, schema

    def get_main_data(
        self,
        test: Test,
        conditions: Conditions | None = None,
    ) -> pl.DataFrame:
        """Fetch and transform a test's main measurements.

        Args:
            test: Metadata identifying the test tables and device type.
            conditions: Optional filters applied before fetching.

        Returns:
            The transformed main measurements.
        """
        stmt, schema = self.select_main_data(test, conditions=conditions)
        return transform_data(
            self.query(stmt, schema_overrides=schema),
            self.get_version(),
            test["dev_uid"],
        )

    def get_aux_data(
        self,
        test: Test,
        conditions: Conditions | None = None,
    ) -> pl.DataFrame | None:
        """Fetch and transform auxiliary measurements when available.

        Args:
            test: Metadata identifying the test tables and device type.
            conditions: Optional filters applied before fetching.

        Returns:
            The transformed auxiliary measurements, or ``None`` if the test
            has no auxiliary table.
        """
        stmt, schema = self.select_aux_data(test, conditions=conditions)
        if stmt is None:
            return None
        else:
            return transform_data(
                self.query(stmt, schema_overrides=schema),
                self.get_version(),
                test["dev_uid"],
            )

    def get_step_count(self, test: Test):
        """Return distinct test steps with a sequential ``step_count`` column.

        Args:
            test: Metadata identifying the main test tables.

        Returns:
            A DataFrame with one row per distinct cycle, step ID, and change
            count, ordered by those columns.
        """
        q1 = f"SELECT DISTINCT cycle, step_id, step_changecount FROM {test['main_first_table']} WHERE unit_id={test['unit_id']} AND chl_id={test['chl_id']} AND test_id={test['test_id']}"
        q2 = (
            f"SELECT DISTINCT cycle, step_id, step_changecount FROM {test['main_second_table']}"
            if test["main_second_table"] is not None
            else None
        )
        if q2 is None:
            query = q1
        else:
            query = f"{q1} UNION {q2}"
        return (
            self.query(query)
            .sort("cycle", "step_id", "step_changecount")
            .with_row_count("step_count")
        )

    def get_first_atime(self, test: Test) -> datetime.datetime:
        """Return the earliest timestamp recorded for a test.

        Args:
            test: Metadata identifying the main test table and database keys.

        Returns:
            The earliest ``test_atime`` value.
        """
        query = f"SELECT MIN(test_atime) AS first_atime FROM {test['main_first_table']} WHERE unit_id={test['unit_id']} AND chl_id={test['chl_id']} AND test_id={test['test_id']}"
        result = self.query(query, schema_overrides={"first_atime": datetime.datetime})
        return result.select("first_atime").to_series().item()

    def get_record_count(self, test: Test):
        """Return the maximum sequence ID recorded for a test.

        Args:
            test: Metadata identifying the main test tables and database keys.

        Returns:
            The maximum ``seq_id`` value.
        """
        if test["main_second_table"] is None:
            query = f"SELECT MAX(seq_id) AS record_count FROM {test['main_first_table']} WHERE unit_id={test['unit_id']} AND chl_id={test['chl_id']} AND test_id={test['test_id']}"
        else:
            query = (
                f"SELECT MAX(seq_id) AS record_count FROM {test['main_second_table']}"
            )
        result = self.query(query, schema_overrides={"record_count": int})
        return result.select("record_count").to_series().item()

    def get_cycle_count(self, test: Test):
        """Return the maximum cycle number recorded for a test.

        Args:
            test: Metadata identifying the main test tables and database keys.

        Returns:
            The maximum ``cycle`` value.
        """
        if test["main_second_table"] is None:
            query = f"SELECT MAX(cycle) AS cycle_count FROM {test['main_first_table']} WHERE unit_id={test['unit_id']} AND chl_id={test['chl_id']} AND test_id={test['test_id']}"
        else:
            query = f"SELECT MAX(cycle) AS cycle_count FROM {test['main_second_table']}"
        result = self.query(query, schema_overrides={"cycle_count": int})
        return result.select("cycle_count").to_series().item()

    def get_counts(self, test: Test) -> dict:
        """Return the maximum record sequence and cycle values for a test.

        Args:
            test: Metadata identifying the main test tables and database keys.

        Returns:
            A dictionary with ``record_count`` and ``cycle_count`` values.
        """
        if test["main_second_table"] is None:
            query = f"SELECT MAX(seq_id) AS record_count, MAX(cycle) AS cycle_count FROM {test['main_first_table']} WHERE unit_id={test['unit_id']} AND chl_id={test['chl_id']} AND test_id={test['test_id']}"
        else:
            query = f"SELECT MAX(seq_id) AS record_count, MAX(cycle) AS cycle_count FROM {test['main_second_table']}"
        result = self.query(
            query, schema_overrides={"record_count": int, "cycle_count": int}
        )
        return result.to_dicts()[0]  # Return the first (and only) row as a dictionary

    def get_data(
        self,
        test: Test,
        conditions: Conditions | None = None,
        naming: bdf.Naming = bdf.DEFAULT,
        threaded: bool = False,
    ) -> TestData:
        """Fetch, transform, and combine a test's main and auxiliary data.

        Args:
            test: Metadata identifying the test and its database tables.
            conditions: Optional filters keyed by BTS names, BDF codes, or
                exact BDF labels. Values use raw database units.
            naming: Output naming convention for the returned ``TestData``.
            threaded: Whether to fetch the data components concurrently.

        Returns:
            A ``TestData`` object containing main and optional auxiliary data.

        Raises:
            ValueError: If ``naming`` is not a supported convention.
        """
        if naming not in bdf.VALID:
            raise ValueError(f"Invalid naming: {naming}")
        if conditions is None:
            main_conditions = None
            aux_conditions = None
        else:
            schemas = get_data_schema(self.get_version(), test["dev_uid"])
            conditions_bts = conditions_to_bts(conditions)
            main_conditions = {
                col: cond
                for col, cond in conditions_bts.items()
                if col in schemas["main"]
            }
            aux_conditions = {
                col: cond
                for col, cond in conditions_bts.items()
                if col in schemas["aux"]
            }
        if threaded:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    "step_count": executor.submit(
                        self.get_step_count,
                        test=test,
                    ),
                    "first_atime": executor.submit(
                        self.get_first_atime,
                        test=test,
                    ),
                    "main": executor.submit(
                        self.get_main_data,
                        test=test,
                        conditions=main_conditions,
                    ),
                    "aux": executor.submit(
                        self.get_aux_data,
                        test=test,
                        conditions=aux_conditions,
                    ),
                }

                steps = futures["step_count"].result()
                first_atime = futures["first_atime"].result()
                main = futures["main"].result()
                aux = futures["aux"].result()
        else:
            steps = self.get_step_count(
                test=test,
            )
            first_atime = self.get_first_atime(test)

            main = self.get_main_data(
                test=test,
                conditions=main_conditions,
            )

            aux = self.get_aux_data(
                test=test,
                conditions=aux_conditions,
            )
        return TestData(
            main=main.with_columns(
                (pl.col("test_atime") - first_atime)
                .dt.total_seconds()
                .alias("test_totaltime")
            ).join(steps, on=("cycle", "step_id", "step_changecount"), how="left"),
            aux=aux,
            naming=naming,
        )
