from __future__ import annotations

import concurrent.futures
import os
from functools import cache
from typing import Self

import polars as pl
import sqlalchemy as sa

from newaresql import bdf, transformations
from newaresql.defaults import JOIN, NAMING, TRANSFORM
from newaresql.schemas import get_data_schema
from newaresql.types import Columns, Conditions, Join, Naming, Test


def build_conditions(conditions: Conditions) -> sa.ColumnElement[bool]:
    clauses = []
    for col, cond in conditions.items():
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
    test = connector.get_tables("test", "h_test")
    note = connector.get_table("test_note")
    return test.join(note, on=["dev_uid", "unit_id", "chl_id", "test_id"], how="left")


def get_tests_0800(connector: Connector) -> pl.DataFrame:
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
    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._engine.dispose()

    @cache
    def get_version(self) -> str:
        query = "SELECT DISTINCT version FROM db_ver"
        versions = self.query(query).select("version").to_series().unique().to_list()
        if len(versions) != 1:
            raise ValueError(
                f"Expected one version, got {len(versions)} versions: {versions}"
            )
        return str(versions[0])

    def list_tables(self) -> list[str]:
        with self._engine.connect() as connection:
            return sa.inspect(connection).get_table_names()

    def get_tests(self) -> pl.DataFrame:
        version = self.get_version()
        if version not in GET_TESTS:
            raise ValueError(f"No get_tests function for version {version}")
        return GET_TESTS[version](self)

    def list_tests(self) -> list[dict]:
        return self.get_tests().to_dicts()

    def query(
        self,
        query: str | sa.Selectable | sa.TextClause,
        **options,
    ):
        """
        Wraps pl.read_database
        pl.read_database(query, connection=connection, schema_overrides=schema, **options)
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
        if len(tables) == 0:
            raise ValueError("At least one table must be specified")
        elif len(tables) == 1:
            if isinstance(conditions, list):
                conditions = conditions[0]

            return self.select_table(tables[0], columns=columns, conditions=conditions)
        else:
            if not isinstance(conditions, list):
                conditions = [conditions] * len(tables)
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
        """
        Fetches a table
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
        """
        Fetches multiple tables via union all
        """
        stmt = self.select_tables(*tables, columns=columns, conditions=conditions)
        return self.query(stmt, **options)

    def select_main(
        self,
        test: Test,
        columns: Columns | None = None,
        conditions: Conditions | None = None,
    ) -> sa.Select | sa.CompoundSelect:

        schema = get_data_schema(self.get_version(), test["dev_uid"])["main"]
        if columns is None:
            columns = list(schema.keys())

        if conditions is None:
            conditions = {}

        if test["main_second_table"] is None:
            stmt = self.select_table(
                test["main_first_table"],
                columns=columns,
                conditions={
                    **conditions,
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
                        **conditions,
                        "unit_id": test["unit_id"],
                        "chl_id": test["chl_id"],
                        "test_id": test["test_id"],
                    },
                    conditions,
                ],
            )
        return stmt

    def select_aux(
        self,
        test: Test,
        columns: Columns | None = None,
        conditions: Conditions | None = None,
    ) -> sa.Select | sa.CompoundSelect | None:

        schema = get_data_schema(self.get_version(), test["dev_uid"])["aux"]
        if columns is None:
            columns = list(schema.keys())

        if conditions is None:
            conditions = {}

        if (test["aux_first_table"] is None) and (test["aux_second_table"] is None):
            stmt = None
        elif (test["aux_first_table"] is not None) and (
            test["aux_second_table"] is None
        ):
            stmt = self.select_table(
                test["aux_first_table"],
                columns=columns,
                conditions={
                    **conditions,
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
                conditions=conditions,
            )
        else:
            stmt = self.select_tables(
                test["aux_first_table"],
                test["aux_second_table"],
                columns=columns,
                conditions=[
                    {
                        **conditions,
                        "unit_id": test["unit_id"],
                        "chl_id": test["chl_id"],
                        "test_id": test["test_id"],
                    },
                    conditions,
                ],
            )
        return stmt

    def select_data(
        self,
        test: Test,
        conditions: Conditions | None = None,
    ) -> sa.Select | sa.CompoundSelect | None:
        schemas = get_data_schema(self.get_version(), test["dev_uid"])
        main_columns = list(schemas["main"].keys())
        main_columns.remove("test_tmp")
        aux_columns = ["seq_id", "auxchl_id", "test_tmp"]
        if conditions is None:
            main_conditions = None
            aux_conditions = None
        else:
            main_conditions = {
                col: value
                for col, value in conditions.items()
                if col in schemas["main"]
            }
            aux_conditions = {
                col: value for col, value in conditions.items() if col in schemas["aux"]
            }
        main_stmt = self.select_main(
            test,
            columns=main_columns,
            conditions=main_conditions,
        )
        aux_stmt = self.select_aux(
            test,
            columns=aux_columns,
            conditions=aux_conditions,
        )

        if aux_stmt is None:
            data_stmt = main_stmt
        else:
            main_subquery = main_stmt.subquery()
            aux_subquery = aux_stmt.subquery()
            data_join = sa.join(
                main_subquery,
                aux_subquery,
                main_subquery.c.seq_id == aux_subquery.c.seq_id,
            )
            data_stmt = sa.select(
                *main_subquery.c,
                *(column for column in aux_subquery.c if column.name != "seq_id"),
            ).select_from(data_join)

        return data_stmt

    def get_main(
        self,
        test: Test,
        columns: Columns | None = None,
        conditions: Conditions | None = None,
        naming: Naming = NAMING,
        transform: bool = TRANSFORM,
        **options,
    ) -> pl.DataFrame:
        schema = get_data_schema(self.get_version(), test["dev_uid"])["main"]
        if columns is None:
            columns = list(schema.keys())
        stmt = self.select_main(test, columns=columns, conditions=conditions)
        main = self.query(stmt, **options).cast(
            {col: typ for col, typ in schema.items() if col in columns}
        )
        if transform:
            main = transformations.transform(main, self.get_version(), test["dev_uid"])
        return bdf.convert(main, "bts", naming)

    def get_aux(
        self,
        test: Test,
        columns: Columns | None = None,
        conditions: Conditions | None = None,
        naming: Naming = NAMING,
        transform: bool = TRANSFORM,
        **options,
    ) -> pl.DataFrame:
        schema = get_data_schema(self.get_version(), test["dev_uid"])["aux"]
        if columns is None:
            columns = list(schema.keys())
        stmt = self.select_aux(test, columns=columns, conditions=conditions)
        if stmt is None:
            aux = pl.DataFrame(
                schema={col: typ for col, typ in schema.items() if col in columns}
            )
        else:
            aux = self.query(stmt, **options).cast(
                {col: typ for col, typ in schema.items() if col in columns}
            )
        if transform:
            aux = transformations.transform(aux, self.get_version(), test["dev_uid"])

        return bdf.convert(aux, "bts", naming)

    def _get_data_local(
        self,
        test: Test,
        main_columns: Columns | None = None,
        aux_columns: Columns | None = None,
        main_conditions: Conditions | None = None,
        aux_conditions: Conditions | None = None,
        **options,
    ) -> pl.DataFrame:

        with concurrent.futures.ThreadPoolExecutor() as executor:
            main_future = executor.submit(
                self.get_main,
                test=test,
                columns=main_columns,
                conditions=main_conditions,
                naming="bts",
                transform=False,
                **options,
            )
            aux_future = executor.submit(
                self.get_aux,
                test=test,
                columns=aux_columns,
                conditions=aux_conditions,
                naming="bts",
                transform=False,
                **options,
            )
            main_result = main_future.result()
            aux_result = aux_future.result()
        return main_result.join(aux_result, on="seq_id", how="left")

    def _get_data_database(
        self,
        test: Test,
        main_columns: Columns | None = None,
        aux_columns: Columns | None = None,
        main_conditions: Conditions | None = None,
        aux_conditions: Conditions | None = None,
        **options,
    ) -> pl.DataFrame:
        main_stmt = self.select_main(
            test,
            columns=main_columns,
            conditions=main_conditions,
            **options,
        )
        aux_stmt = self.select_aux(
            test,
            columns=aux_columns,
            conditions=aux_conditions,
            **options,
        )
        if aux_stmt is None:
            data_stmt = main_stmt
        else:
            main_subquery = main_stmt.subquery()
            aux_subquery = aux_stmt.subquery()
            data_join = sa.join(
                main_subquery,
                aux_subquery,
                main_subquery.c.seq_id == aux_subquery.c.seq_id,
            )
            data_stmt = sa.select(
                *main_subquery.c,
                *(column for column in aux_subquery.c if column.name != "seq_id"),
            ).select_from(data_join)
        schemas = get_data_schema(self.get_version(), test["dev_uid"])
        schema = {**schemas["main"], **schemas["aux"]}
        data = self.query(data_stmt, **options)
        return data.cast(
            {col: typ for col, typ in schema.items() if col in data.columns}
        )

    def get_data(
        self,
        test: Test,
        conditions: Conditions | None = None,
        naming: Naming = NAMING,
        transform: bool = TRANSFORM,
        join: Join = JOIN,
        **options,
    ) -> pl.DataFrame:
        schemas = get_data_schema(self.get_version(), test["dev_uid"])
        main_columns = list(schemas["main"].keys())
        main_columns.remove("test_tmp")
        aux_columns = ["seq_id", "auxchl_id", "test_tmp"]
        if conditions is None:
            main_conditions = None
            aux_conditions = None
        else:
            main_conditions = {
                col: value
                for col, value in conditions.items()
                if col in schemas["main"]
            }
            aux_conditions = {
                col: value for col, value in conditions.items() if col in schemas["aux"]
            }
        if join == "database":
            data = self._get_data_database(
                test=test,
                main_columns=main_columns,
                aux_columns=aux_columns,
                main_conditions=main_conditions,
                aux_conditions=aux_conditions,
                **options,
            )
        elif join == "local":
            data = self._get_data_local(
                test=test,
                main_columns=main_columns,
                aux_columns=aux_columns,
                main_conditions=main_conditions,
                aux_conditions=aux_conditions,
                **options,
            )
        else:
            raise ValueError(f"Invalid join type: {join}")

        if transform:
            data = transformations.transform(data, self.get_version(), test["dev_uid"])

        return bdf.convert(data, "bts", naming)

    def get_stats(self, test: Test) -> dict:
        """
        Not safe for missing data,
        i.e. returns
        - record_count: MAX(seq_id)
        - cycle_count: MAX(cycle)
        """
        if test["main_second_table"] is None:
            table = test["main_first_table"]
            conditions = {
                "unit_id": test["unit_id"],
                "chl_id": test["chl_id"],
                "test_id": test["test_id"],
            }
        else:
            table = test["main_second_table"]
            conditions = None

        t = self.select_table(
            table, columns=["seq_id", "cycle"], conditions=conditions
        ).subquery()
        stmt = sa.select(
            sa.func.max(t.c["seq_id"]).label("record_count"),
            sa.func.max(t.c["cycle"]).label("cycle_count"),
        )
        result = self.query(stmt)
        return result.to_dicts()[0]
