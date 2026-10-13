import concurrent.futures
import os

import polars as pl
import sqlalchemy as sa

from .schemas import get_data_schema
from .types import Conditions, Test


def build_conditions(conditions: Conditions) -> sa.sql.elements.BooleanClauseList:
    clauses = []
    for col, cond in conditions.items():
        if isinstance(cond, list):
            clauses.append(sa.column(col).in_(cond))
        elif isinstance(cond, tuple):
            if len(cond) == 2:
                if cond[0] is None:
                    clauses.append(sa.column(col) <= cond[1])
                elif cond[1] is None:
                    clauses.append(sa.column(col) >= cond[0])
                else:
                    clauses.append(sa.column(col).between(cond[0], cond[1]))
        else:
            clauses.append(sa.column(col) == cond)
    return sa.and_(*clauses)


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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._engine.dispose()

    def list_tables(self) -> list[str]:
        with self._engine.connect() as connection:
            return sa.inspect(connection).get_table_names()

    def query(
        self,
        query: str | sa.Selectable | sa.TextClause,
        **options,
    ):
        """
        Wraps pl.read_database
        """
        with self._engine.connect() as connection:
            return pl.read_database(query, connection=connection, **options)

    def select_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
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
        columns: str | list[str] | None = None,
        conditions: Conditions | None = None,
    ) -> sa.Selectable:
        stmts = [
            self.select_table(table, columns=columns, conditions=conditions)
            for table in tables
        ]
        return sa.union_all(*stmts)

    def get_table(
        self,
        table: str,
        columns: str | list[str] | None = None,
        conditions: Conditions | None = None,
        **options,
    ) -> pl.DataFrame:
        stmt = self.select_table(table, columns=columns, conditions=conditions)
        return self.query(stmt, **options)

    def get_main(
        self,
        test: Test,
        columns: list[str] | None = None,
        conditions: Conditions | None = None,
        **options,
    ) -> pl.DataFrame:

        schema = get_data_schema("..", test["dev_uid"])["main"]
        if columns is None:
            columns = list(schema.keys())

        if conditions is None:
            conditions = {}
        conditions = dict(
            conditions,
            unit_id=test["unit_id"],
            chl_id=test["chl_id"],
            test_id=test["test_id"],
        )

        tables = [
            table
            for table in [test["main_first_table"], test["main_second_table"]]
            if table is not None
        ]

        queries = [
            self.select_table(table, columns=columns, conditions=conditions)
            for table in tables
            if table is not None
        ]
        if len(queries) == 0:
            main = pl.DataFrame(schema={col: schema[col] for col in columns})
        else:
            query = sa.union_all(*queries)
            main = self.query(
                query,
                schema_overrides=schema,
                **options,
            )

        return main

    def get_aux(
        self,
        test: Test,
        columns: list[str] | None = None,
        conditions: Conditions | None = None,
        **options,
    ) -> pl.DataFrame:

        schema = get_data_schema("..", test["dev_uid"])["aux"]
        if columns is None:
            columns = list(schema.keys())

        if conditions is None:
            conditions = {}
        conditions = dict(
            conditions,
            unit_id=test["unit_id"],
            chl_id=test["chl_id"],
            test_id=test["test_id"],
        )

        tables = [
            table
            for table in [test["aux_first_table"], test["aux_second_table"]]
            if table is not None
        ]
        queries = [
            self.select_table(table, columns=columns, conditions=conditions)
            for table in tables
            if table is not None
        ]
        if len(queries) == 0:
            aux = pl.DataFrame(schema={col: schema[col] for col in columns})
        else:
            query = sa.union_all(*queries)
            aux = self.query(
                query,
                schema_overrides=schema,
                **options,
            )

        return aux

    def get_data(
        self,
        test: Test,
        main_conditions: Conditions | None = None,
        aux_conditions: Conditions | None = None,
        **options,
    ) -> pl.DataFrame:

        schemas = get_data_schema("..", test["dev_uid"])
        main_columns = list(schemas["main"].keys())
        aux_columns = ["seq_id", "auxchl_id", "test_tmp"]
        if "test_tmp" in main_columns:
            main_columns.remove("test_tmp")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            main_future = executor.submit(
                self.get_main,
                test=test,
                columns=main_columns,
                conditions=main_conditions,
                **options,
            )
            aux_future = executor.submit(
                self.get_aux,
                test=test,
                columns=aux_columns,
                conditions=aux_conditions,
                **options,
            )
            main_result = main_future.result()
            aux_result = aux_future.result()
        return main_result.join(aux_result, on="seq_id")

    def get_stats(self, test: Test) -> dict:
        """
        Not safe for missing data,
        i.e. returns
        - record_count: MAX(seq_id)
        - cycle_count: MAX(cycle)
        """

        table = self.select_table(
            test["main_second_table"] or test["main_first_table"],
            conditions={
                "unit_id": test["unit_id"],
                "chl_id": test["chl_id"],
                "test_id": test["test_id"],
            },
        )
        stmt = sa.select(
            sa.func.max(table.c["seq_id"]).alias("record_count"),
            sa.func.max(table.c["cycle"]).alias("cycle_count"),
        )
        result = self.query(stmt)
        return result.to_dicts()[0]
