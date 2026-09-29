from __future__ import annotations

from functools import cache
from typing import Iterator

import polars as pl
import sqlalchemy as sa

import newaresql.defaults as defaults
from newaresql.bdf import convert
from newaresql.bts.transformations import extend_data, transform_aux, transform_main
from newaresql.connectors.sql import SQLConnector
from newaresql.schemas import get_data_schema
from newaresql.transformations import transform
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

    # This is the main on
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
        return self.get_tests().to_dicts()

    def get_tests(self) -> pl.DataFrame:
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
        query = make_main_statement(
            test=test,
            engine=self.engine,
            where=where,
            columns=columns,
        )
        schema = get_data_schema(self.get_version(), test["dev_uid"])["main"]

        # Apply where, columns, and naming transformations to the query as needed
        # data = transform_main(
        #     self.get_query(query, schema_overrides=pl.Schema(schema)),
        #     self.get_version(),
        #     test["dev_uid"],
        # )
        data = transform(
            self.get_query(query, schema_overrides=pl.Schema(schema)),
            self.get_version(),
            test["dev_uid"],
        )
        if extend:
            data = extend_data(data)
        return convert(data, "bts", naming)

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
            data = transform_aux(
                self.get_query(query, schema_overrides=pl.Schema(schema)),
                self.get_version(),
                test["dev_uid"],
            )
        if extend:
            data = extend_data(data)
        return convert(data, "bts", naming)

    def get_data(
        self,
        test: Test,
        *,
        where: Where | None = None,
        naming: Naming = defaults.NAMING,
        extend: bool = True,
    ) -> pl.DataFrame:
        main_data = self.get_main_data(
            test,
            where=where,
            naming="bts",
            extend=False,
        )
        aux_data = self.get_aux_data(
            test,
            where=where,
            naming="bts",
            extend=False,
        )
        data = main_data.drop("test_tmp").join(aux_data, on="seq_id", how="left")
        if extend:
            data = extend_data(data)
        return convert(data, "bts", naming)

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
            chunk = transform_main(chunk, self.get_version(), test["dev_uid"])
            if extend:
                chunk = extend_data(chunk)

            yield convert(chunk, "bts", naming)
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
            chunk = transform_aux(chunk, self.get_version(), test["dev_uid"])
            if extend:
                chunk = extend_data(chunk)

            yield convert(chunk, "bts", naming)
        return
