from __future__ import annotations

from functools import cache
from typing import Any

import polars as pl

from newaresql.bdf import convert
from newaresql.bts.schemas import get_data_schema
from newaresql.bts.transform import extend_data, transform_aux, transform_main
from newaresql.core.connector import SQLConnector
from newaresql.core.orm import make_select_query
from newaresql.core.utils import get_config


class BTSConnector(SQLConnector):
    def __init__(
        self,
        host: str | None = None,
        port: int | str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        if host is None:
            host = get_config("bts_host", value=host)
        if port is None:
            port = get_config("bts_port", value=port)
        if user is None:
            user = get_config("bts_user", value=user)
        if password is None:
            password = get_config("bts_password", value=password)
        if database is None:
            database = get_config("bts_database", value=database)
        if isinstance(port, str):
            port = int(port)
        if not all([host, port, user, password, database]):
            raise ValueError(
                "Missing required connection parameters: host, port, user, password, database"
            )

        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        super().__init__(url=url)
        return

    def __enter__(self) -> BTSConnector:
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().__exit__(exc_type, exc_value, traceback)
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
        return self.get_tests().to_dicts()

    def _get_main_raw(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> pl.DataFrame | None:

        if where is None:
            where = {}
        where.update(
            {
                "unit_id": test["unit_id"],
                "chl_id": test["chl_id"],
                "test_id": test["test_id"],
            }
        )
        if test["main_first_table"] is None:
            q1 = None
        else:
            q1 = make_select_query(
                table=test["main_first_table"],
                columns=columns,
                where=where,
            )
        if test["main_second_table"] is None:
            q2 = None
        else:
            q2 = make_select_query(
                table=test["main_second_table"],
                columns=columns,
                where=where,
            )

        if q1 and q2:
            q = f"{q1} UNION ALL {q2}"
        elif q1:
            q = q1
        elif q2:
            q = q2
        else:
            q = None

        return self.get_query(q, schema=schema) if q else None

    def _get_aux_raw(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> pl.DataFrame | None:
        if where is None:
            where = {}
        where.update(
            {
                "unit_id": test["unit_id"],
                "chl_id": test["chl_id"],
                "test_id": test["test_id"],
            }
        )
        if test["aux_first_table"] is None:
            q1 = None
        else:
            q1 = make_select_query(
                table=test["aux_first_table"],
                columns=columns,
                where=where,
            )

        if test["aux_second_table"] is None:
            q2 = None
        else:
            q2 = make_select_query(
                table=test["aux_second_table"],
                columns=columns,
                where=where,
            )
        if q1 and q2:
            q = f"{q1} UNION ALL {q2}"
        elif q1:
            q = q1
        elif q2:
            q = q2
        else:
            q = None
        return self.get_query(q, schema=schema) if q else None

    def get_data(
        self,
        test: dict,
        columns: str | list[str] | None = None,
        where: dict[str, Any | list[Any] | tuple[Any | None, Any | None]] | None = None,
    ) -> pl.DataFrame:

        schema = get_data_schema(self.get_version(), test["dev_uid"])
        main_columns = list(schema["main"].keys())
        main_columns.remove("test_tmp")
        aux_columns = ["seq_id", "auxchl_id", "test_tmp"]

        main = self._get_main_raw(
            test, columns=main_columns, where=where, schema=schema["main"]
        )
        aux = self._get_aux_raw(
            test, columns=aux_columns, where=where, schema=schema["aux"]
        )
        if (aux is not None) and (aux.height == 0):
            aux = None

        if main is None:
            main = pl.DataFrame(schema=schema["main"])
        else:
            main = transform_main(main, self.get_version(), test["dev_uid"])
        if aux is not None:
            aux = transform_aux(aux, self.get_version(), test["dev_uid"])
            data = main.join(aux, on="seq_id", how="left", maintain_order="left")
        else:
            data = main.with_columns(
                auxchl_id=pl.lit(None).cast(pl.Int64),
                test_tmp=pl.lit(None).cast(pl.Int64),
            )
        data = extend_data(data)

        return convert(data, "bts", "label")

    def get_stats(self, test: dict) -> dict:
        stats = {}
        if test["main_second_table"] is None:
            t = test["main_first_table"]
            where = {
                "unit_id": test["unit_id"],
                "chl_id": test["chl_id"],
                "test_id": test["test_id"],
            }
        else:
            t = test["main_second_table"]
            where = None

        q = make_select_query(table=t, columns="MAX(seq_id) as max_seq_id", where=where)
        stats["seq_id"] = self.get_query(q).select("max_seq_id").to_series().item()
        return stats


class BTS0760Connector(BTSConnector):
    def __enter__(self) -> BTS0760Connector:
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().__exit__(exc_type, exc_value, traceback)
        return

    def get_tests(self) -> pl.DataFrame:

        test = self.read_table("test")
        h_test = self.read_table("h_test")
        test_note = self.read_table("test_note")
        tests = (
            pl.concat([test, h_test], how="diagonal_relaxed")
            .join(test_note, on=["dev_uid", "unit_id", "chl_id", "test_id"], how="left")
            .sort(["dev_uid", "unit_id", "chl_id", "test_id"])
        )

        return tests


class BTS0800Connector(BTSConnector):
    def __enter__(self) -> BTS0800Connector:
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        super().__exit__(exc_type, exc_value, traceback)
        return

    def get_tests(self) -> pl.DataFrame:
        tables = [
            "test",
            *sorted(t for t in self.list_tables() if t.startswith("h_test")),
        ]
        tests = pl.concat(
            [self.read_table(table) for table in tables],
            how="diagonal_relaxed",
        ).sort(["dev_uid", "unit_id", "chl_id", "test_id"])

        return tests


CONNECTORS = {
    "0760": BTS0760Connector,
    "0800": BTS0800Connector,
}
