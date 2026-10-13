from newaresql.connector import Connector


def connect(
    host: str | None = None,
    port: int | str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
) -> Connector:
    return Connector(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
    )


# # import pathlib
# # from typing import Literal
# from collections.abc import Callable

# from newaresql.models import Test
# from newaresql.protocols import Connector

# # import polars as pl

# # import newaresql.defaults as defaults
# # from newaresql.connectors.bts import BTSConnector
# # from newaresql.connectors.file import FileConnector

# # # from newaresql.connectors.sql import SQLiteConnector
# # from newaresql.interface import Test
# # from newaresql.protocols import Connector
# # from newaresql.types import Naming, Test, Where

# # _TARGETS = {
# #     "bts": BTSConnector,
# #     "files": FileConnector,
# #     # "sqlite": SQLiteConnector,
# # }


# # def connect(
# #     target: Literal["bts", "files"] = "bts",
# #     options: dict | None = None,
# # ) -> Connector:
# #     """
# #     Valid targets are "bts" and "files".
# #     Valid options are target dependent
# #         bts: host, port, username, password, database
# #         files: root, file_format
# #         sqlite: path


# #     """
# #     # Future support targets: duckdb, azureblob, ..., databrickssql
# #     if target not in _TARGETS:
# #         raise ValueError(
# #             f"Invalid target: {target}. Valid values are: {list(_TARGETS.keys())}"
# #         )

# #     if options is None:
# #         options = {}
# #     return _TARGETS[target](**options)


# # def list_tests(
# #     connector: Connector | None = None,
# #     target: Literal["bts", "files"] = "bts",
# #     options: dict | None = None,
# # ) -> list[dict]:
# #     if connector is None:
# #         with connect(target, options) as connector:
# #             return list_tests(connector, target=target, options=options)
# #     return connector.list_tests()


# # def get_tests(
# #     connector: Connector | None = None,
# #     target: Literal["bts", "files"] = "bts",
# #     options: dict | None = None,
# # ) -> pl.DataFrame:
# #     if connector is None:
# #         with connect(target, options) as connector:
# #             return get_tests(connector, target=target, options=options)
# #     return connector.get_tests()


# # def get_data(
# #     test: Test,
# #     *,
# #     where: Where | None = None,
# #     naming: Naming = defaults.NAMING,
# #     extend: bool = True,
# #     connector: Connector | None = None,
# #     target: Literal["bts", "files"] = "bts",
# #     options: dict | None = None,
# # ) -> pl.DataFrame:
# #     if connector is None:
# #         with connect(target, options) as connector:
# #             return get_data(
# #                 test,
# #                 where=where,
# #                 naming=naming,
# #                 extend=extend,
# #                 connector=connector,
# #                 target=target,
# #                 options=options,
# #             )
# #     return connector.get_data(test, where=where, naming=naming, extend=extend)


# def map_tests(connector: Connector) -> dict[str, Test]:
#     return {
#         Test(metadata=test, connector=connector).name: Test(
#             metadata=test, connector=connector
#         )
#         for test in connector.list_tests()
#     }


# def search_tests(
#     connector: Connector, callback: Callable[[Test], bool]
# ) -> dict[str, Test]:
#     """
#     Search for tests that match the given field values.
#     Recommended keywords are batch_no and remark (correspond to P/N name and comment fields in BTS)
#     """
#     return {
#         Test(metadata=test, connector=connector).name: Test(
#             metadata=test, connector=connector
#         )
#         for test in connector.list_tests()
#         if callback(Test(metadata=test, connector=connector))
#     }
