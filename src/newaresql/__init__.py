import pathlib
from typing import Literal

import newaresql.bts as bts
import newaresql.local as local


def connect(
    path: str | pathlib.Path | None = None,
    format: str | None = None,
    host: str | None = None,
    port: str | int | None = None,
    user: str | None = None,
    password: str | None = None,
    database: str | None = None,
    version: str | None = None,
    target: Literal["bts", "files", "sqlite"] = "files",
):

    if target == "bts":
        return bts.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            version=version,
        )
    elif target in ["files", "sqlite"]:
        return local.connect(path=path, format=format, target=target)

    raise ValueError(
        f"Invalid target: {target}. Valid values are: ['bts', 'files', 'sqlite']"
    )
