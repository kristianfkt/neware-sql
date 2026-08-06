import pathlib
from typing import Literal

from newaresql.local.connector import FileConnector, SQLiteConnector


def connect(
    path: str | pathlib.Path | None = None,
    format: str | None = None,
    target: Literal["files", "sqlite"] = "files",
) -> FileConnector | SQLiteConnector:

    if target == "files":
        return FileConnector(path=path, format=format)
    elif target == "sqlite":
        return SQLiteConnector(path=path)

    raise ValueError("target must be either 'files' or 'sqlite'")
