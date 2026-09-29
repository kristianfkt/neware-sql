import logging
from pathlib import Path

import polars as pl
import sqlalchemy as sa

import newaresql.defaults as defaults
from newaresql.connectors.blob import BlobConnector
from newaresql.connectors.file import FileConnector, filter_lazyframe
from newaresql.connectors.sql import SQLConnector
from newaresql.types import FileFormat
from newaresql.utils.config import get_config
from newaresql.utils.sql import wrap_table

logger = logging.getLogger(__name__)


class SQLSink(SQLConnector):
    def contains_table(self, table: str) -> bool:
        return table in self.list_tables()

    def get_max_seq_id(self, table: str, combo: dict | None = None) -> int:
        """
        Fetch the maximum 'seq_id' of a table, or a subset of a table
        """
        t = wrap_table(table, self.engine)
        stmt = sa.select(sa.func.max(t.c.seq_id).label("max_seq_id"))
        if combo:
            stmt = stmt.where(sa.and_(*(t.c[k] == v for k, v in combo.items())))
        return self.get_query(stmt).select("max_seq_id").to_series().item()


class FileSink(FileConnector):
    def __init__(
        self,
        root: str | Path | None = None,
        file_format: FileFormat = defaults.FILE_FORMAT,
    ):
        root = get_config("NEWARE_FILESINK_ROOT", value=root)
        file_format = (
            get_config(
                "NEWARE_FILESINK_FILEFORMAT",
                value=file_format,
                default=defaults.FILE_FORMAT,
            )
            or defaults.FILE_FORMAT
        )
        if not root:
            raise ValueError("Missing required FileSink root path")

        super().__init__(root=root, file_format=file_format)
        return

    def list_tables(self) -> list[str]:
        return [f.stem for f in self.list_folders()]

    def contains_table(self, table: str) -> bool:
        return table in self.list_tables()

    def scan_table(self, table: str) -> pl.LazyFrame:
        return self.scan_folder(table)

    def get_max_seq_id(self, table: str, combo: dict | None = None) -> int | None:
        """
        Fetch the maximum 'seq_id' of a table, or a subset of a table
        """
        if not self.contains_table(table):
            return None
        lazy = self.scan_table(table)
        if combo:
            lazy = filter_lazyframe(lazy, where=combo)
        result = (
            lazy.select(pl.col("seq_id").max())
            .collect(engine="streaming")
            .select("seq_id")
            .item()
        )
        return result

    def write_table(self, table: str, data: pl.DataFrame, append: bool = True) -> None:
        self.write_folder(table, data, append=append)
        return

    def update_data(self, table: str, data: pl.DataFrame) -> None:
        self.write_table(table, data, append=True)
        return

    def update_meta(self, table: str, data: pl.DataFrame) -> None:
        self.write_table(table, data, append=False)
        return


class BlobSink(BlobConnector):
    # This one just need more fancy credentials compared to the default one
    def get_max_seq_id(self, table: str, combo: dict | None = None) -> int:
        """
        Fetch the maximum 'seq_id' of a table, or a subset of a table
        """
        return 0
