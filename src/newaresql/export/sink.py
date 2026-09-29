import polars as pl

from newaresql.connectors.blob import BlobConnector
from newaresql.connectors.file import FileConnector
from newaresql.connectors.sql import SQLConnector
from newaresql.types import Test
from newaresql.utils.test import get_test_name


class SQLSink(SQLConnector):
    pass


class FileSink(FileConnector):
    def get_tests(self):
        if not self.has_folder("tests"):
            return pl.DataFrame([])
        return self.scan_folder("tests").collect(engine="streaming")

    def list_tests(self):
        if self.has_folder("tests"):
            return self.get_tests().to_dicts()
        return []

    def contains_test(self, test: Test) -> bool:
        """
        Checks if root/test/*.file_format exists
        """
        name = get_test_name(test)
        files = list(self.root.joinpath(name).rglob(f"*.{self.file_format}"))
        return len(files) > 0

    def get_status(self, test: Test):
        pass

    def update_data(self, test: Test, data: pl.DataFrame):
        name = get_test_name(test)
        self.write_folder(name, data, append=True)
        return

    def update_meta(self, test: Test):
        # Writes new row-file, or overwrites existing one
        name = get_test_name(test)
        self.write_folder("tests", pl.DataFrame([test]), append=True, name=name)
        return

    def get_max_seq_id(self, test: Test) -> int | None:
        name = get_test_name(test)
        frame = self.scan_folder(name)
        seq_id = (
            frame.select(pl.col("seq_id").max())
            .collect(engine="streaming")
            .select("seq_id")
            .item()
        )
        return seq_id


class BlobSink(BlobConnector):
    pass
