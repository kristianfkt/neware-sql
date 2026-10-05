import uuid

import polars as pl
from adlfs import AzureBlobFileSystem

from newaresql import defaults
from newaresql.types import FileFormat
from newaresql.utils.file import SCAN, WRITE


class BlobConnector:
    def __init__(
        self,
        *,
        container: str,
        storage_options: dict,
        file_format: FileFormat = defaults.FILE_FORMAT,
    ):
        self._container = container
        self._storage_options = storage_options
        self._file_format = file_format
        self._file_system = AzureBlobFileSystem(**storage_options)

    @property
    def container(self) -> str:
        return self._container

    @property
    def storage_options(self) -> dict:
        return self._storage_options

    @property
    def file_format(self) -> str:
        return self._file_format

    @property
    def file_system(self) -> AzureBlobFileSystem:
        return self._file_system

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def has_folder(self, folder: str) -> bool:
        # This is a placeholder implementation. Replace it with actual logic to check if a folder exists in the blob container.
        return False

    def list_folders(self) -> list[str]:
        # This is a placeholder implementation. Replace it with actual logic to list folders in the blob container.
        blobs = self._file_system.ls(self.container, detail=True)
        tables = {
            b["name"].removeprefix(f"{self.container}/")
            for b in blobs
            if b["type"] == "directory"
        }
        return list(tables)

    def scan_folder(self, folder: str) -> pl.LazyFrame:
        pattern = f"az://{self._container}/{folder}/*.{self._file_format}"
        return SCAN[self._file_format](pattern, storage_options=self._storage_options)

    def read_folder(self, folder: str) -> pl.DataFrame:
        return self.scan_folder(folder).collect(engine="streaming")

    def write_folder(
        self,
        folder: str,
        data: pl.DataFrame,
        append: bool = False,
        name: str | None = None,
    ) -> None:
        """
        ...
        """
        if not append:
            for file in self._file_system.glob(
                f"{self.container}/{folder}/*.{self._file_format}"
            ):
                self._file_system.rm(file)

        if name is None:
            name = str(uuid.uuid4())
        path = f"az://{self._container}/{folder}/{name}.{self._file_format}"
        WRITE[self._file_format](data, path, storage_options=self._storage_options)
        return
