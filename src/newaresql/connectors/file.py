from __future__ import annotations

import uuid
from pathlib import Path
from typing import Self

import polars as pl

import newaresql.defaults as defaults
from newaresql.types import Columns, FileFormat, Where

WRITE = {
    "parquet": pl.DataFrame.write_parquet,
    "csv": pl.DataFrame.write_csv,
    "feather": pl.DataFrame.write_ipc,
    "ipc": pl.DataFrame.write_ipc,
}

SCAN = {
    "parquet": pl.scan_parquet,
    "csv": pl.scan_csv,
    "feather": pl.scan_ipc,
    "ipc": pl.scan_ipc,
}


def filter_lazyframe(
    lazy: pl.LazyFrame,
    columns: Columns | None = None,
    where: Where | None = None,
) -> pl.LazyFrame:
    if where is None:
        where = {}
    clauses = []
    for col, pred in where.items():
        if isinstance(pred, list):
            clauses.append(pl.col(col).is_in(pred))
        elif isinstance(pred, tuple) and len(pred) == 2:
            lo, hi = pred
            if (lo is not None) and (hi is not None):
                clauses.append(pl.col(col).is_between(lo, hi))
            elif lo is not None:
                clauses.append(pl.col(col) >= lo)
            elif hi is not None:
                clauses.append(pl.col(col) <= hi)
        else:
            clauses.append(pl.col(col) == pred)
    if len(clauses) == 1:
        predicate = clauses[0]
    elif len(clauses) > 1:
        predicate = clauses[0].and_(*clauses[1:])
    else:
        predicate = None
    if predicate:
        lazy = lazy.filter(predicate)
    if isinstance(columns, str):
        columns = [columns]
    if columns is not None:
        lazy = lazy.select(columns)
    return lazy


class FileConnector:
    def __init__(
        self,
        *,
        root: str | Path,
        file_format: FileFormat = defaults.FILE_FORMAT,
    ):

        if isinstance(root, str):
            root = Path(root)
        self._root = root
        self._file_format = file_format
        return

    @property
    def root(self) -> Path:
        return self._root

    @property
    def file_format(self) -> str:
        return self._file_format

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return

    def has_folder(self, folder: str) -> bool:
        return (
            self._root.joinpath(folder).exists()
            and self._root.joinpath(folder).is_dir()
        )

    def list_folders(self) -> list[Path]:
        if not self._root.exists():
            return []
        return [f for f in self._root.iterdir() if f.is_dir()]

    def list_files(self) -> list[Path]:
        if not self._root.exists():
            return []
        return [f for f in self._root.rglob(f"*.{self.file_format}")]

    def scan_folder(self, folder: str) -> pl.LazyFrame:
        return SCAN[self._file_format](self._root.joinpath(folder))

    def write_folder(
        self,
        folder: str,
        data: pl.DataFrame,
        append: bool = True,
        name: str | None = None,
    ) -> None:
        """
        Write a DataFrame to a folder, optionally appending to existing files.
        Parameters:
        ----------
        folder : str
            The name of the folder to write the data to.
        data : pl.DataFrame
            The DataFrame to write.
        append : bool, optional
            Whether to append to existing files in the folder (default is True).
            False will delete all existing files, but keep the folder itself.
        name : str | None, optional
            The name of the file to write. If None, a UUID-based name will be generated.
        """
        path = self.root.joinpath(folder)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        if not append:
            for f in path.glob(f"*.{self.file_format}"):
                f.unlink()

        if name is None:
            name = f"{uuid.uuid4()}.{self.file_format}"
        file = self.root.joinpath(folder).joinpath(name)
        WRITE[self.file_format](data, file)
        return

    # def list_tests(self) -> list[dict]:
    #     files = self.root.rglob("*.json")
    #     return [load_json(f) for f in files]

    # def get_tests(self) -> pl.DataFrame:
    #     return pl.DataFrame(self.list_tests())

    # def get_stats(self, test: Test) -> dict:
    #     lazy = self.scan_main_data(test)
    #     return (
    #         lazy.select(pl.col("seq_id").max().alias("max_seq_id"))
    #         .collect(engine="streaming")
    #         .to_dict(as_series=False)
    #     )

    # def scan_main_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     columns: Columns | None = None,
    #     naming: Naming = defaults.NAMING,
    # ) -> pl.LazyFrame:
    #     keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
    #     folder = "_".join(["main"] + [str(test[k]) for k in keys])
    #     lazy = SCAN[self._file_format](self._root.joinpath(folder))
    #     if columns or where:
    #         lazy = filter_lazyframe(lazy, columns=columns, where=where)
    #     return lazy

    # def scan_aux_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     columns: Columns | None = None,
    #     naming: Naming = defaults.NAMING,
    # ) -> pl.LazyFrame:
    #     keys = ["dev_uid", "unit_id", "chl_id", "test_id"]
    #     folder = "_".join(["aux"] + [str(test[k]) for k in keys])
    #     lazy = SCAN[self._file_format](self._root.joinpath(folder))
    #     if columns or where:
    #         lazy = filter_lazyframe(lazy, columns=columns, where=where)
    #     return lazy

    # def scan_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     naming: Naming = defaults.NAMING,
    # ) -> pl.LazyFrame:
    #     main = self.scan_main_data(
    #         test,
    #         where=where,
    #         naming=naming,
    #     )
    #     aux = self.scan_aux_data(
    #         test,
    #         where=where,
    #         naming=naming,
    #     )
    #     return main.join(aux, on="seq_id", how="left")

    # def get_main_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     columns: Columns | None = None,
    #     naming: Naming = defaults.NAMING,
    #     extend: bool = True,
    # ) -> pl.DataFrame:
    #     data = convert(
    #         self.scan_main_data(
    #             test,
    #             where=where,
    #             columns=columns,
    #             naming=naming,
    #         ).collect(engine="streaming"),
    #         "bts",
    #         naming,
    #     )
    #     if extend:
    #         data = extend_data(data)
    #     return data

    # def get_aux_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     columns: Columns | None = None,
    #     naming: Naming = defaults.NAMING,
    #     extend: bool = True,
    # ) -> pl.DataFrame:
    #     data = convert(
    #         self.scan_aux_data(
    #             test,
    #             where=where,
    #             columns=columns,
    #             naming=naming,
    #         ).collect(engine="streaming"),
    #         "bts",
    #         naming,
    #     )
    #     if extend:
    #         data = extend_data(data)
    #     return data

    # def get_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     naming: Naming = defaults.NAMING,
    #     extend: bool = True,
    # ) -> pl.DataFrame:
    #     data = self.scan_data(
    #         test,
    #         where=where,
    #         naming=naming,
    #     ).collect(engine="streaming")
    #     if extend:
    #         data = extend_data(data)
    #     return data

    # def stream_main_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     columns: Columns | None = None,
    #     naming: Naming = defaults.NAMING,
    #     extend: bool = True,
    #     chunk_size: int = defaults.CHUNK_SIZE,
    # ) -> Iterator[pl.DataFrame]:
    #     lazy = self.scan_main_data(
    #         test,
    #         where=where,
    #         columns=columns,
    #         naming=naming,
    #     )
    #     for chunk in lazy.collect_batches(
    #         chunk_size=chunk_size, maintain_order=True, engine="streaming"
    #     ):
    #         if extend:
    #             chunk = extend_data(chunk)
    #         yield chunk

    # def stream_aux_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     columns: Columns | None = None,
    #     naming: Naming = defaults.NAMING,
    #     extend: bool = True,
    #     chunk_size: int = defaults.CHUNK_SIZE,
    # ) -> Iterator[pl.DataFrame]:
    #     lazy = self.scan_aux_data(
    #         test,
    #         where=where,
    #         columns=columns,
    #         naming=naming,
    #     )
    #     for chunk in lazy.collect_batches(
    #         chunk_size=chunk_size, maintain_order=True, engine="streaming"
    #     ):
    #         if extend:
    #             chunk = extend_data(chunk)
    #         yield chunk

    # def stream_data(
    #     self,
    #     test: Test,
    #     *,
    #     where: Where | None = None,
    #     naming: Naming = defaults.NAMING,
    #     extend: bool = True,
    #     chunk_size: int = defaults.CHUNK_SIZE,
    # ) -> Iterator[pl.DataFrame]:
    #     lazy = self.scan_data(
    #         test,
    #         where=where,
    #         naming=naming,
    #     )
    #     for chunk in lazy.collect_batches(
    #         chunk_size=chunk_size, maintain_order=True, engine="streaming"
    #     ):
    #         if extend:
    #             chunk = extend_data(chunk)
    #         yield chunk
