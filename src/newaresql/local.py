"""Read and synchronize local Parquet copies of Neware test data."""

import concurrent.futures
import logging
import os
from collections.abc import Callable
from pathlib import Path

import polars as pl

from newaresql import bdf
from newaresql.connector import Connector
from newaresql.data import pivot_aux
from newaresql.types import Test

logger = logging.getLogger(__name__)


def _test_name(test: Test) -> str:
    """Build the directory name from a test's device and channel identifiers."""
    return f"{test['dev_uid']}-{test['unit_id']}-{test['chl_id']}-{test['test_id']}"


def _test_path(test: Test, root: Path) -> Path:
    """Return the directory for a test under the local root."""
    name = _test_name(test)
    return root.joinpath(name)


def _get_root(root: str | Path | None = None) -> Path:
    """Resolve the local root from an argument or the environment.

    Args:
        root: Local root directory. Defaults to ``NEWARE_LOCAL_ROOT``.

    Returns:
        The local root as a path.

    Raises:
        ValueError: If neither a root nor the environment variable is set.
    """
    if root is None:
        root = os.getenv("NEWARE_LOCAL_ROOT")
    if isinstance(root, str):
        root = Path(root)
    if root is None:
        raise ValueError(
            "Local root directory must be specified either via argument or NEWARE_LOCAL_ROOT environment variable."
        )
    return root


class LocalData:
    """Hold local test frames and expose them in a selectable naming style."""

    def __init__(
        self,
        main: pl.DataFrame | pl.LazyFrame,
        aux: pl.DataFrame | pl.LazyFrame | None = None,
        naming: bdf.Naming = bdf.DEFAULT,
        lazy: bool = False,
    ):
        """Initialize a local test data result.

        Args:
            main: Main measurements in BTS naming.
            aux: Optional auxiliary measurements in BTS naming.
            naming: Naming convention used when accessing frame properties.
            lazy: Whether the supplied frames are being handled lazily. This
                does not convert the frames during initialization.
        """
        self._main = main
        self._aux = aux
        self._lazy = lazy
        self._naming: bdf.Naming = naming
        return

    @property
    def naming(self) -> bdf.Naming:
        """Return the naming convention used by the frame properties."""
        return self._naming

    @property
    def lazy(self) -> bool:
        """Return whether the data is being handled lazily."""
        return self._lazy

    @lazy.setter
    def lazy(self, value: bool) -> None:
        """Convert the stored frames to lazy or eager frames.

        Args:
            value: Whether to use lazy frames. Setting this to ``False``
                collects any lazy frames.
        """
        self._lazy = value
        if self._lazy:
            if isinstance(self._main, pl.DataFrame):
                self._main = self._main.lazy()
            if isinstance(self._aux, pl.DataFrame):
                self._aux = self._aux.lazy()
        else:
            if isinstance(self._main, pl.LazyFrame):
                self._main = self._main.collect()
            if self._aux is not None and isinstance(self._aux, pl.LazyFrame):
                self._aux = self._aux.collect()
        return

    @naming.setter
    def naming(self, value: bdf.Naming) -> None:
        """Set the naming convention used by subsequent property access.

        Args:
            value: One of ``"bts"``, ``"code"``, or ``"label"``.

        Raises:
            ValueError: If ``value`` is not a supported naming convention.
        """
        if value not in bdf.VALID:
            raise ValueError(f"Invalid naming: {value}")
        self._naming = value

    @property
    def main(self) -> pl.DataFrame | pl.LazyFrame:
        """Return the main measurements using the current naming convention."""
        return bdf.convert(self._main, "bts", self._naming)

    @property
    def aux(self) -> pl.DataFrame | pl.LazyFrame | None:
        """Return pivoted auxiliary measurements, or ``None`` if unavailable.

        For lazy frames, channel IDs are collected to construct the pivot.
        """
        if self._aux is None:
            return None

        return bdf.convert(pivot_aux(self._aux), "bts", self._naming)

    @property
    def data(self) -> pl.DataFrame | pl.LazyFrame:
        """Return main data joined with pivoted auxiliary measurements.

        When auxiliary data is available, the joined frame includes individual
        temperature channels and their per-record mean. For lazy auxiliary
        data, channel IDs are collected to construct the pivot.
        """
        if self._aux is None:
            return self.main
        else:
            return bdf.convert(
                self._main.drop("test_tmp").join(
                    pivot_aux(self._aux),  # type: ignore
                    on="seq_id",
                    how="left",
                ),
                "bts",
                self._naming,
            )


def get_tests(root: str | Path | None = None) -> pl.DataFrame | None:
    """Read metadata for locally stored tests.

    Args:
        root: Local root directory. Defaults to ``NEWARE_LOCAL_ROOT``.

    Returns:
        A diagonally concatenated metadata frame, or ``None`` if the root does
        not exist or no matching metadata Parquet files are found beneath it.

    Raises:
        ValueError: If no root is specified or configured.
    """
    root = _get_root(root)
    if not root.exists():
        logger.debug("Local root does not exist: %s", root)
        return None

    files = list(root.rglob("*metadata.parquet"))
    logger.debug("Found %d metadata files under %s", len(files), root)
    if not files:
        return None
    else:
        return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def list_tests(root: str | Path | None = None) -> list[Test] | None:
    """List locally stored test metadata as dictionaries.

    Args:
        root: Local root directory. Defaults to ``NEWARE_LOCAL_ROOT``.

    Returns:
        A list of test metadata dictionaries, or ``None`` if the root does
        not exist or no metadata files are found.

    Raises:
        ValueError: If no root is specified or configured.
    """
    tests = get_tests(root)
    if tests is None:
        return None
    else:
        return tests.to_dicts()


def get_data(
    test: Test,
    naming: bdf.Naming = bdf.DEFAULT,
    lazy: bool = True,
    root: str | Path | None = None,
) -> LocalData:
    """Load locally stored main and optional auxiliary measurements.

    Args:
        test: Test metadata containing ``dev_uid``, ``unit_id``, ``chl_id``,
            and ``test_id`` to identify the local test directory.
        naming: Naming convention used when accessing frame properties.
        lazy: Whether to return lazy frames instead of collecting them.
        root: Local root directory. Defaults to ``NEWARE_LOCAL_ROOT``.

    Returns:
        A container for the main and, when available, auxiliary frames.

    Raises:
        ValueError: If no root is specified or configured.
        FileNotFoundError: If the root does not exist or the test has no main
            Parquet files.
    """
    root = _get_root(root)
    if not root.exists():
        raise FileNotFoundError(f"Local root directory '{root}' does not exist.")

    path = _test_path(test, root)
    main_files = list(path.glob("main-*.parquet"))
    aux_files = list(path.glob("aux-*.parquet"))
    logger.debug(
        "Loading data from %s: %d main files, %d auxiliary files, lazy=%s",
        path,
        len(main_files),
        len(aux_files),
        lazy,
    )
    if not main_files:
        raise FileNotFoundError(f"No main parquet files found in '{path}'.")
    if not aux_files:
        aux_data = None
    else:
        aux_data = pl.scan_parquet(aux_files)
    main_data = pl.scan_parquet(main_files)
    if not lazy:
        main_data = main_data.collect()
        if aux_data is not None:
            aux_data = aux_data.collect()

    return LocalData(main=main_data, aux=aux_data, naming=naming, lazy=lazy)


def _update_test_main(
    *,
    test: Test,
    path: Path,
    connector: Connector,
    chunk_size: int = 100_000,
):
    """Append main measurements in sequence-ID chunks to a test directory."""
    # We name each file with the actual max seq_id
    # main-start-end.parquet
    name = _test_name(test)
    files = list(path.glob("main-*.parquet"))
    seq_ids = [int(f.stem.split("-")[2]) for f in files]
    if seq_ids:
        i = max(seq_ids) + 1
    else:
        i = 1
    while True:
        logger.debug(
            f"Fetching main data for {name} with seq_id range {i}-{i + chunk_size - 1}"
        )
        data = connector.get_main_data(
            test, conditions={"seq_id": (i, i + chunk_size - 1)}
        )
        if data.is_empty():
            break
        j = data.select(pl.col("seq_id").max()).item()
        data.write_parquet(path / f"main-{i}-{j}.parquet")
        i = j + 1

    return


def _update_test_aux(
    test: Test,
    path: Path,
    connector: Connector,
    chunk_size: int = 100_000,
):
    """Append auxiliary measurements in sequence-ID chunks to a test directory."""
    # We name each file with the actual max seq_id
    # aux-start-end.parquet
    # Each file will be twice the chunk size if auxchl_id has 2 values
    name = _test_name(test)
    files = list(path.glob("aux-*.parquet"))
    seq_ids = [int(f.stem.split("-")[2]) for f in files]
    if seq_ids:
        i = max(seq_ids) + 1
    else:
        i = 1
    while True:
        logger.debug(
            f"Fetching auxiliary data for {name} with seq_id range {i}-{i + chunk_size - 1}"
        )
        data = connector.get_aux_data(
            test, conditions={"seq_id": (i, i + chunk_size - 1)}
        )
        if (data is None) or data.is_empty():
            break
        j = data.select(pl.col("seq_id").max()).item()
        data.write_parquet(path / f"aux-{i}-{j}.parquet")
        i = j + 1

    return


def _update_test(
    *,
    test: Test,
    root: Path,
    connector: Connector,
    chunk_size: int = 100_000,
):
    """Update a test's data and metadata, logging data-update failures."""
    path = _test_path(test, root)
    name = _test_name(test)
    logger.info("Updating test %s", name)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    try:
        _update_test_main(
            test=test, path=path, connector=connector, chunk_size=chunk_size
        )
        logger.info(f"Updated main data for {name}")
    except Exception:
        logger.exception("Failed to update main data for %s", name)
        return

    try:
        _update_test_aux(
            test=test, path=path, connector=connector, chunk_size=chunk_size
        )
        logger.info(f"Updated auxiliary data for {name}")
    except Exception:
        logger.exception("Failed to update auxiliary data for %s", name)
        return
    pl.DataFrame([test]).write_parquet(path / "metadata.parquet")
    logger.info("Updated metadata for %s", name)
    return


def update(
    root: str | Path | None = None,
    connector: Connector | None = None,
    host: str | None = None,
    port: int | str | None = None,
    username: str | None = None,
    password: str | None = None,
    database: str | None = None,
    callback: Callable[[Test], bool] | None = None,
    chunk_size: int = 100_000,
    workers: int | None = None,
):
    """Synchronize selected database tests to the local root.

    Existing data is extended from the largest sequence ID in each set of
    main and auxiliary files. Main and auxiliary update failures are logged
    per test. Other per-test errors propagate in sequential mode but are
    logged in threaded mode. Connection, test listing, callback, and root
    directory creation errors can propagate in either mode.

    Args:
        root: Local root directory, created if missing. Defaults to
            ``NEWARE_LOCAL_ROOT``.
        connector: Existing database connector. If omitted, a connector is
            created using the connection arguments below.
        host: Database host, falling back to ``NEWARE_BTS_HOST``.
        port: Database port, falling back to ``NEWARE_BTS_PORT``.
        username: Database username, falling back to ``NEWARE_BTS_USERNAME``.
        password: Database password, falling back to ``NEWARE_BTS_PASSWORD``.
        database: Database name, falling back to ``NEWARE_BTS_DATABASE``.
        callback: Predicate receiving each test; only tests for which it
            returns true are updated. Defaults to updating every test.
        chunk_size: Number of sequence IDs requested per data chunk.
        workers: Number of update threads. ``None``, 0, and 1 run sequentially;
            -1 uses the thread pool's default worker count.

    Raises:
        ValueError: If no root is specified or configured, or ``workers`` is
            an integer less than -1.
    """
    # One recusrive call - will fail is credentials are not set correctly
    # if connector cannot be established.
    if connector is None:
        with Connector(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        ) as connector:
            return update(
                root=root,
                connector=connector,
                callback=callback,
                chunk_size=chunk_size,
                workers=workers,
            )

    root = _get_root(root)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    if workers in (0, 1):
        workers = None
    if isinstance(workers, int) and (workers < -1):
        raise ValueError("Invalid number of workers: must be -1 or greater")

    logger.info("Synchronizing tests to %s with workers=%s", root, workers)
    if workers is None:
        for test in connector.list_tests():
            name = _test_name(test)
            logger.info(f"Processing {name}")
            if (callback is not None) and not callback(test):
                logger.info(f"Skipping {name}")
                continue
            _update_test(
                test=test, root=root, connector=connector, chunk_size=chunk_size
            )
        logger.info("Finished synchronizing tests to %s", root)
        return

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=None if workers == -1 else workers
    ) as executor:
        futures = {}
        for test in connector.list_tests():
            name = _test_name(test)
            if (callback is not None) and not callback(test):
                logger.info("Skipping %s", name)
                continue
            future = executor.submit(
                _update_test,
                test=test,
                root=root,
                connector=connector,
                chunk_size=chunk_size,
            )
            futures[future] = name
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception:
                logger.exception("Error updating %s", name)
        logger.info("Finished synchronizing tests to %s", root)
        return


def delete(
    test: Test,
    root: str | Path | None = None,
):
    """Delete a test's local Parquet files and its directory.

    Args:
        test: Test metadata containing ``dev_uid``, ``unit_id``, ``chl_id``,
            and ``test_id`` to identify the local test directory.
        root: Local root directory. Defaults to ``NEWARE_LOCAL_ROOT``.

    Raises:
        ValueError: If no root is specified or configured.
        OSError: If the directory cannot be removed, including when it still
            contains non-Parquet files or subdirectories.
    """
    root = _get_root(root)
    path = _test_path(test, root)
    if path.exists():
        logger.info("Deleting local test data from %s", path)
        for file in path.glob("*.parquet"):
            file.unlink()
        path.rmdir()
    return


def clean(root: str | Path | None = None):
    """Remove immediate subdirectories without a ``metadata.parquet`` file.

    Parquet files in each matching directory are deleted before the directory
    is removed. A missing root is ignored.

    Args:
        root: Local root directory. Defaults to ``NEWARE_LOCAL_ROOT``.

    Raises:
        ValueError: If no root is specified or configured.
        OSError: If a directory cannot be removed, including when it still
            contains non-Parquet files or subdirectories.
    """

    root = _get_root(root)
    if not root.exists():
        return
    for item in root.iterdir():
        if item.is_dir() and not item.joinpath("metadata.parquet").exists():
            logger.info("Cleaning incomplete local test directory %s", item)
            for file in item.glob("*.parquet"):
                file.unlink()
            item.rmdir()


def purge(root: str | Path | None = None):
    """Remove all immediate subdirectories of the local root.

    Parquet files in each directory are deleted before the directory is
    removed. Files directly in the root are left untouched; a missing root is
    ignored.

    Args:
        root: Local root directory. Defaults to ``NEWARE_LOCAL_ROOT``.

    Raises:
        ValueError: If no root is specified or configured.
        OSError: If a directory cannot be removed, including when it still
            contains non-Parquet files or subdirectories.
    """
    root = _get_root(root)
    if not root.exists():
        return
    for item in root.iterdir():
        if item.is_dir():
            logger.info("Purging local directory %s", item)
            for file in item.glob("*.parquet"):
                file.unlink()
            item.rmdir()


__all__ = [
    "get_data",
    "list_tests",
    "get_tests",
    "update",
    "delete",
    "clean",
    "purge",
]
