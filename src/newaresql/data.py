"""Result container for transformed main and auxiliary test data."""

from typing import overload

import polars as pl

import newaresql.bdf as bdf


@overload
def pivot_aux(aux: pl.DataFrame) -> pl.DataFrame: ...
@overload
def pivot_aux(aux: pl.LazyFrame) -> pl.LazyFrame: ...
def pivot_aux(aux: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Pivot auxiliary measurements into one row per record.

    Returns:
        A frame with one column per auxiliary channel and a mean
        ``test_tmp`` column.

    Raises:
        ValueError: If auxiliary data is unavailable.
    """
    if isinstance(aux, pl.DataFrame):
        on_columns = aux.select("auxchl_id").to_series().unique().to_list()
    else:
        on_columns = aux.select("auxchl_id").unique().collect().to_series().to_list()

    return aux.pivot(
        "auxchl_id",
        on_columns=on_columns,
        index="seq_id",
        values="test_tmp",
        separator="_",
        column_naming="combine",
        aggregate_function=None,  # error on duplicates
    ).with_columns(
        pl.mean_horizontal(pl.selectors.starts_with("test_tmp_")).alias("test_tmp")
    )


class TestData:
    """Hold fetched test frames and expose them in a selectable naming style."""

    def __init__(
        self,
        main: pl.DataFrame,
        aux: pl.DataFrame | None = None,
        naming: bdf.Naming = bdf.DEFAULT,
    ):
        """Initialize a test data result.

        Args:
            main: Transformed main test measurements in BTS naming.
            aux: Optional transformed auxiliary measurements in BTS naming.
            naming: Naming convention used when accessing the frame properties.
        """
        self._main = main
        self._aux = aux
        self._naming: bdf.Naming = naming
        return

    @property
    def naming(self) -> bdf.Naming:
        """Return the naming convention used by the frame properties."""
        return self._naming

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
    def main(self) -> pl.DataFrame:
        """Return the main measurements using the current naming convention."""
        return bdf.convert(self._main, "bts", self._naming)

    @property
    def aux(self) -> pl.DataFrame | None:
        """Return pivoted auxiliary measurements, or ``None`` if unavailable."""
        if self._aux is None:
            return None

        return bdf.convert(pivot_aux(self._aux), "bts", self._naming)

    @property
    def data(self) -> pl.DataFrame:
        """Return main data joined with pivoted auxiliary measurements.

        When auxiliary data is available, the joined frame includes individual
        temperature channels and their per-record mean.
        """
        if self._aux is None:
            return self.main
        else:
            return bdf.convert(
                self._main.drop("test_tmp").join(
                    pivot_aux(self._aux),
                    on="seq_id",
                    how="left",
                ),
                "bts",
                self._naming,
            )
