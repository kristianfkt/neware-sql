import polars as pl

import newaresql.bdf as bdf


class TestData:
    """
    Class representing test data.
    """

    def __init__(
        self,
        main: pl.DataFrame,
        aux: pl.DataFrame | None = None,
        naming: bdf.Naming = bdf.DEFAULT,
    ):
        self._main = main
        self._aux = aux
        self._naming: bdf.Naming = naming
        return

    @property
    def naming(self) -> bdf.Naming:
        return self._naming

    @naming.setter
    def naming(self, value: bdf.Naming) -> None:
        if value not in bdf.VALID:
            raise ValueError(f"Invalid naming: {value}")
        self._naming = value

    @property
    def main(self) -> pl.DataFrame:
        return bdf.convert(self._main, "bts", self._naming)

    @property
    def aux(self) -> pl.DataFrame | None:
        if self._aux is None:
            return None
        return bdf.convert(self._aux, "bts", self._naming)

    @property
    def data(self) -> pl.DataFrame:
        if self._aux is None:
            return self.main
        else:
            return bdf.convert(
                self._main.drop("test_tmp").join(
                    self._aux.select("auxchl_id", "seq_id", "test_tmp"),
                    on="seq_id",
                    how="left",
                ),
                "bts",
                self._naming,
            )
