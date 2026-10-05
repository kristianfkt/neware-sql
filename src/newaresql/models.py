from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from newaresql.protocols import Connector

if TYPE_CHECKING:
    import matplotlib.figure


@dataclass
class Test:
    metadata: dict
    connector: Connector | None = None

    @property
    def device(self) -> int:
        return self.metadata["dev_uid"]

    @property
    def unit(self) -> int:
        return self.metadata["unit_id"]

    @property
    def channel(self) -> int:
        return self.metadata["chl_id"]

    @property
    def id(self) -> int:
        return self.metadata["id"]

    @property
    def name(self) -> str:
        return f"{self.device}-{self.unit}-{self.channel}-{self.id}"

    @property
    def batch(self) -> str:
        return self.metadata["batch_no"]

    @property
    def remark(self) -> str:
        return self.metadata["remark"]

    @property
    def start(self) -> datetime:
        return self.metadata["start_time"]

    @property
    def end(self) -> datetime:
        return self.metadata["end_time"]

    @property
    def duration(self) -> float:
        return (self.end - self.start).total_seconds()

    def plot(self, show: bool = False) -> matplotlib.figure.Figure:
        if self.connector is None:
            raise ValueError("No connector available for plotting.")
        import matplotlib.pyplot as plt

        data = self.connector.get_data(self.metadata, naming="label", extend=True)
        columns = ["Current / A", "Voltage / V", "Temperature / degC"]
        fig, axs = plt.subplots(len(columns), 1, constrained_layout=True)
        for i, column in enumerate(columns):
            axs[i].plot(
                data.select("Test time / s").to_series(),
                data.select(column).to_series(),
            )
            axs[i].set_ylabel(column)
            axs[i].set_xlabel("Test time / s")
        if show:
            plt.show()
        return fig
