import logging
from dataclasses import dataclass
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class Field:
    """
    Mapping of a column in the database to its label and code representation.
    Attributes:
        neware (str): The name of the column in the Neware database.
        label (str): The human-readable label for the column.
        code (str): The code representation for the column.
    """

    bts: str
    label: str
    code: str


FIELDS = [
    # Counters and indexes
    Field(bts="step_id", label="Step ID / 1", code="step_id"),
    Field(bts="step_type", label="Step Type / 1", code="step_type"),
    Field(bts="step_index", label="Step Index / 1", code="step_index"),
    Field(bts="seq_id", label="Record Count / 1", code="record_count"),
    Field(bts="step_count", label="Step Count / 1", code="step_count"),
    Field(bts="cycle", label="Cycle Count / 1", code="cycle_count"),
    # Time data
    Field(bts="unix_time", label="Unix Time / s", code="unix_time"),
    Field(bts="test_atime", label="Time / datetime", code="time_datetime"),
    Field(bts="test_time", label="Step Time / s", code="step_time"),
    # Timeseries data
    Field(bts="test_vol", label="Voltage / V", code="voltage_volt"),
    Field(bts="test_cur", label="Current / A", code="current_ampere"),
    Field(bts="test_tmp", label="Temperature / degC", code="temperature_celsius"),
    Field(bts="test_pow", label="Power / W", code="power_watt"),
    # Integral data
    Field(
        bts="test_capchg",
        label="Step Charging Capacity / Ah",
        code="step_charging_capacity_ah",
    ),
    Field(
        bts="test_capdchg",
        label="Step Discharging Capacity / Ah",
        code="step_discharging_capacity_ah",
    ),
    Field(
        bts="test_engchg",
        label="Step Charging Energy / Wh",
        code="step_charging_energy_wh",
    ),
    Field(
        bts="test_engdchg",
        label="Step Discharging Energy / Wh",
        code="step_discharging_energy_wh",
    ),
    Field(
        bts="test_cap",
        label="Step Capacity / Ah",
        code="step_capacity_ah",
    ),
    Field(
        bts="test_eng",
        label="Step Energy / Wh",
        code="step_energy_wh",
    ),
]

MAPPINGS = {
    (src, dst): {getattr(field, src): getattr(field, dst) for field in FIELDS}
    for src in ["bts", "code", "label"]
    for dst in ["bts", "code", "label"]
    if src != dst
}


def convert(
    data: pl.DataFrame,
    src: Literal["bts", "code", "label"] = "bts",
    dst: Literal["bts", "code", "label"] = "label",
) -> pl.DataFrame:
    """
    Convert the column names of a DataFrame.
    src - source
    dst - destination
    Valid values for src and dst are: "bts", "code", "label"
    """
    valid = {"bts", "code", "label"}
    if src not in valid:
        raise ValueError(f"Invalid source: {src}. Valid values are: {valid}")
    if dst not in valid:
        raise ValueError(f"Invalid destination: {dst}. Valid values are: {valid}")

    if src == dst:
        return data

    mapping = MAPPINGS.get((src, dst))
    if mapping is None:
        raise ValueError(f"No mapping found for {src} to {dst}")
    return data.rename(mapping, strict=False)
