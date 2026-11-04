from typing import Literal, TypedDict

import polars as pl

Naming = Literal["bts", "code", "label"]
VALID: set[str] = {"bts", "code", "label"}
DEFAULT: Naming = "label"


class Column(TypedDict):
    """
    Columns representing the metadata of a column in the dataset.
    """

    bts: str
    code: str
    label: str


COLUMNS = [
    Column(bts="dev_uid", code="device_uid", label="Device UID / 1"),
    Column(bts="unit_id", code="unit_id", label="Unit ID / 1"),
    Column(bts="chl_id", code="channel_id", label="Channel ID / 1"),
    Column(bts="test_id", code="test_id", label="Test ID / 1"),
    Column(bts="step_id", code="step_id", label="Step ID / 1"),
    Column(bts="step_type", code="step_type", label="Step Type / 1"),
    Column(bts="step_index", code="step_index", label="Step Index / 1"),
    Column(bts="seq_id", code="record_count", label="Record Count / 1"),
    Column(bts="step_count", code="step_count", label="Step Count / 1"),
    Column(bts="cycle", code="cycle_count", label="Cycle Count / 1"),
    Column(bts="auxchl_id", code="aux_channel_id", label="Aux Channel ID / 1"),
    Column(bts="unix_time", code="unix_time", label="Unix Time / s"),
    Column(bts="test_atime", code="time_datetime", label="Time / datetime"),
    Column(bts="test_time", code="step_time", label="Step Time / s"),
    Column(bts="test_totaltime", code="total_time", label="Test Time / s"),
    Column(bts="test_vol", code="voltage_volt", label="Voltage / V"),
    Column(bts="test_cur", code="current_ampere", label="Current / A"),
    Column(bts="test_tmp", code="temperature_celsius", label="Temperature / degC"),
    Column(bts="test_pow", code="power_watt", label="Power / W"),
    Column(
        bts="test_capchg",
        code="step_charging_capacity_ah",
        label="Step Charging Capacity / Ah",
    ),
    Column(
        bts="test_capdchg",
        code="step_discharging_capacity_ah",
        label="Step Discharging Capacity / Ah",
    ),
    Column(
        bts="test_engchg",
        code="step_charging_energy_wh",
        label="Step Charging Energy / Wh",
    ),
    Column(
        bts="test_engdchg",
        code="step_discharging_energy_wh",
        label="Step Discharging Energy / Wh",
    ),
    Column(bts="test_cap", code="step_capacity_ah", label="Step Capacity / Ah"),
    Column(bts="test_eng", code="step_energy_wh", label="Step Energy / Wh"),
]

MAPS = {
    (src, dst): {col[src]: col[dst] for col in COLUMNS}
    for src in ["bts", "code", "label"]
    for dst in ["bts", "code", "label"]
}


def convert(
    data: pl.DataFrame,
    source: Naming,
    destination: Naming,
):
    if source not in VALID:
        raise ValueError(f"Invalid source naming: {source}")
    if destination not in VALID:
        raise ValueError(f"Invalid destination naming: {destination}")
    if source == destination:
        return data

    mapping = MAPS.get((source, destination), {})
    mapping = {k: v for k, v in mapping.items() if k in data.columns}
    return data.rename(mapping, strict=False)
