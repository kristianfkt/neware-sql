"""Column metadata and conversion between BTS and BDF naming conventions."""

from typing import Literal, TypedDict, overload

import polars as pl

Naming = Literal["bts", "code", "label"]

VALID: set[Naming] = {"bts", "code", "label"}
DEFAULT: Naming = "label"


class Column(TypedDict):
    """Names for a dataset column in BTS, BDF code, and BDF label forms."""

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
    Column(bts="seq_id", code="record_index", label="Record Index / 1"),
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
    Column(
        bts="test_tmp_1", code="temperature_celsius_1", label="Temperature T1/ degC"
    ),
    Column(
        bts="test_tmp_2", code="temperature_celsius_2", label="Temperature T2/ degC"
    ),
    Column(
        bts="test_tmp_3", code="temperature_celsius_3", label="Temperature T3/ degC"
    ),
    Column(
        bts="test_tmp_4", code="temperature_celsius_4", label="Temperature T4/ degC"
    ),
    Column(
        bts="test_tmp_5", code="temperature_celsius_5", label="Temperature T5/ degC"
    ),
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


@overload
def convert(
    data: pl.DataFrame,
    source: Naming,
    destination: Naming,
) -> pl.DataFrame: ...
@overload
def convert(
    data: pl.LazyFrame,
    source: Naming,
    destination: Naming,
) -> pl.LazyFrame: ...
def convert(
    data: pl.DataFrame | pl.LazyFrame,
    source: Naming,
    destination: Naming,
) -> pl.DataFrame | pl.LazyFrame:
    """Convert mapped DataFrame columns to another naming convention.

    Args:
        data: DataFrame whose columns use the source convention.
        source: Naming convention currently used by the columns.
        destination: Naming convention to use in the returned frame.

    Returns:
        The input frame when both conventions match; otherwise, a renamed
        frame containing columns with known mappings.

    Raises:
        ValueError: If either naming convention is unsupported.
    """
    if source not in VALID:
        raise ValueError(f"Invalid source naming: {source}")
    if destination not in VALID:
        raise ValueError(f"Invalid destination naming: {destination}")
    if source == destination:
        return data

    mapping = MAPS.get((source, destination), {})
    schema = (
        data.columns
        if isinstance(data, pl.DataFrame)
        else data.collect_schema().names()
    )
    mapping = {k: v for k, v in mapping.items() if k in schema}
    return data.rename(mapping, strict=False).select(*mapping.values())
