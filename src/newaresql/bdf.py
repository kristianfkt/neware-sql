import logging

import polars as pl

logger = logging.getLogger(__name__)


NEWARE_MAPPING = {
    "unix_time": "Unix Time / s",
    "test_atime": "Time / datetime",
    "seq_id": "Record Count / 1",
    "cycle": "Cycle Count / 1",
    "step_type": "Step Type / 1",
    "step_id": "Step ID / 1",
    "test_time": "Step Time / s",
    "test_vol": "Voltage / V",
    "test_cur": "Current / A",
    "test_tmp": "Temperature / degC",
    "test_capchg": "Step Charging Capacity / Ah",
    "test_capdchg": "Step Discharging Capacity / Ah",
    "test_engchg": "Step Charging Energy / Wh",
    "test_engdchg": "Step Discharging Energy / Wh",
    "test_cap": "Step Capacity / Ah",
    "test_eng": "Step Energy / Wh",
    "test_pow": "Power / W",
    "auxchl_id": "Aux Channel ID / 1",
    "test_totaltime": "Test Time / s",
    "step_index": "Step Index / 1",
    "step_count": "Step Count / 1",
}


def to_bdf(data: pl.DataFrame) -> pl.DataFrame:
    """
    Convert column names to BDF format using the NEWARE_MAPPING dictionary. If a column name is not found in the mapping, it will remain unchanged.
    """
    return data.rename(NEWARE_MAPPING, strict=False)
