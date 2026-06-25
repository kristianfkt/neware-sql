import polars as pl

from newaresql.bts import BTSVersion

# ==============================
# Voltage transformation mappings
# ==============================
MAPPING_VOLTAGE = {
    BTSVersion.BTS63: pl.col("test_vol") / pl.lit(1000),
    BTSVersion.BTS84: pl.col("test_vol") / pl.lit(1000),
}

# ==============================
# Current transformation mappings
# ==============================
MAPPING_CURRENT = {
    BTSVersion.BTS63: pl.col("test_cur") / pl.lit(1000),
    BTSVersion.BTS84: pl.col("test_cur") / pl.lit(1000),
}

# =============================
# Time transformation mappings
# =============================

MAPPING_TIME = {
    BTSVersion.BTS63: pl.col("test_time") / pl.lit(1000),
    BTSVersion.BTS84: pl.col("test_time") / pl.lit(1000),
}

# =============================
# Temperature transformation mappings
# =============================
MAPPING_TEMPERATURE = {
    BTSVersion.BTS63: pl.col("test_temp") / pl.lit(10),
    BTSVersion.BTS84: pl.col("test_temp") / pl.lit(10),
}


BDF_MAPPING = {
    "seq_id": "Record Count / 1",
    "cycle": "Cycle Count / 1",
    "step_type": "Step Type / 1",
    "test_time": "Step Time / s",
    "test_vol": "Voltage / V",
    "test_cur": "Current / A",
    "test_temp": "Temperature / degC",
}


def to_bdf(data: pl.DataFrame) -> pl.DataFrame:
    return data.rename(columns=BDF_MAPPING)


TRANSFORM_MAPPING = {
    "test_cur": MAPPING_CURRENT,
    "test_vol": MAPPING_VOLTAGE,
    "test_time": MAPPING_TIME,
    "test_tmp": MAPPING_TEMPERATURE,
}


def transform(data, version: BTSVersion) -> pl.DataFrame:
    expressions = [
        mapping[column][version].alias(column)
        for column, mapping in TRANSFORM_MAPPING.items()
        if column in data.columns
    ]
    return data.with_columns(expressions)
