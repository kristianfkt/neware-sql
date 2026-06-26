import polars as pl

BDF_MAPPING = {
    "seq_id": "Record Count / 1",
    "cycle": "Cycle Count / 1",
    "step_type": "Step Type / 1",
    "test_time": "Step Time / s",
    "test_vol": "Voltage / V",
    "test_cur": "Current / A",
    "test_tmp": "Temperature / degC",
}


def to_bdf(data: pl.DataFrame) -> pl.DataFrame:
    return data.rename(BDF_MAPPING, strict=False)


def _transform_24(data: pl.DataFrame) -> pl.DataFrame:
    expressions = [
        (pl.col("test_vol") / 10000.0).alias("test_vol"),
        (pl.col("test_cur") / 10000.0).alias("test_cur"),
        (pl.col("test_time") / 1000.0).alias("test_time"),
        (pl.col("test_tmp") / 10.0).alias("test_tmp"),
    ]

    return data.with_columns(*expressions)


def _transform_26(data: pl.DataFrame) -> pl.DataFrame:
    expressions = [
        (pl.col("test_vol") / 10000.0).alias("test_vol"),
        (pl.col("test_cur") / 10000.0).alias("test_cur"),
        (pl.col("test_time") / 1000.0).alias("test_time"),
        (pl.col("test_tmp") / 10.0).alias("test_tmp"),
    ]

    return data.with_columns(*expressions)


DISPATCH = {
    "24": _transform_24,
    "26": _transform_26,
}


def transform(data, test: dict) -> pl.DataFrame:
    dev_uid = test.get("dev_uid")
    dev_type = str(dev_uid)[0:2]
    if dev_type not in DISPATCH:
        raise ValueError(f"Unsupported device type: {dev_type}")
    return to_bdf(DISPATCH[dev_type](data)).select(*list(BDF_MAPPING.values()))
