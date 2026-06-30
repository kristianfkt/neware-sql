import logging

import polars as pl

logger = logging.getLogger(__name__)
_BDF_MAPPING = {
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


def wrap_nullcheck(expression: pl.Expr) -> pl.Expr:
    """
    Wraps a polars expression with a null check.
    If any of the columns in the expression are null, the entire expression will return None.
    This is useful for ensuring that calculations do not crash with incomplete data.
    """
    columns = set(expression.meta.root_names())
    if columns:
        logger.debug(f"Wrapping expression with null check for columns: {columns}")
        nullcheck = pl.any_horizontal(*(pl.col(col).is_null() for col in columns))
        expression = pl.when(nullcheck).then(pl.lit(None)).otherwise(expression)
    else:
        logger.debug("No columns found in expression; skipping null check wrapping.")
    return expression


def check_required(data: pl.DataFrame, expression: pl.Expr) -> bool:
    """
    Checks if all columns required by the expression are present in the DataFrame.
    Returns True if all required columns are present, False otherwise.
    """
    required = set(expression.meta.root_names())
    return all(col in data.columns for col in required)


def to_bdf(data: pl.DataFrame) -> pl.DataFrame:
    """
    Convert column names to BDF format using the _BDF_MAPPING dictionary. If a column name is not found in the mapping, it will remain unchanged.
    """
    return data.rename(_BDF_MAPPING, strict=False)


def _transform_24(data: pl.DataFrame) -> pl.DataFrame:
    """
    Transform in three steps:
    1. Calculate the absolute value of cur_step_range and store it in a new column d_cur_step_range.
    2. Create new columns for current_scale, capchg_scale, capdchg_scale, engchg_scale, and engdchg_scale based on the value of cur_step_range.
    3. Apply the scaling factors to the relevant columns and drop the temporary columns.
    """

    CUR_SCALE_10 = 10
    CUR_SCALE_100 = 100
    CUR_SCALE_1000 = 1000

    CUR_SCALE_FACTOR_10 = 10000.0
    CUR_SCALE_FACTOR_100 = 1000.0
    CUR_SCALE_FACTOR_1000 = 100.0
    CUR_SCALE_FACTOR_MAX = 10.0

    step_type_mapping = {
        1: "CC Charge",
        2: "CC Discharge",
        4: "Rest",
        5: "Cycle",
        6: "End",
        7: "CC-CV Charge",
        8: "CP Discharge",
        9: "CP Charge",
        10: "CR Discharge",
        20: "CC-CV Discharge",
    }

    # pre-define column variables for use in expressions
    cur_step_range = pl.col("cur_step_range")
    d_cur_step_range = pl.col("d_cur_step_range")
    scale_cur = pl.col("scale_cur")
    scale_capchg = pl.col("scale_capchg")
    scale_capdchg = pl.col("scale_capdchg")
    scale_engchg = pl.col("scale_engchg")
    scale_engdchg = pl.col("scale_engdchg")
    factor_engchg = pl.col("factor_engchg")
    factor_engdchg = pl.col("factor_engdchg")
    factor_capchg = pl.col("factor_capchg")
    factor_capdchg = pl.col("factor_capdchg")

    expressions = {
        "d_cur_step_range": pl.when(
            cur_step_range.abs().is_between(0, 999999, closed="right")
        )
        .then(cur_step_range.abs())
        .when(cur_step_range.abs().is_between(1000000, 999999999, closed="both"))
        .then(cur_step_range // 1000000000.0)
        .otherwise(0),
        "scale_cur": pl.when(cur_step_range > 0)
        .then(
            pl.when(cur_step_range < CUR_SCALE_10)
            .then(CUR_SCALE_FACTOR_10)
            .when(cur_step_range < CUR_SCALE_100)
            .then(CUR_SCALE_FACTOR_100)
            .when(cur_step_range < CUR_SCALE_1000)
            .then(CUR_SCALE_FACTOR_1000)
            .otherwise(CUR_SCALE_FACTOR_MAX)
        )
        .otherwise(
            pl.when(d_cur_step_range < 0.01)
            .then(100000000.0)
            .when(d_cur_step_range < 0.1)
            .then(10000000.0)
            .when(d_cur_step_range < 1)
            .then(1000000.0)
            .when(d_cur_step_range < 10)
            .then(100000.0)
            .when(d_cur_step_range < 100)
            .then(10000.0)
            .when(d_cur_step_range < 1000)
            .then(1000.0)
            .otherwise(100)
        ),
        "scale_capchg": pl.when(factor_capchg == 0)
        .then(scale_cur * 1e3 * 3600)
        .when(factor_capchg == 1)
        .then(scale_cur)
        .when(factor_capchg == 2)
        .then(scale_cur * 1e3),
        "scale_capdchg": pl.when(factor_capdchg == 0)
        .then(scale_cur * 1e3 * 3600)
        .when(factor_capdchg == 1)
        .then(scale_cur)
        .when(factor_capdchg == 2)
        .then(scale_cur * 1e3),
        "scale_engchg": pl.when(factor_engchg == 0)
        .then(scale_cur * 1e3 * 3600)
        .when(factor_engchg == 1)
        .then(scale_cur)
        .when(factor_engchg == 2)
        .then(scale_cur * 1e3),
        "scale_engdchg": pl.when(factor_engdchg == 0)
        .then(scale_cur * 1e3 * 3600)
        .when(factor_engdchg == 1)
        .then(scale_cur)
        .when(factor_engdchg == 2)
        .then(scale_cur * 1e3),
        "test_time": pl.col("test_time") / 1e3,
        "test_vol": pl.col("test_vol") / 1e4,
        "test_cur": pl.col("test_cur") / (scale_cur * 1e3),
        "test_tmp": pl.col("test_tmp") / 1e1,
        "test_capchg": pl.col("test_capchg") / scale_capchg,
        "test_capdchg": pl.col("test_capdchg") / scale_capdchg,
        "test_engchg": pl.col("test_engchg") / scale_engchg,
        "test_engdchg": pl.col("test_engdchg") / scale_engdchg,
        "test_pow": pl.col("test_cur") * pl.col("test_vol"),
        "test_cap": pl.col("test_capchg") - pl.col("test_capdchg"),
        "test_eng": pl.col("test_engchg") - pl.col("test_engdchg"),
        "unix_time": pl.col("test_atime").dt.epoch("s"),
        "step_type": pl.col("step_type").replace_strict(
            step_type_mapping, default="Unknown"
        ),
    }

    for name, expr in expressions.items():
        if check_required(data, expr):
            logger.info(f"Transforming data with {name} column")
            data = data.with_columns(wrap_nullcheck(expr).alias(name))
        else:
            logger.info(
                f"Skipping transformation of {name} column due to missing required columns"
            )

    return data


def _transform_26(data: pl.DataFrame) -> pl.DataFrame:
    step_type_mapping = {
        1: "CC Charge",
        2: "CC Discharge",
        4: "Rest",
        5: "Cycle",
        6: "End",
        7: "CC-CV Charge",
        8: "CP Discharge",
        9: "CP Charge",
        10: "CR Discharge",
        20: "CC-CV Discharge",
    }

    expressions = {
        "test_time": pl.col("test_time") / 1e3,
        "test_vol": pl.col("test_vol") / 1e4,
        "test_cur": pl.col("test_cur") / 1e3,
        "test_tmp": pl.col("test_tmp") / 1e1,
        "test_pow": pl.col("test_vol") * pl.col("test_cur"),
        "test_capchg": pl.col("test_capchg") / 3600 / 1e3,
        "test_capdchg": pl.col("test_capdchg") / 3600 / 1e3,
        "test_engchg": pl.col("test_engchg") / 3600 / 1e3,
        "test_engdchg": pl.col("test_engdchg") / 3600 / 1e3,
        "total_cap": pl.col("total_cap") / 3600 / 1e3,
        "total_eng": pl.col("total_eng") / 3600 / 1e3,
        "test_cap": pl.col("test_capchg") - pl.col("test_capdchg"),
        "test_eng": pl.col("test_engchg") - pl.col("test_engdchg"),
        "unix_time": pl.col("test_atime").dt.epoch("s"),
        "step_type": pl.col("step_type").replace_strict(
            step_type_mapping, default="Unknown"
        ),
    }
    for name, expr in expressions.items():
        if check_required(data, expr):
            logger.info(f"Transforming data with {name} column")
            data = data.with_columns(wrap_nullcheck(expr).alias(name))
        else:
            logger.info(
                f"Skipping transformation of {name} column due to missing required columns"
            )
    return data


_TRANSFORM_MAPPING = {
    "24": _transform_24,
    "26": _transform_26,
}


def transform(data, test: dict) -> pl.DataFrame:
    dev_uid = test.get("dev_uid")
    dev_type = str(dev_uid)[0:2]
    if dev_type not in _TRANSFORM_MAPPING:
        raise ValueError(f"Unsupported device type: {dev_type}")
    return _TRANSFORM_MAPPING[dev_type](data)


def enrich(data: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates additional columns based on existing data, such as power, step count, step index, and Unix time.
    It's advised to only enrich full datasets, as the step count and step index calculations rely on the entire dataset to be accurate.
    """
    expressions = {
        "step_count": pl.when(pl.col("test_time") == 0).then(1).otherwise(0).cum_sum(),
        "step_index": (pl.col("seq_id") - pl.col("seq_id").min() + 1).over(
            "step_count"
        ),
        "test_totaltime": (pl.col("unix_time") - pl.col("unix_time").min()),
    }
    for name, expr in expressions.items():
        if check_required(data, expr):
            logger.info(f"Enriching data with {name} column")
            data = data.with_columns(wrap_nullcheck(expr).alias(name))
        else:
            logger.info(
                f"Skipping enrichment of {name} column due to missing required columns"
            )
    return data
