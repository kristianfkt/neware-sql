from __future__ import annotations

import polars as pl

from newaresql.transformations.transformation import Registry

CUR_SCALE_10 = 10
CUR_SCALE_100 = 100
CUR_SCALE_1000 = 1000

CUR_SCALE_FACTOR_10 = 10000.0
CUR_SCALE_FACTOR_100 = 1000.0
CUR_SCALE_FACTOR_1000 = 100.0
CUR_SCALE_FACTOR_MAX = 10.0

STEP_TYPE_MAPPING = {
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

TRANSFORMATIONS_24 = Registry()

step_time = TRANSFORMATIONS_24.add(
    alias="step_time",
    expression=pl.col("test_time") / 1000,
)

test_vol = TRANSFORMATIONS_24.add(
    alias="test_vol",
    expression=pl.col("test_vol") / 10000,
)
test_time = TRANSFORMATIONS_24.add(
    alias="test_time",
    expression=pl.col("test_time") / 1000,
)
test_tmp = TRANSFORMATIONS_24.add(
    alias="test_tmp",
    expression=pl.col("test_tmp") / 10,
)
test_pow = TRANSFORMATIONS_24.add(
    alias="test_pow",
    expression=pl.col("test_vol") * pl.col("test_cur"),
)
test_cap = TRANSFORMATIONS_24.add(
    alias="test_cap",
    expression=pl.col("test_capchg") - pl.col("test_capdchg"),
)
test_eng = TRANSFORMATIONS_24.add(
    alias="test_eng",
    expression=pl.col("test_engchg") - pl.col("test_engdchg"),
)

d_cur_step_range = TRANSFORMATIONS_24.add(
    alias="d_cur_step_range",
    expression=pl.when(
        pl.col("cur_step_range").abs().is_between(0, 999999, closed="right")
    )
    .then(pl.col("cur_step_range").abs())
    .when(pl.col("cur_step_range").abs().is_between(1000000, 999999999, closed="both"))
    .then(pl.col("cur_step_range") // 1000000000.0)
    .otherwise(0),
)

scale_cur = TRANSFORMATIONS_24.add(
    alias="scale_cur",
    expression=pl.when(pl.col("cur_step_range") > 0)
    .then(
        pl.when(pl.col("cur_step_range") < CUR_SCALE_10)
        .then(CUR_SCALE_FACTOR_10)
        .when(pl.col("cur_step_range") < CUR_SCALE_100)
        .then(CUR_SCALE_FACTOR_100)
        .when(pl.col("cur_step_range") < CUR_SCALE_1000)
        .then(CUR_SCALE_FACTOR_1000)
        .otherwise(CUR_SCALE_FACTOR_MAX)
    )
    .otherwise(
        pl.when(pl.col("d_cur_step_range") < 0.01)
        .then(100000000.0)
        .when(pl.col("d_cur_step_range") < 0.1)
        .then(10000000.0)
        .when(pl.col("d_cur_step_range") < 1)
        .then(1000000.0)
        .when(pl.col("d_cur_step_range") < 10)
        .then(100000.0)
        .when(pl.col("d_cur_step_range") < 100)
        .then(10000.0)
        .when(pl.col("d_cur_step_range") < 1000)
        .then(1000.0)
        .otherwise(100),
    ),
)

scale_capchg = TRANSFORMATIONS_24.add(
    alias="scale_capchg",
    expression=pl.when(pl.col("factor_capchg") == 0)
    .then(pl.col("scale_cur") * 1e3 * 3600)
    .when(pl.col("factor_capchg") == 1)
    .then(pl.col("scale_cur"))
    .when(pl.col("factor_capchg") == 2)
    .then(pl.col("scale_cur") * 1e3),
)
scale_capdchg = TRANSFORMATIONS_24.add(
    alias="scale_capdchg",
    expression=pl.when(pl.col("factor_capdchg") == 0)
    .then(pl.col("scale_cur") * 1e3 * 3600)
    .when(pl.col("factor_capdchg") == 1)
    .then(pl.col("scale_cur"))
    .when(pl.col("factor_capdchg") == 2)
    .then(pl.col("scale_cur") * 1e3),
)
scale_engchg = TRANSFORMATIONS_24.add(
    alias="scale_engchg",
    expression=pl.when(pl.col("factor_engchg") == 0)
    .then(pl.col("scale_cur") * 1e3 * 3600)
    .when(pl.col("factor_engchg") == 1)
    .then(pl.col("scale_cur"))
    .when(pl.col("factor_engchg") == 2)
    .then(pl.col("scale_cur") * 1e3),
)
scale_engdchg = TRANSFORMATIONS_24.add(
    alias="scale_engdchg",
    expression=pl.when(pl.col("factor_engdchg") == 0)
    .then(pl.col("scale_cur") * 1e3 * 3600)
    .when(pl.col("factor_engdchg") == 1)
    .then(pl.col("scale_cur"))
    .when(pl.col("factor_engdchg") == 2)
    .then(pl.col("scale_cur") * 1e3),
)

test_cur = TRANSFORMATIONS_24.add(
    alias="test_cur",
    expression=pl.col("test_cur") / (pl.col("scale_cur") * 1e3),
)
test_capchg = TRANSFORMATIONS_24.add(
    alias="test_capchg",
    expression=pl.col("test_capchg") / pl.col("scale_capchg"),
)
test_capdchg = TRANSFORMATIONS_24.add(
    alias="test_capdchg",
    expression=pl.col("test_capdchg") / pl.col("scale_capdchg"),
)
test_engchg = TRANSFORMATIONS_24.add(
    alias="test_engchg",
    expression=pl.col("test_engchg") / pl.col("scale_engchg"),
)
test_engdchg = TRANSFORMATIONS_24.add(
    alias="test_engdchg",
    expression=pl.col("test_engdchg") / pl.col("scale_engdchg"),
)

unix_time = TRANSFORMATIONS_24.add(
    alias="unix_time",
    expression=pl.col("test_atime").dt.epoch("s"),
)
step_type = TRANSFORMATIONS_24.add(
    alias="step_type",
    expression=pl.col("step_type").replace_strict(
        STEP_TYPE_MAPPING,
        default="Unknown",
    ),
)
