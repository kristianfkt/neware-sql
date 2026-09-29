from __future__ import annotations

import polars as pl

from newaresql.transformations.transformation import Registry

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

TRANSFORMATIONS_26 = Registry()


test_time = TRANSFORMATIONS_26.add(
    alias="test_time",
    expression=pl.col("test_time") / 1e3,
)
test_vol = TRANSFORMATIONS_26.add(
    alias="test_vol",
    expression=pl.col("test_vol") / 1e4,
)
test_cur = TRANSFORMATIONS_26.add(
    alias="test_cur",
    expression=pl.col("test_cur") / 1e3,
)
test_tmp = TRANSFORMATIONS_26.add(
    alias="test_tmp",
    expression=pl.col("test_tmp") / 1e1,
)
test_pow = TRANSFORMATIONS_26.add(
    alias="test_pow",
    expression=pl.col("test_vol") * pl.col("test_cur"),
)
test_capchg = TRANSFORMATIONS_26.add(
    alias="test_capchg",
    expression=pl.col("test_capchg") / 3600 / 1e3,
)
test_capdchg = TRANSFORMATIONS_26.add(
    alias="test_capdchg",
    expression=pl.col("test_capdchg") / 3600 / 1e3,
)
test_engchg = TRANSFORMATIONS_26.add(
    alias="test_engchg",
    expression=pl.col("test_engchg") / 3600 / 1e3,
)
test_engdchg = TRANSFORMATIONS_26.add(
    alias="test_engdchg",
    expression=pl.col("test_engdchg") / 3600 / 1e3,
)
total_cap = TRANSFORMATIONS_26.add(
    alias="total_cap",
    expression=pl.col("total_cap") / 3600 / 1e3,
)
total_eng = TRANSFORMATIONS_26.add(
    alias="total_eng",
    expression=pl.col("total_eng") / 3600 / 1e3,
)
test_cap = TRANSFORMATIONS_26.add(
    alias="test_cap",
    expression=pl.col("test_capchg") - pl.col("test_capdchg"),
)
test_eng = TRANSFORMATIONS_26.add(
    alias="test_eng",
    expression=pl.col("test_engchg") - pl.col("test_engdchg"),
)
unix_time = TRANSFORMATIONS_26.add(
    alias="unix_time",
    expression=pl.col("test_atime").dt.epoch("s"),
)
step_type = TRANSFORMATIONS_26.add(
    alias="step_type",
    expression=pl.col("step_type").replace_strict(
        STEP_TYPE_MAPPING,
        default="Unknown",
    ),
)
