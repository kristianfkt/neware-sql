"""Polars transformation expressions for BTS device type 26."""

from __future__ import annotations

import polars as pl

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

TRANSFORMATIONS: dict[str, pl.Expr] = {
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
        STEP_TYPE_MAPPING,
        default="Unknown",
    ),
}
