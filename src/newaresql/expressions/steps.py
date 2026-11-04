import polars as pl

STEPS = {
    "first_test_vol": pl.col("test_vol").first(),
    "last_test_vol": pl.col("test_vol").last(),
    "min_test_vol": pl.col("test_vol").min(),
    "max_test_vol": pl.col("test_vol").max(),
    "mean_test_vol": pl.col("test_vol").mean(),
    "std_test_vol": pl.col("test_vol").std(),
    #
    "first_test_cur": pl.col("test_cur").first(),
    "last_test_cur": pl.col("test_cur").last(),
    "min_test_cur": pl.col("test_cur").min(),
    "max_test_cur": pl.col("test_cur").max(),
    "mean_test_cur": pl.col("test_cur").mean(),
    "std_test_cur": pl.col("test_cur").std(),
    #
    "first_test_pow": pl.col("test_pow").first(),
    "last_test_pow": pl.col("test_pow").last(),
    "min_test_pow": pl.col("test_pow").min(),
    "max_test_pow": pl.col("test_pow").max(),
    "mean_test_pow": pl.col("test_pow").mean(),
    "std_test_pow": pl.col("test_pow").std(),
    #
    "test_cap": pl.col("test_cap").last(),
    "test_eng": pl.col("test_eng").last(),
}


def get_expressions(bts_build: str, dev_uid: int) -> dict[str, pl.Expr]:
    return STEPS
