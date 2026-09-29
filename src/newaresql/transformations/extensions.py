import polars as pl

from newaresql.transformations.transformation import Registry

EXTENSIONS = Registry()

step_count = EXTENSIONS.add(
    "step_count", pl.when(pl.col("test_time") == 0).then(1).otherwise(0).cum_sum()
)
step_index = EXTENSIONS.add(
    "step_index", (pl.col("seq_id") - pl.col("seq_id").min() + 1).over("step_count")
)


test_time = EXTENSIONS.add(
    "test_time", (pl.col("unix_time") - pl.col("unix_time").min())
)
