import polars as pl

from newaresql.expressions import transformations_24, transformations_26

TRANSFORMATIONS = {
    "0760-24": transformations_24.TRANSFORMATIONS,
    "0800-24": transformations_24.TRANSFORMATIONS,
    "0800-26": transformations_26.TRANSFORMATIONS,
}


def get_expressions(bts_build: str, dev_uid: int) -> dict[str, pl.Expr]:
    dev_type = str(dev_uid)[:2]
    return TRANSFORMATIONS.get(f"{bts_build}-{dev_type}", {})
