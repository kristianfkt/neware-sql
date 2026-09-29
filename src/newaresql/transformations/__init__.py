from typing import overload

import polars as pl

from newaresql.transformations.extensions import EXTENSIONS
from newaresql.transformations.transformations_24 import TRANSFORMATIONS_24
from newaresql.transformations.transformations_26 import TRANSFORMATIONS_26

TRANSFORMATIONS = {
    "0760-24": TRANSFORMATIONS_24,
    "0800-24": TRANSFORMATIONS_24,
    "0800-26": TRANSFORMATIONS_26,
}


@overload
def transform(data: pl.LazyFrame, bts_build: str, dev_uid: int) -> pl.LazyFrame: ...
@overload
def transform(data: pl.DataFrame, bts_build: str, dev_uid: int) -> pl.DataFrame: ...


def transform(
    data: pl.LazyFrame | pl.DataFrame, bts_build: str, dev_uid: int
) -> pl.LazyFrame | pl.DataFrame:
    dev_type = str(dev_uid)[:2]
    key = f"{bts_build}-{dev_type}"

    if isinstance(data, pl.DataFrame):
        return transform(data.lazy(), bts_build, dev_uid).collect(engine="streaming")
    return TRANSFORMATIONS[key].apply(data)


@overload
def extend(data: pl.LazyFrame) -> pl.LazyFrame: ...
@overload
def extend(data: pl.DataFrame) -> pl.DataFrame: ...


def extend(data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame | pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return extend(data.lazy()).collect(engine="streaming")
    return EXTENSIONS.apply(data)
