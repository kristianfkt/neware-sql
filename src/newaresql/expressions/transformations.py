"""Select transformation expressions for supported build and device pairs."""

import polars as pl

from newaresql.expressions import transformations_24, transformations_26

TRANSFORMATIONS = {
    "0760-24": transformations_24.TRANSFORMATIONS,
    "0800-24": transformations_24.TRANSFORMATIONS,
    "0800-26": transformations_26.TRANSFORMATIONS,
}


def get_expressions(bts_build: str, dev_uid: int) -> dict[str, pl.Expr]:
    """Return expressions for a supported BTS build and device type.

    Args:
        bts_build: BTS build version.
        dev_uid: Device UID whose leading digits identify the device type.

    Returns:
        A mapping of output names to Polars expressions, or an empty mapping
        for an unsupported combination.
    """
    dev_type = str(dev_uid)[:2]
    return TRANSFORMATIONS.get(f"{bts_build}-{dev_type}", {})
