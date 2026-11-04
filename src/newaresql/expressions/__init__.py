import logging
from typing import overload

import polars as pl

import newaresql.expressions.cycles as cycles
import newaresql.expressions.steps as steps
import newaresql.expressions.transformations as transformations

logger = logging.getLogger(__name__)


def get_dependencies(expression: pl.Expr) -> set[str]:
    """
    Fetch all dependencies for the given expression.
        test_vol = test_vol / 10000 depends only on itself
        test_pow = test_vol * test_cur depends on test_vol and test_cur
    """
    logger.debug("Getting dependencies for expression: %s", expression)
    dependencies = set(expression.meta.root_names())
    logger.debug("Dependencies found for expression: %s: %s", expression, dependencies)
    return dependencies


def check_dependencies(
    alias: str, expression: pl.Expr, data: pl.LazyFrame, waiting: set[str]
) -> bool:
    """
    Checks if all dependency requirements are satisfied:
        No dependencies: returns True

        Dependencies met: returns True
        Dependencies not met: returns False
    """
    logger.debug("Checking dependencies for expression: %s", alias)
    dependencies = get_dependencies(expression)
    if not dependencies:
        logger.debug("No dependencies found for expression: %s", alias)
        return True
    schema = data.collect_schema()
    for dependency in list(set(dependencies).difference({alias})):
        logger.debug("Processing dependency: %s for expression: %s", dependency, alias)
        # Dependency not found in schema or waiting set - cannot proceed with this expression
        if (dependency not in schema) and (dependency not in waiting):
            logger.debug(
                "Dependency %s for expression %s not found in schema or waiting set",
                dependency,
                alias,
            )
            return False

        # Dependency found in schema and in waiting - can not proceed this iteration
        if (dependency in schema) and (dependency in waiting):
            logger.debug(
                "Dependency %s for expression %s found in schema and waiting set, cannot proceed this iteration",
                dependency,
                alias,
            )
            return False
    # At this point, all dependencies should be met
    return True


@overload
def apply_expressions(
    expressions: dict[str, pl.Expr],
    data: pl.LazyFrame,
) -> pl.LazyFrame: ...
@overload
def apply_expressions(
    expressions: dict[str, pl.Expr],
    data: pl.DataFrame,
) -> pl.DataFrame: ...


def apply_expressions(
    expressions: dict[str, pl.Expr],
    data: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Applies expressions recursively
    """
    if isinstance(data, pl.DataFrame):
        return apply_expressions(expressions, data.lazy()).collect()

    def _check(dependencies: set[str], pending: set[str], present: set[str]):
        for dependency in dependencies:
            if dependency in pending:
                return False
            elif dependency in present:
                continue
        return True

    pending = set(expressions.keys())
    present = set(data.collect_schema().names())
    ready = {}
    deferred = {}

    for alias, expression in expressions.items():
        logger.debug("Processing expression %s", alias)
        # Get all dependencies for the current expression, excluding itself
        dependencies = set(expression.meta.root_names()).difference({alias})
        if _check(dependencies, pending, present):
            logger.debug("Expression %s is ready to be applied", alias)
            ready[alias] = expression
        else:
            logger.debug("Expression %s is deferred", alias)
            deferred[alias] = expression
    if ready and deferred:
        logger.debug(
            "Applying expressions %s and deferring %s",
            list(ready.keys()),
            list(deferred.keys()),
        )

        return apply_expressions(deferred, data.with_columns(**ready))
    elif ready:
        logger.debug("Applying expressions %s", list(ready.keys()))
        return data.with_columns(**ready)
    else:
        logger.debug("No expressions are ready to be applied, returning data as is")
    logger.debug("Finished applying expressions")
    return data

    # if not ready:
    #     logger.debug("No expressions are ready to be applied, returning data as is")
    #     return data

    # dispatch = {}  # Expressions ready to be applied immediately
    # retry = {}  # Expressions that need to be retried in the next iteration
    # waiting = set(expressions.keys())  # Expressions that are waiting to be processed
    # for alias, expression in expressions.items():
    #     # Fetch all dependencies - remove self
    #     dependencies = list(set(get_dependencies(expression)).difference({alias}))

    #     if _check(dependencies, waiting, set(data.collect_schema())):
    #         dispatch[alias] = expression
    #     else:
    #         retry[alias] = expression

    # logger.debug(f"Dispatch: {list(dispatch.keys())}")
    # logger.debug(f"Retry: {list(retry.keys())}")

    # if retry and dispatch:
    #     logger.debug(
    #         "Dispatching expressions: %s, retrying expressions: %s",
    #         dispatch,
    #         retry,
    #     )
    #     waiting.difference_update(dispatch)
    #     return apply_expressions(retry, data.with_columns(**dispatch))
    # elif dispatch:
    #     logger.debug(
    #         "Dispatching expressions: %s",
    #         dispatch,
    #     )
    #     return data.with_columns(**dispatch)
    # else:
    #     logger.debug("No expressions to dispatch, returning final results")
    #     return data


@overload
def transform_data(
    data: pl.DataFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.DataFrame: ...
@overload
def transform_data(
    data: pl.LazyFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.LazyFrame: ...
def transform_data(
    data: pl.LazyFrame | pl.DataFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.LazyFrame | pl.DataFrame:
    """
    Applies the required transformations to the data based on the BTS version and device UID.
    """
    if isinstance(data, pl.DataFrame):
        return transform_data(data.lazy(), bts_version, dev_uid).collect()

    expressions = transformations.get_expressions(bts_version, dev_uid)
    result = apply_expressions(expressions, data)
    return result


@overload
def aggregate_steps(
    data: pl.LazyFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.LazyFrame: ...
@overload
def aggregate_steps(
    data: pl.DataFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.DataFrame: ...
def aggregate_steps(
    data: pl.DataFrame | pl.LazyFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.DataFrame | pl.LazyFrame:
    if isinstance(data, pl.DataFrame):
        return aggregate_steps(data.lazy(), bts_version, dev_uid).collect()
    expressions = steps.get_expressions(bts_version, dev_uid)
    return data.group_by("cycle", "step_id", "step_changecount").agg(**expressions)


@overload
def aggregate_cycles(
    data: pl.LazyFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.LazyFrame: ...
@overload
def aggregate_cycles(
    data: pl.DataFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.DataFrame: ...
def aggregate_cycles(
    data: pl.DataFrame | pl.LazyFrame,
    bts_version: str,
    dev_uid: int,
) -> pl.DataFrame | pl.LazyFrame:
    if isinstance(data, pl.DataFrame):
        return aggregate_cycles(data.lazy(), bts_version, dev_uid).collect()
    expressions = cycles.get_expressions(bts_version, dev_uid)
    return data.group_by("cycle").agg(**expressions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    data = pl.DataFrame(
        {
            "a": [0, 1, 2, 3],
            "b": [0, 4, 5, 6],
        }
    ).lazy()
    expressions = {
        "k": pl.lit(1),
        "s": pl.when(pl.col("a") > 1).then(pl.lit(1)).otherwise(pl.lit(0)),
        "c": pl.col("a") + pl.col("b"),
        "d": pl.col("a") * pl.col("s"),
    }

    chain_length = 100
    for depth in reversed(range(chain_length)):
        alias = f"chain_{depth:03}"
        source = "d" if depth == 0 else f"chain_{depth - 1:03}"
        expressions[alias] = pl.col(source) + pl.col("k")

    branch_count = 32
    branch_aliases = []
    for branch in range(branch_count):
        source_depth = branch * (chain_length - 1) // (branch_count - 1)
        alias = f"branch_{branch:03}"
        expressions[alias] = pl.col(f"chain_{source_depth:03}") * (branch + 1)
        branch_aliases.append(alias)

    expressions["total"] = pl.sum_horizontal(
        [pl.col(alias) for alias in branch_aliases]
    )

    print(data.collect())
    print(apply_expressions(expressions, data).collect())
