from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self):
        self.transformations: dict[str, Transformation] = {}

    def add(self, alias: str, expression: pl.Expr):
        t = Transformation(self, alias, expression)
        self.transformations[alias] = t
        return t

    def contains(self, alias: str) -> bool:
        return alias in self.transformations

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        applied = set()
        for alias, t in self.transformations.items():
            if alias in applied:
                continue
            data, applied = t.apply(data, applied=applied)
        return data


@dataclass
class Transformation:
    registry: Registry = field(repr=False, compare=False)
    alias: str
    expression: pl.Expr

    @property
    def dependencies(self) -> list[str]:
        return list(self.expression.meta.root_names())

    def apply(
        self, data: pl.LazyFrame, applied: set[str]
    ) -> tuple[pl.LazyFrame, set[str]]:
        """
        Apply the transformation to the data, ensuring all dependencies are met.
        """
        logger.info(f"Applying transformation: {self.alias}")

        # Early exit
        if self.alias in applied:
            logger.info(f"Transformation {self.alias} already applied.")
            return data, applied

        for col in self.dependencies:
            logger.info(f"Checking dependency: {col}")
            if col == self.alias:
                logger.info(
                    f"Dependency {col} is the same as the transformation alias {self.alias}, skipping."
                )
                continue

            if (col not in applied) and (self.registry.contains(col)):
                logger.info(f"Applying dependency transformation: {col}")
                trn = self.registry.transformations[col]
                data, applied = trn.apply(data, applied=applied)

            elif (col not in data.collect_schema()) and (
                not self.registry.contains(col)
            ):
                raise ValueError(
                    f"Column '{col}' is missing from the data and has no transformation."
                )
        data = data.with_columns(self.expression.alias(self.alias))
        applied.add(self.alias)
        return data, applied
