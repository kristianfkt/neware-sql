from __future__ import annotations

import logging
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)


class Registry:
    def __init__(self):
        self.transformations: dict[str, Transformation] = {}

    def __getitem__(self, alias: str) -> Transformation:
        return self.transformations[alias]

    def __contains__(self, alias: str) -> bool:
        return alias in self.transformations

    def add(self, alias: str, expression: pl.Expr):
        t = Transformation(self, alias, expression)
        self.transformations[alias] = t
        return t

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        applied = set()
        for alias, t in self.transformations.items():
            if alias in applied:
                logger.info("Skip %s: already applied", alias)
                continue
            if not t.possible(data):
                logger.info("Skip %s: dependencies unavailable", alias)
                continue
            logger.info("Apply transformation: %s", alias)
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

    def possible(self, data: pl.LazyFrame) -> bool:
        """
        Check if all dependencies are available, or can be applied
        """
        schema = data.collect_schema()
        dependencies = self.dependencies
        logger.info("Checking dependencies for transformation: %s", self.alias)
        logger.info("Dependencies: %s", dependencies)
        logger.info("Current schema: %s", schema)
        for dep in dependencies:
            logger.info("Check dependency: %s", dep)
            if dep in schema:  # dependency already exists in the schema
                logger.info("Skip existing dependency: %s", dep)
                continue

            if (dep == self.alias) and (dep not in schema):  # missing self-dependency
                logger.info("Skip %s: self-dependency is missing", self.alias)
                return False

            if (dep in self.registry) and self.registry[dep].possible(data):
                logger.info("Dependency can be applied: %s", dep)
                continue
            else:
                logger.info("Missing dependency: %s", dep)
                return False

        logger.info("Dependencies met: %s", self.alias)
        return True

    def apply(
        self, data: pl.LazyFrame, applied: set[str]
    ) -> tuple[pl.LazyFrame, set[str]]:
        """
        Apply the transformation to the data, ensuring all dependencies are met.
        """
        logger.info("Apply transformation: %s", self.alias)
        if self.alias in applied:
            logger.info("Skip %s: already applied", self.alias)
            return data, applied

        for col in self.dependencies:
            logger.info("Check %s for %s", col, self.alias)
            if col == self.alias:
                logger.info("Skip self-dependency: %s", self.alias)
                continue

            if (col not in applied) and (col in self.registry):
                logger.info("Apply dependency: %s", col)
                trn = self.registry.transformations[col]
                data, applied = trn.apply(data, applied=applied)

            # Throw an error if the dependency is missing and cannot be applied
            if (col not in data.collect_schema()) and (col not in applied):
                raise ValueError(
                    f"Column '{col}' is missing from the data and cannot be applied."
                )

        data = data.with_columns(self.expression.alias(self.alias))
        applied.add(self.alias)
        return data, applied
