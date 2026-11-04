"""Shared type aliases for database conditions and test metadata."""

import datetime
from typing import Any

Scalar = int | float | str | bool | datetime.datetime | datetime.date
Equal = Scalar
In = list[Scalar]
Between = tuple[Scalar, Scalar]
LessOrEqual = tuple[None, Scalar]
GreaterOrEqual = tuple[Scalar, None]

Condition = Equal | In | Between | LessOrEqual | GreaterOrEqual
Conditions = dict[str, Condition]


Test = dict[str, Any]
Columns = str | list[str]
