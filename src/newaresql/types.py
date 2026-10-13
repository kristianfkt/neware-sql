import datetime
from typing import Any, Literal, TypeAlias

Scalar: TypeAlias = int | float | str | bool | datetime.datetime | datetime.date
Equal: TypeAlias = Scalar
In: TypeAlias = list[Scalar]
Between: TypeAlias = tuple[Scalar, Scalar]
LessOrEqual: TypeAlias = tuple[None, Scalar]
GreaterOrEqual: TypeAlias = tuple[Scalar, None]

Condition: TypeAlias = Equal | In | Between | LessOrEqual | GreaterOrEqual
Conditions: TypeAlias = dict[str, Condition]


Test: TypeAlias = dict[str, Any]
Naming: TypeAlias = Literal["bts", "code", "label"]
Columns: TypeAlias = str | list[str]
