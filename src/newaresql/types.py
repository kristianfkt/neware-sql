import datetime
from typing import Any, Literal

"""
types.py contains a pile of type aliases that hopefully make the code more readable and maintainable.
It defined str as a scalar, but includes the same scalar in a LessOrEqual or GreaterOrEqual tuple. This will crash I guess. 
"""

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
