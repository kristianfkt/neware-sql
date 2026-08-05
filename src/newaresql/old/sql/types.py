import datetime

import sqlalchemy as sa

PYTYPES: dict[type[sa.types.TypeEngine], type] = {
    sa.types.Integer: int,
    sa.types.Float: float,
    sa.types.Numeric: float,
    sa.types.String: str,
    sa.types.Text: str,
    sa.types.DateTime: datetime.datetime,
    sa.types.Date: datetime.date,
    sa.types.Boolean: bool,
}


def to_pytype(sqltype: sa.types.TypeEngine) -> type:
    """
    Convert a SQLAlchemy type to a Python type.
    """

    return PYTYPES.get(sqltype._type_affinity, object)
