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


def get_table_schema(table: str, engine: sa.Engine) -> dict[str, type]:
    # Table object
    t = sa.Table(table, sa.MetaData(), autoload_with=engine)
    return {col.name: to_pytype(col.type) for col in t.columns}


def make_select_query(
    table: str,
    columns: str | list[str] | None = None,
    where: dict[str, tuple | list] | None = None,
) -> str:
    """
    Maybe I should use SQLAlchemy to build the query .. it's like almost no change, so whatever.
    """

    # Columns
    if columns is None:
        columns = "*"
    if isinstance(columns, list):
        columns = ", ".join(columns)

    if where is None:
        where = {}
    predicates = []
    for col, val in where.items():
        if isinstance(val, list):
            predicates.append(f"{col} IN ({', '.join(map(str, val))})")
        elif isinstance(val, tuple) and len(val) == 2:
            lo, hi = val
            if (hi is not None) and (lo is not None):
                predicates.append(f"{col} BETWEEN {lo} AND {hi}")
            elif hi is not None:
                predicates.append(f"{col} <= {hi}")
            elif lo is not None:
                predicates.append(f"{col} >= {lo}")
        else:
            predicates.append(f"{col} = {val}")
    if predicates:
        query = f"SELECT {columns} FROM {table} WHERE {' AND '.join(predicates)}"
    else:
        query = f"SELECT {columns} FROM {table}"
    return query
