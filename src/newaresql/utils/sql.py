import datetime
import logging

import sqlalchemy as sa

from newaresql.types import Columns, Where

logger = logging.getLogger(__name__)

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

    affinity = getattr(sqltype, "_type_affinity", None)

    # non-clause is only due to type checker complaining
    if (affinity in PYTYPES) and (affinity is not None):
        _t = PYTYPES[affinity]
    else:
        try:
            _t = sqltype.python_type()
        except NotImplementedError:
            _t = object
    return _t


def wrap_table(
    table: str,
    engine: sa.engine.Engine,
) -> sa.Table:
    """
    Wrap a table name as a SQLAlchemy Table object.
    """
    return sa.Table(table, sa.MetaData(), autoload_with=engine)


def select_table(
    table: sa.Table,
    columns: Columns | None = None,
) -> sa.Select:
    """
    Select a table with optionally specified columns.
    """

    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        stmt = sa.select(table)
    else:
        stmt = sa.select(*[table.c[col] for col in columns])
    return stmt


def build_predicate(
    table: sa.Table,
    where: Where,
) -> sa.ColumnElement:
    predicates = []
    for c, p in where.items():
        if isinstance(p, list):
            predicates.append(table.c[c].in_(p))
        elif isinstance(p, tuple) and len(p) == 2:
            lo, hi = p
            if (lo is not None) and (hi is not None):
                predicates.append(table.c[c].between(lo, hi))
            elif hi is not None:
                predicates.append(table.c[c] <= hi)
            elif lo is not None:
                predicates.append(table.c[c] >= lo)
        else:
            predicates.append(table.c[c] == p)
    return sa.and_(*predicates)


def wrap_query(query: str) -> sa.TextClause:
    """
    Wrap a query string as a SQLAlchemy TextClause object.
    """
    return sa.text(query)


def compile_statement(stmt: sa.Select, engine: sa.engine.Engine) -> str:
    """
    Compile a SQLAlchemy statement to a SQL string.
    """
    return str(stmt.compile(engine, compile_kwargs={"literal_binds": True}))


def make_select_stmt(
    table: str | sa.Table,
    engine: sa.engine.Engine,
    where: Where | None = None,
    columns: Columns | None = None,
) -> sa.Select:

    if isinstance(table, str):
        table = wrap_table(table, engine)
    stmt = select_table(table, columns=columns)
    if where is not None:
        stmt = stmt.where(build_predicate(table, where))
    return stmt


def get_table_schema(
    table: str | sa.Table,
    engine: sa.engine.Engine,
) -> dict[str, type]:
    """
    Fetch the schema of a table as a dictionary mapping column names to Python types.
    """
    if isinstance(table, str):
        table = wrap_table(table, engine)
    return {col.name: to_pytype(col.type) for col in table.columns}


def union_statements(
    *stmts: sa.Select,
) -> sa.CompoundSelect:
    """
    Union multiple SQLAlchemy statements into a single statement.
    """
    if not stmts:
        raise ValueError("At least one statement is required for union.")
    return sa.union_all(*stmts)
