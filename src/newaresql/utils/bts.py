from __future__ import annotations

import sqlalchemy as sa

from newaresql.connectors.sql import make_select_stmt
from newaresql.types import Columns, Test, Where


def make_main_statement(
    test: Test,
    engine: sa.engine.Engine,
    where: Where | None = None,
    columns: Columns | None = None,
) -> sa.Select | sa.CompoundSelect:
    if where is None:
        where = {}
    where["unit_id"] = test["unit_id"]
    where["chl_id"] = test["chl_id"]
    where["test_id"] = test["test_id"]

    q1 = make_select_stmt(
        test["main_first_table"],
        engine,
        where=where,
        columns=columns,
    )
    if test["main_second_table"] is None:
        q2 = None
    else:
        q2 = make_select_stmt(
            test["main_second_table"],
            engine,
            where=where,
            columns=columns,
        )
    if q2 is None:
        q = q1
    else:
        q = q1.union_all(q2)
    return q


def make_aux_statement(
    test: Test,
    engine: sa.engine.Engine,
    where: Where | None = None,
    columns: Columns | None = None,
) -> sa.Select | sa.CompoundSelect | None:
    if where is None:
        where = {}
    where["unit_id"] = test["unit_id"]
    where["chl_id"] = test["chl_id"]
    where["test_id"] = test["test_id"]

    if test["aux_first_table"] is None:
        q1 = None
    else:
        q1 = make_select_stmt(
            test["aux_first_table"],
            engine,
            where=where,
            columns=columns,
        )
    if test["aux_second_table"] is None:
        q2 = None
    else:
        q2 = make_select_stmt(
            test["aux_second_table"],
            engine,
            where=where,
            columns=columns,
        )
    if (q1 is None) and (q2 is None):
        q = None
    elif q2 is None:
        q = q1
    elif q1 is None:
        q = q2
    else:
        q = q1.union_all(q2)
    return q
