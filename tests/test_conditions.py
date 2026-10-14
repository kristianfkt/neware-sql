import unittest
from unittest.mock import Mock, patch

import polars as pl
import sqlalchemy as sa

from newaresql.connector import Connector, build_conditions, conditions_to_bts


class ConditionNamesTests(unittest.TestCase):
    def test_conversion_preserves_values_and_input(self):
        cycles = [4, 5]
        conditions = {
            "Cycle Count / 1": (4, 100),
            "record_count": (None, 100),
            "step_id": cycles,
            "unknown": 7,
            "Cycle count / 1": 8,
        }
        original = conditions.copy()
        converted = conditions_to_bts(conditions)
        self.assertEqual(
            converted,
            {
                "cycle": (4, 100),
                "seq_id": (None, 100),
                "step_id": cycles,
                "unknown": 7,
                "Cycle count / 1": 8,
            },
        )
        self.assertEqual(conditions, original)
        self.assertIs(converted["step_id"], cycles)
        self.assertIsNot(converted, conditions)
        self.assertEqual(conditions_to_bts(converted), converted)
        self.assertEqual(conditions_to_bts({}), {})

    def test_last_alias_wins(self):
        self.assertEqual(
            conditions_to_bts({"cycle": 1, "cycle_count": 2}), {"cycle": 2}
        )
        self.assertEqual(
            conditions_to_bts({"Cycle Count / 1": 2, "cycle": 1}), {"cycle": 1}
        )

    def test_sql_predicates_accept_all_names(self):
        cases = [
            (4, "cycle = 4"),
            ([4, 5], "cycle IN (4, 5)"),
            ((4, 100), "cycle BETWEEN 4 AND 100"),
            ((None, 100), "cycle <= 100"),
            ((4, None), "cycle >= 4"),
        ]
        for name in ("cycle", "cycle_count", "Cycle Count / 1"):
            for value, expected in cases:
                with self.subTest(name=name, value=value):
                    predicate = build_conditions({name: value})
                    sql = str(predicate.compile(compile_kwargs={"literal_binds": True}))
                    self.assertEqual(sql, expected)


class ConditionRoutingTests(unittest.TestCase):
    def setUp(self):
        self.connector = object.__new__(Connector)
        self.connector.get_version = Mock(return_value="0800")
        self.test = {"dev_uid": 260001}
        schemas = {
            "main": {"seq_id": int, "cycle": int, "test_tmp": float},
            "aux": {
                "seq_id": int,
                "cycle": int,
                "auxchl_id": int,
                "test_tmp": float,
            },
        }
        schema_patch = patch(
            "newaresql.connector.get_data_schema", return_value=schemas
        )
        schema_patch.start()
        self.addCleanup(schema_patch.stop)

    def condition_cases(self):
        return [
            ({name: (4, 100)}, {"cycle": (4, 100)})
            for name in ("cycle", "cycle_count", "Cycle Count / 1")
        ] + [
            ({"Record Count / 1": (None, 100)}, {"seq_id": (None, 100)}),
            (None, None),
            ({}, {}),
            ({"unknown": 4}, {}),
        ]

    def test_get_data_normalizes_before_both_join_routes(self):
        data = pl.DataFrame({"seq_id": [1]})
        for join in ("local", "database"):
            for conditions, expected in self.condition_cases():
                with self.subTest(join=join, conditions=conditions):
                    original = None if conditions is None else conditions.copy()
                    with patch.object(
                        self.connector, f"_get_data_{join}", return_value=data
                    ) as fetch:
                        result = self.connector.get_data(
                            self.test,
                            conditions=conditions,
                            naming="bts",
                            transform=False,
                            join=join,
                        )
                    self.assertIs(result, data)
                    self.assertEqual(
                        fetch.call_args.kwargs["main_conditions"], expected
                    )
                    self.assertEqual(fetch.call_args.kwargs["aux_conditions"], expected)
                    self.assertEqual(conditions, original)

    def test_select_data_normalizes_before_schema_split(self):
        main = sa.select(sa.column("seq_id"), sa.column("cycle"))
        aux = sa.select(sa.column("seq_id"), sa.column("test_tmp"))
        for conditions, expected in self.condition_cases():
            with self.subTest(conditions=conditions):
                original = None if conditions is None else conditions.copy()
                with (
                    patch.object(
                        self.connector, "select_main", return_value=main
                    ) as select_main,
                    patch.object(
                        self.connector, "select_aux", return_value=aux
                    ) as select_aux,
                ):
                    self.connector.select_data(self.test, conditions=conditions)
                self.assertEqual(select_main.call_args.kwargs["conditions"], expected)
                self.assertEqual(select_aux.call_args.kwargs["conditions"], expected)
                self.assertEqual(conditions, original)


if __name__ == "__main__":
    unittest.main()
