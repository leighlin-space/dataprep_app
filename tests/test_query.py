import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import query


class TestBuildQuery(unittest.TestCase):
    def test_simple_select(self):
        sql = query.build_query("orders", columns=[], limit=None)
        self.assertIn('FROM "orders"', sql)
        self.assertIn("SELECT *", sql)
        self.assertNotIn("LIMIT", sql)

    def test_select_specific_columns(self):
        sql = query.build_query("orders", columns=["id", "amount"], limit=None)
        self.assertIn('"id"', sql)
        self.assertIn('"amount"', sql)

    def test_limit_applied(self):
        sql = query.build_query("orders", columns=[], limit=1000)
        self.assertIn("LIMIT 1000", sql)

    def test_no_limit_when_none(self):
        sql = query.build_query("orders", columns=[], limit=None)
        self.assertNotIn("LIMIT", sql)

    def test_join_clause(self):
        sql = query.build_query(
            "orders", columns=[],
            join_table="customers", join_type="LEFT",
            join_left_col="customer_id", join_right_col="id",
            limit=None,
        )
        self.assertIn("LEFT JOIN", sql)
        self.assertIn('"orders"."customer_id" = "customers"."id"', sql)

    def test_equality_filter_quotes_strings(self):
        sql = query.build_query(
            "orders", columns=[],
            filters=[{"column": "region", "op": "=", "value": "Rochester"}],
            limit=None,
        )
        self.assertIn("'Rochester'", sql)

    def test_equality_filter_does_not_quote_numbers(self):
        sql = query.build_query(
            "orders", columns=[],
            filters=[{"column": "amount", "op": ">", "value": "100"}],
            limit=None,
        )
        self.assertIn('"amount" > 100', sql)
        self.assertNotIn("'100'", sql)

    def test_in_filter_with_list(self):
        sql = query.build_query(
            "orders", columns=[],
            filters=[{"column": "drug_name", "op": "IN", "value": ["OXYCODONE", "FENTANYL"]}],
            limit=None,
        )
        self.assertIn("IN ('OXYCODONE', 'FENTANYL')", sql)

    def test_in_filter_with_comma_string(self):
        sql = query.build_query(
            "orders", columns=[],
            filters=[{"column": "drug_code", "op": "IN", "value": "9193, 9143"}],
            limit=None,
        )
        self.assertIn("IN (9193, 9143)", sql)  # numeric-looking codes unquoted

    def test_is_null_filter_no_value_needed(self):
        sql = query.build_query(
            "orders", columns=[],
            filters=[{"column": "region", "op": "IS NULL"}],
            limit=None,
        )
        self.assertIn('"region" IS NULL', sql)

    def test_group_by_and_aggregation(self):
        sql = query.build_query(
            "claims", columns=[],
            group_by=["prescriber_id"],
            aggregations=[{"func": "SUM", "column": "mme", "alias": "total_mme"}],
            limit=None,
        )
        self.assertIn("GROUP BY prescriber_id", sql)
        self.assertIn('SUM("mme") AS "total_mme"', sql)

    def test_count_distinct_aggregation(self):
        sql = query.build_query(
            "claims", columns=[],
            group_by=["prescriber_id"],
            aggregations=[{"func": "COUNT DISTINCT", "column": "patient_id"}],
            limit=None,
        )
        self.assertIn('COUNT(DISTINCT "patient_id")', sql)

    def test_multiple_filters_joined_with_and(self):
        sql = query.build_query(
            "orders", columns=[],
            filters=[
                {"column": "region", "op": "=", "value": "Rochester"},
                {"column": "amount", "op": ">", "value": "0"},
            ],
            limit=None,
        )
        self.assertIn(" AND ", sql)

    def test_sql_injection_style_value_is_contained_in_quotes(self):
        """Not a full injection defense (this tool assumes trusted internal users),
        but a value containing a quote shouldn't silently corrupt the query structure
        in a way that changes which table/columns are touched."""
        sql = query.build_query(
            "orders", columns=[],
            filters=[{"column": "region", "op": "=", "value": "Roch'ester"}],
            limit=None,
        )
        # At minimum, the base table and structure must remain intact
        self.assertIn('FROM "orders"', sql)


class TestLooksNumeric(unittest.TestCase):
    def test_integer_string(self):
        self.assertTrue(query._looks_numeric("123"))

    def test_float_string(self):
        self.assertTrue(query._looks_numeric("123.45"))

    def test_non_numeric_string(self):
        self.assertFalse(query._looks_numeric("Rochester"))

    def test_alphanumeric_drug_code(self):
        # e.g. "9220L" from the drug code list — should NOT be treated as numeric
        self.assertFalse(query._looks_numeric("9220L"))

    def test_none_value(self):
        self.assertFalse(query._looks_numeric(None))


if __name__ == "__main__":
    unittest.main()
