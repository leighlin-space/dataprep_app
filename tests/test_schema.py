import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import schema


class TestDetectPrimaryKey(unittest.TestCase):
    def test_finds_unique_column(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "name": ["a", "b", "a", "c"]})
        self.assertEqual(schema.detect_primary_key(df), "id")

    def test_no_key_when_nothing_unique_enough(self):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})
        self.assertIsNone(schema.detect_primary_key(df))

    def test_empty_dataframe_returns_none(self):
        df = pd.DataFrame({"a": []})
        self.assertIsNone(schema.detect_primary_key(df))

    def test_picks_the_most_unique_when_multiple_candidates(self):
        df = pd.DataFrame({
            "almost_unique": [1, 2, 3, 3],   # 75% unique
            "fully_unique": [1, 2, 3, 4],    # 100% unique
        })
        self.assertEqual(schema.detect_primary_key(df), "fully_unique")


class TestDetectRelationships(unittest.TestCase):
    def test_finds_matching_key_between_tables(self):
        orders = pd.DataFrame({"order_id": [1, 2, 3], "customer_id": [10, 20, 10]})
        customers = pd.DataFrame({"id": [10, 20, 30], "name": ["A", "B", "C"]})
        rels = schema.detect_relationships({"orders": orders, "customers": customers})
        matches = [r for r in rels if r.from_column == "customer_id" and r.to_column == "id"]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].overlap_pct, 100.0)

    def test_no_relationship_when_no_overlap(self):
        t1 = pd.DataFrame({"a": [1, 2, 3]})
        t2 = pd.DataFrame({"b": [100, 200, 300]})
        rels = schema.detect_relationships({"t1": t1, "t2": t2})
        self.assertEqual(rels, [])

    def test_single_column_table_does_not_relate_to_itself(self):
        t1 = pd.DataFrame({"a": [1, 2, 3]})
        rels = schema.detect_relationships({"t1": t1})
        self.assertEqual(rels, [])

    def test_low_cardinality_column_not_flagged(self):
        """A column with only 1 distinct value (e.g. all 'Y') shouldn't
        spuriously match everything — guarded by len(vals1) < 2 check."""
        t1 = pd.DataFrame({"flag": ["Y", "Y", "Y"]})
        t2 = pd.DataFrame({"other_flag": ["Y", "Y", "Y"]})
        rels = schema.detect_relationships({"t1": t1, "t2": t2})
        self.assertEqual(rels, [])

    def test_partial_overlap_below_threshold_not_flagged(self):
        t1 = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        t2 = pd.DataFrame({"b": [1, 2, 99, 98, 97]})  # only 40% overlap
        rels = schema.detect_relationships({"t1": t1, "t2": t2})
        self.assertEqual(rels, [])


class TestDedupeRelationships(unittest.TestCase):
    def test_keeps_stronger_direction(self):
        rels = [
            schema.Relationship("a", "x", "b", "y", overlap_pct=100.0),
            schema.Relationship("b", "y", "a", "x", overlap_pct=80.0),
        ]
        deduped = schema._dedupe_relationships(rels)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].overlap_pct, 100.0)

    def test_distinct_relationships_both_kept(self):
        rels = [
            schema.Relationship("a", "x", "b", "y", overlap_pct=100.0),
            schema.Relationship("a", "z", "c", "w", overlap_pct=95.0),
        ]
        deduped = schema._dedupe_relationships(rels)
        self.assertEqual(len(deduped), 2)


class TestToMermaid(unittest.TestCase):
    def test_produces_er_diagram_header(self):
        tables = {"orders": pd.DataFrame({"id": [1, 2]})}
        result = schema.to_mermaid(tables, [])
        self.assertTrue(result.startswith("erDiagram"))
        self.assertIn("orders", result)

    def test_marks_primary_key(self):
        tables = {"orders": pd.DataFrame({"order_id": [1, 2, 3], "amt": [10, 10, 10]})}
        result = schema.to_mermaid(tables, [])
        self.assertIn("order_id PK", result)

    def test_includes_relationship_lines(self):
        tables = {"a": pd.DataFrame({"x": [1]}), "b": pd.DataFrame({"y": [1]})}
        rel = schema.Relationship("a", "x", "b", "y", overlap_pct=100.0)
        result = schema.to_mermaid(tables, [rel])
        self.assertIn("a }o--o{ b", result)

    def test_caps_columns_shown_at_12(self):
        wide_df = pd.DataFrame({f"col_{i}": [1, 2] for i in range(20)})
        result = schema.to_mermaid({"wide": wide_df}, [])
        # Count column lines inside the wide table's block (rough check: at most 12 "col_" mentions)
        col_mentions = result.count("col_")
        self.assertLessEqual(col_mentions, 12)


# ---------------------------------------------------------------------------
# SQL-based functions need a real DuckDB connection — run on your machine.
# ---------------------------------------------------------------------------
try:
    import duckdb as _real_duckdb
    HAS_DUCKDB = not getattr(_real_duckdb, "__is_stub__", False)
except ImportError:
    HAS_DUCKDB = False


@unittest.skipUnless(HAS_DUCKDB, "Requires a real duckdb install — run on your machine, not the sandbox")
class TestSchemaSqlFunctions(unittest.TestCase):
    def setUp(self):
        from modules import db
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = db.DB_PATH
        self._orig_temp_dir = db.TEMP_DIR
        db.DB_PATH = Path(self._tmp) / "test_warehouse.duckdb"
        db.TEMP_DIR = Path(self._tmp) / "spill"
        self.db = db
        self.con = db.get_connection()

    def tearDown(self):
        self.con.close()
        self.db.DB_PATH = self._orig_db_path
        self.db.TEMP_DIR = self._orig_temp_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_sql_and_pandas_versions_agree_on_primary_key(self):
        """The SQL and pandas versions of detect_primary_key should agree
        on the same data — a divergence would mean one of them has a bug."""
        df = pd.DataFrame({"id": range(100), "val": ["x"] * 100})
        self.db.register_table(self.con, "t", df)
        pandas_result = schema.detect_primary_key(df)
        sql_result = schema.detect_primary_key_sql(self.con, "t")
        self.assertEqual(pandas_result, sql_result)

    def test_sql_and_pandas_versions_agree_on_relationships(self):
        orders = pd.DataFrame({"order_id": range(50), "customer_id": [i % 10 for i in range(50)]})
        customers = pd.DataFrame({"id": range(10), "name": [f"C{i}" for i in range(10)]})
        self.db.register_table(self.con, "orders", orders)
        self.db.register_table(self.con, "customers", customers)

        pandas_rels = schema.detect_relationships({"orders": orders, "customers": customers})
        sql_rels = schema.detect_relationships_sql(self.con, ["orders", "customers"], candidate_only=False)

        pandas_pairs = {(r.from_column, r.to_column) for r in pandas_rels}
        sql_pairs = {(r.from_column, r.to_column) for r in sql_rels}
        self.assertEqual(pandas_pairs, sql_pairs)

    def test_relationship_detection_on_empty_table_does_not_crash(self):
        empty = pd.DataFrame({"a": pd.array([], dtype="Int64")})
        other = pd.DataFrame({"b": [1, 2, 3]})
        self.db.register_table(self.con, "empty_t", empty)
        self.db.register_table(self.con, "other_t", other)
        try:
            rels = schema.detect_relationships_sql(self.con, ["empty_t", "other_t"])
        except Exception as e:
            self.fail(f"detect_relationships_sql raised on empty table: {e}")
        self.assertEqual(rels, [])

    def test_regression_cross_type_column_comparison(self):
        """
        Regression test: comparing a string ID in one table against an
        integer ID in another (e.g. 'SR #' as VARCHAR vs 'PRCNG_EVENT_ID'
        as BIGINT — two real data sources with the same semantic key
        stored differently) used to raise:
        BinderException: Cannot compare values of type VARCHAR and BIGINT
        in IN/ANY/ALL clause - an explicit cast is required.
        Must not raise, and should still detect the relationship.
        """
        t1 = pd.DataFrame({"SR #": ["1001", "1002", "1003", "1004", "1005"]})
        t2 = pd.DataFrame({"PRCNG_EVENT_ID": [1001, 1002, 1003, 1004, 9999]})
        self.db.register_table(self.con, "t1", t1)
        self.db.register_table(self.con, "t2", t2)
        try:
            rels = schema.detect_relationships_sql(self.con, ["t1", "t2"], candidate_only=False)
        except Exception as e:
            self.fail(f"detect_relationships_sql raised on cross-type comparison: {e}")
        # 4 of 5 values overlap (80%) — below the 90% threshold, so no
        # relationship is expected here; the point of this test is that
        # it runs at all without a BinderException, not the exact count.
        self.assertIsInstance(rels, list)

    def test_regression_cross_type_comparison_detects_full_overlap(self):
        """Same cross-type scenario, but with 100% overlap — should actually be flagged."""
        t1 = pd.DataFrame({"SR #": ["1001", "1002", "1003"]})
        t2 = pd.DataFrame({"PRCNG_EVENT_ID": [1001, 1002, 1003]})
        self.db.register_table(self.con, "t1", t1)
        self.db.register_table(self.con, "t2", t2)
        rels = schema.detect_relationships_sql(self.con, ["t1", "t2"], candidate_only=False)
        matches = [r for r in rels if {r.from_column, r.to_column} == {"SR #", "PRCNG_EVENT_ID"}]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].overlap_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
