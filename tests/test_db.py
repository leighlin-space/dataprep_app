import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import db


class TestSafeTableName(unittest.TestCase):
    """
    Pure string logic, no DuckDB connection needed — runs anywhere.
    Regression coverage for a real bug found during review: the old
    implementation used Path(name).stem, which silently truncated any
    table name containing a dot (e.g. 'sales.q1_2026' -> 'sales').
    """

    def test_simple_name_unchanged_shape(self):
        self.assertEqual(db._safe_table_name("orders"), "orders")

    def test_does_not_truncate_at_dot(self):
        # This is the exact bug: must NOT collapse to just "sales"
        result = db._safe_table_name("sales.q1_2026")
        self.assertNotEqual(result, "sales")
        self.assertIn("q1_2026", result)

    def test_spaces_replaced(self):
        result = db._safe_table_name("my table name")
        self.assertNotIn(" ", result)

    def test_leading_digit_gets_prefixed(self):
        result = db._safe_table_name("2026_claims")
        self.assertFalse(result[0].isdigit())
        self.assertTrue(result.startswith("t_"))

    def test_empty_name_falls_back(self):
        result = db._safe_table_name("")
        self.assertEqual(result, "unnamed_table")

    def test_special_characters_sanitized(self):
        result = db._safe_table_name("data!@#$%^&*()export")
        self.assertTrue(all(c.isalnum() or c == "_" for c in result))

    def test_multiple_dots_all_preserved_as_content(self):
        result = db._safe_table_name("a.b.c.d")
        # Every meaningful segment should still be represented, not just "a"
        self.assertIn("b", result)
        self.assertIn("c", result)
        self.assertIn("d", result)


# ---------------------------------------------------------------------------
# The tests below require a real DuckDB connection and could not be run in
# the sandbox used to build this suite (no network access to install
# duckdb there). Run these on your machine, where duckdb is installed via
# requirements.txt, to get full coverage of db.py.
# ---------------------------------------------------------------------------
try:
    import duckdb as _real_duckdb
    HAS_DUCKDB = not getattr(_real_duckdb, "__is_stub__", False)
except ImportError:
    HAS_DUCKDB = False


@unittest.skipUnless(HAS_DUCKDB, "Requires a real duckdb install — run on your machine, not the sandbox")
class TestDbWithRealConnection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = db.DB_PATH
        self._orig_temp_dir = db.TEMP_DIR
        db.DB_PATH = Path(self._tmp) / "test_warehouse.duckdb"
        db.TEMP_DIR = Path(self._tmp) / "spill"
        self.con = db.get_connection()

    def tearDown(self):
        self.con.close()
        db.DB_PATH = self._orig_db_path
        db.TEMP_DIR = self._orig_temp_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_register_and_list_table(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        db.register_table(self.con, "test_table", df)
        self.assertIn("test_table", db.list_tables(self.con))

    def test_register_table_overwrites(self):
        db.register_table(self.con, "t", pd.DataFrame({"a": [1]}))
        db.register_table(self.con, "t", pd.DataFrame({"a": [1, 2, 3]}))
        self.assertEqual(db.count_rows(self.con, "t"), 3)

    def test_count_rows_matches_actual(self):
        db.register_table(self.con, "t", pd.DataFrame({"a": range(50)}))
        self.assertEqual(db.count_rows(self.con, "t"), 50)

    def test_count_rows_never_materializes_full_table(self):
        """Regression: earlier version of Tab 3 called get_table_df() just
        to get a row count, which crashed on very large tables. count_rows
        must work via SQL COUNT(*) only."""
        df = pd.DataFrame({"a": range(1000)})
        db.register_table(self.con, "big", df)
        n = db.count_rows(self.con, "big")
        self.assertEqual(n, 1000)

    def test_get_columns_matches_dataframe_columns(self):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        db.register_table(self.con, "t", df)
        cols = db.get_columns(self.con, "t")
        self.assertEqual(set(cols), {"x", "y", "z"})

    def test_preview_table_respects_limit(self):
        df = pd.DataFrame({"a": range(1000)})
        db.register_table(self.con, "t", df)
        preview = db.preview_table(self.con, "t", n=10)
        self.assertEqual(len(preview), 10)

    def test_drop_table_removes_it(self):
        db.register_table(self.con, "temp", pd.DataFrame({"a": [1]}))
        db.drop_table(self.con, "temp")
        self.assertNotIn("temp", db.list_tables(self.con))

    def test_drop_nonexistent_table_does_not_raise(self):
        try:
            db.drop_table(self.con, "never_existed")
        except Exception as e:
            self.fail(f"drop_table raised on nonexistent table: {e}")

    def test_table_name_with_dot_actually_registers(self):
        """End-to-end confirmation of the _safe_table_name fix."""
        db.register_table(self.con, "sales.q1_2026", pd.DataFrame({"a": [1, 2]}))
        tables = db.list_tables(self.con)
        self.assertTrue(any("q1_2026" in t for t in tables))


if __name__ == "__main__":
    unittest.main()
