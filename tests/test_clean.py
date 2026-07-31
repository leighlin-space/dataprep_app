import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import clean


class TestProfileDataframe(unittest.TestCase):
    def test_basic_profile(self):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", None]})
        profiles = clean.profile_dataframe(df)
        by_name = {p.name: p for p in profiles}
        self.assertEqual(by_name["id"].n_rows, 3)
        self.assertTrue(by_name["id"].is_likely_key)
        self.assertAlmostEqual(by_name["name"].null_pct, 33.33, places=1)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"a": [], "b": []})
        profiles = clean.profile_dataframe(df)
        # Should not raise a ZeroDivisionError
        for p in profiles:
            self.assertEqual(p.null_pct, 0.0)


class TestSuggestCleaning(unittest.TestCase):
    def test_flags_fully_empty_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "empty_col": [None, None, None]})
        profiles = clean.profile_dataframe(df)
        suggestions = clean.suggest_cleaning(df, profiles)
        ids = [s.id for s in suggestions]
        self.assertIn("drop_empty_empty_col", ids)

    def test_flags_high_null_but_not_full(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "sparse": [1, None, None, None, None]})
        profiles = clean.profile_dataframe(df)
        suggestions = clean.suggest_cleaning(df, profiles)
        ids = [s.id for s in suggestions]
        self.assertIn("flag_nulls_sparse", ids)
        self.assertNotIn("drop_empty_sparse", ids)

    def test_flags_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        profiles = clean.profile_dataframe(df)
        suggestions = clean.suggest_cleaning(df, profiles)
        ids = [s.id for s in suggestions]
        self.assertIn("drop_duplicates", ids)

    def test_no_false_positive_on_clean_data(self):
        df = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        profiles = clean.profile_dataframe(df)
        suggestions = clean.suggest_cleaning(df, profiles)
        self.assertEqual(suggestions, [])

    def test_flags_whitespace(self):
        df = pd.DataFrame({"state": [" CA", "NY ", "TX"]})
        profiles = clean.profile_dataframe(df)
        suggestions = clean.suggest_cleaning(df, profiles)
        ids = [s.id for s in suggestions]
        self.assertIn("trim_state", ids)


class TestApplySuggestions(unittest.TestCase):
    def test_apply_only_accepted(self):
        df = pd.DataFrame({"a": [1, 1, 2], "empty": [None, None, None]})
        profiles = clean.profile_dataframe(df)
        suggestions = clean.suggest_cleaning(df, profiles)

        # Accept nothing -> dataframe unchanged
        result = clean.apply_suggestions(df, suggestions, accepted_ids=set())
        pd.testing.assert_frame_equal(result, df)

        # Accept drop_duplicates + drop_empty -> both applied
        accepted = {s.id for s in suggestions}
        result = clean.apply_suggestions(df, suggestions, accepted_ids=accepted)
        self.assertNotIn("empty", result.columns)
        self.assertEqual(len(result), 2)  # duplicate dropped

    def test_trim_actually_trims(self):
        df = pd.DataFrame({"state": [" CA", "NY "]})
        profiles = clean.profile_dataframe(df)
        suggestions = clean.suggest_cleaning(df, profiles)
        accepted = {s.id for s in suggestions}
        result = clean.apply_suggestions(df, suggestions, accepted)
        self.assertEqual(result["state"].tolist(), ["CA", "NY"])


class TestFindDuplicates(unittest.TestCase):
    def test_finds_all_instances_not_just_extras(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
        dupes = clean.find_duplicates(df)
        self.assertEqual(len(dupes), 2)  # both copies of the (1, x) row

    def test_no_duplicates_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        dupes = clean.find_duplicates(df)
        self.assertEqual(len(dupes), 0)

    def test_empty_dataframe_does_not_crash(self):
        df = pd.DataFrame({"a": []})
        dupes = clean.find_duplicates(df)
        self.assertEqual(len(dupes), 0)


if __name__ == "__main__":
    unittest.main()
