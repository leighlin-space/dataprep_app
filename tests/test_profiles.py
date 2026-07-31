import sys
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import profiles


class TestDatasetProfiles(unittest.TestCase):
    def setUp(self):
        # Redirect PROFILE_DIR to a throwaway temp dir so tests never touch real saved profiles
        self._tmp = tempfile.mkdtemp()
        self._orig_dir = profiles.PROFILE_DIR
        profiles.PROFILE_DIR = Path(self._tmp) / "profiles" / "datasets"

    def tearDown(self):
        profiles.PROFILE_DIR = self._orig_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        p = profiles.DatasetProfile(name="test_format", mapping={"patient_id": "Patient_ID", "mme": "MME"})
        profiles.save_profile(p)
        loaded = profiles.load_profile("test_format")
        self.assertEqual(loaded.mapping["patient_id"], "Patient_ID")
        self.assertEqual(loaded.mapping["mme"], "MME")

    def test_list_profiles(self):
        profiles.save_profile(profiles.DatasetProfile(name="format_a", mapping={}))
        profiles.save_profile(profiles.DatasetProfile(name="format_b", mapping={}))
        names = profiles.list_profiles()
        self.assertIn("format_a", names)
        self.assertIn("format_b", names)

    def test_delete_profile(self):
        profiles.save_profile(profiles.DatasetProfile(name="temp_format", mapping={}))
        self.assertIn("temp_format", profiles.list_profiles())
        profiles.delete_profile("temp_format")
        self.assertNotIn("temp_format", profiles.list_profiles())

    def test_missing_required_roles_detects_gap(self):
        mapping = {"patient_id": "Patient_ID"}  # prescriber_id missing
        missing = profiles.missing_required_roles(mapping)
        self.assertIn("prescriber_id", missing)
        self.assertNotIn("patient_id", missing)

    def test_missing_required_roles_none_missing(self):
        mapping = {"patient_id": "PID", "prescriber_id": "DocID"}
        missing = profiles.missing_required_roles(mapping)
        self.assertEqual(missing, [])

    def test_missing_required_roles_treats_none_as_missing(self):
        mapping = {"patient_id": "PID", "prescriber_id": None}
        missing = profiles.missing_required_roles(mapping)
        self.assertIn("prescriber_id", missing)

    def test_apply_profile_adds_role_columns(self):
        df = pd.DataFrame({"Doctor_ID": ["A1", "A2"], "MME": [10, 20]})
        mapping = {"prescriber_id": "Doctor_ID", "mme": "MME"}
        out = profiles.apply_profile(df, mapping)
        self.assertIn("prescriber_id", out.columns)
        self.assertIn("mme", out.columns)
        self.assertEqual(out["prescriber_id"].tolist(), ["A1", "A2"])

    def test_apply_profile_ignores_unmapped_roles(self):
        df = pd.DataFrame({"Doctor_ID": ["A1"]})
        mapping = {"prescriber_id": "Doctor_ID", "npi": None}
        out = profiles.apply_profile(df, mapping)
        self.assertIn("prescriber_id", out.columns)
        self.assertNotIn("npi", out.columns)

    def test_apply_profile_ignores_column_not_in_df(self):
        """If the mapped source column doesn't actually exist in this df
        (e.g. stale profile applied to the wrong table), should not crash."""
        df = pd.DataFrame({"a": [1, 2]})
        mapping = {"patient_id": "Nonexistent_Col"}
        try:
            out = profiles.apply_profile(df, mapping)
        except Exception as e:
            self.fail(f"apply_profile raised on missing source column: {e}")
        self.assertNotIn("patient_id", out.columns)

    def test_safe_filename_handles_special_characters(self):
        unsafe = "My Profile / v2.0 (final)!"
        safe = profiles._safe_filename(unsafe)
        self.assertNotIn("/", safe)
        self.assertNotIn(" ", safe)
        self.assertNotIn("(", safe)

    def test_field_roles_have_no_duplicate_role_names(self):
        role_names = [r["role"] for r in profiles.FIELD_ROLES]
        self.assertEqual(len(role_names), len(set(role_names)), "Duplicate role defined in FIELD_ROLES")


if __name__ == "__main__":
    unittest.main()
