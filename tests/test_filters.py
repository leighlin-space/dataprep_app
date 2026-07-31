import sys
import unittest
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import filters


class TestFilterProfiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_dir = filters.PROFILE_DIR
        filters.PROFILE_DIR = Path(self._tmp) / "profiles" / "filters"

    def tearDown(self):
        filters.PROFILE_DIR = self._orig_dir
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_seed_creates_all_three_defaults(self):
        filters.seed_default_if_missing()
        names = filters.list_profiles()
        self.assertIn("opioid_8_drug", names)
        self.assertIn("drug_code_12", names)
        self.assertIn("drug_code_14", names)

    def test_seed_is_idempotent(self):
        filters.seed_default_if_missing()
        filters.seed_default_if_missing()  # should not error or duplicate
        names = filters.list_profiles()
        self.assertEqual(names.count("opioid_8_drug"), 1)

    def test_opioid_8_drug_has_8_entries(self):
        filters.seed_default_if_missing()
        fp = filters.load_profile("opioid_8_drug")
        self.assertEqual(len(fp.drug_list), 8)
        self.assertEqual(fp.drug_list_role, "drug_name")

    def test_drug_code_14_is_superset_of_12(self):
        self.assertTrue(set(filters.DRUG_CODE_12).issubset(set(filters.DRUG_CODE_14)))
        self.assertEqual(len(filters.DRUG_CODE_14), len(filters.DRUG_CODE_12) + 2)

    def test_drug_code_profiles_tagged_correctly(self):
        filters.seed_default_if_missing()
        fp12 = filters.load_profile("drug_code_12")
        fp14 = filters.load_profile("drug_code_14")
        self.assertEqual(fp12.drug_list_role, "drug_code")
        self.assertEqual(fp14.drug_list_role, "drug_code")

    def test_save_and_load_custom_profile(self):
        fp = filters.FilterProfile(
            name="custom_test", drug_list=["ASPIRIN"], drug_list_role="drug_name",
            region_values=["Rochester"], date_start="2025-01-01", date_end="2025-12-31",
        )
        filters.save_profile(fp)
        loaded = filters.load_profile("custom_test")
        self.assertEqual(loaded.drug_list, ["ASPIRIN"])
        self.assertEqual(loaded.region_values, ["Rochester"])
        self.assertEqual(loaded.date_start, "2025-01-01")

    def test_delete_profile(self):
        filters.save_profile(filters.FilterProfile(name="throwaway"))
        filters.delete_profile("throwaway")
        self.assertNotIn("throwaway", filters.list_profiles())

    def test_empty_profile_defaults_are_safe(self):
        """A FilterProfile with no args shouldn't crash on save/load (mutable-default trap check)."""
        fp1 = filters.FilterProfile(name="a")
        fp2 = filters.FilterProfile(name="b")
        fp1.drug_list.append("SHOULD_NOT_LEAK")
        self.assertEqual(fp2.drug_list, [])  # confirms field(default_factory=list) isn't shared


if __name__ == "__main__":
    unittest.main()
