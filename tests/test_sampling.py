import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import inventory as inv
from modules import sampling as smp

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import duckdb as _d
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class _DataCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.data = Path(self._tmp) / "data"
        self.data.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _sorted_parquet(self, n_files=3, per_file=10_000, row_group_size=1000):
        """
        Deliberately SORTED data across files — the worst case for a
        clustered sample, and the case that reveals bias if any exists.
        """
        total = 0
        for f in range(n_files):
            ids = list(range(total, total + per_file))
            pq.write_table(pa.table({"idx": pa.array(ids),
                                     "pad": pa.array(["x" * 8] * per_file)}),
                           self.data / f"p{f}.parquet",
                           row_group_size=row_group_size)
            total += per_file
        entries, _ = inv.scan_files(inv.list_folder(str(self.data)))
        return entries, total


@unittest.skipUnless(HAS_PYARROW, "needs pyarrow")
class TestPlanning(_DataCase):
    def test_small_group_takes_all_rows(self):
        pq.write_table(pa.table({"a": pa.array(range(50))}), self.data / "a.parquet")
        entries, _ = inv.scan_files(inv.list_folder(str(self.data)))
        plan = smp.plan_sample(entries, n=1000, seed=1)
        self.assertEqual(plan.strategy, "all_rows")
        df = smp.execute_plan(plan)
        self.assertEqual(len(df), 50)

    def test_large_group_forced_two_stage_touches_few_row_groups(self):
        entries, total = self._sorted_parquet()
        plan = smp.plan_sample(entries, n=1000, seed=1, rows_per_group=25,
                               force_strategy="two_stage")
        self.assertEqual(plan.strategy, "two_stage")
        self.assertLessEqual(plan.n_row_groups_touched, 40)
        self.assertLess(plan.n_row_groups_touched, plan.n_row_groups_total)

    def test_rows_per_group_trades_spread_for_bytes(self):
        entries, _ = self._sorted_parquet()
        wide = smp.plan_sample(entries, n=1000, seed=1, rows_per_group=10,
                               force_strategy="two_stage")
        narrow = smp.plan_sample(entries, n=1000, seed=1, rows_per_group=200,
                                 force_strategy="two_stage")
        self.assertGreater(wide.n_row_groups_touched, narrow.n_row_groups_touched)
        self.assertGreater(wide.est_bytes_read, narrow.est_bytes_read)

    def test_auto_picks_exact_for_small_groups(self):
        pq.write_table(pa.table({"a": pa.array(range(5000))}), self.data / "a.parquet")
        entries, _ = inv.scan_files(inv.list_folder(str(self.data)))
        self.assertEqual(smp.plan_sample(entries, n=1000, seed=1).strategy, "exact")

    def test_planning_reads_no_data(self):
        """Planning happens off footer metadata so the UI can show cost first."""
        entries, _ = self._sorted_parquet()
        orig_rg, orig_read = pq.ParquetFile.read_row_group, pq.ParquetFile.read

        def _boom(*a, **k):
            raise AssertionError("planning read data pages")

        pq.ParquetFile.read_row_group = _boom
        pq.ParquetFile.read = _boom
        try:
            smp.plan_sample(entries, n=1000, seed=1, force_strategy="two_stage")
        finally:
            pq.ParquetFile.read_row_group = orig_rg
            pq.ParquetFile.read = orig_read


@unittest.skipUnless(HAS_PYARROW, "needs pyarrow")
class TestTwoStageExecution(_DataCase):
    def test_returns_exactly_the_target_row_count(self):
        entries, _ = self._sorted_parquet()
        plan = smp.plan_sample(entries, n=1000, seed=7, force_strategy="two_stage")
        self.assertEqual(len(smp.execute_plan(plan)), 1000)

    def test_same_seed_reproduces_the_same_rows(self):
        entries, _ = self._sorted_parquet()
        a = smp.execute_plan(smp.plan_sample(entries, n=500, seed=42,
                                             force_strategy="two_stage"))
        b = smp.execute_plan(smp.plan_sample(entries, n=500, seed=42,
                                             force_strategy="two_stage"))
        self.assertEqual(a["idx"].tolist(), b["idx"].tolist())

    def test_different_seeds_give_different_rows(self):
        entries, _ = self._sorted_parquet()
        a = smp.execute_plan(smp.plan_sample(entries, n=500, seed=1,
                                             force_strategy="two_stage"))
        b = smp.execute_plan(smp.plan_sample(entries, n=500, seed=2,
                                             force_strategy="two_stage"))
        self.assertNotEqual(a["idx"].tolist(), b["idx"].tolist())

    def test_no_duplicate_rows_within_a_row_group(self):
        entries, _ = self._sorted_parquet()
        plan = smp.plan_sample(entries, n=1000, seed=3, force_strategy="two_stage")
        for pick in plan.picks:
            self.assertEqual(len(pick["offsets"]), len(set(pick["offsets"])))

    def test_offsets_are_inside_their_row_group(self):
        entries, _ = self._sorted_parquet()
        plan = smp.plan_sample(entries, n=1000, seed=3, force_strategy="two_stage")
        sizes = {}
        for e in entries:
            for i, rg in enumerate(e.row_groups):
                sizes[(e.path, i)] = rg["rows"]
        for pick in plan.picks:
            limit = sizes[(pick["path"], pick["row_group"])]
            self.assertLess(max(pick["offsets"]), limit)

    def test_execution_only_reads_the_planned_row_groups(self):
        """
        The cost guarantee: reading more row groups than planned is what
        turns a seconds-long sample into an hours-long scan.
        """
        entries, _ = self._sorted_parquet()
        plan = smp.plan_sample(entries, n=1000, seed=5, rows_per_group=25,
                               force_strategy="two_stage")
        calls = []
        orig = pq.ParquetFile.read_row_group

        def _counted(self_, i, *a, **k):
            calls.append(i)
            return orig(self_, i, *a, **k)

        pq.ParquetFile.read_row_group = _counted
        try:
            smp.execute_plan(plan)
        finally:
            pq.ParquetFile.read_row_group = orig
        self.assertEqual(len(calls), len(plan.picks))

    def test_sample_is_unbiased_on_sorted_data(self):
        """
        Statistical check, seeded so it's deterministic. Sorted data means a
        clustered sample would show up as bias if the weighting were wrong:
        row groups must be chosen with probability proportional to their row
        count, or early files get over-sampled.

        Tolerance is loose on purpose — this catches a broken weighting
        scheme, not a subtle variance change.
        """
        entries, total = self._sorted_parquet()
        truth = (total - 1) / 2
        means = []
        for seed in range(40):
            plan = smp.plan_sample(entries, n=1000, seed=seed, rows_per_group=25,
                                   force_strategy="two_stage")
            means.append(smp.execute_plan(plan)["idx"].mean())
        grand = sum(means) / len(means)
        self.assertLess(abs(grand - truth), 0.05 * total,
                        f"mean of sample means {grand:.0f} vs truth {truth:.0f} — "
                        f"suggests the row-group weighting is wrong")

    def test_every_file_can_be_reached(self):
        """No file should be structurally excluded from selection."""
        entries, _ = self._sorted_parquet()
        seen = set()
        for seed in range(30):
            plan = smp.plan_sample(entries, n=1000, seed=seed,
                                   force_strategy="two_stage")
            seen.update(p["path"] for p in plan.picks)
        self.assertEqual(len(seen), 3)


@unittest.skipUnless(HAS_PYARROW and HAS_DUCKDB, "needs pyarrow + duckdb")
class TestExactExecution(_DataCase):
    def test_exact_returns_target_rows_and_is_reproducible(self):
        entries, _ = self._sorted_parquet(n_files=1, per_file=5000)
        a = smp.execute_plan(smp.plan_sample(entries, n=1000, seed=11,
                                            force_strategy="exact"))
        b = smp.execute_plan(smp.plan_sample(entries, n=1000, seed=11,
                                            force_strategy="exact"))
        self.assertEqual(len(a), 1000)
        self.assertEqual(sorted(a["idx"]), sorted(b["idx"]))

    def test_exact_has_no_duplicates(self):
        entries, _ = self._sorted_parquet(n_files=1, per_file=5000)
        df = smp.execute_plan(smp.plan_sample(entries, n=1000, seed=1,
                                              force_strategy="exact"))
        self.assertEqual(len(df), len(set(df["idx"])))


@unittest.skipUnless(HAS_DUCKDB, "needs duckdb")
class TestMixedFormatGroups(_DataCase):
    def test_csv_in_group_forces_exact(self):
        pd.DataFrame({"a": range(100)}).to_csv(self.data / "a.csv", index=False)
        entries, _ = inv.scan_files(inv.list_folder(str(self.data)))
        plan = smp.plan_sample(entries, n=10, seed=1)
        self.assertEqual(plan.strategy, "exact")
        self.assertTrue(plan.footerless_files)
        self.assertEqual(len(smp.execute_plan(plan)), 10)

    def test_csv_group_row_count_is_still_known(self):
        """
        'No footer' costs us row-group BOUNDARIES for sampling, not the row
        count — the inventory already measured that exactly.
        """
        pd.DataFrame({"a": range(100)}).to_csv(self.data / "a.csv", index=False)
        entries, _ = inv.scan_files(inv.list_folder(str(self.data)))
        self.assertEqual(entries[0].row_count, 100)
        plan = smp.plan_sample(entries, n=10, seed=1)
        self.assertIn("row COUNT is already known", plan.note)

    def test_excel_sheet_is_sampled_by_name(self):
        path = self.data / "book.xlsx"
        with pd.ExcelWriter(path) as w:
            pd.DataFrame({"a": range(200), "tag": ["first"] * 200}).to_excel(
                w, sheet_name="First", index=False)
            pd.DataFrame({"a": range(200), "tag": ["second"] * 200}).to_excel(
                w, sheet_name="Second", index=False)
        entries = inv.scan_file(inv.list_folder(str(self.data))[0])
        second = [e for e in entries if e.sheet == "Second"]
        plan = smp.plan_sample(second, n=50, seed=1)
        df = smp.execute_plan(plan)
        self.assertEqual(len(df), 50)
        self.assertEqual(set(df["tag"]), {"second"},
                         "sampling pulled the wrong sheet")

    def test_mixed_excel_and_parquet_group_samples_from_both(self):
        if not HAS_PYARROW:
            self.skipTest("needs pyarrow")
        pq.write_table(pa.table({"a": pa.array(range(500)),
                                 "src": pa.array(["pq"] * 500)}),
                       self.data / "a.parquet")
        with pd.ExcelWriter(self.data / "b.xlsx") as w:
            pd.DataFrame({"a": range(500), "src": ["xl"] * 500}).to_excel(
                w, sheet_name="S", index=False)
        entries, _ = inv.scan_files(inv.list_folder(str(self.data)))
        plan = smp.plan_sample(entries, n=200, seed=1)
        df = smp.execute_plan(plan)
        self.assertEqual(len(df), 200)
        self.assertEqual(set(df["src"]), {"pq", "xl"},
                         "a mixed group must sample from both sources, "
                         "not just the SQL-readable one")


class TestManifestAndNaming(unittest.TestCase):
    def test_manifest_records_what_is_needed_to_reproduce(self):
        plan = smp.SamplePlan(strategy="two_stage", n_target=1000, n_total_rows=5_000_000,
                              seed=123, rows_per_group=25,
                              picks=[{"path": "/x/a.parquet", "row_group": 3,
                                      "offsets": [1, 2, 3]}],
                              n_row_groups_touched=1, n_row_groups_total=50,
                              files=["/x/a.parquet"])
        m = smp.manifest(plan, "S01", label="cigna", n_returned=1000)
        self.assertEqual(m["seed"], 123)
        self.assertEqual(m["strategy"], "two_stage")
        self.assertAlmostEqual(m["selection_probability"], 1000 / 5_000_000)
        self.assertEqual(m["picks"][0]["row_group"], 3)
        self.assertEqual(m["picks"][0]["file"], "a.parquet")

    def test_exact_manifest_has_no_pick_list(self):
        plan = smp.SamplePlan(strategy="exact", n_target=10, n_total_rows=100, seed=1)
        self.assertIsNone(smp.manifest(plan, "S01")["picks"])

    def test_sample_table_name_is_a_safe_identifier(self):
        self.assertEqual(smp.sample_table_name("S01"), "S01_sample")
        self.assertEqual(smp.sample_table_name("S01", "cigna 018 v2"), "cigna_018_v2_sample")
        self.assertTrue(smp.sample_table_name("S01", "2026 claims")[0].isalpha())

    def test_sample_table_names_are_distinct_per_group(self):
        names = {smp.sample_table_name(f"S{i:02d}") for i in range(500)}
        self.assertEqual(len(names), 500)


if __name__ == "__main__":
    unittest.main()
