"""
Scale guarantees for 500+ datasets.

The point of these tests is NOT throughput — nobody cares whether the test
machine reads 500 small files in 0.4s or 4s. The point is **complexity**:
at 500 files and 500 schemas, anything accidentally quadratic turns into a
multi-second lag on every single Streamlit rerun, and Streamlit reruns the
whole script on every click.

Two hot spots were found this way and are now guarded below:

  * the schema library counted files by re-scanning every entry for every
    schema — O(schemas² × files), measured at 2.7s per rerun at 500/500.
  * suggest_merges compared every schema pair — 124,750 pairs at 500
    schemas, 8.4s to compute, and one UI expander per pair would have hung
    the browser outright.

Where a test asserts a time, the limit is deliberately loose (a slow CI box
should still pass); the assertions that actually pin the complexity are the
ones counting operations.
"""

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import inventory as inv
from modules import sampling as smp

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

N_FILES = 500
N_SCHEMAS_WIDE = 500
N_COLUMNS = 210          # matches the real Dataset1 width


def _fake_schemas(n, cols_per_schema=N_COLUMNS, share_nameset=False):
    """Build n schema records in memory — no files needed."""
    schemas = {}
    for i in range(n):
        cols = [{"name": f"col_{j}", "type": "int64", "physical": "INT64"}
                for j in range(cols_per_schema)]
        if share_nameset:
            # Same names everywhere, differing types only: the pathological
            # input for pairwise merge suggestion.
            cols[i % cols_per_schema] = dict(cols[i % cols_per_schema],
                                             type=f"decimal(18,{i % 9})")
        else:
            cols[0] = {"name": f"unique_{i}", "type": "int64", "physical": "INT64"}
        fs, fn, fset = inv.fingerprints(cols)
        sid = f"S{i:04d}"
        schemas[sid] = inv.SchemaRecord(schema_id=sid, fp_strict=fs, fp_names=fn,
                                        fp_nameset=fset, n_fields=len(cols),
                                        columns=cols)
    return schemas


def _fake_entries(schemas, files_per_schema=1):
    entries = []
    k = 0
    for sid, sc in schemas.items():
        for _ in range(files_per_schema):
            e = inv.FileEntry(path=f"/fake/f{k:05d}.parquet", name=f"f{k:05d}.parquet",
                              folder="/fake", ext=".parquet", size_bytes=1024,
                              source_path=f"/fake/f{k:05d}.parquet")
            e.row_count = 10_000
            e.row_count_source = "footer"
            e.n_fields = sc.n_fields
            e.columns = sc.columns
            e.row_groups = [{"rows": 10_000, "bytes": 1024}]
            e.fp_strict, e.fp_names, e.fp_nameset = sc.fp_strict, sc.fp_names, sc.fp_nameset
            entries.append(e)
            k += 1
    return entries


def _state():
    return dict(inv.DEFAULT_STATE, folders=["/fake"], excluded=[],
                merges={}, file_overrides={})


class TestGroupingAtScale(unittest.TestCase):
    """In-memory: isolates the algorithms from disk speed entirely."""

    def test_500_files_one_schema_groups_into_one_dataset(self):
        schemas = _fake_schemas(1)
        entries = _fake_entries(schemas, files_per_schema=N_FILES)
        ds = [d for d in inv.build_datasets(entries, schemas, _state()) if d["n_files"]]
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["n_files"], N_FILES)
        self.assertEqual(ds[0]["row_count"], N_FILES * 10_000)

    def test_500_distinct_schemas_group_into_500_datasets(self):
        schemas = _fake_schemas(N_SCHEMAS_WIDE)
        entries = _fake_entries(schemas)
        ds = [d for d in inv.build_datasets(entries, schemas, _state()) if d["n_files"]]
        self.assertEqual(len(ds), N_SCHEMAS_WIDE)

    def test_build_datasets_scales_linearly_not_quadratically(self):
        """
        Doubling the input should roughly double the work. A quadratic
        implementation shows up as a 4x jump, which this catches with room
        to spare for timing noise.
        """
        def _time(n):
            schemas = _fake_schemas(n)
            entries = _fake_entries(schemas)
            state = _state()
            t0 = time.perf_counter()
            inv.build_datasets(entries, schemas, state)
            return time.perf_counter() - t0

        small = _time(250)
        large = _time(500)
        self.assertLess(large, max(small * 3.0, 0.5),
                        f"build_datasets went from {small:.3f}s to {large:.3f}s on 2x "
                        f"input — looks worse than linear")

    def test_dataset_numbering_is_contiguous_at_scale(self):
        schemas = _fake_schemas(N_SCHEMAS_WIDE)
        entries = _fake_entries(schemas)
        ds = [d for d in inv.build_datasets(entries, schemas, _state()) if d["n_files"]]
        numbers = sorted(d["dataset_no"] for d in ds)
        self.assertEqual(numbers, list(range(1, N_SCHEMAS_WIDE + 1)))


class TestSchemaLibraryCounting(unittest.TestCase):
    def test_regression_counting_is_one_pass_not_per_schema(self):
        """
        Regression: the schema library did
            for sid in schemas: sum(1 for e in entries if lookup(e) == sid)
        where lookup itself scanned every schema — O(schemas² × files),
        2.7s per rerun at 500 schemas / 500 files. count_files_by_schema
        must do a single pass.
        """
        schemas = _fake_schemas(N_SCHEMAS_WIDE)
        entries = _fake_entries(schemas)
        state = _state()

        lookups = []
        orig = inv.schema_id_for_entry

        def _counted(e, sc, st, index=None):
            lookups.append(1)
            return orig(e, sc, st, index=index)

        inv.schema_id_for_entry = _counted
        try:
            counts = inv.count_files_by_schema(entries, schemas, state)
        finally:
            inv.schema_id_for_entry = orig

        self.assertEqual(len(lookups), len(entries),
                         "one lookup per entry — not one per (schema, entry) pair")
        self.assertEqual(sum(counts.values()), len(entries))

    def test_counting_500x500_is_fast(self):
        schemas = _fake_schemas(N_SCHEMAS_WIDE)
        entries = _fake_entries(schemas)
        t0 = time.perf_counter()
        inv.count_files_by_schema(entries, schemas, _state())
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 1.0,
                        f"counting took {elapsed:.2f}s — this runs on every rerun")

    def test_fingerprint_index_lookup_does_not_scan_schemas(self):
        schemas = _fake_schemas(N_SCHEMAS_WIDE)
        entries = _fake_entries(schemas)
        index = inv.fingerprint_index(schemas)
        self.assertEqual(len(index), N_SCHEMAS_WIDE)
        for e in entries[:50]:
            self.assertIsNotNone(
                inv.schema_id_for_entry(e, schemas, _state(), index=index))


class TestSuggestMergesAtScale(unittest.TestCase):
    def test_regression_pathological_nameset_is_capped_and_fast(self):
        """
        Regression: 500 schemas sharing one column-name set produced 124,750
        candidate pairs in 8.4s, and the UI rendered one expander per pair.
        Must be capped, and must say it was capped rather than silently
        showing a subset.
        """
        schemas = _fake_schemas(N_SCHEMAS_WIDE, share_nameset=True)
        state = _state()
        t0 = time.perf_counter()
        out = inv.suggest_merges(schemas, list(schemas), state)
        elapsed = time.perf_counter() - t0

        self.assertLess(elapsed, 3.0, f"suggest_merges took {elapsed:.1f}s at 500 schemas")
        self.assertLessEqual(len(out), inv.MAX_MERGE_SUGGESTIONS + 1)
        self.assertTrue(out[-1].get("truncated"),
                        "a truncated list must announce itself")
        self.assertIn("candidate pairs", out[-1]["note"])

    def test_no_shared_namesets_means_no_pairs_compared(self):
        """
        Bucketing by nameset first is what keeps this tractable: schemas
        with different column names can never be candidates, so they must
        never reach diff_schemas.
        """
        schemas = _fake_schemas(N_SCHEMAS_WIDE)   # every schema unique
        calls = []
        orig = inv.diff_schemas
        inv.diff_schemas = lambda a, b: (calls.append(1) or orig(a, b))
        try:
            out = inv.suggest_merges(schemas, list(schemas), _state())
        finally:
            inv.diff_schemas = orig
        self.assertEqual(out, [])
        self.assertEqual(calls, [], "compared schemas that can't possibly match")

    def test_genuine_candidates_still_found_among_500(self):
        schemas = _fake_schemas(N_SCHEMAS_WIDE)
        # inject one real false-split pair into an otherwise-unique set
        cols = [dict(c) for c in schemas["S0000"].columns]
        cols[5] = dict(cols[5], type="int32")
        fs, fn, fset = inv.fingerprints(cols)
        schemas["SDUP"] = inv.SchemaRecord(schema_id="SDUP", fp_strict=fs, fp_names=fn,
                                           fp_nameset=fset, n_fields=len(cols),
                                           columns=cols)
        out = inv.suggest_merges(schemas, list(schemas), _state())
        pairs = {frozenset((s["a"], s["b"])) for s in out if not s.get("truncated")}
        self.assertIn(frozenset(("S0000", "SDUP")), pairs)


class TestSamplingKeepsTheAppSmall(unittest.TestCase):
    def test_total_rows_bounded_by_groups_times_sample_size(self):
        """
        The architectural promise: however large the production, the app
        holds n_groups x 1000 rows. 500 datasets = 500k rows, not 5 billion.
        """
        schemas = _fake_schemas(N_SCHEMAS_WIDE)
        entries = _fake_entries(schemas)
        ds = [d for d in inv.build_datasets(entries, schemas, _state()) if d["n_files"]]
        population = sum(d["row_count"] for d in ds)
        ceiling = len(ds) * smp.SAMPLE_ROWS
        self.assertEqual(population, N_SCHEMAS_WIDE * 10_000)
        self.assertLess(ceiling, population,
                        "sampling should be a reduction at this scale")
        self.assertEqual(ceiling, 500_000)

    def test_plan_cost_is_independent_of_population_size(self):
        """
        Doubling the rows per file must not change how much a two_stage plan
        reads — that independence is the entire reason the strategy exists.
        """
        def _plan(rows_per_file):
            schemas = _fake_schemas(1)
            entries = _fake_entries(schemas, files_per_schema=20)
            for e in entries:
                e.row_count = rows_per_file
                e.row_groups = [{"rows": rows_per_file // 10, "bytes": 1000}
                                for _ in range(10)]
            return smp.plan_sample(entries, n=1000, seed=1, rows_per_group=25,
                                   force_strategy="two_stage")

        small = _plan(100_000)
        big = _plan(100_000_000)
        self.assertEqual(small.n_row_groups_touched, big.n_row_groups_touched)
        self.assertEqual(small.est_bytes_read, big.est_bytes_read)

    def test_selection_probability_shrinks_with_population(self):
        schemas = _fake_schemas(1)
        entries = _fake_entries(schemas, files_per_schema=N_FILES)
        for e in entries:
            e.row_count = 10_000_000
            e.row_groups = [{"rows": 1_000_000, "bytes": 10_000} for _ in range(10)]
        plan = smp.plan_sample(entries, n=1000, seed=1, force_strategy="two_stage")
        self.assertEqual(plan.n_total_rows, N_FILES * 10_000_000)
        m = smp.manifest(plan, "S01", n_returned=1000)
        self.assertAlmostEqual(m["selection_probability"], 1000 / plan.n_total_rows)


@unittest.skipUnless(HAS_PYARROW, "needs pyarrow")
class TestRealFilesAtScale(unittest.TestCase):
    """
    The one test that actually touches 500 files on disk. Files are tiny —
    what's being measured is per-file overhead, which is what scales.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp()
        cls.data = Path(cls._tmp) / "prod"
        cls.data.mkdir(parents=True)
        # 500 files across 5 schemas, mimicking one production
        for i in range(N_FILES):
            variant = i % 5
            cols = {f"c{j}": pa.array([1, 2], pa.int64()) for j in range(3 + variant)}
            pq.write_table(pa.table(cols), cls.data / f"CIGNA_IRM{600000 + i}.parquet")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def setUp(self):
        self._orig = (inv.STATE_DIR, inv.SCHEMAS_PATH, inv.STATE_PATH, inv.CACHE_PATH)
        base = Path(self._tmp) / "state"
        inv.STATE_DIR = base
        inv.SCHEMAS_PATH = base / "schemas.json"
        inv.STATE_PATH = base / "state.json"
        inv.CACHE_PATH = base / "scan_cache.json"

    def tearDown(self):
        (inv.STATE_DIR, inv.SCHEMAS_PATH, inv.STATE_PATH, inv.CACHE_PATH) = self._orig

    def test_lists_all_500_files(self):
        self.assertEqual(len(inv.list_folder(str(self.data))), N_FILES)

    def test_scan_500_files_without_reading_data(self):
        recs = inv.list_folder(str(self.data))
        orig_rg, orig_read = pq.ParquetFile.read_row_group, pq.ParquetFile.read

        def _boom(*a, **k):
            raise AssertionError("scanned data pages at scale")

        pq.ParquetFile.read_row_group = _boom
        pq.ParquetFile.read = _boom
        try:
            t0 = time.perf_counter()
            entries, cache = inv.scan_files(recs)
            elapsed = time.perf_counter() - t0
        finally:
            pq.ParquetFile.read_row_group = orig_rg
            pq.ParquetFile.read = orig_read

        self.assertEqual(len(entries), N_FILES)
        self.assertTrue(all(e.ok for e in entries))
        self.assertLess(elapsed, 60.0, f"scanning 500 footers took {elapsed:.1f}s")

    def test_end_to_end_500_files_to_5_datasets(self):
        recs = inv.list_folder(str(self.data))
        entries, cache = inv.scan_files(recs)
        inv.save_cache(cache)
        schemas, new = inv.register_schemas(entries, {})
        state = dict(inv.DEFAULT_STATE, folders=[str(self.data)], excluded=[],
                     merges={}, file_overrides={})
        inv.save_state(state)

        self.assertEqual(len(schemas), 5)
        ds = [d for d in inv.build_datasets(entries, schemas, state) if d["n_files"]]
        self.assertEqual(len(ds), 5)
        self.assertEqual(sum(d["n_files"] for d in ds), N_FILES)
        self.assertEqual(sum(d["row_count"] for d in ds), N_FILES * 2)

        payload = inv.to_json_payload(ds, entries, schemas, state)
        self.assertEqual(payload["totals"]["files"], N_FILES)
        self.assertEqual(payload["totals"]["rows"], N_FILES * 2)

    def test_second_scan_of_500_files_reads_no_footers(self):
        recs = inv.list_folder(str(self.data))
        _, cache = inv.scan_files(recs)
        calls = []
        orig = inv.scan_file
        inv.scan_file = lambda rec: (calls.append(1) or orig(rec))
        try:
            t0 = time.perf_counter()
            inv.scan_files(recs, cache=cache)
            elapsed = time.perf_counter() - t0
        finally:
            inv.scan_file = orig
        self.assertEqual(calls, [], "cache did not prevent a re-read at scale")
        self.assertLess(elapsed, 10.0)

    def test_entries_from_cache_at_scale(self):
        recs = inv.list_folder(str(self.data))
        _, cache = inv.scan_files(recs)
        inv.save_cache(cache)
        state = dict(inv.DEFAULT_STATE, folders=[str(self.data)])
        inv.save_state(state)
        t0 = time.perf_counter()
        entries = inv.entries_from_cache(state)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(entries), N_FILES)
        self.assertLess(elapsed, 15.0, f"cache load took {elapsed:.1f}s per rerun")


if __name__ == "__main__":
    unittest.main()
