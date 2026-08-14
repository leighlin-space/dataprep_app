import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules import inventory as inv

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


def _write_parquet(path, cols, row_group_size=None):
    tbl = pa.table(cols)
    pq.write_table(tbl, path, row_group_size=row_group_size)


class _TmpStateCase(unittest.TestCase):
    """Redirects the state dir so tests never touch a real inventory."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig = (inv.STATE_DIR, inv.SCHEMAS_PATH, inv.STATE_PATH, inv.CACHE_PATH)
        base = Path(self._tmp) / "inventory_state"
        inv.STATE_DIR = base
        inv.SCHEMAS_PATH = base / "schemas.json"
        inv.STATE_PATH = base / "state.json"
        inv.CACHE_PATH = base / "scan_cache.json"
        self.data = Path(self._tmp) / "data"
        self.data.mkdir()

    def tearDown(self):
        (inv.STATE_DIR, inv.SCHEMAS_PATH,
         inv.STATE_PATH, inv.CACHE_PATH) = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)


class TestFingerprints(unittest.TestCase):
    def test_identical_columns_same_fingerprint(self):
        cols = [{"name": "a", "type": "int64"}, {"name": "b", "type": "string"}]
        self.assertEqual(inv.fingerprints(cols), inv.fingerprints(list(cols)))

    def test_type_change_splits_strict_but_not_names(self):
        a = [{"name": "a", "type": "int64"}]
        b = [{"name": "a", "type": "int32"}]
        fa = inv.fingerprints(a)
        fb = inv.fingerprints(b)
        self.assertNotEqual(fa[0], fb[0], "strict fingerprint must notice the type change")
        self.assertEqual(fa[1], fb[1], "name fingerprint must ignore types")
        self.assertEqual(fa[2], fb[2])

    def test_order_change_splits_strict_and_names_but_not_nameset(self):
        a = [{"name": "a", "type": "int64"}, {"name": "b", "type": "int64"}]
        b = list(reversed(a))
        fa, fb = inv.fingerprints(a), inv.fingerprints(b)
        self.assertNotEqual(fa[0], fb[0])
        self.assertNotEqual(fa[1], fb[1])
        self.assertEqual(fa[2], fb[2], "nameset fingerprint must be order-insensitive")


class TestInferType(unittest.TestCase):
    def test_ints_stay_int(self):
        self.assertEqual(inv._infer_type([1, 2, 3]), "int64")

    def test_int_and_float_promote_to_double(self):
        self.assertEqual(inv._infer_type([1, 2.5]), "double")

    def test_mixed_falls_back_to_varchar(self):
        self.assertEqual(inv._infer_type([1, "x"]), "varchar")

    def test_all_blank_is_empty(self):
        self.assertEqual(inv._infer_type([None, "", None]), "empty")

    def test_bool_not_treated_as_int(self):
        """bool is an int subclass in Python — must not be reported as int64."""
        self.assertEqual(inv._infer_type([True, False]), "boolean")


@unittest.skipUnless(HAS_PYARROW, "needs pyarrow")
class TestParquetScan(_TmpStateCase):
    def test_exact_row_count_and_fields_from_footer(self):
        _write_parquet(self.data / "a.parquet",
                       {"id": pa.array(range(500)), "v": pa.array(["x"] * 500)})
        entries = inv.scan_file(inv.list_folder(str(self.data))[0])
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e.row_count, 500)
        self.assertEqual(e.row_count_source, "footer")
        self.assertEqual(e.n_fields, 2)

    def test_row_groups_recorded(self):
        _write_parquet(self.data / "a.parquet", {"id": pa.array(range(1000))},
                       row_group_size=100)
        e = inv.scan_file(inv.list_folder(str(self.data))[0])[0]
        self.assertEqual(len(e.row_groups), 10)
        self.assertEqual(sum(g["rows"] for g in e.row_groups), 1000)

    def test_scan_never_reads_data_pages(self):
        """
        The whole premise of the inventory step: it must read footers only.
        If anything ever calls read_row_group/read_table during a scan, a
        1.02 TB production becomes an hours-long scan instead of seconds.
        """
        _write_parquet(self.data / "a.parquet", {"id": pa.array(range(100))})

        orig_rg = pq.ParquetFile.read_row_group
        orig_read = pq.ParquetFile.read

        def _boom(*a, **k):
            raise AssertionError("scan read data pages — it must use the footer only")

        pq.ParquetFile.read_row_group = _boom
        pq.ParquetFile.read = _boom
        try:
            e = inv.scan_file(inv.list_folder(str(self.data))[0])[0]
            self.assertEqual(e.row_count, 100)
            self.assertEqual(e.error, "")
        finally:
            pq.ParquetFile.read_row_group = orig_rg
            pq.ParquetFile.read = orig_read

    def test_unreadable_file_is_reported_not_raised(self):
        (self.data / "broken.parquet").write_bytes(b"not a parquet file")
        entries = inv.scan_file(inv.list_folder(str(self.data))[0])
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0].ok)
        self.assertIn("parquet", entries[0].error.lower())


@unittest.skipUnless(HAS_DUCKDB, "needs duckdb")
class TestDelimitedScan(_TmpStateCase):
    def test_csv_row_count_is_exact(self):
        pd.DataFrame({"id": range(321), "v": ["a"] * 321}).to_csv(
            self.data / "x.csv", index=False)
        e = inv.scan_file(inv.list_folder(str(self.data))[0])[0]
        self.assertEqual(e.row_count, 321)
        self.assertEqual(e.row_count_source, "scan")

    def test_regression_quoted_newlines_not_miscounted(self):
        """
        Counting newlines would be faster and wrong: RFC-4180 allows
        newlines inside quoted fields. On this file `wc -l` reports 2x the
        real row count. The inventory must report the true number.
        """
        n = 200
        path = self.data / "quoted.csv"
        pd.DataFrame({"id": range(n), "note": ["line1\nline2"] * n}).to_csv(
            path, index=False)
        e = inv.scan_file(inv.list_folder(str(self.data))[0])[0]
        self.assertEqual(e.row_count, n)
        physical_lines = path.read_text().count("\n")
        self.assertGreater(physical_lines, n,
                           "test file should actually contain embedded newlines")

    def test_json_row_count(self):
        pd.DataFrame({"id": range(50)}).to_json(
            self.data / "r.json", orient="records")
        e = inv.scan_file(inv.list_folder(str(self.data))[0])[0]
        self.assertEqual(e.row_count, 50)

    def test_no_format_reports_unknown_row_count(self):
        """Every supported format must produce a real row count."""
        pd.DataFrame({"a": [1, 2]}).to_csv(self.data / "a.csv", index=False)
        pd.DataFrame({"a": [1, 2, 3]}).to_json(self.data / "b.json", orient="records")
        if HAS_PYARROW:
            _write_parquet(self.data / "c.parquet", {"a": pa.array([1])})
        entries, _ = inv.scan_files(inv.list_folder(str(self.data)))
        for e in entries:
            self.assertIsNotNone(e.row_count, f"{e.name} reported an unknown row count")
            self.assertIn(e.row_count_source, ("footer", "scan"))


class TestExcelScan(_TmpStateCase):
    def test_each_sheet_is_its_own_entry(self):
        path = self.data / "book.xlsx"
        with pd.ExcelWriter(path) as w:
            pd.DataFrame({"a": range(30), "b": range(30)}).to_excel(
                w, sheet_name="First", index=False)
            pd.DataFrame({"x": range(7), "y": range(7), "z": range(7)}).to_excel(
                w, sheet_name="Second", index=False)
        entries = inv.scan_file(inv.list_folder(str(self.data))[0])
        self.assertEqual(len(entries), 2)
        by_sheet = {e.sheet: e for e in entries}
        self.assertEqual(by_sheet["First"].row_count, 30)
        self.assertEqual(by_sheet["First"].n_fields, 2)
        self.assertEqual(by_sheet["Second"].row_count, 7)
        self.assertEqual(by_sheet["Second"].n_fields, 3)

    def test_sheets_with_different_layouts_get_different_fingerprints(self):
        path = self.data / "book.xlsx"
        with pd.ExcelWriter(path) as w:
            pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="A", index=False)
            pd.DataFrame({"b": [1], "c": [2]}).to_excel(w, sheet_name="B", index=False)
        entries = inv.scan_file(inv.list_folder(str(self.data))[0])
        self.assertNotEqual(entries[0].fp_strict, entries[1].fp_strict)

    def test_entry_ids_are_unique_per_sheet(self):
        path = self.data / "book.xlsx"
        with pd.ExcelWriter(path) as w:
            pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="A", index=False)
            pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="B", index=False)
        entries = inv.scan_file(inv.list_folder(str(self.data))[0])
        self.assertEqual(len({e.path for e in entries}), 2)
        for e in entries:
            self.assertEqual(e.source_path, str(path))


@unittest.skipUnless(HAS_PYARROW, "needs pyarrow")
class TestCacheAndRescan(_TmpStateCase):
    def _mk(self):
        _write_parquet(self.data / "a.parquet", {"id": pa.array(range(10))})
        return inv.list_folder(str(self.data))

    def test_unchanged_file_is_not_rescanned(self):
        recs = self._mk()
        _, cache = inv.scan_files(recs)
        calls = []
        orig = inv.scan_file
        inv.scan_file = lambda rec: (calls.append(rec["path"]) or orig(rec))
        try:
            inv.scan_files(recs, cache=cache)
        finally:
            inv.scan_file = orig
        self.assertEqual(calls, [], "cached file should not be re-read")

    def test_changed_file_is_rescanned(self):
        recs = self._mk()
        _, cache = inv.scan_files(recs)
        _write_parquet(self.data / "a.parquet", {"id": pa.array(range(99))})
        recs2 = inv.list_folder(str(self.data))
        entries, _ = inv.scan_files(recs2, cache=cache)
        self.assertEqual(entries[0].row_count, 99)

    def test_regression_cache_holds_a_list_per_file(self):
        """
        Regression: the cache changed from one dict per file to a LIST
        (one file can yield several entries — Excel sheets). Consumers
        that still did FileEntry(**cached) crashed with
        'argument after ** must be a mapping, not list'.
        """
        recs = self._mk()
        _, cache = inv.scan_files(recs)
        for value in cache.values():
            self.assertIsInstance(value, list)
        state = dict(inv.DEFAULT_STATE, folders=[str(self.data)])
        inv.save_cache(cache)
        entries = inv.entries_from_cache(state)   # must not raise
        self.assertEqual(len(entries), 1)

    def test_legacy_dict_shaped_cache_still_loads(self):
        """A stale cache is a performance problem, not a crash."""
        recs = self._mk()
        _, cache = inv.scan_files(recs)
        legacy = {k: v[0] for k, v in cache.items()}
        inv.save_cache(legacy)
        state = dict(inv.DEFAULT_STATE, folders=[str(self.data)])
        self.assertEqual(len(inv.entries_from_cache(state)), 1)

    def test_garbage_cache_does_not_crash(self):
        recs = self._mk()
        inv.save_cache({recs[0]["path"]: "junk"})
        state = dict(inv.DEFAULT_STATE, folders=[str(self.data)])
        self.assertEqual(inv.entries_from_cache(state), [])


@unittest.skipUnless(HAS_PYARROW, "needs pyarrow")
class TestSchemaRegistryAndGrouping(_TmpStateCase):
    def _build(self):
        # 3 files, 2 schemas: two identical, one with an extra column
        for i in range(2):
            _write_parquet(self.data / f"same{i}.parquet",
                           {"a": pa.array([1, 2]), "b": pa.array(["x", "y"])})
        _write_parquet(self.data / "other.parquet",
                       {"a": pa.array([1]), "b": pa.array(["x"]), "c": pa.array([1.0])})
        entries, cache = inv.scan_files(inv.list_folder(str(self.data)))
        inv.save_cache(cache)
        schemas, new = inv.register_schemas(entries, {})
        state = dict(inv.DEFAULT_STATE, folders=[str(self.data)],
                     excluded=[], merges={}, file_overrides={})
        return entries, schemas, state

    def test_identical_files_share_one_schema(self):
        entries, schemas, state = self._build()
        self.assertEqual(len(schemas), 2)
        groups = [d for d in inv.build_datasets(entries, schemas, state) if d["n_files"]]
        by_files = sorted(d["n_files"] for d in groups)
        self.assertEqual(by_files, [1, 2])

    def test_row_and_byte_totals_are_summed_per_group(self):
        entries, schemas, state = self._build()
        groups = {d["group_id"]: d for d in inv.build_datasets(entries, schemas, state)}
        two = [d for d in groups.values() if d["n_files"] == 2][0]
        self.assertEqual(two["row_count"], 4)
        self.assertGreater(two["total_bytes"], 0)

    def test_rescan_preserves_existing_labels(self):
        entries, schemas, state = self._build()
        sid = next(iter(schemas))
        schemas[sid].label = "my_label"
        schemas, new = inv.register_schemas(entries, schemas)
        self.assertEqual(new, [], "no new schemas on a repeat scan")
        self.assertEqual(schemas[sid].label, "my_label")

    def test_schema_survives_excluding_all_its_files(self):
        entries, schemas, state = self._build()
        target = [e for e in entries if e.name == "other.parquet"][0]
        sid = inv.schema_id_for_entry(target, schemas, state)
        state["excluded"] = [target.path]
        ds = inv.build_datasets(entries, schemas, state)
        kept = [d for d in ds if d["group_id"] == sid]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["n_files"], 0)
        self.assertIn("schema retained, 0 files currently included", kept[0]["flags"])

    def test_zero_file_group_gets_no_dataset_number(self):
        """A numbered row with no files reads as a produced dataset. It isn't."""
        entries, schemas, state = self._build()
        target = [e for e in entries if e.name == "other.parquet"][0]
        state["excluded"] = [target.path]
        ds = inv.build_datasets(entries, schemas, state)
        for d in ds:
            if d["n_files"] == 0:
                self.assertIsNone(d["dataset_no"])
            else:
                self.assertIsNotNone(d["dataset_no"])

    def test_merge_folds_groups_and_records_the_diff(self):
        entries, schemas, state = self._build()
        ids = list(schemas)
        state["merges"] = {ids[1]: {"into": ids[0], "note": "test", "at": "now"}}
        ds = [d for d in inv.build_datasets(entries, schemas, state) if d["n_files"]]
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["n_files"], 3)
        self.assertIn(ids[1], ds[0]["merged_from"])
        self.assertTrue(ds[0]["merge_diffs"], "a non-identical merge must record its diff")
        self.assertTrue(any("NOT identical" in f for f in ds[0]["flags"]))

    def test_file_override_reports_under_another_schema_but_keeps_truth(self):
        entries, schemas, state = self._build()
        ids = list(schemas)
        target = [e for e in entries if e.name == "other.parquet"][0]
        real_sid = inv.schema_id_for_entry(target, schemas, state)
        other_sid = [i for i in ids if i != real_sid][0]
        state["file_overrides"] = {target.path: {"schema_id": other_sid,
                                                "note": "belongs here", "at": "now"}}
        ds = {d["group_id"]: d for d in inv.build_datasets(entries, schemas, state)}
        self.assertEqual(ds[other_sid]["n_files"], 3)
        ov = ds[other_sid]["overridden_files"]
        self.assertEqual(len(ov), 1)
        self.assertEqual(ov[0]["actual_fingerprint"], target.fp_strict,
                         "the file's real fingerprint must stay visible")

    def test_merge_cycle_does_not_hang(self):
        entries, schemas, state = self._build()
        a, b = list(schemas)[:2]
        state["merges"] = {a: {"into": b}, b: {"into": a}}
        inv.build_datasets(entries, schemas, state)   # must terminate

    def test_count_files_by_schema_matches_per_schema_scan(self):
        entries, schemas, state = self._build()
        fast = inv.count_files_by_schema(entries, schemas, state)
        slow = {sid: sum(1 for e in entries if e.ok
                         and inv.schema_id_for_entry(e, schemas, state) == sid)
                for sid in schemas}
        self.assertEqual(fast, slow)


class TestSuggestMerges(unittest.TestCase):
    def _schema(self, sid, cols):
        fs, fn, fset = inv.fingerprints(cols)
        return inv.SchemaRecord(schema_id=sid, fp_strict=fs, fp_names=fn,
                                fp_nameset=fset, n_fields=len(cols), columns=cols)

    def test_type_only_difference_is_suggested(self):
        a = self._schema("S01", [{"name": "id", "type": "int64"}])
        b = self._schema("S02", [{"name": "id", "type": "int32"}])
        out = inv.suggest_merges({"S01": a, "S02": b}, ["S01", "S02"],
                                 dict(inv.DEFAULT_STATE))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["diff"]["kind"], "types_only")

    def test_different_columns_not_suggested(self):
        a = self._schema("S01", [{"name": "id", "type": "int64"}])
        b = self._schema("S02", [{"name": "other", "type": "int64"}])
        out = inv.suggest_merges({"S01": a, "S02": b}, ["S01", "S02"],
                                 dict(inv.DEFAULT_STATE))
        self.assertEqual(out, [])

    def test_order_only_difference_is_suggested(self):
        cols = [{"name": "a", "type": "int64"}, {"name": "b", "type": "int64"}]
        a = self._schema("S01", cols)
        b = self._schema("S02", list(reversed(cols)))
        out = inv.suggest_merges({"S01": a, "S02": b}, ["S01", "S02"],
                                 dict(inv.DEFAULT_STATE))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["diff"]["kind"], "order_only")

    def test_already_merged_schema_is_not_suggested_again(self):
        a = self._schema("S01", [{"name": "id", "type": "int64"}])
        b = self._schema("S02", [{"name": "id", "type": "int32"}])
        state = dict(inv.DEFAULT_STATE, merges={"S02": {"into": "S01"}})
        self.assertEqual(inv.suggest_merges({"S01": a, "S02": b}, ["S01", "S02"], state), [])


@unittest.skipUnless(HAS_PYARROW, "needs pyarrow")
class TestJsonPayload(_TmpStateCase):
    def test_payload_is_json_serialisable_and_carries_provenance(self):
        _write_parquet(self.data / "a.parquet", {"a": pa.array([1, 2, 3])})
        entries, cache = inv.scan_files(inv.list_folder(str(self.data)))
        inv.save_cache(cache)
        schemas, _ = inv.register_schemas(entries, {})
        state = dict(inv.DEFAULT_STATE, folders=[str(self.data)],
                     excluded=[], merges={}, file_overrides={})
        ds = inv.build_datasets(entries, schemas, state)
        payload = inv.to_json_payload(ds, entries, schemas, state)
        text = json.dumps(payload)          # must not raise on sets etc.
        self.assertIn("row_count_method", text)
        self.assertEqual(payload["totals"]["rows"], 3)
        self.assertEqual(payload["datasets"][0]["row_count_method"], "footer")

    def test_retained_schema_is_not_labelled_as_a_dataset(self):
        _write_parquet(self.data / "a.parquet", {"a": pa.array([1])})
        entries, cache = inv.scan_files(inv.list_folder(str(self.data)))
        inv.save_cache(cache)
        schemas, _ = inv.register_schemas(entries, {})
        state = dict(inv.DEFAULT_STATE, folders=[str(self.data)],
                     excluded=[entries[0].path], merges={}, file_overrides={})
        ds = inv.build_datasets(entries, schemas, state)
        payload = inv.to_json_payload(ds, entries, schemas, state)
        self.assertEqual(payload["totals"]["datasets"], 0)
        self.assertEqual(payload["datasets"][0]["dataset"], "(retained schema, 0 files)")


class TestFolderListing(_TmpStateCase):
    def test_missing_folder_raises_clearly(self):
        with self.assertRaises(FileNotFoundError):
            inv.list_folder(str(self.data / "nope"))

    def test_non_recursive(self):
        (self.data / "sub").mkdir()
        pd.DataFrame({"a": [1]}).to_csv(self.data / "top.csv", index=False)
        pd.DataFrame({"a": [1]}).to_csv(self.data / "sub" / "nested.csv", index=False)
        names = [r["name"] for r in inv.list_folder(str(self.data))]
        self.assertEqual(names, ["top.csv"])

    def test_unrecognised_extensions_ignored(self):
        (self.data / "notes.docx").write_text("x")
        pd.DataFrame({"a": [1]}).to_csv(self.data / "a.csv", index=False)
        names = [r["name"] for r in inv.list_folder(str(self.data))]
        self.assertEqual(names, ["a.csv"])


class TestPicker(unittest.TestCase):
    """
    The folder dialog runs in a subprocess on the machine hosting Streamlit.
    Every failure mode below is stubbed rather than staged, because the real
    conditions (no display, user cancels, dialog crashes, dialog left open)
    can't all be produced on any one machine — an environment-dependent skip
    would leave the error paths untested on exactly the machine that runs
    the app.
    """

    def setUp(self):
        self._orig_available = inv.picker_available

    def tearDown(self):
        inv.picker_available = self._orig_available

    def test_reports_availability_without_guessing(self):
        ok, why = inv.picker_available()
        self.assertIsInstance(ok, bool)
        if not ok:
            self.assertTrue(why, "an unavailable picker must explain itself")

    def test_unavailable_returns_error_instead_of_raising(self):
        inv.picker_available = lambda: (False, "no display detected")
        path, err = inv.pick_directory()
        self.assertEqual(path, "")
        self.assertIn("no display", err)

    def test_cancel_returns_empty_path_and_no_error(self):
        """
        Cancelling is not a failure. An error string here would show the user
        a warning for having simply changed their mind.
        """
        inv.picker_available = lambda: (True, "")
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        orig = subprocess.run
        subprocess.run = lambda *a, **k: completed
        try:
            path, err = inv.pick_directory()
        finally:
            subprocess.run = orig
        self.assertEqual((path, err), ("", ""))

    def test_selected_path_is_returned_stripped(self):
        inv.picker_available = lambda: (True, "")
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="/Users/x/prod\n", stderr="")
        orig = subprocess.run
        subprocess.run = lambda *a, **k: completed
        try:
            path, err = inv.pick_directory()
        finally:
            subprocess.run = orig
        self.assertEqual(path, "/Users/x/prod")
        self.assertEqual(err, "")

    def test_dialog_crash_surfaces_stderr(self):
        inv.picker_available = lambda: (True, "")
        completed = subprocess.CompletedProcess(
            args=[], returncode=2, stdout="", stderr="tkinter unavailable: no module")
        orig = subprocess.run
        subprocess.run = lambda *a, **k: completed
        try:
            path, err = inv.pick_directory()
        finally:
            subprocess.run = orig
        self.assertEqual(path, "")
        self.assertIn("tkinter", err)

    def test_dialog_left_open_times_out_without_hanging_streamlit(self):
        inv.picker_available = lambda: (True, "")
        orig = subprocess.run

        def _timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="picker", timeout=1)

        subprocess.run = _timeout
        try:
            path, err = inv.pick_directory(timeout=1)
        finally:
            subprocess.run = orig
        self.assertEqual(path, "")
        self.assertIn("cancelled", err.lower())

    def test_subprocess_launch_failure_is_caught(self):
        """Nothing the dialog does may propagate into the Streamlit thread."""
        inv.picker_available = lambda: (True, "")
        orig = subprocess.run

        def _explode(*a, **k):
            raise OSError("exec format error")

        subprocess.run = _explode
        try:
            path, err = inv.pick_directory()
        finally:
            subprocess.run = orig
        self.assertEqual(path, "")
        self.assertIn("OSError", err)


if __name__ == "__main__":
    unittest.main()