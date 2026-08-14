# Test Suite

186 tests across every module, including the Inventory and Sampling phases.
`unittest` only — no install beyond `requirements.txt`.

## Run everything

```powershell
python tests/run_tests.py
```

## What's covered

| File | Tests | Needs |
|---|---|---|
| `test_clean.py` | Auto-cleaning, profiling, duplicate detection | — |
| `test_export.py` | Excel export, incl. the NaN/nullable-dtype crash | — |
| `test_query.py` | SQL generation (joins, filters, `IN`, aggregation) | — |
| `test_profiles.py` | Dataset Profile save/load/apply | — |
| `test_filters.py` | Filter Profile save/load, seeded defaults | — |
| `test_db.py` | Table naming + connection-based operations | duckdb |
| `test_schema.py` | PK / relationship detection (pandas + SQL) | duckdb |
| `test_inventory.py` | Fingerprints, all-format scanning, schema registry, merges, overrides, cache | pyarrow, duckdb, openpyxl |
| `test_sampling.py` | Plan strategies, statistical unbiasedness, reproducibility, mixed-format groups | pyarrow, duckdb |
| `test_scale.py` | **500-file / 500-schema complexity guarantees** | pyarrow |

Tests needing a library that isn't importable skip cleanly rather than fail.
With everything in `requirements.txt` installed, the suite reports **0 skipped**.

The folder-dialog tests are stubbed, not staged: no display, user cancels,
dialog crashes, and dialog left open can't all be produced on one machine, and
an environment-dependent skip would leave those error paths untested on exactly
the machine that runs the app.

## The scale tests — what they actually prove

`test_scale.py` exists because of one specific question: does this hold up at
500 datasets? It does not measure throughput — nobody cares whether 500 tiny
files scan in 0.4s or 4s on a given box. It measures **complexity**, because
Streamlit re-runs the entire script on every click, so anything accidentally
quadratic becomes multi-second lag on every interaction.

What's pinned:

- 500 files, one schema → 1 dataset; 500 distinct schemas → 500 datasets, with
  contiguous numbering.
- `build_datasets` stays linear: doubling the input must not triple the time.
- Scanning 500 files **never touches a data page** — `read_row_group` and
  `read` are monkeypatched to raise. This is the guarantee that keeps a 1.02 TB
  production a seconds-long scan rather than an hours-long one.
- A second scan of 500 files reads **zero** footers (cache by size + mtime).
- Sampling reads only the row groups the plan selected — counted, not timed.
- Plan cost is independent of population size: 100k rows/file and 100M
  rows/file produce the same `est_bytes_read`.
- Warehouse ceiling: 500 datasets × 1000 rows = 500,000 rows, whatever the
  production holds.

## Regression tests worth knowing about

Each of these is a bug that actually happened, not a hypothetical:

- `test_export.py::test_regression_nullable_dtype_with_nan` — Excel export
  crash from nullable pandas dtypes with missing values.
- `test_db.py::test_does_not_truncate_at_dot` — `"sales.q1_2026"` silently
  became `"sales"`.
- `test_db.py::test_count_rows_never_materializes_full_table` — guards against
  reintroducing the full-table pandas pull.
- `test_schema.py::test_regression_cross_type_column_comparison` — VARCHAR vs
  BIGINT comparison raised BinderException.
- `test_inventory.py::test_regression_cache_holds_a_list_per_file` — the cache
  changed from one dict per file to a list (Excel workbooks yield one entry per
  sheet); consumers still doing `FileEntry(**cached)` crashed with *argument
  after ** must be a mapping, not list*.
- `test_inventory.py::test_regression_quoted_newlines_not_miscounted` — counting
  newlines to get a CSV row count is faster and wrong. RFC-4180 permits newlines
  inside quoted fields; on the test file `wc -l` reports double the true count.
- `test_scale.py::test_regression_counting_is_one_pass_not_per_schema` — the
  schema library counted files by re-scanning every entry for every schema:
  O(schemas² × files), 250,000 lookups and 3.0s per rerun at 500/500.
- `test_scale.py::test_regression_pathological_nameset_is_capped_and_fast` — 500
  schemas sharing one column-name set produced 124,750 candidate pairs in 8.4s,
  and the UI would have rendered one expander per pair.

Each of the last two was confirmed to **fail** against the pre-fix code before
the fix went in. A guard that has never failed isn't a guard.

## Adding a test when you find a new bug

Same pattern every time: write a test that reproduces the bug and fails, fix the
code, confirm it passes, keep the test.