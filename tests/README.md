# Test Suite

94 tests covering every module. Uses `unittest` (standard library) — no extra install needed beyond what's already in `requirements.txt`.

## Run everything

```powershell
python tests/run_tests.py
```

or, equivalently:

```powershell
python -m unittest discover -s tests -v
```

## What's covered

| File | Tests | Needs DuckDB? |
|---|---|---|
| `test_clean.py` | Auto-cleaning, profiling, duplicate detection | No |
| `test_export.py` | Excel export, incl. regression test for the NaN/nullable-dtype crash | No |
| `test_query.py` | SQL generation (joins, filters, `IN`, aggregation) | No (query building is pure string logic) |
| `test_profiles.py` | Dataset Profile save/load/apply | No |
| `test_filters.py` | Filter Profile save/load, seeded defaults | No |
| `test_db.py` | Table naming (pure logic) + connection-based operations | Partial — connection tests skip without duckdb, run automatically once it's installed |
| `test_schema.py` | Primary key / relationship detection (pandas + SQL versions) | Partial — same as above |

Tests that need a live DuckDB connection are written to **skip cleanly** rather than fail if duckdb isn't importable, so `run_tests.py` always gives a clean pass/fail signal. On your machine, since duckdb is already installed, every test runs for real — nothing should show as "skipped."

## Regression tests worth knowing about

These exist because they're bugs that actually happened, not hypothetical edge cases:

- `test_export.py::test_regression_nullable_dtype_with_nan` — the Excel export crash from nullable pandas dtypes with missing values.
- `test_db.py::test_does_not_truncate_at_dot` — a table name like `"sales.q1_2026"` used to silently become `"sales"` (fixed via `_safe_table_name`).
- `test_db.py::test_count_rows_never_materializes_full_table` — guards against reintroducing the full-table pandas pull that used to crash on large tables in Tab 3.

## Adding a test when you find a new bug

Same pattern every time: write a test that reproduces the bug and fails, fix the code, confirm the test now passes, keep the test. That way a fixed bug can't silently come back in a later change.
