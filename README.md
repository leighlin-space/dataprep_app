# Internal Data Prep Tool — Phase 1 Prototype

A local, no-external-dependency tool to load raw data, auto-suggest
cleaning steps, land it in a local DuckDB warehouse, and see inferred
relationships between tables.

## Setup

```bash
cd dataprep_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

This opens a local browser tab. Nothing leaves your machine — the
warehouse is a single file, `warehouse.duckdb`, created next to `app.py`.

## What's implemented (Phase 1 + early Phase 2)

- **Tab 1 — Load & Clean**: upload CSV/Excel/JSON/Parquet, see an
  auto-generated column profile, review/accept cleaning suggestions
  (empty columns, high-null columns, exact duplicates, whitespace),
  then load the cleaned result into the DuckDB warehouse as a table.
- **Tab 2 — Schema**: infers primary-key candidates per table and
  foreign-key relationships across tables via value-overlap heuristics,
  and renders a Mermaid ER diagram description.
- **Tab 3 — Warehouse Tables**: browse/preview/drop tables currently
  in the warehouse.

## Known limitations (by design, for a Phase 1 prototype)

- Relationship detection is O(n²) in column count — fine for tens of
  tables, would need indexing/sampling for hundreds.
- No SQL query builder yet (Phase 3).
- No Excel exhibit export yet (Phase 5).
- Mermaid diagram is shown as code, not rendered inline — add
  `streamlit-mermaid` to requirements.txt to render it directly in the
  browser instead of copy-pasting into mermaid.live.

## Next steps (in order)

1. **Query builder tab**: pick tables → join keys (pre-filled from
   Tab 2, editable) → filters/group-by/aggregations → live preview.
   Add a "Show SQL" toggle for the mixed no-code/SQL audience.
2. **Pipeline save/replay**: serialize a built query as JSON so it can
   be re-run on refreshed source files with one click.
3. **Excel export**: `xlsxwriter`-based exhibit templates (formatted
   headers, number formats, optional charts) generated from the final
   dataframe.
4. **Packaging**: Dockerize for shared internal hosting once the
   single-user workflow is validated.
