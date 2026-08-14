"""
Phase 2 Sampling — pull a small random sample per schema group.

Design constraint that shapes everything here: **a row group is the
smallest unit parquet lets you read.** There is no API, in any library,
that reads row 3,182,449,001 without decompressing the row group that
contains it. So "1000 random rows" out of 4.95B has exactly two honest
implementations, and they are not interchangeable:

  STRATEGY "exact"  — DuckDB reservoir sampling over the whole dataset.
      Every row has probability n/N, and the 1000 rows are a simple
      random sample in the textbook sense. Cost: reads every byte.
      1.02 TB over a network drive is hours. Correct choice for small
      and medium groups.

  STRATEGY "two_stage" — pick row groups with probability proportional
      to their row count, then draw rows uniformly inside each selected
      group. Cost: reads only the selected row groups, so seconds to
      minutes. Every row still has probability exactly n/N — this is
      PPS-with-replacement sampling and it is unbiased, not an
      approximation.

      What it is NOT is a *spread-out* sample. The 1000 rows come from
      `n_slots` physical locations. If the production is sorted (by
      date, by source system, by claim number — and productions usually
      are), a sample drawn from 20 row groups can miss whole ranges of
      the sort column entirely. The estimate of a mean stays unbiased;
      its variance rises, and any statement of the form "the sample
      covers 2019-2024" can be plain false. `rows_per_group` is the
      dial: lower it to touch more row groups (better spread, more
      bytes read), raise it to read less.

Both strategies take a seed and emit a manifest recording every
(file, row_group, row_offset) selected, so a sample can be reproduced
and defended months later. That matters more here than usual: a figure
derived from a sample of a produced dataset should be traceable to the
exact rows it came from.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

SAMPLE_ROWS = 1000            # per schema group — the whole app stays tiny
DEFAULT_ROWS_PER_GROUP = 25   # 1000/25 = 40 row groups touched by default
# Below this total size, just do the exact full-scan sample — it's fast
# enough that there's no reason to accept clustering.
EXACT_SCAN_MAX_BYTES = 2 * 1024 ** 3


@dataclass
class SamplePlan:
    """What we're about to read, decided before a single data page is touched."""
    strategy: str                     # "exact" | "two_stage" | "all_rows"
    n_target: int
    n_total_rows: int
    seed: int
    rows_per_group: int = 0
    # {(path, rg_index): [row offsets within that row group]}
    picks: list = field(default_factory=list)   # [{"path","row_group","offsets"}]
    est_bytes_read: int = 0
    n_row_groups_touched: int = 0
    n_row_groups_total: int = 0
    files: list = field(default_factory=list)          # display names/paths
    file_specs: list = field(default_factory=list)     # [{"path","sheet"}] to read
    footerless_files: list = field(default_factory=list)
    note: str = ""


def _row_group_index(entries) -> tuple[list, int, int, list]:
    """
    Flatten every row group across every file in the group into one list.
    Returns (row_groups, total_rows, total_bytes, footerless_paths).

    footerless_paths are files with no row-group metadata — CSV and JSON.
    They cannot be sampled by the metadata route at all (no footer means
    no row count and no row-group boundaries), so their presence forces
    a full read. Reporting them separately rather than silently skipping
    them is the difference between "we sampled the group" and "we sampled
    the parquet part of the group and said nothing".
    """
    rgs = []
    total_rows = 0
    total_bytes = 0
    footerless = []
    for e in entries:
        if not e.ok:
            continue
        if not e.row_groups:
            footerless.append(e.path)
            continue
        for i, rg in enumerate(e.row_groups):
            rgs.append({"path": e.path, "rg": i,
                        "rows": rg["rows"], "bytes": rg.get("bytes", 0)})
            total_rows += rg["rows"]
            total_bytes += rg.get("bytes", 0)
    return rgs, total_rows, total_bytes, footerless


def plan_sample(entries, n: int = SAMPLE_ROWS, seed: int | None = None,
                rows_per_group: int = DEFAULT_ROWS_PER_GROUP,
                force_strategy: str | None = None) -> SamplePlan:
    """
    Decide how to draw `n` rows from the files in one schema group.
    Reads nothing — this is planning off the footer metadata already in
    hand, so the UI can show the cost before committing.
    """
    seed = seed if seed is not None else random.randrange(2 ** 31)
    rgs, total_rows, total_bytes, footerless = _row_group_index(entries)
    file_paths = sorted({e.path for e in entries if e.ok})
    # An Excel entry is one SHEET of a workbook, so the reader needs the
    # sheet name alongside the path — the entry id alone isn't enough.
    specs = []
    seen = set()
    for e in entries:
        if not e.ok:
            continue
        key = (e.source_path or e.path, e.sheet)
        if key in seen:
            continue
        seen.add(key)
        specs.append({"path": e.source_path or e.path, "sheet": e.sheet})

    plan = SamplePlan(
        strategy="", n_target=n, n_total_rows=total_rows, seed=seed,
        n_row_groups_total=len(rgs), files=file_paths, file_specs=specs,
    )

    plan.footerless_files = footerless

    if footerless:
        # A CSV/JSON in the group means no footer, so no row count and no
        # row-group boundaries — the metadata route is simply unavailable.
        # Fall back to the full read rather than sampling only the parquet
        # files and calling the result a sample of the group.
        plan.strategy = "exact"
        plan.est_bytes_read = total_bytes
        plan.n_row_groups_touched = len(rgs)
        plan.note = (
            f"{len(footerless)} file(s)/sheet(s) in this group are not parquet, so they have "
            f"no row groups and the metadata route doesn't apply to them. Reading the group "
            f"in full and reservoir-sampling {n:,} rows — still a true simple random sample, "
            f"just at the cost of a full read. (The row COUNT is already known exactly from "
            f"the inventory scan; what's missing here is the row-group boundaries sampling "
            f"needs.)"
        )
        return plan

    if total_rows == 0:
        plan.strategy = "all_rows"
        plan.note = "No readable rows in this group."
        return plan

    if total_rows <= n:
        plan.strategy = "all_rows"
        plan.est_bytes_read = total_bytes
        plan.n_row_groups_touched = len(rgs)
        plan.note = (f"Group holds {total_rows:,} rows, at or below the {n:,}-row target — "
                     f"taking all of them. No sampling, so nothing to be random about.")
        return plan

    use_exact = (force_strategy == "exact") or (
        force_strategy is None and total_bytes <= EXACT_SCAN_MAX_BYTES)

    if use_exact:
        plan.strategy = "exact"
        plan.est_bytes_read = total_bytes
        plan.n_row_groups_touched = len(rgs)
        plan.note = ("Full-scan reservoir sample: a true simple random sample, "
                     "every row read once.")
        return plan

    # --- two-stage PPS ---
    rows_per_group = max(1, min(rows_per_group, n))
    n_slots = math.ceil(n / rows_per_group)

    rng = random.Random(seed)
    weights = [rg["rows"] for rg in rgs]
    # PPS *with replacement*: a row group may be drawn more than once, in
    # which case it just contributes more rows. Keeping replacement is what
    # makes the per-row probability exactly n/N with no finite-population
    # correction to argue about.
    chosen = rng.choices(range(len(rgs)), weights=weights, k=n_slots)

    draws: dict[int, int] = {}
    for idx in chosen:
        draws[idx] = draws.get(idx, 0) + rows_per_group

    picks = []
    remaining = n
    for idx, want in draws.items():
        rg = rgs[idx]
        take = min(want, rg["rows"], remaining)
        if take <= 0:
            continue
        offsets = sorted(rng.sample(range(rg["rows"]), take))
        picks.append({"path": rg["path"], "row_group": rg["rg"], "offsets": offsets})
        remaining -= take
        if remaining <= 0:
            break

    plan.strategy = "two_stage"
    plan.rows_per_group = rows_per_group
    plan.picks = picks
    plan.n_row_groups_touched = len(picks)
    plan.est_bytes_read = sum(rgs[i]["bytes"] for i in draws)
    plan.note = (
        f"Every row has probability {n}/{total_rows:,} of selection — unbiased. But the "
        f"rows come from {len(picks)} of {len(rgs):,} row groups, so if the data is sorted "
        f"the sample is clustered on the sort column. Lower rows-per-group to spread it."
    )
    return plan


def _promote_concat(tables):
    """
    Concatenate row-group slices. A schema group may have been manually
    merged across files whose types differ (int32 vs int64 is the usual
    one), so strict concat would fail — permissive promotion is required,
    not optional.
    """
    import pyarrow as pa

    if len(tables) == 1:
        return tables[0]
    try:
        return pa.concat_tables(tables, promote_options="permissive")
    except TypeError:
        return pa.concat_tables(tables, promote=True)   # pyarrow < 14


def execute_plan(plan: SamplePlan, progress=None):
    """
    Run a plan. Returns a pandas DataFrame of at most plan.n_target rows.

    progress: optional callable(i, total, label).
    """
    import pyarrow.parquet as pq

    if plan.strategy in ("exact", "all_rows"):
        # Both read whole files, and a group may legitimately mix parquet
        # with CSV (a manual merge, or a production that shipped both), so
        # these go through DuckDB's per-file readers rather than pyarrow —
        # pointing read_parquet at a .csv is a hard error.
        return _read_full(
            plan.file_specs or [{"path": p, "sheet": ""} for p in plan.files],
            n=(plan.n_target if plan.strategy == "exact" else None),
            seed=plan.seed)

    # two_stage: read only the selected row groups, keep only the selected rows
    by_file: dict[str, list] = {}
    for p in plan.picks:
        by_file.setdefault(p["path"], []).append(p)

    tables = []
    done = 0
    for path, picks in by_file.items():
        pf = pq.ParquetFile(path)
        for p in picks:
            tbl = pf.read_row_group(p["row_group"])
            tables.append(tbl.take(p["offsets"]))
            done += 1
            if progress:
                progress(done, len(plan.picks), f"{Path(path).name} rg{p['row_group']}")

    if not tables:
        import pandas as pd
        return pd.DataFrame()

    out = _promote_concat(tables)
    if out.num_rows > plan.n_target:
        out = out.slice(0, plan.n_target)
    return out.to_pandas()


EXCEL_EXTS = {".xlsx", ".xlsm", ".xls", ".ods"}


def _reader_for(path: str) -> str:
    """DuckDB reader expression for one file, dispatched on extension."""
    esc = path.replace("'", "''")
    ext = Path(path).suffix.lower()
    if ext in (".parquet", ".pq"):
        return f"read_parquet('{esc}')"
    if ext == ".csv":
        return f"read_csv_auto('{esc}', sample_size=-1)"
    if ext in (".tsv", ".txt"):
        return "read_csv_auto('%s', sample_size=-1, delim='\\t')" % esc
    if ext in (".json", ".jsonl"):
        return f"read_json_auto('{esc}')"
    raise ValueError(f"No DuckDB reader for {ext}")


def _read_full(specs: list[dict], n: int | None, seed: int):
    """
    Read whole files/sheets, optionally sampling n rows from the result.

    Two paths, chosen by what the group actually contains:

      * All DuckDB-readable (parquet / CSV / JSON) -> one SQL statement with
        UNION ALL BY NAME and a reservoir sample. DuckDB streams and
        materialises only the n sampled rows, so a large CSV group is read
        without being held in memory. UNION ALL BY NAME rather than a path
        list, because it tolerates the column differences a manually merged
        group can contain, filling absent columns with NULL instead of
        failing outright.

      * Any Excel involved -> pandas, since DuckDB has no native Excel
        reader. Here the frames ARE materialised before sampling. That is
        acceptable only because Excel's own row ceiling (~1.05M per sheet)
        bounds how bad it can get.

    In a mixed group the sample is drawn AFTER both sources are combined —
    pushing it into SQL would sample only the SQL side and then append all
    the Excel rows, which weights the two sources wrongly and quietly
    breaks the "every row has probability n/N" guarantee.

    Throwaway connection either way: drawing a sample never touches the
    warehouse, only storing the result does.
    """
    import pandas as pd

    if not specs:
        return pd.DataFrame()

    excel = [s for s in specs if Path(s["path"]).suffix.lower() in EXCEL_EXTS]
    other = [s for s in specs if Path(s["path"]).suffix.lower() not in EXCEL_EXTS]

    frames = []

    if other:
        import duckdb

        parts = [f"SELECT * FROM {_reader_for(s['path'])}" for s in other]
        union = " UNION ALL BY NAME ".join(parts) if len(parts) > 1 else parts[0]
        sql = f"SELECT * FROM ({union})"
        if n is not None and not excel:
            sql += f" USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})"
        con = duckdb.connect()
        try:
            frames.append(con.execute(sql).df())
        finally:
            con.close()

    for s in excel:
        ext = Path(s["path"]).suffix.lower()
        engine = {".xls": "xlrd", ".ods": "odf"}.get(ext)
        frames.append(pd.read_excel(s["path"], sheet_name=(s["sheet"] or 0),
                                    engine=engine))

    if not frames:
        return pd.DataFrame()

    out = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    if n is not None and (excel or len(frames) > 1) and len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out


def manifest(plan: SamplePlan, group_id: str, label: str = "",
             n_returned: int | None = None) -> dict:
    """
    The reproducibility record. With the seed and the pick list, this
    sample can be regenerated byte-for-byte, or audited row by row.
    """
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "schema_group": group_id,
        "label": label,
        "strategy": plan.strategy,
        "seed": plan.seed,
        "rows_requested": plan.n_target,
        "rows_returned": n_returned,
        "population_rows": plan.n_total_rows,
        "selection_probability": (None if not plan.n_total_rows
                                  else plan.n_target / plan.n_total_rows),
        "row_groups_touched": plan.n_row_groups_touched,
        "row_groups_total": plan.n_row_groups_total,
        "rows_per_group": plan.rows_per_group,
        "bytes_read_estimate": plan.est_bytes_read,
        "files": [Path(p).name for p in plan.files],
        "footerless_files": [Path(p).name for p in plan.footerless_files],
        "caveat": plan.note,
        "picks": ([{"file": Path(p["path"]).name, "row_group": p["row_group"],
                    "offsets": p["offsets"]} for p in plan.picks]
                  if plan.strategy == "two_stage" else None),
    }


def sample_table_name(group_id: str, label: str = "") -> str:
    """Warehouse table name for a group's sample — e.g. S01_sample."""
    base = (label or group_id).strip()
    cleaned = "".join(c if c.isalnum() else "_" for c in base).strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return f"{cleaned or group_id}_sample"


def save_manifest(m: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sample_{m['schema_group']}.json"
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    return path
