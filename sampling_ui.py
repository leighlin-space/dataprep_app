"""
UI for the Phase 2 Sampling tab.

Reads the inventory produced by the Inventory tab, then draws one small
random sample per schema group and lands it in the warehouse. After this
tab, every downstream tab (Clean, Schema board, Query Builder, Export)
is working on a few dozen 1000-row tables — the whole point.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from modules import db, inventory as inv, sampling as smp

MANIFEST_DIR = inv.STATE_DIR / "samples"


def _fmt_bytes(n) -> str:
    if not n:
        return "0 B"
    for unit, div in (("TB", 1024 ** 4), ("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:,.2f} {unit}"
    return f"{n:,} B"


def render(con) -> None:
    st.subheader("Phase 2 Sampling — 1000 random rows per schema")
    st.caption(
        "One sample per schema group, landed in the warehouse as its own table. "
        "Nothing else is ever loaded: the warehouse holds a few dozen 1000-row tables, "
        "not the production."
    )

    state = inv.load_state()
    schemas = inv.load_schemas()
    entries = inv.entries_from_cache(state)

    if not entries:
        st.info("Nothing scanned yet — run the Inventory tab first.")
        return

    datasets = [d for d in inv.build_datasets(entries, schemas, state) if d["n_files"] > 0]
    if not datasets:
        st.info("No dataset groups with files. Check the Inventory tab.")
        return

    # -----------------------------------------------------------------
    # Global controls
    # -----------------------------------------------------------------
    st.markdown("### Sampling settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_rows = st.number_input("Rows per sample", 100, 10000, smp.SAMPLE_ROWS, step=100,
                                 key="smp_n")
    with c2:
        strategy = st.selectbox(
            "Strategy",
            ["auto (exact when small)", "exact — full scan, true SRS",
             "two_stage — metadata-driven"],
            key="smp_strategy",
        )
    with c3:
        seed = st.number_input("Seed", 0, 2 ** 31 - 1, 20260812, key="smp_seed")

    force = None
    if strategy.startswith("exact"):
        force = "exact"
    elif strategy.startswith("two_stage"):
        force = "two_stage"

    rows_per_group = st.slider(
        "two_stage: rows drawn per selected row group", 1, 250,
        smp.DEFAULT_ROWS_PER_GROUP, key="smp_rpg",
        help="Lower = more row groups touched = better spread across the file, more bytes "
             "read. Only affects the two_stage strategy.",
    )

    with st.expander("ℹ️ What 'random' means for each strategy — read once"):
        st.markdown(
            "**Both strategies are unbiased**: every row in the group has probability "
            "exactly `rows / population` of being selected. They differ in *spread*, "
            "because a row group is the smallest unit parquet allows reading.\n\n"
            "- **exact** — DuckDB reservoir sample over the whole group. A textbook "
            "simple random sample. Reads every byte, so it is only practical below a "
            "few GB.\n"
            "- **two_stage** — row groups are chosen with probability proportional to "
            "their row count, then rows are drawn uniformly inside each. Reads only the "
            "selected row groups. The sample is still unbiased, but it comes from a "
            "limited number of physical locations in the file. **If the data is sorted "
            "(by date, claim number, source system), a clustered sample can miss whole "
            "ranges of that column** — the mean stays unbiased, its variance rises, and "
            "any claim about the sample's coverage of the sort column may be false.\n\n"
            "On a deliberately sorted 300k-row test set, sample means had no measurable "
            "bias under either strategy; the spread of those means was ~4.6× wider at 25 "
            "rows/group and ~15× wider at 250 rows/group than a true simple random "
            "sample. Every draw records its seed and its exact row selections."
        )

    # -----------------------------------------------------------------
    # Per-group plans
    # -----------------------------------------------------------------
    st.markdown("### Dataset groups")
    existing_tables = set(db.list_tables(con))

    plans = {}
    rows = []
    for d in datasets:
        gid = d["group_id"]
        grp_entries = [e for e in entries if e.path in set(d["files"])]
        plan = smp.plan_sample(grp_entries, n=int(n_rows), seed=int(seed),
                               rows_per_group=int(rows_per_group), force_strategy=force)
        plans[gid] = (plan, d, grp_entries)
        tname = smp.sample_table_name(gid, d["label"] if d["label"] != gid else "")
        rows.append({
            "Dataset": f"Dataset{d['dataset_no']}",
            "Schema": gid,
            "Label": d["label"],
            "Files": d["n_files"],
            "Population rows": ("unknown" if d["row_count"] is None
                                else f"{d['row_count']:,}"),
            "Strategy": plan.strategy,
            "Row groups touched": f"{plan.n_row_groups_touched} / {plan.n_row_groups_total}",
            "Est. read": _fmt_bytes(plan.est_bytes_read),
            "Sample table": tname + (" ✅" if tname in existing_tables else ""),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    unknown_rows = [d for _, d, _ in plans.values() if d["row_count"] is None]
    if unknown_rows:
        st.warning(
            f"{len(unknown_rows)} group(s) contain files with no row count (CSV/JSON have no "
            f"footer). Those groups can't be sampled by the metadata route — they need the "
            f"exact strategy, which reads them in full."
        )

    # -----------------------------------------------------------------
    # Draw
    # -----------------------------------------------------------------
    st.markdown("### Draw samples")
    b1, b2 = st.columns([1, 3])
    with b1:
        do_all = st.button("🎲 Sample all groups", key="smp_all", use_container_width=True)
    with b2:
        st.caption("Each sample overwrites its own table. The manifest — seed, strategy, and "
                   "the exact (file, row group, row offset) list — is written next to it so "
                   "the draw can be reproduced or audited.")

    def _draw(gid: str) -> None:
        plan, d, grp_entries = plans[gid]
        label = d["label"] if d["label"] != gid else ""
        tname = smp.sample_table_name(gid, label)
        bar = st.progress(0.0, text=f"{gid}: reading...")

        def _prog(i, total, lbl):
            bar.progress(i / max(total, 1), text=f"{gid}: {i}/{total} — {lbl}")

        try:
            df = smp.execute_plan(plan, progress=_prog)
            bar.empty()
            if df.empty:
                st.error(f"{gid}: sample came back empty.")
                return
            db.register_table(con, tname, df)
            m = smp.manifest(plan, gid, label=d["label"], n_returned=len(df))
            smp.save_manifest(m, MANIFEST_DIR)
            st.success(f"{gid} → table `{tname}` — {len(df):,} rows × {df.shape[1]} columns "
                       f"({plan.strategy}, seed {plan.seed})")
        except Exception as e:
            bar.empty()
            st.error(f"{gid}: {type(e).__name__}: {e}")

    if do_all:
        for gid in plans:
            _draw(gid)
        st.rerun()

    for gid, (plan, d, _) in plans.items():
        with st.expander(f"Dataset{d['dataset_no']} · {gid} · {d['label']}"):
            st.caption(plan.note)
            i1, i2, i3 = st.columns(3)
            i1.metric("Population", "unknown" if d["row_count"] is None
                      else f"{d['row_count']:,}")
            i2.metric("Selection p", "—" if not plan.n_total_rows
                      else f"{plan.n_target / plan.n_total_rows:.2e}")
            i3.metric("Est. read", _fmt_bytes(plan.est_bytes_read))
            if st.button(f"🎲 Sample {gid}", key=f"smp_draw_{gid}"):
                _draw(gid)

            mpath = MANIFEST_DIR / f"sample_{gid}.json"
            if mpath.exists():
                st.download_button("⬇️ manifest.json", data=mpath.read_text(),
                                   file_name=mpath.name, mime="application/json",
                                   key=f"smp_man_{gid}")

    # -----------------------------------------------------------------
    # Existing samples
    # -----------------------------------------------------------------
    sample_tables = sorted(t for t in db.list_tables(con) if t.endswith("_sample"))
    if sample_tables:
        st.markdown("### Samples in the warehouse")
        st.dataframe(pd.DataFrame([{
            "Table": t,
            "Rows": db.count_rows(con, t),
            "Columns": db.count_columns(con, t),
        } for t in sample_tables]), use_container_width=True, hide_index=True)
        st.caption("These are what the Clean, Schema, Query Builder and Export tabs operate on.")
