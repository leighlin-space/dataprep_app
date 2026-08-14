"""
UI for the Phase 1 Inventory tab.

Kept in its own file rather than inlined into app.py — app.py is already
~950 lines and this tab is the largest one yet. app.py needs three
lines: import it, add a tab, call render().
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from modules import inventory as inv


def _fmt_bytes(n: int) -> str:
    if n is None:
        return "—"
    for unit, div in (("TB", 1024 ** 4), ("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:,.2f} {unit}"
    return f"{n:,} B"


def _fmt_rows(n) -> str:
    return "unknown" if n is None else f"{n:,}"


def render() -> None:
    st.subheader("Phase 1 Inventory — scan folders, group by schema")
    st.caption(
        "Metadata only: parquet footers and filesystem stat. No data pages are read, "
        "nothing is loaded into the warehouse. Row counts come from parquet metadata; "
        "CSV/JSON files have no footer, so their row count is reported as unknown "
        "rather than scanned for."
    )

    state = inv.load_state()
    schemas = inv.load_schemas()

    # -----------------------------------------------------------------
    # 1. Sources
    # -----------------------------------------------------------------
    st.markdown("### 1️⃣ Folders")
    state["batch_label"] = st.text_input(
        "Batch / production label (free text, attached to newly found schemas)",
        value=state.get("batch_label", ""), key="inv_batch",
        placeholder="IRM-018",
    )


    # --- Add a folder: native dialog first, manual path as fallback ---
    picker_ok, picker_why = inv.picker_available()

    br_col, add_col, btn_col = st.columns([1.2, 3, 1])

    with br_col:
        st.write("")
        if st.button("📂 Browse...", key="inv_browse", use_container_width=True,
                     disabled=not picker_ok,
                     help=picker_why or "Opens a folder dialog on the machine running this app."):
            # Start the dialog in the last-used folder's parent when we have one
            last = state["folders"][-1] if state["folders"] else ""
            initial = str(Path(last).parent) if last else ""
            with st.spinner("Waiting for the folder dialog... (check for a window "
                            "behind your browser)"):
                picked, err = inv.pick_directory(initial=initial)
            if err:
                st.session_state["inv_pick_err"] = err
            elif picked:
                try:
                    inv.list_folder(picked)
                    if picked not in state["folders"]:
                        state["folders"].append(picked)
                        inv.save_state(state)
                    st.session_state.pop("inv_pick_err", None)
                    st.rerun()
                except Exception as e:
                    st.session_state["inv_pick_err"] = str(e)
            else:
                st.session_state["inv_pick_err"] = ""  # cancelled — say nothing

    with add_col:
        new_folder = st.text_input(
            "...or type / paste a path (scanned non-recursively)",
            value="", key="inv_new_folder",
            placeholder=r"Z:\Productions\CIGNA\IRM-018",
        )
    with btn_col:
        st.write("")
        if st.button("➕ Add", key="inv_add_folder", use_container_width=True):
            if new_folder:
                try:
                    inv.list_folder(new_folder)  # validate before saving
                    if new_folder not in state["folders"]:
                        state["folders"].append(new_folder)
                        inv.save_state(state)
                    st.rerun()
                except Exception as e:
                    st.error(f"{e}")

    if st.session_state.get("inv_pick_err"):
        st.warning(st.session_state["inv_pick_err"])
    if not picker_ok:
        st.caption(f"📂 Browse is unavailable: {picker_why}")
    else:
        st.caption("📂 Browse opens a dialog on the machine running this app — fine when you "
                   "run it locally. If this ever gets hosted or containerised, the dialog "
                   "would open on the server, so use the path field instead. "
                   "Mapped drives and UNC paths (`\\\\server\\share`) both work.")

    if not state["folders"]:
        st.info("Add at least one folder. Each is listed non-recursively — add subfolders separately.")
        return

    for f in list(state["folders"]):
        c1, c2, c3 = st.columns([6, 2, 1])
        try:
            n = len(inv.list_folder(f))
            c1.write(f"📁 `{f}`")
            c2.caption(f"{n} scannable file(s)")
        except Exception as e:
            c1.write(f"📁 `{f}`")
            c2.error("unreachable")
        if c3.button("🗑️", key=f"inv_rmfolder_{f}"):
            state["folders"].remove(f)
            inv.save_state(state)
            st.rerun()

    # -----------------------------------------------------------------
    # 2. File preview + include/exclude
    # -----------------------------------------------------------------
    st.markdown("### 2️⃣ File list")
    st.caption("Uncheck a file to leave it out of the inventory. Exclusions persist, "
               "and a file's schema stays in the library even after you exclude every "
               "file that produced it.")

    listing = []
    for f in state["folders"]:
        try:
            listing.extend(inv.list_folder(f))
        except Exception:
            continue

    if not listing:
        st.warning("No scannable files found in these folders "
                   f"(looking for: {', '.join(sorted(inv.SCANNABLE_EXTS))}).")
        return

    excluded = set(state.get("excluded", []))

    # Fold in whatever the last scan learned, so the file list shows real
    # numbers instead of just names and sizes. One row per file here; a
    # multi-sheet workbook is summarised (its sheets appear as separate
    # datasets further down).
    cached = inv.load_cache()

    def _facts(path: str) -> tuple[str, str, str]:
        got = cached.get(path)
        if not isinstance(got, list) or not got:
            return "—", "—", "not scanned"
        errs = [g for g in got if g.get("error")]
        if errs:
            return "—", "—", "error"
        rows = sum(g.get("row_count") or 0 for g in got)
        fields = max(g.get("n_fields") or 0 for g in got)
        src = got[0].get("row_count_source") or "—"
        sheets = [g.get("sheet") for g in got if g.get("sheet")]
        label = f"{len(sheets)} sheets" if len(sheets) > 1 else src
        return f"{rows:,}", str(fields), label

    rows_for_df = []
    for r in listing:
        rc, nf, src = _facts(r["path"])
        rows_for_df.append({
            "Include": r["path"] not in excluded,
            "File": r["name"],
            "Size": _fmt_bytes(r["size_bytes"]),
            "Rows": rc,
            "Fields": nf,
            "Row count from": src,
            "Folder": r["folder"],
            "path": r["path"],
        })
    preview_df = pd.DataFrame(rows_for_df)

    edited = st.data_editor(
        preview_df, use_container_width=True, hide_index=True, height=280,
        disabled=["File", "Size", "Rows", "Fields", "Row count from", "Folder", "path"],
        column_config={"path": None}, key="inv_file_editor",
    )
    new_excluded = set(edited.loc[~edited["Include"], "path"].tolist())
    if new_excluded != excluded:
        state["excluded"] = sorted(new_excluded)
        inv.save_state(state)
        excluded = new_excluded

    included_records = [r for r in listing if r["path"] not in excluded]
    st.caption(f"{len(included_records)} of {len(listing)} files included — "
               f"{_fmt_bytes(sum(r['size_bytes'] for r in included_records))}")

    # -----------------------------------------------------------------
    # 3. Scan
    # -----------------------------------------------------------------
    st.markdown("### 3️⃣ Scan metadata")
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        do_scan = st.button("🔍 Scan", key="inv_scan", use_container_width=True)
    with sc2:
        st.caption("Cached by size + mtime — unchanged files are not re-read. Cost is not "
                   "uniform: **parquet** answers from its footer (milliseconds per file, "
                   "regardless of size), while **CSV / JSON / Excel** have no footer and must "
                   "be read through once to get an exact row count. Every format still "
                   "reports a real row count — nothing shows as unknown.")

    if do_scan:
        cache = inv.load_cache()
        bar = st.progress(0.0, text="Reading footers...")

        def _prog(i, total, name):
            bar.progress(i / max(total, 1), text=f"{i}/{total} — {name}")

        entries, cache = inv.scan_files(included_records, cache=cache, progress=_prog)
        inv.save_cache(cache)
        bar.empty()

        schemas, new_ids = inv.register_schemas(entries, schemas, batch=state.get("batch_label", ""))
        inv.save_schemas(schemas)
        st.session_state["inv_entries"] = [e.__dict__ for e in entries]
        if new_ids:
            st.success(f"Scanned {len(entries)} file(s). New schema(s) registered: {', '.join(new_ids)}")
        else:
            st.success(f"Scanned {len(entries)} file(s). No new schemas.")

    if "inv_entries" not in st.session_state:
        # Rebuild from cache so the page survives a rerun without re-scanning.
        # Shared loader — the cache's shape is inventory.py's business.
        rebuilt = inv.entries_from_cache(state)
        if rebuilt:
            st.session_state["inv_entries"] = [e.__dict__ for e in rebuilt]
        else:
            st.info("Click Scan to read metadata for the included files.")
            return

    entries = [inv.FileEntry(**d) for d in st.session_state["inv_entries"]]
    entries = [e for e in entries
               if e.path not in excluded and (e.source_path or e.path) not in excluded]

    failed = [e for e in entries if not e.ok]
    if failed:
        with st.expander(f"⚠️ {len(failed)} file(s) failed metadata read"):
            st.table([{"file": e.name, "error": e.error} for e in failed])

    # -----------------------------------------------------------------
    # 4. Datasets
    # -----------------------------------------------------------------
    st.markdown("### 4️⃣ Datasets")
    datasets = inv.build_datasets(entries, schemas, state)

    st.dataframe(pd.DataFrame([{
        "Dataset": (f"Dataset{d['dataset_no']}" if d["dataset_no"] else "—"),
        "Label": d["label"],
        "Schema": d["group_id"],
        "Files": d["n_files"],
        "Size": _fmt_bytes(d["total_bytes"]),
        "Size (Bytes)": d["total_bytes"],
        "Row Count": _fmt_rows(d["row_count"]),
        "Count from": d.get("row_count_method", "—"),
        "Data Fields": d["n_fields"],
        "Flags": "; ".join(d["flags"]),
    } for d in datasets]), use_container_width=True, hide_index=True)

    tot_bytes = sum(d["total_bytes"] for d in datasets)
    tot_rows = None if any(d["row_count"] is None for d in datasets) \
        else sum(d["row_count"] for d in datasets)
    st.caption(f"**Total** — {sum(d['n_files'] for d in datasets)} files, "
               f"{_fmt_bytes(tot_bytes)} ({tot_bytes:,} bytes), rows: {_fmt_rows(tot_rows)}")

    for d in datasets:
        if d["merge_diffs"] or d["overridden_files"]:
            _dl = f"Dataset{d['dataset_no']}" if d["dataset_no"] else d["group_id"]
            with st.expander(f"🔎 {_dl} — manual grouping detail"):
                for md in d["merge_diffs"]:
                    st.write(f"**{md['schema_id']}** folded in — difference: `{md['kind']}`")
                    if md["only_in_a"]:
                        st.caption(f"only in canonical: {', '.join(md['only_in_a'])}")
                    if md["only_in_b"]:
                        st.caption(f"only in {md['schema_id']}: {', '.join(md['only_in_b'])}")
                    if md["type_diff"]:
                        st.table(md["type_diff"])
                for ov in d["overridden_files"]:
                    st.write(f"📌 `{ov['file']}` manually assigned to {ov['assigned_to']} "
                             f"— {ov['note'] or 'no note'}")

    # -----------------------------------------------------------------
    # 5. Suggested merges
    # -----------------------------------------------------------------
    active_ids = [d["group_id"] for d in datasets if d["group_id"] in schemas]
    suggestions = inv.suggest_merges(schemas, list(schemas.keys()), state)
    if suggestions:
        st.markdown("### 5️⃣ Possible false splits")
        st.caption("Same column names, different strict fingerprint — the strict grouping split "
                   "these on type or column order alone. Review before merging: merging "
                   "non-identical schemas is a judgement call, and it gets recorded as one.")
        for s in suggestions:
            if s.get("truncated"):
                st.warning(s["note"])
                continue
            a, b, d = s["a"], s["b"], s["diff"]
            with st.expander(f"{a} ↔ {b} — differ by `{d['kind']}`"):
                if d["type_diff"]:
                    st.table(d["type_diff"])
                if d["order_differs"]:
                    st.caption("Column order differs between the two schemas.")
                note = st.text_input("Note (required — goes in the audit record)",
                                     key=f"inv_sugnote_{a}_{b}")
                mc1, mc2 = st.columns(2)
                if mc1.button(f"Merge {b} → {a}", key=f"inv_merge_{a}_{b}", disabled=not note):
                    state["merges"][b] = {"into": a, "note": note, "at": inv._now()}
                    inv.save_state(state)
                    st.rerun()
                if mc2.button(f"Merge {a} → {b}", key=f"inv_merge_{b}_{a}", disabled=not note):
                    state["merges"][a] = {"into": b, "note": note, "at": inv._now()}
                    inv.save_state(state)
                    st.rerun()

    # -----------------------------------------------------------------
    # 6. Manual grouping
    # -----------------------------------------------------------------
    st.markdown("### 6️⃣ Manual grouping")
    schema_ids = sorted(schemas.keys())

    with st.expander("Merge two schemas into one Dataset"):
        m1, m2 = st.columns(2)
        src = m1.selectbox("Fold this schema...", schema_ids, key="inv_merge_src")
        dst = m2.selectbox("...into this one", schema_ids, key="inv_merge_dst")
        if src and dst and src != dst:
            d = inv.diff_schemas(schemas[dst], schemas[src])
            st.caption(f"Difference: `{d['kind']}`")
            if d["only_in_b"]:
                st.caption(f"Columns only in {src}: {', '.join(d['only_in_b'][:20])}")
            if d["only_in_a"]:
                st.caption(f"Columns only in {dst}: {', '.join(d['only_in_a'][:20])}")
            if d["type_diff"]:
                st.table(d["type_diff"][:20])
            mnote = st.text_input("Note", key="inv_manual_merge_note")
            if st.button("Merge", key="inv_manual_merge", disabled=not mnote):
                state["merges"][src] = {"into": dst, "note": mnote, "at": inv._now()}
                inv.save_state(state)
                st.rerun()

    if state.get("merges"):
        with st.expander(f"Active merges ({len(state['merges'])})"):
            for src, m in list(state["merges"].items()):
                c1, c2 = st.columns([5, 1])
                c1.write(f"`{src}` → `{m['into']}` — {m.get('note','')} _{m.get('at','')}_")
                if c2.button("↩️ Undo", key=f"inv_unmerge_{src}"):
                    del state["merges"][src]
                    inv.save_state(state)
                    st.rerun()

    with st.expander("Assign a file to an existing schema"):
        st.caption("Use when a file's own fingerprint differs but it belongs with an existing "
                   "dataset. The file's real fingerprint is kept and reported alongside the "
                   "reassignment, so the override is visible rather than hidden.")
        ok_entries = [e for e in entries if e.ok]
        if ok_entries:
            fsel = st.selectbox("File", [e.name for e in ok_entries], key="inv_ov_file")
            target = st.selectbox("Report under schema", schema_ids, key="inv_ov_schema")
            onote = st.text_input("Note", key="inv_ov_note")
            chosen = next(e for e in ok_entries if e.name == fsel)
            cur = inv.schema_id_for_entry(chosen, schemas, state)
            st.caption(f"Currently reported under: `{cur or '(unregistered)'}` — "
                       f"own fingerprint `{chosen.fp_strict}`, {chosen.n_fields} fields")
            oc1, oc2 = st.columns(2)
            if oc1.button("Assign", key="inv_ov_apply", disabled=not onote):
                state["file_overrides"][chosen.path] = {
                    "schema_id": target, "note": onote, "at": inv._now()}
                inv.save_state(state)
                st.rerun()
            if chosen.path in state.get("file_overrides", {}):
                if oc2.button("↩️ Clear override", key="inv_ov_clear"):
                    del state["file_overrides"][chosen.path]
                    inv.save_state(state)
                    st.rerun()

    # -----------------------------------------------------------------
    # 7. Schema library
    # -----------------------------------------------------------------
    st.markdown("### 7️⃣ Schema library")
    st.caption("Schemas persist independently of files. Excluding or deleting every file of a "
               "schema leaves the record here, flagged as 0 files — delete it explicitly if you "
               "want it gone.")
    if not schemas:
        st.info("No schemas registered yet.")
    # One pass for every schema's file count — the per-schema re-scan this
    # replaced cost 2.7s per rerun at 500 schemas.
    schema_counts = inv.count_files_by_schema(entries, schemas, state)
    for sid in schema_ids:
        s = schemas[sid]
        n_files = schema_counts.get(sid, 0)
        merged_note = f" → merged into {state['merges'][sid]['into']}" if sid in state.get("merges", {}) else ""
        with st.expander(f"{sid} — {s.label or '(unlabelled)'} — {s.n_fields} fields, "
                         f"{n_files} file(s){merged_note}"):
            e1, e2 = st.columns(2)
            new_label = e1.text_input("Label", value=s.label, key=f"inv_lbl_{sid}")
            new_batch = e2.text_input("Batch", value=s.batch, key=f"inv_bat_{sid}")
            new_note = st.text_area("Note", value=s.note, height=70, key=f"inv_note_{sid}")
            b1, b2 = st.columns([1, 1])
            if b1.button("💾 Save", key=f"inv_save_{sid}"):
                s.label, s.batch, s.note = new_label, new_batch, new_note
                inv.save_schemas(schemas)
                st.rerun()
            if b2.button("🗑️ Delete schema record", key=f"inv_del_{sid}"):
                if n_files:
                    st.error(f"{n_files} included file(s) still report under {sid} — "
                             f"exclude them first, or reassign them.")
                else:
                    del schemas[sid]
                    state["merges"] = {k: v for k, v in state.get("merges", {}).items()
                                       if k != sid and v.get("into") != sid}
                    inv.save_schemas(schemas)
                    inv.save_state(state)
                    st.rerun()
            st.caption(f"fingerprint `{s.fp_strict}` · first seen {s.first_seen}")
            st.dataframe(pd.DataFrame(s.columns), use_container_width=True,
                         hide_index=True, height=200)

    # -----------------------------------------------------------------
    # 8. JSON output
    # -----------------------------------------------------------------
    st.markdown("### 8️⃣ Export JSON")
    st.caption("Plain JSON handoff — per-dataset totals, per-file names, full schema column "
               "lists, and every manual override with its note. Nothing is written to or read "
               "from Excel; type the figures into your template from this.")
    inc_files = st.checkbox("Include per-file name lists", value=True, key="inv_json_files")
    payload = inv.to_json_payload(datasets, entries, schemas, state,
                                 include_file_detail=inc_files)
    js = json.dumps(payload, indent=2, ensure_ascii=False)
    st.download_button("⬇️ Download inventory.json", data=js,
                       file_name="phase1_inventory.json", mime="application/json")
    with st.expander("Preview"):
        st.code(js[:4000] + ("\n... truncated ..." if len(js) > 4000 else ""), language="json")
