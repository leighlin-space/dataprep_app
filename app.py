"""
Internal Data Prep Tool — Phase 1 + early Phase 2

Run locally with:
    streamlit run app.py

Everything here is local-only: files stay on disk, the warehouse is a
local DuckDB file (warehouse.duckdb), no external calls are made.
"""

import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
from pathlib import Path
from modules import ingest, clean, schema, db, query, export, profiles, filters
import inventory_ui
import sampling_ui

LARGE_FILE_THRESHOLD_MB = 50  # above this, route through DuckDB's native reader instead of pandas

def _render_schema_board(tables_data: dict, relationships: list, height: int = 700) -> None:
    """
    'Detective board' style schema view: every table is a draggable box
    showing ALL its columns (scrollable if long), auto-detected
    relationships are drawn as solid lines. Manual connections have 3
    states: click column -> column draws a DASHED line ("potential");
    click that line once -> becomes SOLID ("confirmed"); click it again
    -> removed entirely, as if never drawn.

    tables_data: {table_name: [{"name": col, "type": dtype, "pk": bool}, ...]}
    relationships: list of schema.Relationship

    Note: manual connections + colors live only in this render's browser
    session — they reset if you click "Detect relationships" again or
    reload the page.
    """
    import json

    tables_json = json.dumps(tables_data).replace("</", "<\\/")
    rels_json = json.dumps([
        {"from_table": r.from_table, "from_col": r.from_column,
         "to_table": r.to_table, "to_col": r.to_column, "overlap": r.overlap_pct}
        for r in relationships
    ]).replace("</", "<\\/")

    palette = ["#E8873A", "#5B9BD5", "#E15759", "#59A14F", "#B07AA1",
               "#EDC948", "#76B7B2", "#FF6FA6", "#9C755F", "#4C4C4C"]
    palette_json = json.dumps(palette)

    html = f"""
    <div id="toolbar" style="display:flex; align-items:center; gap:6px; margin-bottom:8px; flex-wrap:wrap;">
      <span style="font-size:13px; color:#555; margin-right:4px;">Connection color:</span>
      <div id="swatches" style="display:flex; gap:4px;"></div>
      <button id="clear-manual-btn" style="margin-left:14px;">Clear my connections</button>
      <button id="reset-layout-btn" style="margin-left:6px;">Reset layout</button>
    </div>
    <div id="board-wrap" style="width:100%; height:{height-60}px; overflow:auto; border:1px solid #ddd;
         border-radius:8px; position:relative; background:#fafafa;">
      <div id="board-canvas" style="position:relative; width:2600px; height:1800px;">
        <svg id="board-svg" style="position:absolute; top:0; left:0; width:2600px; height:1800px;"></svg>
      </div>
    </div>
    <div style="margin-top:6px; font-size:12px; color:#777;">
      Click a column, then click another column, to draw a dashed "potential" connection in the selected color.
      Click that line once to confirm it (solid). Click a confirmed line again to remove it. Drag a table by its header to move it.
    </div>

    <script>
    (function() {{
        const tablesData = {tables_json};
        const autoRels = {rels_json};
        const palette = {palette_json};
        const canvas = document.getElementById("board-canvas");
        const wrap = document.getElementById("board-wrap");
        const svg = document.getElementById("board-svg");
        const boxEls = {{}};
        const boxPos = {{}};
        let manualLinks = [];  // {{from_table, from_col, to_table, to_col, color}}
        let pendingSelection = null;
        let activeColor = palette[0];

        const tableNames = Object.keys(tablesData);
        const COLS_PER_ROW = 4, BOX_W = 240, BOX_H = 280, GAP_X = 60, GAP_Y = 50;

        // --- toolbar swatches ---
        const swatchWrap = document.getElementById("swatches");
        function renderSwatches() {{
            swatchWrap.innerHTML = "";
            palette.forEach(c => {{
                const sw = document.createElement("div");
                const selected = (c === activeColor);
                sw.style.cssText = `width:20px; height:20px; border-radius:50%; background:${{c}};
                    cursor:pointer; border:${{selected ? "3px solid #222" : "1px solid #999"}};
                    box-sizing:border-box;`;
                sw.title = c;
                sw.addEventListener("click", () => {{ activeColor = c; renderSwatches(); }});
                swatchWrap.appendChild(sw);
            }});
        }}
        renderSwatches();

        function layoutDefaults() {{
            tableNames.forEach((t, i) => {{
                boxPos[t] = {{
                    x: 30 + (i % COLS_PER_ROW) * (BOX_W + GAP_X),
                    y: 30 + Math.floor(i / COLS_PER_ROW) * (BOX_H + GAP_Y)
                }};
            }});
        }}

        function buildBoxes() {{
            canvas.querySelectorAll(".table-box").forEach(el => el.remove());
            tableNames.forEach(t => {{
                const cols = tablesData[t];
                const box = document.createElement("div");
                box.className = "table-box";
                box.style.cssText = `position:absolute; left:${{boxPos[t].x}}px; top:${{boxPos[t].y}}px;
                    width:${{BOX_W}}px; background:white; border:1px solid #999; border-radius:6px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.12); font-family:sans-serif; font-size:12px; z-index:2;`;

                const header = document.createElement("div");
                header.textContent = t;
                header.style.cssText = `background:#1D9E75; color:white; padding:6px 8px; border-radius:6px 6px 0 0;
                    cursor:move; font-weight:bold; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;`;
                box.appendChild(header);

                const list = document.createElement("div");
                list.className = "col-list";
                list.style.cssText = `max-height:${{BOX_H - 30}}px; overflow-y:auto;`;
                list.addEventListener("scroll", drawLines);
                cols.forEach(c => {{
                    const row = document.createElement("div");
                    row.dataset.table = t;
                    row.dataset.col = c.name;
                    row.style.cssText = `padding:3px 8px; border-bottom:1px solid #eee; cursor:pointer;
                        display:flex; justify-content:space-between; gap:6px;`;
                    row.innerHTML = `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                        ${{c.pk ? "🔑 " : ""}}${{c.name}}</span><span style="color:#999; flex-shrink:0;">${{c.type}}</span>`;
                    row.addEventListener("click", () => onColumnClick(t, c.name, row));
                    list.appendChild(row);
                }});
                box.appendChild(list);
                canvas.appendChild(box);
                boxEls[t] = box;
                makeDraggable(box, header, t);
            }});
        }}

        function makeDraggable(box, handle, tableName) {{
            let dragging = false, startX, startY, origX, origY;
            handle.addEventListener("mousedown", (e) => {{
                dragging = true; startX = e.clientX; startY = e.clientY;
                origX = boxPos[tableName].x; origY = boxPos[tableName].y;
                e.preventDefault();
            }});
            document.addEventListener("mousemove", (e) => {{
                if (!dragging) return;
                boxPos[tableName].x = origX + (e.clientX - startX);
                boxPos[tableName].y = origY + (e.clientY - startY);
                box.style.left = boxPos[tableName].x + "px";
                box.style.top = boxPos[tableName].y + "px";
                drawLines();
            }});
            document.addEventListener("mouseup", () => {{ dragging = false; }});
        }}

        function onColumnClick(table, col, rowEl) {{
            document.querySelectorAll(".col-list div[data-col]").forEach(r => r.style.background = "");
            if (!pendingSelection) {{
                pendingSelection = {{ table, col }};
                rowEl.style.background = "#FFE9CC";
                return;
            }}
            if (pendingSelection.table === table && pendingSelection.col === col) {{
                pendingSelection = null;
                return;
            }}
            manualLinks.push({{
                from_table: pendingSelection.table, from_col: pendingSelection.col,
                to_table: table, to_col: col, color: activeColor, state: "potential"
            }});
            pendingSelection = null;
            drawLines();
        }}

        // Pick which side of each box a line should leave/enter from, based on
        // actual current horizontal position — this is what makes lines look
        // right instead of backwards after dragging boxes around.
        function pickSides(fromTable, toTable) {{
            const fromCenter = boxPos[fromTable].x + BOX_W / 2;
            const toCenter = boxPos[toTable].x + BOX_W / 2;
            return fromCenter <= toCenter
                ? {{ fromSide: "right", toSide: "left" }}
                : {{ fromSide: "left", toSide: "right" }};
        }}

        function getAnchor(table, col, side) {{
            const box = boxEls[table];
            const list = box ? box.querySelector(".col-list") : null;
            const row = box ? box.querySelector(`[data-col="${{CSS.escape(col)}}"]`) : null;
            if (!box || !row || !list) return null;
            const boxRect = box.getBoundingClientRect();
            const listRect = list.getBoundingClientRect();
            const rowRect = row.getBoundingClientRect();
            const canvasRect = canvas.getBoundingClientRect();

            // If the field is currently scrolled out of view within its box,
            // clamp the line to the visible list's top/bottom edge instead of
            // the field's true (invisible) position — keeps the line pointing
            // sensibly toward "it's above/below here" rather than jumping
            // somewhere that looks disconnected from the box entirely.
            const rowCenterY = rowRect.top + rowRect.height / 2;
            const clampedY = Math.min(Math.max(rowCenterY, listRect.top), listRect.bottom);

            const x = (side === "right" ? boxRect.right : boxRect.left) - canvasRect.left;
            const y = clampedY - canvasRect.top;
            return {{ x, y }};
        }}

        function drawLine(a, b, color, dashed, onClick) {{
            if (!a || !b) return;
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
            line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
            line.setAttribute("stroke", color);
            line.setAttribute("stroke-width", "3");
            if (dashed) line.setAttribute("stroke-dasharray", "6,4");
            if (onClick) {{
                line.style.cursor = "pointer";
                line.style.pointerEvents = "stroke";
                line.addEventListener("click", onClick);
            }} else {{
                line.style.pointerEvents = "none";
            }}
            svg.appendChild(line);
        }}

        function drawLines() {{
            svg.innerHTML = "";
            autoRels.forEach(r => {{
                const sides = pickSides(r.from_table, r.to_table);
                const a = getAnchor(r.from_table, r.from_col, sides.fromSide);
                const b = getAnchor(r.to_table, r.to_col, sides.toSide);
                drawLine(a, b, r.overlap >= 99 ? "#1D9E75" : "#5B9BD5", false, null);
            }});
            manualLinks.forEach((l, idx) => {{
                const sides = pickSides(l.from_table, l.to_table);
                const a = getAnchor(l.from_table, l.from_col, sides.fromSide);
                const b = getAnchor(l.to_table, l.to_col, sides.toSide);
                const isDashed = (l.state === "potential");
                drawLine(a, b, l.color, isDashed, () => {{
                    if (l.state === "potential") {{
                        l.state = "confirmed";
                    }} else {{
                        manualLinks.splice(manualLinks.indexOf(l), 1);
                    }}
                    drawLines();
                }});
            }});
        }}

        document.getElementById("clear-manual-btn").addEventListener("click", () => {{
            manualLinks = []; drawLines();
        }});
        document.getElementById("reset-layout-btn").addEventListener("click", () => {{
            layoutDefaults();
            tableNames.forEach(t => {{
                boxEls[t].style.left = boxPos[t].x + "px";
                boxEls[t].style.top = boxPos[t].y + "px";
            }});
            drawLines();
        }});
        wrap.addEventListener("scroll", drawLines);

        layoutDefaults();
        buildBoxes();
        setTimeout(drawLines, 50);
    }})();
    </script>
    """
    components.html(html, height=height, scrolling=False)

st.set_page_config(page_title="Data Prep Tool", page_icon="🆂", layout="wide")
st.title(" 🪐 Saturn Ion - in memory of my first car")

filters.seed_default_if_missing()
con = db.get_connection()
with st.sidebar.expander("Loaded Tables", expanded=False):
    _sidebar_tables = db.list_tables(con)
    if not _sidebar_tables:
        st.caption("No tables loaded yet.")
    else:
        for _t in _sidebar_tables:
            _editing_key = f"sidebar_editing_{_t}"

            if st.session_state.get(_editing_key):
                _new_name = st.text_input(
                    "Rename", value=_t, key=f"sidebar_rename_input_{_t}", label_visibility="collapsed"
                )
                _c1, _c2 = st.columns(2)
                if _c1.button("✅ Save", key=f"sidebar_confirm_{_t}"):
                    if _new_name and _new_name != _t:
                        db.rename_table(con, _t, _new_name)
                    st.session_state[_editing_key] = False
                    st.rerun()
                if _c2.button("✖️ Cancel", key=f"sidebar_cancel_{_t}"):
                    st.session_state[_editing_key] = False
                    st.rerun()
            else:
                _col1, _col2, _col3 = st.columns([3, 1, 1])
                _col1.write(_t)
                if _col2.button("✏️", key=f"sidebar_edit_{_t}"):
                    st.session_state[_editing_key] = True
                    st.rerun()
                if _col3.button("🗑️", key=f"sidebar_drop_{_t}"):
                    db.drop_table(con, _t)
                    st.rerun()
(tab_inventory, tab_sample, tab_load, tab_schema, tab_tables,
 tab_profiles, tab_query, tab_export) = st.tabs(
    ["1. Inventory", "2. Sample", "3. Load & Clean", "4. Schema",
     "5. Warehouse Tables", "6. Profiles", "7. Query Builder", "8. Export"]
)

# ---------------------------------------------------------------------------
# TAB 1: Inventory
# ---------------------------------------------------------------------------
with tab_inventory:
    inventory_ui.render()

with tab_sample:
    sampling_ui.render(con)

with tab_load:
    st.subheader("Load from server path (recommended for very large files)")
    st.caption("For files too large to upload through the browser (50GB, 100GB+), point directly at the "
               "path on your company server/network drive — DuckDB reads it in place, nothing uploads.")

    server_path = st.text_input("Full file path", value="", placeholder=r"Z:\Users\rliu\Opioid\...\file.csv")
    if server_path:
        path_obj = Path(server_path)
        ext = path_obj.suffix.lstrip(".").lower()

        if not path_obj.exists():
            st.error("Path not found — check it's reachable from this machine (mapped drive, etc.).")
        elif ext not in ("csv", "tsv", "parquet", "json"):
            st.error(f"'.{ext}' has no native DuckDB reader for this path — Excel files should use the "
                     f"browser uploader below instead.")
        else:
            size_gb = path_obj.stat().st_size / (1024 ** 3)
            st.caption(f"{size_gb:.2f} GB")

            convert_first = ext in ("csv", "tsv", "json") and st.checkbox(
                "Convert to Parquet first (recommended above ~10GB — faster for every query after)",
                value=size_gb > 10,
            )

            server_table_name = st.text_input("Table name in warehouse", value=path_obj.stem, key="server_table_name")

            if st.button("✅ Load from server path"):
                try:
                    load_path, load_type = server_path, ("csv" if ext in ("csv", "tsv") else ext)
                    if convert_first:
                        dest = str(path_obj.with_suffix(".parquet"))
                        with st.spinner(f"Converting to Parquet at {dest} ..."):
                            db.convert_to_parquet(server_path, load_type, dest)
                        load_path, load_type = dest, "parquet"
                        st.success(f"Converted → {dest}")

                    with st.spinner("Loading into warehouse via DuckDB native reader..."):
                        n_rows = db.register_table_from_path(con, server_table_name, load_path, load_type)
                    st.success(f"Loaded '{server_table_name}' — {n_rows:,} rows.")
                except Exception as e:
                    st.error(f"Load failed: {e}")

    st.markdown("---")
    st.subheader("Upload raw data")
    st.caption(f"Files under {LARGE_FILE_THRESHOLD_MB}MB get the full cleaning workflow (pandas). "
               f"Larger CSV/Parquet/JSON files load straight into DuckDB's native reader — faster and "
               f"lighter on memory, with cleaning suggestions run on a sample instead of the full file.")
    uploaded = st.file_uploader(
        "CSV, Excel, JSON, or Parquet",
        type=["csv", "xlsx", "xls", "json", "parquet", "tsv", "txt"],
        accept_multiple_files=True,
    )

    if uploaded:
        for f in uploaded:
            st.markdown(f"---\n### 📄 {f.name}")
            size_mb = f.size / (1024 * 1024)
            ext = f.name.lower().rsplit(".", 1)[-1]
            native_supported = ext in ("csv", "tsv", "parquet", "json")
            table_name_default = f.name.rsplit(".", 1)[0]

            # ---- LARGE FILE PATH: DuckDB native reader, bypass pandas ----
            if size_mb > LARGE_FILE_THRESHOLD_MB and native_supported:
                st.info(f"📦 {size_mb:.0f}MB — large-file mode: loading directly via DuckDB "
                        f"(no full pandas load).")
                table_name = st.text_input("Table name in warehouse", value=table_name_default, key=f"name_{f.name}")

                if st.button(f"✅ Load '{f.name}' into warehouse (DuckDB native)", key=f"load_large_{f.name}"):
                    tmp_path = None
                    try:
                        suffix = "." + ext
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.getbuffer())
                            tmp_path = tmp.name
                        filetype = "csv" if ext in ("csv", "tsv") else ext
                        n_rows = db.register_table_from_path(con, table_name, tmp_path, filetype)
                        st.success(f"Loaded '{table_name}' — {n_rows:,} rows, via DuckDB native reader.")
                    except Exception as e:
                        st.error(f"Could not load {f.name}: {e}")
                    finally:
                        if tmp_path:
                            os.unlink(tmp_path)

                # Lightweight preview + duplicate check on a sample only, once table exists
                if table_name in db.list_tables(con):
                    sample = db.preview_table(con, table_name, n=500)
                    st.caption(f"Preview (first 500 rows of {db.count_rows(con, table_name):,} total)")
                    st.dataframe(sample.head(10), use_container_width=True)
                    dupes = clean.find_duplicates(sample)
                    if len(dupes) > 0:
                        st.warning(f"⚠️ {len(dupes)} duplicate rows found in the 500-row sample — "
                                   f"full-file duplicate check available via SQL in Tab 5 (Query Builder).")
                    else:
                        st.caption("✅ No duplicates in sample (full-file check not run at this size — use Query Builder for an exact count).")
                continue

            # ---- SMALL FILE PATH: full pandas load + cleaning workflow (unchanged) ----
            try:
                raw_df = ingest.load_file(f)
            except Exception as e:
                st.error(f"Could not load {f.name}: {e}")
                continue

            st.caption(f"{raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
            st.dataframe(raw_df.head(10), use_container_width=True)

            dupes = clean.find_duplicates(raw_df)
            if len(dupes) > 0:
                n_extra = len(dupes) - dupes.drop_duplicates().shape[0]
                st.warning(f"⚠️ {len(dupes)} rows are part of an exact duplicate set "
                           f"({n_extra} extra copies beyond the first occurrence).")
                with st.expander(f"🔍 View duplicate rows — {f.name}"):
                    st.dataframe(dupes, use_container_width=True)
            else:
                st.caption("✅ No exact duplicate rows found.")

            col_profiles = clean.profile_dataframe(raw_df)
            suggestions = clean.suggest_cleaning(raw_df, col_profiles)

            with st.expander(f"📊 Column profile — {f.name}"):
                st.table([{
                    "column": p.name, "dtype": p.dtype, "% null": p.null_pct,
                    "unique values": p.n_unique, "likely key?": p.is_likely_key,
                } for p in col_profiles])

            accepted_ids = set()
            if suggestions:
                st.markdown("**Suggested cleaning steps** (checked = will apply)")
                for s in suggestions:
                    default_checked = s.action != "flag_only"
                    checked = st.checkbox(s.description, value=default_checked, key=f"{f.name}_{s.id}")
                    if checked:
                        accepted_ids.add(s.id)
            else:
                st.success("No issues detected.")

            table_name = st.text_input(
                "Table name in warehouse", value=f.name.rsplit(".", 1)[0], key=f"name_{f.name}"
            )

            if st.button(f"✅ Clean & load '{f.name}' into warehouse", key=f"load_{f.name}"):
                cleaned_df = clean.apply_suggestions(raw_df, suggestions, accepted_ids)
                db.register_table(con, table_name, cleaned_df)
                st.success(f"Loaded as table '{table_name}' ({cleaned_df.shape[0]:,} rows).")

# ---------------------------------------------------------------------------
# TAB 2: Schema
# ---------------------------------------------------------------------------
with tab_schema:
    st.subheader("Schema board — explore & annotate relationships")
    table_names = db.list_tables(con)

    if len(table_names) < 1:
        st.info("Load at least one table in Tab 1 to see the schema here.")
    else:
        st.caption("This is a viewing/exploration space — to rename or remove columns, "
                   "use Tab 3 (Warehouse Tables) instead.")

        if st.button("🔍 Detect relationships"):
            with st.spinner("Comparing columns across tables (in DuckDB, not pandas)..."):
                rels = schema.detect_relationships_sql(con, table_names)
            st.session_state["relationships"] = rels

        rels = st.session_state.get("relationships", [])
        if not rels:
            st.caption("No relationships detected yet — click 'Detect relationships' above, "
                       "or connect fields manually on the board below.")

        # Build full column data per table — no 12-column cap, board scrolls internally instead
        tables_data = {}
        for name in table_names:
            pk = schema.detect_primary_key_sql(con, name)
            col_info = con.execute(f'DESCRIBE "{name}"').fetchall()
            tables_data[name] = [
                {"name": col, "type": str(dtype), "pk": (col == pk)}
                for col, dtype, *_ in col_info
            ]

        _render_schema_board(tables_data, rels)
# ---------------------------------------------------------------------------
# TAB 3: Warehouse tables
# ---------------------------------------------------------------------------
with tab_tables:
    st.subheader("Tables currently in the warehouse")
    table_names = db.list_tables(con)
    if not table_names:
        st.info("No tables loaded yet.")
    for name in table_names:
        with st.expander(name):
            n_rows = db.count_rows(con, name)
            n_cols = db.count_columns(con, name)
            st.caption(f"{n_rows:,} rows × {n_cols} columns")
            preview = db.preview_table(con, name, n=20)
            st.dataframe(preview, use_container_width=True)
            if st.button(f"🗑️ Drop '{name}'", key=f"drop_{name}"):
                db.drop_table(con, name)
                st.rerun()

# ---------------------------------------------------------------------------
# TAB 4: Query Builder
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# TAB 4: Profiles (Dataset Profile + Filter Profile)
# ---------------------------------------------------------------------------
with tab_profiles:
    st.subheader("Dataset Profiles — map columns to standard roles, once per format")
    table_names = db.list_tables(con)

    prof_col1, prof_col2 = st.columns([1, 1])

    with prof_col1:
        st.markdown("**Create / update a profile**")
        if not table_names:
            st.info("Load a table in Tab 1 first — the mapping UI reads its column names.")
        else:
            source_table = st.selectbox("Map columns from table", table_names, key="profile_source_table")
            source_cols = ["(none)"] + db.get_columns(con, source_table)

            profile_name = st.text_input("Profile name", value=f"{source_table}_format")
            mapping = {}
            for role_def in profiles.FIELD_ROLES:
                role, label, required = role_def["role"], role_def["label"], role_def["required"]
                display_label = f"{label} {'*' if required else ''}"
                # Best-effort default: exact case-insensitive column-name match
                default_idx = 0
                for i, c in enumerate(source_cols):
                    if c.lower().replace("_", "") == role.lower().replace("_", ""):
                        default_idx = i
                        break
                choice = st.selectbox(display_label, source_cols, index=default_idx, key=f"map_{role}")
                mapping[role] = None if choice == "(none)" else choice

            missing = profiles.missing_required_roles(mapping)
            if missing:
                st.warning(f"Required roles not yet mapped: {', '.join(missing)}")

            if st.button("💾 Save profile") and not missing:
                profiles.save_profile(profiles.DatasetProfile(name=profile_name, mapping=mapping))
                st.success(f"Saved profile '{profile_name}'.")
                st.rerun()

    with prof_col2:
        st.markdown("**Saved profiles**")
        saved = profiles.list_profiles()
        if not saved:
            st.info("No profiles saved yet.")
        for name in saved:
            with st.expander(name):
                p = profiles.load_profile(name)
                st.table([{"role": k, "column": v} for k, v in p.mapping.items() if v])
                if st.button(f"🗑️ Delete '{name}'", key=f"del_profile_{name}"):
                    profiles.delete_profile(name)
                    st.rerun()

    st.markdown("---")
    st.markdown("---")
    st.subheader("Filter Profiles")
    st.caption("Pick pieces from each section below, then save the combination as a new filter profile. "
               "Or skip straight to the custom builder at the bottom to define everything from scratch.")

    # =========================================================================
    # SECTION 1: TIME
    # =========================================================================
    st.markdown("### ⏱️ Time")
    qf_col1, qf_col2 = st.columns(2)
    with qf_col1:
        qf_date_start = st.text_input("Start date (YYYY-MM-DD)", value="", key="qf_date_start")
    with qf_col2:
        qf_date_end = st.text_input("End date (YYYY-MM-DD)", value="", key="qf_date_end")

    # =========================================================================
    # SECTION 2: REGION
    # =========================================================================
    st.markdown("### 📍 Region")
    qf_region_method = st.radio(
        "Define region by:",
        ["From a loaded table's ZIP/location column", "Upload a file of location names"],
        horizontal=True, key="qf_region_method",
    )

    if qf_region_method == "From a loaded table's ZIP/location column":
        qf_region_tables = db.list_tables(con)
        if not qf_region_tables:
            st.info("Load a table in Tab 1 first.")
        else:
            qf_rt_col1, qf_rt_col2 = st.columns(2)
            with qf_rt_col1:
                qf_region_table = st.selectbox("Table", qf_region_tables, key="qf_region_table")
            with qf_rt_col2:
                qf_region_col = st.selectbox("Column", db.get_columns(con, qf_region_table), key="qf_region_col")
            if st.button("📥 Load distinct values from this column", key="qf_load_region_col"):
                qf_sql = f'SELECT DISTINCT "{qf_region_col}" AS v FROM "{qf_region_table}" WHERE "{qf_region_col}" IS NOT NULL LIMIT 5000'
                qf_result = query.run_sql(con, qf_sql)
                st.session_state["qf_region_values"] = qf_result["v"].astype(str).tolist()
    else:
        qf_region_file = st.file_uploader("File with one location per line (CSV or TXT)",
                                           type=["csv", "txt"], key="qf_region_file")
        if qf_region_file is not None:
            try:
                if qf_region_file.name.lower().endswith(".csv"):
                    qf_rdf = ingest.load_file(qf_region_file)
                    qf_parsed = qf_rdf[qf_rdf.columns[0]].dropna().astype(str).unique().tolist()
                else:
                    qf_parsed = [ln.decode("utf-8").strip() for ln in qf_region_file.readlines()]
                    qf_parsed = [p for p in qf_parsed if p]
                st.session_state["qf_region_values"] = qf_parsed
                st.success(f"Loaded {len(qf_parsed)} location values from {qf_region_file.name}.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    qf_region_values = st.session_state.get("qf_region_values", [])
    if qf_region_values:
        _preview = ", ".join(qf_region_values[:10])
        st.caption(f"✅ {len(qf_region_values)} region values loaded: {_preview}"
                   f"{' ...' if len(qf_region_values) > 10 else ''}")
        if st.button("Clear region selection", key="qf_clear_region"):
            st.session_state["qf_region_values"] = []
            st.rerun()

    # =========================================================================
    # SECTION 3: DRUGS — saved drug-list profiles as buttons
    # =========================================================================
    st.markdown("### 💊 Drugs")
    qf_drug_profile_names = [n for n in filters.list_profiles() if filters.load_profile(n).drug_list]

    if not qf_drug_profile_names:
        st.info("No saved drug lists yet — create one in the custom builder at the bottom.")
    else:
        qf_selected_drug = st.session_state.get("qf_selected_drug")
        qf_btn_cols = st.columns(len(qf_drug_profile_names))
        for i, qf_dname in enumerate(qf_drug_profile_names):
            with qf_btn_cols[i]:
                qf_dp = filters.load_profile(qf_dname)
                qf_is_selected = (qf_selected_drug == qf_dname)
                if st.button(
                    f"{qf_dname}\n({len(qf_dp.drug_list)})",
                    key=f"qf_drugbtn_{qf_dname}",
                    use_container_width=True,
                    type="primary" if qf_is_selected else "secondary",
                ):
                    st.session_state["qf_selected_drug"] = None if qf_is_selected else qf_dname
                    st.rerun()

        qf_selected_drug = st.session_state.get("qf_selected_drug")
        if qf_selected_drug:
            qf_dp = filters.load_profile(qf_selected_drug)
            st.caption(f"Selected: **{qf_selected_drug}** — {len(qf_dp.drug_list)} entries, "
                       f"role: `{qf_dp.drug_list_role}`")

    # =========================================================================
    # Save the Time + Region + Drug combination picked above
    # =========================================================================
    st.markdown("---")
    qf_combo_name = st.text_input("Save this combination as a new Filter Profile", value="", key="qf_combo_name")
    if st.button("💾 Save combined filter profile", key="qf_save_combo") and qf_combo_name:
        qf_selected_drug = st.session_state.get("qf_selected_drug")
        qf_drug_list, qf_drug_role = [], "drug_name"
        if qf_selected_drug:
            qf_base = filters.load_profile(qf_selected_drug)
            qf_drug_list, qf_drug_role = qf_base.drug_list, qf_base.drug_list_role
        filters.save_profile(filters.FilterProfile(
            name=qf_combo_name,
            drug_list=qf_drug_list, drug_list_role=qf_drug_role,
            region_values=st.session_state.get("qf_region_values", []),
            date_start=qf_date_start, date_end=qf_date_end,
        ))
        st.success(f"Saved '{qf_combo_name}'.")
        st.rerun()

    # =========================================================================
    # All saved filter profiles — browse / delete
    # =========================================================================
    with st.expander("📁 All saved filter profiles"):
        for _name in filters.list_profiles():
            _fp = filters.load_profile(_name)
            st.write(f"**{_name}** — drugs: {len(_fp.drug_list)} ({_fp.drug_list_role}) | "
                     f"regions: {len(_fp.region_values)} | dates: {_fp.date_start or '...'} to {_fp.date_end or '...'}")
            if st.button(f"🗑️ Delete '{_name}'", key=f"qf_del_{_name}"):
                filters.delete_profile(_name)
                st.rerun()

    # =========================================================================
    # CUSTOMIZABLE FILTER — build one entirely from scratch, always last
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🛠️ Build a custom filter profile")
    filter_name = st.text_input("Filter profile name", value="", key="filter_name")
    drug_text = st.text_area("Drug list (one per line)", value="", height=150, key="custom_drug_text")
    custom_drug_role = st.radio("This list matches:", ["drug_name", "drug_code"], horizontal=True, key="custom_drug_role")
    region_text = st.text_area("Region values (one per line, optional)", value="", height=80, key="custom_region_text")
    date_start = st.text_input("Date range start (optional, YYYY-MM-DD)", value="", key="custom_date_start")
    date_end = st.text_input("Date range end (optional, YYYY-MM-DD)", value="", key="custom_date_end")

    if st.button("💾 Save custom filter profile", key="custom_save") and filter_name:
        filters.save_profile(filters.FilterProfile(
            name=filter_name,
            drug_list=[d.strip() for d in drug_text.splitlines() if d.strip()],
            drug_list_role=custom_drug_role,
            region_values=[r.strip() for r in region_text.splitlines() if r.strip()],
            date_start=date_start, date_end=date_end,
        ))
        st.success(f"Saved filter profile '{filter_name}'.")
        st.rerun()

with tab_query:
    st.subheader("Build a query")
    table_names = db.list_tables(con)

    if not table_names:
        st.info("Load at least one table in Tab 1 first.")
    else:
        advanced = st.toggle("Advanced: write/edit raw SQL", value=False)

        if advanced:
            default_sql = st.session_state.get("last_built_sql", f'SELECT * FROM "{table_names[0]}" LIMIT 100')
            sql_text = st.text_area("SQL", value=default_sql, height=150)
            if st.button("▶️ Run SQL"):
                try:
                    result = query.run_sql(con, sql_text)
                    st.session_state["query_result"] = result
                    st.session_state["last_built_sql"] = sql_text
                except Exception as e:
                    st.error(f"Query failed: {e}")

        else:
            base_table = st.selectbox("Base table", table_names)
            base_cols = db.get_columns(con, base_table)

            do_join = st.checkbox("Join with another table")
            join_table = join_type = join_left = join_right = None
            join_cols = []
            if do_join:
                other_tables = [t for t in table_names if t != base_table]
                join_table = st.selectbox("Join table", other_tables)
                join_type = st.selectbox("Join type", query.JOIN_TYPES)

                # Pre-fill from inferred relationships if available
                rels = st.session_state.get("relationships", [])
                suggested = next(
                    (r for r in rels if {r.from_table, r.to_table} == {base_table, join_table}),
                    None,
                )
                join_cols = db.get_columns(con, join_table)
                col1, col2 = st.columns(2)
                with col1:
                    default_left = suggested.from_column if suggested and suggested.from_table == base_table else base_cols[0]
                    join_left = st.selectbox(f"{base_table} column", base_cols, index=base_cols.index(default_left) if default_left in base_cols else 0)
                with col2:
                    default_right = suggested.to_column if suggested and suggested.to_table == join_table else join_cols[0]
                    join_right = st.selectbox(f"{join_table} column", join_cols, index=join_cols.index(default_right) if default_right in join_cols else 0)
                if suggested:
                    st.caption(f"Pre-filled from detected relationship ({suggested.overlap_pct}% value overlap)")

            available_cols = base_cols + join_cols
            columns = st.multiselect("Columns to include (empty = all)", available_cols)


            st.markdown("**Apply a Filter Profile** (optional)")
            st.caption("Resolves roles (drug name/code, region, date) to actual columns via a Dataset Profile.")
            profile_filters = []
            fp_col1, fp_col2 = st.columns(2)
            with fp_col1:
                dataset_profile_names = ["(none)"] + profiles.list_profiles()
                chosen_ds_profile = st.selectbox("Dataset Profile (for this table's column mapping)", dataset_profile_names)
            with fp_col2:
                filter_profile_names = ["(none)"] + filters.list_profiles()
                chosen_filter_profile = st.selectbox("Filter Profile", filter_profile_names)

            if chosen_ds_profile != "(none)" and chosen_filter_profile != "(none)":
                ds_map = profiles.load_profile(chosen_ds_profile).mapping
                fp = filters.load_profile(chosen_filter_profile)

                if fp.drug_list:
                    drug_col = ds_map.get(fp.drug_list_role)
                    if drug_col and drug_col in available_cols:
                        profile_filters.append({"column": drug_col, "op": "IN", "value": fp.drug_list})
                        st.caption(f"✅ Drug filter → `{drug_col}` IN ({len(fp.drug_list)} values, role: {fp.drug_list_role})")
                    else:
                        st.warning(f"Filter profile expects a '{fp.drug_list_role}' column, but the Dataset Profile "
                                   f"doesn't map one (or it's not in the base/join table) — drug filter skipped.")

                if fp.region_values:
                    region_col = ds_map.get("region")
                    if region_col and region_col in available_cols:
                        profile_filters.append({"column": region_col, "op": "IN", "value": fp.region_values})
                        st.caption(f"✅ Region filter → `{region_col}` IN ({', '.join(fp.region_values)})")
                    else:
                        st.warning("Filter profile has region values, but no 'region' column is mapped — skipped.")

                if fp.date_start or fp.date_end:
                    date_col = ds_map.get("fill_date")
                    if date_col and date_col in available_cols:
                        if fp.date_start:
                            profile_filters.append({"column": date_col, "op": ">=", "value": fp.date_start})
                        if fp.date_end:
                            profile_filters.append({"column": date_col, "op": "<=", "value": fp.date_end})
                        st.caption(f"✅ Date filter → `{date_col}` between {fp.date_start or '...'} and {fp.date_end or '...'}")
                    else:
                        st.warning("Filter profile has a date range, but no 'fill_date' column is mapped — skipped.")

            st.markdown("**Additional manual filters**")
            n_filters = st.number_input("Number of filters", 0, 5, 0)
            manual_filters = []
            for i in range(n_filters):
                c1, c2, c3 = st.columns(3)
                with c1:
                    fcol = st.selectbox("Column", available_cols, key=f"fcol_{i}")
                with c2:
                    fop = st.selectbox("Operator", query.FILTER_OPS, key=f"fop_{i}")
                with c3:
                    fval = st.text_input("Value", key=f"fval_{i}", disabled=fop in ("IS NULL", "IS NOT NULL"))
                manual_filters.append({"column": fcol, "op": fop, "value": fval})

            all_filters = profile_filters + manual_filters

            st.markdown("**Group by / aggregate** (optional)")
            group_by = st.multiselect("Group by columns", available_cols)
            n_aggs = st.number_input("Number of aggregations", 0, 5, 0)
            aggregations = []
            for i in range(n_aggs):
                c1, c2 = st.columns(2)
                with c1:
                    afunc = st.selectbox("Function", query.AGG_FUNCS, key=f"afunc_{i}")
                with c2:
                    acol = st.selectbox("Column", available_cols, key=f"acol_{i}")
                aggregations.append({"func": afunc, "column": acol})

            is_aggregating = bool(group_by or aggregations)
            st.markdown("**Row limit**")
            if is_aggregating:
                st.caption("This query aggregates, so the result is already small — "
                           "'No limit' is usually safe here.")
            else:
                st.caption("Raw (non-aggregated) query — keep a limit unless you specifically "
                           "need the full result set, to avoid pulling huge amounts into memory.")
            limit_choice = st.radio(
                "Limit", ["1,000 rows", "10,000 rows", "100,000 rows", "No limit"],
                index=0, horizontal=True, label_visibility="collapsed",
            )
            limit_map = {"1,000 rows": 1000, "10,000 rows": 10000, "100,000 rows": 100000, "No limit": None}
            row_limit = limit_map[limit_choice]

            if row_limit is None and not is_aggregating:
                st.warning("⚠️ No limit on a raw (non-aggregated) query — this will pull every "
                           "matching row into memory. Only do this if you're confident in the size.")

            built_sql = query.build_query(
                base_table=base_table, columns=columns,
                join_table=join_table, join_type=join_type or "INNER",
                join_left_col=join_left, join_right_col=join_right,
                filters=all_filters, group_by=group_by, aggregations=aggregations,
                limit=row_limit,
            )
            st.code(built_sql, language="sql")

            live_preview = st.checkbox("🔄 Live preview — auto-update as filters change", value=True)
            safe_to_autorun = row_limit is not None or is_aggregating
            run_clicked = st.button("▶️ Run query")

            if live_preview and not safe_to_autorun:
                st.caption("Live preview paused: no row limit on a non-aggregated query. "
                           "Set a limit above, or click Run manually.")

            if run_clicked or (live_preview and safe_to_autorun):
                try:
                    result = query.run_sql(con, built_sql)
                    st.session_state["query_result"] = result
                    st.session_state["last_built_sql"] = built_sql
                except Exception as e:
                    st.error(f"Query failed: {e}")

        if "query_result" in st.session_state:
            result = st.session_state["query_result"]
            st.markdown("---")
            st.markdown(f"**Result** — {result.shape[0]:,} rows × {result.shape[1]} columns")
            st.dataframe(result, use_container_width=True)

            numeric_cols = result.select_dtypes(include="number").columns.tolist()
            non_numeric_cols = [c for c in result.columns if c not in numeric_cols]
            if numeric_cols and non_numeric_cols and 1 < result.shape[0] <= 500:
                with st.expander("📊 Quick chart"):
                    x_col = st.selectbox("X axis", non_numeric_cols, key="chart_x")
                    y_col = st.selectbox("Y axis", numeric_cols, key="chart_y")
                    st.bar_chart(result.set_index(x_col)[y_col])

            save_name = st.text_input("Save result as new table (optional)", value="")
            if st.button("💾 Save as table") and save_name:
                db.register_table(con, save_name, result)
                st.success(f"Saved as '{save_name}'.")

# ---------------------------------------------------------------------------
# TAB 5: Export
# ---------------------------------------------------------------------------
with tab_export:
    st.subheader("Export to Excel")

    if "query_result" not in st.session_state:
        st.info("Run a query in Tab 4 first — the export tab works on your latest query result.")
    else:
        result = st.session_state["query_result"]
        st.caption(f"Exporting current query result: {result.shape[0]:,} rows × {result.shape[1]} columns")
        st.dataframe(result.head(10), use_container_width=True)

        sheet_name = st.text_input("Sheet name", value="Exhibit")
        title = st.text_input("Exhibit title (optional)", value="")

        excel_bytes = export.dataframe_to_excel_bytes(result, sheet_name=sheet_name, title=title or None)
        st.download_button(
            "⬇️ Download Excel exhibit",
            data=excel_bytes,
            file_name=f"{sheet_name.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )