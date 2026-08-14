"""
Phase 1 Inventory — metadata-only folder scan.

The point of this module: answer "what did we receive?" WITHOUT reading
any data. For a 499-file / 4.95B-row / 1.02TB parquet production,
loading anything into the warehouse is not an option, and it isn't
necessary — row count, column list and column types all live in the
parquet footer, and file size lives in the filesystem.

Nothing here touches warehouse.duckdb. Nothing here calls
db.get_connection(). Metadata reads go through pyarrow (footer only);
DuckDB is used only as a fallback and only via a throwaway in-memory
connection.

Three persistent stores under inventory_state/, deliberately separate:

  schemas.json    Schema records, keyed by a stable schema_id (S01, S02...).
                  A schema record OUTLIVES the files that produced it —
                  removing every file of a schema from the scan leaves the
                  record intact, flagged as retained-with-0-files. This is
                  what makes "upload, then drop the files, keep the schema"
                  work, and it's why schemas are not derived on the fly.

  state.json      Folder sources, per-file exclusions, and the manual
                  overrides (schema->group merges, file->schema
                  reassignments). Every override carries a note, because
                  the output of this feeds a sworn inventory and a manual
                  regrouping has to be explainable months later.

  scan_cache.json Last scan result per file path (size, mtime, row count,
                  fingerprints). Re-reading 499 footers off a network
                  drive is slow; renaming a group shouldn't trigger it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

STATE_DIR = Path(__file__).parent.parent / "inventory_state"
SCHEMAS_PATH = STATE_DIR / "schemas.json"
STATE_PATH = STATE_DIR / "state.json"
CACHE_PATH = STATE_DIR / "scan_cache.json"

# Extensions we'll consider a "produced file". Parquet is the only one
# with a footer, so it's the only one that yields a row count for free.
# Every supported format must yield the same inventory facts — row count,
# field count, size — so nothing shows as "unknown". What differs is cost:
# parquet answers from its footer, everything else has to be read through.
PARQUET_EXTS = {".parquet", ".pq"}
DELIMITED_EXTS = {".csv", ".tsv", ".txt", ".json", ".jsonl"}
EXCEL_EXTS = {".xlsx", ".xlsm", ".xls", ".ods"}
HEADER_ONLY_EXTS = DELIMITED_EXTS            # kept: older name used elsewhere
SCANNABLE_EXTS = PARQUET_EXTS | DELIMITED_EXTS | EXCEL_EXTS


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class FileEntry:
    """One file on disk, after a metadata-only read."""
    path: str            # unique id — "<file>::<sheet>" for multi-sheet workbooks
    name: str
    folder: str
    ext: str
    source_path: str = ""   # the real file on disk
    sheet: str = ""         # Excel sheet name, empty otherwise
    size_bytes: int = 0
    mtime: float = 0.0
    row_count: int | None = None        # None = not knowable from metadata (CSV/JSON)
    # How row_count was obtained. Provenance matters: a figure read from a
    # parquet footer and a figure produced by scanning a CSV are both exact,
    # but they are not the same claim, and a sworn inventory should be able
    # to say which one it is making.
    row_count_source: str = ""          # "footer" | "scan" | "unknown"
    n_fields: int = 0
    # Per-row-group rows + compressed bytes, from the footer. This is what
    # makes sampling possible without a full scan: a row group is the
    # smallest unit parquet lets you read, so the sampler needs to know how
    # many rows each one holds and what reading it will cost.
    row_groups: list = field(default_factory=list)   # [{"rows": int, "bytes": int}]
    fp_strict: str = ""                 # name + type + order
    fp_names: str = ""                  # ordered names only
    fp_nameset: str = ""                # name SET — order-insensitive
    columns: list = field(default_factory=list)   # [{"name","type","physical"}]
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class SchemaRecord:
    """
    A schema, persisted independently of any file.

    label is user-editable and is what shows up as the Dataset name;
    canonical_of is set when this schema has been folded into another
    one by a manual merge.
    """
    schema_id: str
    fp_strict: str
    fp_names: str
    fp_nameset: str
    n_fields: int
    columns: list = field(default_factory=list)
    label: str = ""
    batch: str = ""                     # e.g. "IRM-018" — free text, groups schemas by production
    first_seen: str = ""
    note: str = ""
    retained: bool = True               # kept even with 0 files present


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def _sha(parts: list[str]) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:12]


def fingerprints(columns: list[dict]) -> tuple[str, str, str]:
    """
    Three fingerprints from one column list:
      strict  — name + logical type, in order  (the grouping key)
      names   — names in order, types ignored
      nameset — sorted name set, order AND types ignored

    Computing all three up front is what lets the UI say "these two
    groups differ only by type" or "...only by column order" instead of
    just showing two opaque groups the analyst has to diff by hand.
    """
    strict = _sha([f'{c["name"]}:{c["type"]}' for c in columns])
    names = _sha([c["name"] for c in columns])
    nameset = _sha(sorted(c["name"] for c in columns))
    return strict, names, nameset


# ---------------------------------------------------------------------------
# Folder listing (no metadata read at all — just stat)
# ---------------------------------------------------------------------------

def list_folder(folder: str, exts: set[str] | None = None) -> list[dict]:
    """
    Non-recursive listing of one folder. Cheap: os.scandir + stat only,
    no file is opened. This is what powers the "preview before you scan"
    step — you see the file list and sizes before committing to reading
    499 footers.
    """
    exts = exts or SCANNABLE_EXTS
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    out = []
    with os.scandir(p) as it:
        for e in it:
            if not e.is_file():
                continue
            ext = Path(e.name).suffix.lower()
            if ext not in exts:
                continue
            st = e.stat()
            out.append({
                "path": str(Path(e.path)),
                "name": e.name,
                "folder": str(p),
                "ext": ext,
                "size_bytes": st.st_size,
                "mtime": st.st_mtime,
            })
    return sorted(out, key=lambda r: r["name"])


# ---------------------------------------------------------------------------
# Metadata read (footer only)
# ---------------------------------------------------------------------------

def _infer_type(values: list) -> str:
    """
    Crude type name from sample values, for file formats that carry no
    schema of their own (Excel, and anything else read cell by cell).
    Deliberately coarse: this feeds the schema fingerprint, so it has to
    be stable across files, and a finer-grained guess would split groups
    on nothing more than whether one file happened to have a decimal in
    the first 200 rows.
    """
    import datetime as _dt

    kinds = set()
    for v in values:
        if v is None or v == "":
            continue
        if isinstance(v, bool):
            kinds.add("boolean")
        elif isinstance(v, int):
            kinds.add("int64")
        elif isinstance(v, float):
            kinds.add("double")
        elif isinstance(v, (_dt.datetime, _dt.date)):
            kinds.add("timestamp")
        else:
            kinds.add("varchar")
    if not kinds:
        return "empty"
    if kinds == {"int64"}:
        return "int64"
    if kinds <= {"int64", "double"}:
        return "double"
    if len(kinds) == 1:
        return kinds.pop()
    return "varchar"          # mixed -> the only honest common type


def read_parquet_metadata(path: str) -> list[dict]:
    """
    Read ONE parquet footer. No row group, no column chunk, no data.
    Row count and column types are exact and free — this is the only
    format that gives them without reading the file.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    md = pf.metadata
    arrow_schema = pf.schema_arrow
    parquet_schema = pf.schema

    physical = {}
    for i in range(len(parquet_schema)):
        col = parquet_schema.column(i)
        physical[col.name] = col.physical_type

    columns = [
        {"name": f.name, "type": str(f.type), "physical": physical.get(f.name, "")}
        for f in arrow_schema
    ]
    row_groups = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        row_groups.append({"rows": rg.num_rows, "bytes": rg.total_byte_size})

    return [{
        "part": "", "row_count": md.num_rows, "row_count_source": "footer",
        "columns": columns, "row_groups": row_groups,
    }]


def _duckdb_reader(path: str, ext: str) -> str:
    """DuckDB reader expression for one delimited/JSON file."""
    esc = path.replace("'", "''")
    if ext == ".csv":
        return f"read_csv_auto('{esc}', sample_size=-1)"
    if ext in (".tsv", ".txt"):
        return f"read_csv_auto('{esc}', sample_size=-1, delim='\\t')"
    if ext in (".json", ".jsonl"):
        return f"read_json_auto('{esc}')"
    raise ValueError(f"No DuckDB reader for {ext}")


def read_delimited(path: str, ext: str) -> list[dict]:
    """
    Columns + EXACT row count for CSV/TSV/JSON.

    These formats have no footer, so the row count costs one full pass
    over the bytes. There is no correct shortcut: counting newlines is
    faster but wrong, because RFC-4180 permits newlines inside quoted
    fields — precisely the messy data where the number matters. DuckDB
    parses properly and materialises nothing, so this is a streaming
    read, not a load.
    """
    import duckdb

    reader = _duckdb_reader(path, ext)
    con = duckdb.connect()
    try:
        desc = con.execute(f"DESCRIBE SELECT * FROM {reader} LIMIT 0").fetchall()
        columns = [{"name": r[0], "type": str(r[1]).lower(), "physical": ""} for r in desc]
        n = con.execute(f"SELECT COUNT(*) FROM {reader}").fetchone()[0]
    finally:
        con.close()
    return [{"part": "", "row_count": n, "row_count_source": "scan",
             "columns": columns, "row_groups": []}]


def read_excel(path: str, ext: str, type_sample_rows: int = 200) -> list[dict]:
    """
    One entry PER SHEET, because a workbook is not a dataset — a 5-sheet
    workbook holding five different layouts is five schemas, and folding
    them into one would misreport both the field count and the row count.

    Single streaming pass per sheet with openpyxl read_only: header from
    row 1, types inferred from the first `type_sample_rows` data rows,
    row count from iterating to the end. ws.max_row is deliberately NOT
    trusted — writers other than Excel routinely omit or misstate the
    dimension record.
    """
    out = []

    if ext in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook

        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            for ws in wb.worksheets:
                header = None
                n_rows = 0
                samples: dict[int, list] = {}
                for row in ws.iter_rows(values_only=True):
                    if header is None:
                        header = ["" if c is None else str(c) for c in row]
                        continue
                    if all(c is None for c in row):
                        continue          # trailing blank rows openpyxl still yields
                    n_rows += 1
                    if n_rows <= type_sample_rows:
                        for i, c in enumerate(row):
                            samples.setdefault(i, []).append(c)
                if header is None:
                    out.append({"part": ws.title, "row_count": 0, "row_count_source": "scan",
                                "columns": [], "row_groups": []})
                    continue
                columns = [{"name": (h or f"column_{i+1}"),
                            "type": _infer_type(samples.get(i, [])), "physical": ""}
                           for i, h in enumerate(header)]
                out.append({"part": ws.title, "row_count": n_rows,
                            "row_count_source": "scan", "columns": columns,
                            "row_groups": []})
        finally:
            wb.close()
        return out

    # Legacy .xls and .ods have no streaming reader — pandas loads the whole
    # sheet into memory. Fine for the sizes these formats can hold at all.
    import pandas as pd

    engine = "xlrd" if ext == ".xls" else "odf"
    sheets = pd.read_excel(path, sheet_name=None, engine=engine)
    for name, df in sheets.items():
        columns = [{"name": str(c), "type": str(df[c].dtype), "physical": ""}
                   for c in df.columns]
        out.append({"part": str(name), "row_count": int(len(df)),
                    "row_count_source": "scan", "columns": columns, "row_groups": []})
    return out


def count_rows_scan(path: str, ext: str) -> int:
    """Exact row count for one file, by reading it. Kept for re-counting."""
    import duckdb

    con = duckdb.connect()
    try:
        return con.execute(f"SELECT COUNT(*) FROM {_duckdb_reader(path, ext)}").fetchone()[0]
    finally:
        con.close()


def scan_file(rec: dict) -> list[FileEntry]:
    """
    Metadata scan of a single listed file. Returns a LIST because one file
    can hold more than one dataset (an Excel workbook's sheets). Never
    raises — a failure comes back as a single entry carrying the error.
    """
    ext = rec["ext"]
    base = dict(path=rec["path"], name=rec["name"], folder=rec["folder"], ext=ext,
                size_bytes=rec["size_bytes"], mtime=rec["mtime"])

    try:
        if ext in PARQUET_EXTS:
            parts = read_parquet_metadata(rec["path"])
        elif ext in DELIMITED_EXTS:
            parts = read_delimited(rec["path"], ext)
        elif ext in EXCEL_EXTS:
            parts = read_excel(rec["path"], ext)
        else:
            raise ValueError(f"Unsupported extension: {ext}")
    except Exception as e:
        entry = FileEntry(**base, source_path=rec["path"])
        entry.error = f"{type(e).__name__}: {e}"
        return [entry]

    entries = []
    multi = len(parts) > 1
    for part in parts:
        sheet = part.get("part", "")
        entry = FileEntry(
            **{**base, "path": f'{rec["path"]}::{sheet}' if sheet and multi else rec["path"]},
            source_path=rec["path"], sheet=sheet,
        )
        entry.row_count = part["row_count"]
        entry.row_count_source = part["row_count_source"]
        entry.columns = part["columns"]
        entry.n_fields = len(part["columns"])
        entry.row_groups = part.get("row_groups", [])
        entry.fp_strict, entry.fp_names, entry.fp_nameset = fingerprints(part["columns"])
        entries.append(entry)
    return entries


def scan_files(records: list[dict], cache: dict | None = None,
               progress=None) -> tuple[list[FileEntry], dict]:
    """
    Scan a list of files, reusing cached results where size and mtime are
    unchanged. Cache values are LISTS of entries per file, since one file
    can yield several.

    Cost is not uniform and the caller should say so in the UI: parquet is
    a footer read (milliseconds), everything else is a full pass over the
    file to get an exact row count.
    """
    cache = dict(cache or {})
    entries: list[FileEntry] = []
    total = len(records)

    for i, rec in enumerate(records):
        key = rec["path"]
        hit = cache.get(key)
        fresh = (isinstance(hit, list) and hit
                 and hit[0].get("size_bytes") == rec["size_bytes"]
                 and hit[0].get("mtime") == rec["mtime"])
        if fresh:
            got = [FileEntry(**d) for d in hit]
        else:
            got = scan_file(rec)
            cache[key] = [asdict(e) for e in got]
        entries.extend(got)
        if progress:
            progress(i + 1, total, rec["name"])

    return entries, cache


# ---------------------------------------------------------------------------
# Native folder picker
# ---------------------------------------------------------------------------

# Streamlit is a web app: the browser cannot hand us an absolute path on the
# user's machine. A real folder dialog is only possible because this tool is
# run locally (`streamlit run app.py`), so the server process IS the user's
# machine. The dialog opens wherever the SERVER runs — if this app is ever
# containerised or hosted for shared internal use, the dialog would open on
# the host and the caller would just hang. Hence: availability is checked
# first, failure is reported plainly, and the manual path field always stays
# as the fallback.
#
# The dialog runs in a SUBPROCESS, not in-process. Streamlit executes the
# script on a ScriptRunner thread, and driving Tk from a non-main thread is
# unreliable enough to take down the whole server. A subprocess isolates it:
# the worst case is a non-zero exit code and an error string.

_PICKER_SCRIPT = r"""
import sys
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception as e:
    sys.stderr.write("tkinter unavailable: %s" % e)
    raise SystemExit(2)

root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)   # else it can open behind the browser
except Exception:
    pass
initial = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
path = filedialog.askdirectory(title="Select a production folder",
                               initialdir=initial, mustexist=True)
try:
    root.destroy()
except Exception:
    pass
sys.stdout.write(path or "")
"""


def picker_available() -> tuple[bool, str]:
    """
    Can we realistically open a native dialog on this machine?
    Returns (available, reason_if_not).
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("tkinter") is None:
        return False, ("tkinter is not installed with this Python — on Linux, "
                       "`sudo apt install python3-tk`; on Windows/macOS use a "
                       "python.org build.")
    # A GUI needs a display server. Windows and macOS always have one; Linux
    # (and therefore any container) needs DISPLAY or WAYLAND_DISPLAY set.
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, ("no display detected — this looks like a headless or "
                       "containerised server, so a dialog would have nowhere "
                       "to open. Type or paste the path instead.")
    return True, ""


def pick_directory(initial: str = "", timeout: int = 300) -> tuple[str, str]:
    """
    Open a native folder dialog on the machine running Streamlit.
    Returns (selected_path, error). Both empty means the user cancelled.

    timeout is generous — someone browsing a network drive may take a while,
    and the dialog is modal only to itself, not to the app.
    """
    import subprocess
    import sys

    ok, why = picker_available()
    if not ok:
        return "", why

    kwargs = {}
    if sys.platform == "win32":
        # Suppress the console window that would otherwise flash on Windows
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PICKER_SCRIPT, initial or ""],
            capture_output=True, text=True, timeout=timeout, **kwargs,
        )
    except subprocess.TimeoutExpired:
        return "", f"The folder dialog was left open for over {timeout}s — cancelled."
    except Exception as e:
        return "", f"Could not launch the folder dialog: {type(e).__name__}: {e}"

    if proc.returncode != 0:
        return "", (proc.stderr or "").strip() or f"Dialog exited with code {proc.returncode}."

    return proc.stdout.strip(), ""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _ensure_dir():
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def load_schemas() -> dict[str, SchemaRecord]:
    raw = _read_json(SCHEMAS_PATH, {})
    return {k: SchemaRecord(**v) for k, v in raw.items()}


def save_schemas(schemas: dict[str, SchemaRecord]) -> None:
    _ensure_dir()
    SCHEMAS_PATH.write_text(json.dumps(
        {k: asdict(v) for k, v in schemas.items()}, indent=2))


DEFAULT_STATE = {
    "folders": [],          # list of folder paths (non-recursive each)
    "excluded": [],         # file paths the analyst removed from the inventory
    "merges": {},           # schema_id -> {"into": schema_id, "note": str, "at": ts}
    "file_overrides": {},   # file path -> {"schema_id": str, "note": str, "at": ts}
    "batch_label": "",
}


def load_state() -> dict:
    st = _read_json(STATE_PATH, {})
    out = dict(DEFAULT_STATE)
    out.update(st)
    return out


def save_state(state: dict) -> None:
    _ensure_dir()
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_cache() -> dict:
    return _read_json(CACHE_PATH, {})


def save_cache(cache: dict) -> None:
    _ensure_dir()
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _cache_entries(value) -> list[dict]:
    """
    Normalise one cache value into a list of entry dicts.

    The cache used to hold a single dict per file. It now holds a list,
    because one file can yield several entries (an Excel workbook's
    sheets). Old cache files are still readable rather than throwing —
    a stale cache is a performance problem, not a reason to crash.
    """
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def entries_from_cache(state: dict) -> list[FileEntry]:
    """
    Rebuild scanned entries from the cache, without re-reading anything.

    This is THE way both tabs load entries. It lives here rather than in
    each UI file because the cache's shape is this module's business —
    duplicating the parsing is what let a format change break one tab and
    silently leave the other working.

    Exclusions are applied on both the entry id and the file it came from,
    so excluding a workbook in the file list excludes all of its sheets.
    """
    cache = load_cache()
    excluded = set(state.get("excluded", []))
    entries: list[FileEntry] = []

    for folder in state.get("folders", []):
        try:
            listing = list_folder(folder)
        except Exception:
            continue
        for rec in listing:
            if rec["path"] in excluded:
                continue
            for d in _cache_entries(cache.get(rec["path"])):
                try:
                    e = FileEntry(**d)
                except TypeError:
                    continue        # entry written by an older field set
                if e.path in excluded or (e.source_path or e.path) in excluded:
                    continue
                entries.append(e)
    return entries


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

def _next_schema_id(schemas: dict) -> str:
    n = 1
    while f"S{n:02d}" in schemas:
        n += 1
    return f"S{n:02d}"


def register_schemas(entries: list[FileEntry], schemas: dict[str, SchemaRecord],
                     batch: str = "") -> tuple[dict[str, SchemaRecord], list[str]]:
    """
    Ensure every distinct strict fingerprint in `entries` has a schema
    record. Existing records are matched by fingerprint and left alone
    (labels and notes the analyst wrote are preserved). Returns the
    updated registry and the list of newly created schema_ids.
    """
    by_fp = {s.fp_strict: sid for sid, s in schemas.items()}
    new_ids = []

    for e in entries:
        if not e.ok or not e.fp_strict or e.fp_strict in by_fp:
            continue
        sid = _next_schema_id(schemas)
        schemas[sid] = SchemaRecord(
            schema_id=sid,
            fp_strict=e.fp_strict, fp_names=e.fp_names, fp_nameset=e.fp_nameset,
            n_fields=e.n_fields, columns=e.columns,
            label="", batch=batch, first_seen=_now(),
        )
        by_fp[e.fp_strict] = sid
        new_ids.append(sid)

    return schemas, new_ids


def fingerprint_index(schemas: dict[str, SchemaRecord]) -> dict[str, str]:
    """
    fp_strict -> schema_id, built once.

    Without this, schema_id_for_entry scans every schema for every entry,
    and any caller looping over schemas turns that into O(schemas² × files).
    At 500 files and 500 schemas that measured 2.7s per Streamlit rerun —
    i.e. 2.7s of lag on every single click. Callers in a loop MUST pass an
    index.
    """
    return {sc.fp_strict: sid for sid, sc in schemas.items()}


def schema_id_for_entry(e: FileEntry, schemas: dict[str, SchemaRecord],
                        state: dict, index: dict[str, str] | None = None) -> str | None:
    """
    Which schema does this file report under? A manual file override wins
    over the file's own fingerprint — that's the "assign this file the
    schema we already have for this batch" case.

    Pass `index` (from fingerprint_index) when calling this in a loop.
    """
    ov = state.get("file_overrides", {}).get(e.path)
    if ov and ov.get("schema_id") in schemas:
        return ov["schema_id"]
    if index is not None:
        return index.get(e.fp_strict)
    for sid, sc in schemas.items():
        if sc.fp_strict == e.fp_strict:
            return sid
    return None


def count_files_by_schema(entries: list[FileEntry], schemas: dict[str, SchemaRecord],
                          state: dict) -> dict[str, int]:
    """
    sid -> number of included entries reporting under it, in ONE pass over
    the entries. Replaces the per-schema re-scan the schema library used to
    do, which was the quadratic hot spot at 500 schemas.
    """
    index = fingerprint_index(schemas)
    excluded = set(state.get("excluded", []))
    counts = {sid: 0 for sid in schemas}
    for e in entries:
        if not e.ok:
            continue
        if e.path in excluded or (e.source_path or e.path) in excluded:
            continue
        sid = schema_id_for_entry(e, schemas, state, index=index)
        if sid in counts:
            counts[sid] += 1
    return counts


def resolve_group(schema_id: str, state: dict) -> str:
    """Follow merge pointers to the canonical schema_id (cycle-safe)."""
    merges = state.get("merges", {})
    seen = set()
    cur = schema_id
    while cur in merges and cur not in seen:
        seen.add(cur)
        nxt = merges[cur].get("into")
        if not nxt or nxt == cur:
            break
        cur = nxt
    return cur


# ---------------------------------------------------------------------------
# Schema comparison
# ---------------------------------------------------------------------------

def diff_schemas(a: SchemaRecord, b: SchemaRecord) -> dict:
    """
    Explain why two schemas aren't identical. Used both for the
    merge-suggestion list and for the note attached to a manual merge.
    """
    a_names = [c["name"] for c in a.columns]
    b_names = [c["name"] for c in b.columns]
    a_types = {c["name"]: c["type"] for c in a.columns}
    b_types = {c["name"]: c["type"] for c in b.columns}

    only_a = [n for n in a_names if n not in b_types]
    only_b = [n for n in b_names if n not in a_types]
    type_diff = [
        {"column": n, "a": a_types[n], "b": b_types[n]}
        for n in a_names if n in b_types and a_types[n] != b_types[n]
    ]
    shared_a = [n for n in a_names if n in b_types]
    shared_b = [n for n in b_names if n in a_types]
    order_differs = shared_a != shared_b

    if not only_a and not only_b:
        if type_diff and not order_differs:
            kind = "types_only"
        elif order_differs and not type_diff:
            kind = "order_only"
        elif type_diff and order_differs:
            kind = "types_and_order"
        else:
            kind = "identical"
    else:
        kind = "columns_differ"

    return {
        "kind": kind,
        "only_in_a": only_a, "only_in_b": only_b,
        "type_diff": type_diff, "order_differs": order_differs,
    }


MAX_MERGE_SUGGESTIONS = 200


def suggest_merges(schemas: dict[str, SchemaRecord], active_ids: list[str],
                   state: dict, limit: int = MAX_MERGE_SUGGESTIONS) -> list[dict]:
    """
    Candidate merges: pairs of currently-active schemas whose column
    NAMES match (as a set) but whose strict fingerprints differ — i.e.
    the groups a strict fingerprint split on type or column order alone.

    These are suggestions only. Nothing is merged without an explicit
    click, because merging two non-identical schemas contradicts the
    "one identical schema per dataset" definition and has to be a
    deliberate, noted decision.
    """
    merges = state.get("merges", {})
    ids = [i for i in active_ids if i not in merges and i in schemas]

    # Bucket by nameset FIRST. A candidate pair must share a nameset by
    # definition, so comparing across buckets is wasted work — this is what
    # keeps the cost proportional to (bucket size)² instead of (all
    # schemas)². At 500 schemas the naive version took 8.4s.
    buckets: dict[str, list[str]] = {}
    for sid in ids:
        buckets.setdefault(schemas[sid].fp_nameset, []).append(sid)

    out = []
    truncated = False
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i, a_id in enumerate(bucket):
            for b_id in bucket[i + 1:]:
                a, b = schemas[a_id], schemas[b_id]
                if a.fp_strict == b.fp_strict:
                    continue
                if len(out) >= limit:
                    # A pathological production (500 schemas sharing one
                    # nameset) yields 124,750 pairs. Computing them is slow;
                    # rendering one expander each would hang the browser.
                    # Cap it and say so rather than pretending to list them.
                    truncated = True
                    break
                out.append({"a": a_id, "b": b_id, "diff": diff_schemas(a, b)})
            if truncated:
                break
        if truncated:
            break

    if truncated:
        out.append({"a": None, "b": None, "diff": None, "truncated": True,
                    "note": (f"More than {limit} candidate pairs share a column-name set. "
                             f"Showing the first {limit}. Merge some, or group by batch, "
                             f"to narrow the list.")})
    return out


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_datasets(entries: list[FileEntry], schemas: dict[str, SchemaRecord],
                   state: dict) -> list[dict]:
    """
    Group the included files into reportable Datasets.

    Rules that matter for the output:

      * Totals (files, bytes, rows) come ONLY from files currently
        included in the scan. A schema whose files were all removed
        still appears, with 0 files — flagged, not silently dropped and
        not silently contributing an empty row to a sworn inventory.

      * Row count is reported as None if ANY included file's row count
        is unknown (a CSV, or a footer that failed to read). A partial
        sum presented as a total is worse than an admitted gap here, so
        the unknown count is carried in `rows_unknown_files` instead.

      * n_fields for a manually merged group is taken from the canonical
        schema, and `merged_from` + `merge_diffs` record what was folded
        in. Downstream (the JSON, and your Notes column) gets the diff
        so the number is defensible.
    """
    excluded = set(state.get("excluded", []))
    # A workbook is excluded as a file, but its entries are per sheet, so
    # match on either the entry id or the file it came from.
    included = [e for e in entries
                if e.path not in excluded and (e.source_path or e.path) not in excluded]
    fp_index = fingerprint_index(schemas)

    groups: dict[str, dict] = {}

    def _blank(gid: str) -> dict:
        canon = schemas.get(gid)
        return {
            "group_id": gid,
            "label": (canon.label if canon and canon.label else gid),
            "batch": canon.batch if canon else "",
            "n_fields": canon.n_fields if canon else 0,
            "n_files": 0,
            "total_bytes": 0,
            "row_count": 0,
            "rows_unknown_files": 0,
            "files": [],
            "schemas_included": [],
            "overridden_files": [],
            "errors": [],
        }

    # Every non-merged, retained schema gets a slot up front — this is
    # what keeps a 0-file retained schema visible.
    for sid, s in schemas.items():
        if not s.retained:
            continue
        gid = resolve_group(sid, state)
        groups.setdefault(gid, _blank(gid))
        if gid not in groups[gid]["schemas_included"]:
            groups[gid]["schemas_included"].append(gid)
        if sid != gid and sid not in groups[gid]["schemas_included"]:
            groups[gid]["schemas_included"].append(sid)

    unassigned = []
    for e in included:
        if not e.ok:
            gid = "(unreadable)"
            g = groups.setdefault(gid, _blank(gid))
            g["errors"].append({"file": e.name, "error": e.error})
            g["n_files"] += 1
            g["total_bytes"] += e.size_bytes
            g["files"].append(e.path)
            continue

        sid = schema_id_for_entry(e, schemas, state, index=fp_index)
        if sid is None:
            unassigned.append(e)
            continue

        gid = resolve_group(sid, state)
        g = groups.setdefault(gid, _blank(gid))
        g["n_files"] += 1
        g["total_bytes"] += e.size_bytes
        g["files"].append(e.path)
        if e.row_count is None:
            g["rows_unknown_files"] += 1
        else:
            g["row_count"] += e.row_count
            g.setdefault("row_count_sources", set()).add(e.row_count_source or "footer")
        if sid not in g["schemas_included"]:
            g["schemas_included"].append(sid)

        ov = state.get("file_overrides", {}).get(e.path)
        if ov:
            g["overridden_files"].append({
                "file": e.name, "assigned_to": sid,
                "actual_fingerprint": e.fp_strict, "note": ov.get("note", ""),
            })

    if unassigned:
        g = groups.setdefault("(unregistered)", _blank("(unregistered)"))
        for e in unassigned:
            g["n_files"] += 1
            g["total_bytes"] += e.size_bytes
            g["files"].append(e.path)
            if e.row_count is None:
                g["rows_unknown_files"] += 1
            else:
                g["row_count"] += e.row_count

    # Post-process: merge provenance, unknown-row handling, flags
    out = []
    for gid, g in groups.items():
        merged_from = [s for s in g["schemas_included"] if s != gid]
        g["merged_from"] = merged_from
        g["merge_diffs"] = []
        canon = schemas.get(gid)
        for sid in merged_from:
            other = schemas.get(sid)
            if canon and other:
                d = diff_schemas(canon, other)
                if d["kind"] != "identical":
                    g["merge_diffs"].append({"schema_id": sid, **d})

        if g["rows_unknown_files"]:
            g["row_count_partial"] = g["row_count"]
            g["row_count"] = None

        g["flags"] = []
        if g["n_files"] == 0:
            g["flags"].append("schema retained, 0 files currently included")
        if merged_from:
            g["flags"].append(f"manually merged from {len(merged_from)} other schema(s)")
        if g["merge_diffs"]:
            g["flags"].append("merged schemas are NOT identical — see merge_diffs")
        if g["overridden_files"]:
            g["flags"].append(f"{len(g['overridden_files'])} file(s) manually reassigned")
        if g["rows_unknown_files"]:
            g["flags"].append(f"row count unknown for {g['rows_unknown_files']} file(s)")
        if g.get("row_count_sources"):
            srcs = sorted(g["row_count_sources"])
            g["row_count_sources"] = srcs   # set -> list, so the dict stays JSON-safe
            g["row_count_method"] = " + ".join(srcs)
            if "scan" in srcs and "footer" in srcs:
                g["flags"].append("row count is part footer, part full scan — mixed provenance")
        if g["errors"]:
            g["flags"].append(f"{len(g['errors'])} file(s) failed metadata read")
        out.append(g)

    # Files first, then size — 0-file retained schemas sort to the bottom.
    # Only groups WITH files get a Dataset number: a numbered row with no
    # files reads as a produced dataset in the final inventory, which it
    # isn't. Empty retained schemas keep dataset_no=None and are reported
    # separately.
    out.sort(key=lambda g: (-g["n_files"], -g["total_bytes"], g["group_id"]))
    n = 0
    for g in out:
        if g["n_files"] > 0:
            n += 1
            g["dataset_no"] = n
        else:
            g["dataset_no"] = None
    return out


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def to_json_payload(datasets: list[dict], entries: list[FileEntry],
                    schemas: dict[str, SchemaRecord], state: dict,
                    include_file_detail: bool = True) -> dict:
    """
    Everything needed to fill the Inventory sheet by hand, plus the
    provenance to defend it. No Excel is written and no Excel is read —
    this is a plain JSON handoff.
    """
    excluded = set(state.get("excluded", []))
    payload = {
        "generated_at": _now(),
        "batch_label": state.get("batch_label", ""),
        "method": "metadata only — parquet footer + filesystem stat; no data pages read",
        "grouping": "strict fingerprint (column name + logical type + order), with manual overrides applied",
        "folders": state.get("folders", []),
        "totals": {
            "datasets": len([d for d in datasets if d["n_files"] > 0]),
            "files": sum(d["n_files"] for d in datasets),
            "total_bytes": sum(d["total_bytes"] for d in datasets),
            "rows": (None if any(d["row_count"] is None for d in datasets)
                     else sum(d["row_count"] for d in datasets)),
            "excluded_files": len(excluded),
        },
        "datasets": [],
        "schemas": {},
        "manual_overrides": {
            "merges": state.get("merges", {}),
            "file_overrides": state.get("file_overrides", {}),
            "excluded_files": sorted(excluded),
        },
    }

    for d in datasets:
        row = {
            "dataset": (f"Dataset{d['dataset_no']}" if d["dataset_no"]
                        else "(retained schema, 0 files)"),
            "label": d["label"],
            "batch": d["batch"],
            "files": d["n_files"],
            "size_bytes": d["total_bytes"],
            "size_gb": round(d["total_bytes"] / 1024 ** 3, 3),
            "size_tb": round(d["total_bytes"] / 1024 ** 4, 4),
            "row_count": d["row_count"],
            "row_count_partial": d.get("row_count_partial"),
            "row_count_method": d.get("row_count_method", "footer"),
            "data_fields": d["n_fields"],
            "schema_id": d["group_id"],
            "merged_from": d["merged_from"],
            "merge_diffs": d["merge_diffs"],
            "reassigned_files": d["overridden_files"],
            "flags": d["flags"],
        }
        if include_file_detail:
            row["file_names"] = [Path(p).name for p in sorted(d["files"])]
        payload["datasets"].append(row)

    for sid, s in schemas.items():
        payload["schemas"][sid] = {
            "label": s.label, "batch": s.batch, "n_fields": s.n_fields,
            "first_seen": s.first_seen, "note": s.note, "retained": s.retained,
            "fingerprint_strict": s.fp_strict,
            "columns": s.columns,
        }
    return payload
