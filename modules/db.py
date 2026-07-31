"""
DuckDB connection management.

We use a single persistent .duckdb file as the "database". Every cleaned
table an analyst loads gets registered here as a real table, so later
phases (query builder, joins, exports) can just run SQL against it.
"""

import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "warehouse.duckdb"
TEMP_DIR = Path(__file__).parent.parent / "duckdb_spill"

# Leave headroom for the OS + Streamlit itself rather than letting DuckDB
# claim everything. Adjust MEMORY_LIMIT_GB to roughly (total RAM - 8-16GB).
MEMORY_LIMIT_GB = 48


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Return a connection to the persistent local warehouse, with memory
    capped and a spill directory configured so queries over data larger
    than RAM complete (slower, via disk) instead of crashing.
    """
    con = duckdb.connect(str(DB_PATH))
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{MEMORY_LIMIT_GB}GB'")
    con.execute(f"SET temp_directory='{TEMP_DIR}'")
    return con


def register_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """
    Write a cleaned pandas dataframe into DuckDB as a permanent table.
    Overwrites if a table with the same name already exists.
    """
    safe_name = _safe_table_name(name)
    con.register("tmp_df", df)
    con.execute(f'CREATE OR REPLACE TABLE "{safe_name}" AS SELECT * FROM tmp_df')
    con.unregister("tmp_df")


def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in con.execute("SHOW TABLES").fetchall()]


def get_table_df(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute(f'SELECT * FROM "{name}"').df()


def drop_table(con: duckdb.DuckDBPyConnection, name: str) -> None:
    con.execute(f'DROP TABLE IF EXISTS "{name}"')

def rename_table(con: duckdb.DuckDBPyConnection, old_name: str, new_name: str) -> None:
    """Rename a table in place. Runs the new name through the same
    sanitizer used at creation time, so renaming can't produce an
    invalid SQL identifier."""
    safe_new = _safe_table_name(new_name)
    con.execute(f'ALTER TABLE "{old_name}" RENAME TO "{safe_new}"')    

def convert_to_parquet(source_path: str, filetype: str, dest_path: str) -> str:
    """
    One-time conversion of a large CSV/JSON file to Parquet. Parquet is
    columnar + compressed, so subsequent queries only read the columns
    and row-groups they need instead of scanning the whole file — this
    is the single biggest speed lever at 50GB+.
    Uses a throwaway connection (COPY doesn't need the warehouse).
    """
    src_escaped = source_path.replace("'", "''")
    dest_escaped = dest_path.replace("'", "''")
    reader = f"read_csv_auto('{src_escaped}', sample_size=-1)" if filetype in ("csv", "tsv") \
        else f"read_json_auto('{src_escaped}')"

    tmp_con = duckdb.connect()
    tmp_con.execute(f"SET memory_limit='{MEMORY_LIMIT_GB}GB'")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_con.execute(f"SET temp_directory='{TEMP_DIR}'")
    tmp_con.execute(f"COPY (SELECT * FROM {reader}) TO '{dest_escaped}' (FORMAT PARQUET)")
    tmp_con.close()
    return dest_path


def register_table_from_path(con: duckdb.DuckDBPyConnection, name: str, path: str, filetype: str) -> int:
    """
    Load a file directly into DuckDB using its native readers — streams
    from disk rather than materializing the whole file in pandas/RAM
    first. This is the path large CSV/Parquet/JSON files should take;
    Excel has no native DuckDB reader, so it still goes through pandas
    (register_table) regardless of size.
    Returns the resulting row count.
    """
    safe_name = _safe_table_name(name)
    path_escaped = path.replace("'", "''")

    if filetype == "csv":
        reader = f"read_csv_auto('{path_escaped}', sample_size=-1)"
    elif filetype == "parquet":
        reader = f"read_parquet('{path_escaped}')"
    elif filetype == "json":
        reader = f"read_json_auto('{path_escaped}')"
    else:
        raise ValueError(f"No native DuckDB reader for filetype: {filetype}")

    con.execute(f'CREATE OR REPLACE TABLE "{safe_name}" AS SELECT * FROM {reader}')
    return con.execute(f'SELECT COUNT(*) FROM "{safe_name}"').fetchone()[0]


def get_columns(con: duckdb.DuckDBPyConnection, name: str) -> list[str]:
    """Column names only, via DESCRIBE — never materializes the table."""
    return [r[0] for r in con.execute(f'DESCRIBE "{name}"').fetchall()]


def count_rows(con: duckdb.DuckDBPyConnection, name: str) -> int:
    """Row count via SQL — never materializes the table."""
    return con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]


def count_columns(con: duckdb.DuckDBPyConnection, name: str) -> int:
    return len(con.execute(f'DESCRIBE "{name}"').fetchall())


def preview_table(con: duckdb.DuckDBPyConnection, name: str, n: int = 200) -> pd.DataFrame:
    """Pull only a small sample into pandas — for UI preview, not full profiling."""
    return con.execute(f'SELECT * FROM "{name}" LIMIT {n}').df()


def _safe_table_name(name: str) -> str:
    """
    Sanitize a table name into a safe SQL identifier. Does NOT strip
    file extensions (that's the caller's job, e.g. via rsplit on the
    original filename) — this used to call Path(name).stem, which
    silently truncated any table name containing a dot, e.g.
    'sales.q1_2026' -> 'sales'. Fixed to sanitize the whole string.
    """
    cleaned = "".join(c if c.isalnum() else "_" for c in name)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned or "unnamed_table"
