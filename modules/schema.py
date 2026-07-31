"""
Schema inference: candidate keys, foreign key relationships, Mermaid ER diagram.

Heuristics only — always show these as *suggestions* the analyst can
confirm or correct, never as ground truth.
"""

import pandas as pd
from dataclasses import dataclass

UNIQUENESS_THRESHOLD = 0.98   # column is a PK candidate if this unique
OVERLAP_THRESHOLD = 0.90      # FK candidate if this much value overlap


def _mermaid_safe(text: str) -> str:
    """
    Sanitize a column name or type for use inside a Mermaid erDiagram
    attribute line. Mermaid's attribute syntax breaks on spaces,
    parentheses, '#', etc. — real column names like 'SR #' or types
    like 'DECIMAL(18,2)' will otherwise produce a 'Syntax error in
    text' render failure. Diagram labels don't need to be the exact
    original string (the real name is still shown in the app's table
    previews) — just readable and syntactically safe.
    """
    cleaned = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(text))
    cleaned = cleaned.strip("_") or "col"
    if cleaned[0].isdigit():
        cleaned = f"c_{cleaned}"
    return cleaned


@dataclass
class Relationship:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    overlap_pct: float


def detect_primary_key(df: pd.DataFrame) -> str | None:
    """Return the column name most likely to be this table's primary key."""
    n = len(df)
    if n == 0:
        return None
    best_col, best_ratio = None, 0.0
    for col in df.columns:
        ratio = df[col].nunique(dropna=True) / n
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio
    if best_ratio >= UNIQUENESS_THRESHOLD:
        return best_col
    return None


def detect_relationships(tables: dict[str, pd.DataFrame]) -> list[Relationship]:
    """
    Compare every column pair across tables and flag likely FK -> PK
    relationships based on value-set overlap. O(n^2) in column count —
    fine for the table counts this tool targets (tens, not thousands).
    """
    relationships = []
    table_names = list(tables.keys())

    for i, t1 in enumerate(table_names):
        for t2 in table_names:
            if t1 == t2:
                continue
            for c1 in tables[t1].columns:
                vals1 = set(tables[t1][c1].dropna().unique())
                if not vals1 or len(vals1) < 2:
                    continue
                for c2 in tables[t2].columns:
                    vals2 = set(tables[t2][c2].dropna().unique())
                    if not vals2:
                        continue
                    overlap = len(vals1 & vals2) / len(vals1)
                    if overlap >= OVERLAP_THRESHOLD:
                        relationships.append(Relationship(
                            from_table=t1, from_column=c1,
                            to_table=t2, to_column=c2,
                            overlap_pct=round(overlap * 100, 1),
                        ))
    return _dedupe_relationships(relationships)


def _dedupe_relationships(rels: list[Relationship]) -> list[Relationship]:
    """A<->B and B<->A both getting flagged is expected; keep the stronger direction."""
    seen = {}
    for r in rels:
        key = frozenset([f"{r.from_table}.{r.from_column}", f"{r.to_table}.{r.to_column}"])
        if key not in seen or r.overlap_pct > seen[key].overlap_pct:
            seen[key] = r
    return list(seen.values())


def detect_primary_key_sql(con, table_name: str) -> str | None:
    """SQL version of detect_primary_key — runs inside DuckDB, no pandas."""
    cols = [r[0] for r in con.execute(f'DESCRIBE "{table_name}"').fetchall()]
    n_rows = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    if n_rows == 0:
        return None

    best_col, best_ratio = None, 0.0
    for col in cols:
        n_unique = con.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}"').fetchone()[0]
        ratio = n_unique / n_rows
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio
    return best_col if best_ratio >= UNIQUENESS_THRESHOLD else None


def detect_relationships_sql(con, table_names: list[str], candidate_only: bool = True) -> list[Relationship]:
    """
    SQL-native relationship detection — everything (uniqueness ratios,
    overlap counts) runs as DuckDB queries against the table on disk,
    never materializing full columns in Python. This is the version to
    use once tables approach or exceed available RAM; the pandas-set
    version in detect_relationships() is fine for small in-memory data
    and easier to read, but won't scale past RAM.

    candidate_only=True restricts overlap checks to columns with >50%
    uniqueness per table (key-like columns) rather than every column
    pair — keeps the O(n^2) column-pair cost sane on wide tables.
    """
    relationships = []

    # Pre-compute per-table column stats once
    table_stats = {}
    for t in table_names:
        cols = [r[0] for r in con.execute(f'DESCRIBE "{t}"').fetchall()]
        n_rows = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        if n_rows == 0:
            table_stats[t] = {"rows": 0, "cols": []}
            continue
        keep_cols = []
        for c in cols:
            n_unique = con.execute(f'SELECT COUNT(DISTINCT "{c}") FROM "{t}"').fetchone()[0]
            ratio = n_unique / n_rows if n_rows else 0
            if not candidate_only or ratio > 0.5:
                keep_cols.append(c)
        table_stats[t] = {"rows": n_rows, "cols": keep_cols}

    for t1 in table_names:
        for t2 in table_names:
            if t1 == t2 or table_stats[t1]["rows"] == 0 or table_stats[t2]["rows"] == 0:
                continue
            for c1 in table_stats[t1]["cols"]:
                n_distinct_1 = con.execute(f'SELECT COUNT(DISTINCT "{c1}") FROM "{t1}"').fetchone()[0]
                if n_distinct_1 < 2:
                    continue
                for c2 in table_stats[t2]["cols"]:
                    # Cast both sides to VARCHAR before comparing — matching
                    # columns across data sources can differ in stored type
                    # (e.g. an ID as text in one table, integer in another),
                    # and DuckDB won't silently coerce that in an IN clause.
                    overlap = con.execute(f"""
                        SELECT COUNT(DISTINCT t1v."{c1}")
                        FROM (SELECT DISTINCT "{c1}" FROM "{t1}" WHERE "{c1}" IS NOT NULL) t1v
                        WHERE CAST(t1v."{c1}" AS VARCHAR) IN (
                            SELECT CAST("{c2}" AS VARCHAR) FROM "{t2}" WHERE "{c2}" IS NOT NULL
                        )
                    """).fetchone()[0]
                    ratio = overlap / n_distinct_1
                    if ratio >= OVERLAP_THRESHOLD:
                        relationships.append(Relationship(
                            from_table=t1, from_column=c1,
                            to_table=t2, to_column=c2,
                            overlap_pct=round(ratio * 100, 1),
                        ))
    return _dedupe_relationships(relationships)


def to_mermaid(tables: dict[str, pd.DataFrame], relationships: list[Relationship]) -> str:
    """Generate a Mermaid ER diagram string from tables + inferred relationships (pandas version, small data)."""
    lines = ["erDiagram"]
    for name, df in tables.items():
        pk = detect_primary_key(df)
        safe_name = _mermaid_safe(name)
        lines.append(f"    {safe_name} {{")
        for col in df.columns[:12]:  # cap columns shown for readability
            dtype = _mermaid_safe(df[col].dtype)
            safe_col = _mermaid_safe(col)
            marker = " PK" if col == pk else ""
            lines.append(f"        {dtype} {safe_col}{marker}")
        lines.append("    }")

    for r in relationships:
        lines.append(
            f'    {_mermaid_safe(r.from_table)} }}o--o{{ {_mermaid_safe(r.to_table)} : '
            f'"{_mermaid_safe(r.from_column)}_to_{_mermaid_safe(r.to_column)}"'
        )

    return "\n".join(lines)


def to_mermaid_sql(con, table_names: list[str], relationships: list[Relationship]) -> str:
    """SQL-native version of to_mermaid — reads schema via DESCRIBE, no pandas."""
    lines = ["erDiagram"]
    for name in table_names:
        pk = detect_primary_key_sql(con, name)
        col_info = con.execute(f'DESCRIBE "{name}"').fetchall()
        safe_name = _mermaid_safe(name)
        lines.append(f"    {safe_name} {{")
        for col, dtype, *_ in col_info[:12]:
            dtype_clean = _mermaid_safe(dtype)
            safe_col = _mermaid_safe(col)
            marker = " PK" if col == pk else ""
            lines.append(f"        {dtype_clean} {safe_col}{marker}")
        lines.append("    }")

    for r in relationships:
        lines.append(
            f'    {_mermaid_safe(r.from_table)} }}o--o{{ {_mermaid_safe(r.to_table)} : '
            f'"{_mermaid_safe(r.from_column)}_to_{_mermaid_safe(r.to_column)}"'
        )
    return "\n".join(lines)