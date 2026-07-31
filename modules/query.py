"""
Query builder (Phase 3).

Two modes feed the same execution path:
  - No-code: UI picks table(s), join, filters, group-by, aggregations ->
    we assemble a SQL string.
  - Advanced: the assembled SQL is shown and editable directly.
Either way, `run_sql()` is the single execution point against DuckDB.
"""

import duckdb
import pandas as pd

AGG_FUNCS = ["SUM", "AVG", "COUNT", "MIN", "MAX", "COUNT DISTINCT"]
FILTER_OPS = ["=", "!=", ">", ">=", "<", "<=", "LIKE", "IN", "IS NULL", "IS NOT NULL"]
JOIN_TYPES = ["INNER", "LEFT", "RIGHT", "FULL OUTER"]


def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Single execution point — used by both no-code and advanced modes."""
    return con.execute(sql).df()


def build_query(
    base_table: str,
    columns: list[str],
    join_table: str | None = None,
    join_type: str = "INNER",
    join_left_col: str | None = None,
    join_right_col: str | None = None,
    filters: list[dict] | None = None,
    group_by: list[str] | None = None,
    aggregations: list[dict] | None = None,
    limit: int | None = 1000,
) -> str:
    """
    Assemble a SQL SELECT from no-code UI selections.

    filters: [{"column": "amount", "op": ">", "value": "100"}, ...]
    aggregations: [{"func": "SUM", "column": "amount", "alias": "total_amount"}, ...]
    """
    select_parts = []

    if group_by or aggregations:
        select_parts.extend(group_by or [])
        for agg in (aggregations or []):
            func = agg["func"]
            col = agg["column"]
            alias = agg.get("alias") or f"{func.lower().replace(' ', '_')}_{col}"
            if func == "COUNT DISTINCT":
                select_parts.append(f'COUNT(DISTINCT "{col}") AS "{alias}"')
            else:
                select_parts.append(f'{func}("{col}") AS "{alias}"')
    else:
        select_parts = [f'"{c}"' for c in columns] if columns else ["*"]

    sql = f'SELECT {", ".join(select_parts)} FROM "{base_table}"'

    if join_table and join_left_col and join_right_col:
        sql += (
            f' {join_type} JOIN "{join_table}" '
            f'ON "{base_table}"."{join_left_col}" = "{join_table}"."{join_right_col}"'
        )

    if filters:
        clauses = []
        for f in filters:
            col, op, val = f["column"], f["op"], f.get("value", "")
            if op in ("IS NULL", "IS NOT NULL"):
                clauses.append(f'"{col}" {op}')
            elif op == "LIKE":
                clauses.append(f'"{col}" LIKE \'%{val}%\'')
            elif op == "IN":
                items = val if isinstance(val, list) else [v.strip() for v in str(val).split(",") if v.strip()]
                formatted = ", ".join(v if _looks_numeric(v) else f"'{v}'" for v in items)
                clauses.append(f'"{col}" IN ({formatted})')
            else:
                # Quote value only if it doesn't look numeric
                v = val if _looks_numeric(val) else f"'{val}'"
                clauses.append(f'"{col}" {op} {v}')
        sql += " WHERE " + " AND ".join(clauses)

    if group_by:
        sql += f' GROUP BY {", ".join(group_by)}'

    if limit:
        sql += f" LIMIT {limit}"

    return sql


def _looks_numeric(val: str) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False