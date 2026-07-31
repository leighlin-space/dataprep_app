"""
Lightweight auto-profiling + cleaning suggestions.

Intentionally NOT using a heavy library (e.g. ydata-profiling) for the
MVP: this runs instantly on click, and suggestions are surfaced as an
editable checklist rather than applied silently, so analysts keep an
audit trail of what changed.
"""

import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_pct: float
    n_unique: int
    n_rows: int
    is_likely_key: bool
    sample_values: list = field(default_factory=list)


@dataclass
class Suggestion:
    id: str
    column: str
    description: str
    action: str  # internal action code applied in apply_suggestions()


def profile_dataframe(df: pd.DataFrame) -> list[ColumnProfile]:
    n_rows = len(df)
    profiles = []
    for col in df.columns:
        s = df[col]
        n_null = s.isna().sum()
        n_unique = s.nunique(dropna=True)
        profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(s.dtype),
                null_pct=round(100 * n_null / n_rows, 2) if n_rows else 0.0,
                n_unique=n_unique,
                n_rows=n_rows,
                is_likely_key=(n_unique == n_rows and n_rows > 0),
                sample_values=s.dropna().unique()[:5].tolist(),
            )
        )
    return profiles


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return every row involved in an exact duplicate (all columns match),
    including the first occurrence — so the analyst can see the full set,
    not just count how many extras would be dropped.
    """
    dupe_mask = df.duplicated(keep=False)
    return df[dupe_mask].sort_values(by=list(df.columns))


def suggest_cleaning(df: pd.DataFrame, profiles: list[ColumnProfile]) -> list[Suggestion]:
    suggestions = []

    # Fully empty columns
    for p in profiles:
        if p.null_pct == 100.0:
            suggestions.append(Suggestion(
                id=f"drop_empty_{p.name}",
                column=p.name,
                description=f"'{p.name}' is 100% empty — drop it",
                action="drop_column",
            ))

    # High-null columns (but not fully empty)
    for p in profiles:
        if 30 <= p.null_pct < 100:
            suggestions.append(Suggestion(
                id=f"flag_nulls_{p.name}",
                column=p.name,
                description=f"'{p.name}' is {p.null_pct}% null — review before using in joins/aggregations",
                action="flag_only",
            ))

    # Exact duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        suggestions.append(Suggestion(
            id="drop_duplicates",
            column="(all columns)",
            description=f"{n_dupes} exact duplicate rows found — drop them",
            action="drop_duplicates",
        ))

    # Whitespace-only string inconsistency (e.g. " CA" vs "CA")
    for col in df.select_dtypes(include=["string", "object"]).columns:
        stripped = df[col].astype(str).str.strip()
        if not stripped.equals(df[col].astype(str)):
            suggestions.append(Suggestion(
                id=f"trim_{col}",
                column=col,
                description=f"'{col}' has leading/trailing whitespace — trim it",
                action="trim_whitespace",
            ))

    return suggestions


def apply_suggestions(df: pd.DataFrame, suggestions: list[Suggestion], accepted_ids: set[str]) -> pd.DataFrame:
    """Apply only the suggestions the analyst has checked off."""
    out = df.copy()
    for s in suggestions:
        if s.id not in accepted_ids:
            continue
        if s.action == "drop_column":
            out = out.drop(columns=[s.column], errors="ignore")
        elif s.action == "drop_duplicates":
            out = out.drop_duplicates()
        elif s.action == "trim_whitespace":
            out[s.column] = out[s.column].astype(str).str.strip()
        # "flag_only" intentionally makes no change — informational
    return out
