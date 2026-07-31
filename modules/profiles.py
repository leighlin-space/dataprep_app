"""
Dataset Profiles (Layer 1 — the universal adapter).

A Dataset Profile is a saved mapping from THIS dataset's actual column
names to a fixed set of semantic roles the rest of the app understands.
Map once per source format, save it with a name, reuse forever after —
that's what makes the aggregation templates work across differently
-named datasets without rewriting anything.

Profiles are stored as small JSON files under profiles/datasets/.
"""

import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict

PROFILE_DIR = Path(__file__).parent.parent / "profiles" / "datasets"

# Roles pulled directly from what the Rochester opioid pipeline actually
# used. required=True roles are the minimum needed for the 4 aggregation
# templates to run at all; everything else degrades gracefully if absent.
FIELD_ROLES = [
    {"role": "patient_id",   "label": "Patient ID",             "required": True},
    {"role": "prescriber_id","label": "Prescriber / Doctor ID",  "required": True},
    {"role": "dea",          "label": "DEA Number",              "required": False},
    {"role": "npi",          "label": "NPI Number",              "required": False},
    {"role": "pharmacy_id",  "label": "Pharmacy ID",             "required": False},
    {"role": "carrier_id",   "label": "Carrier ID",              "required": False},
    {"role": "drug_name",    "label": "Drug Name / Base",        "required": False},
    {"role": "drug_code",    "label": "Drug Code",               "required": False},
    {"role": "drug_class",   "label": "Drug Type / Class",       "required": False},
    {"role": "drug_form",    "label": "Drug Form (TAB/CAP/etc)", "required": False},
    {"role": "dosage_unit",  "label": "Dosage Unit",             "required": False},
    {"role": "mme",          "label": "MME",                     "required": False},
    {"role": "fill_date",    "label": "Fill Date",                "required": False},
    {"role": "region",       "label": "Region",                   "required": False},
]
REQUIRED_ROLES = [r["role"] for r in FIELD_ROLES if r["required"]]


@dataclass
class DatasetProfile:
    name: str
    mapping: dict  # role -> actual column name in the source data
    description: str = ""


def _ensure_dir():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def save_profile(profile: DatasetProfile) -> None:
    _ensure_dir()
    path = PROFILE_DIR / f"{_safe_filename(profile.name)}.json"
    path.write_text(json.dumps(asdict(profile), indent=2))


def load_profile(name: str) -> DatasetProfile:
    path = PROFILE_DIR / f"{_safe_filename(name)}.json"
    data = json.loads(path.read_text())
    return DatasetProfile(**data)


def list_profiles() -> list[str]:
    _ensure_dir()
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def delete_profile(name: str) -> None:
    path = PROFILE_DIR / f"{_safe_filename(name)}.json"
    path.unlink(missing_ok=True)


def missing_required_roles(mapping: dict) -> list[str]:
    """Roles marked required that have no column mapped (or mapped to None)."""
    return [r for r in REQUIRED_ROLES if not mapping.get(r)]


def apply_profile(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Return a new dataframe with standardized role-named columns added
    alongside the originals (prefixed nothing — role names are the new
    columns), so downstream code (aggregation templates, filters) can
    always reference df['patient_id'], df['mme'], etc. regardless of
    what the source file called them. Unmapped roles are simply absent.
    """
    out = df.copy()
    for role, source_col in mapping.items():
        if source_col and source_col in df.columns:
            out[role] = df[source_col]
    return out


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "unnamed_profile"