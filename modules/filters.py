"""
Filter Profiles (Layer 2 — the standardization layer).

Saved, named, reusable filter definitions (drug list, region values,
date range). The point: one canonical "Opioid - 8 Drug" filter that
every analyst selects, instead of everyone retyping their own list.
Change it once here, every future run picks up the update.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field

PROFILE_DIR = Path(__file__).parent.parent / "profiles" / "filters"

# Seeded defaults, taken directly from the Rochester opioid pipeline —
# available out of the box, editable/saveable as new profiles.
DEFAULT_OPIOID_DRUGS = [
    "OXYCODONE", "HYDROCODONE", "MORPHINE", "OXYMORPHONE",
    "HYDROMORPHONE", "FENTANYL", "METHADONE", "TAPENTADOL",
]

DRUG_CODE_12 = [
    "9193", "9143", "9300", "9050", "9150", "9801",
    "9652", "9780", "9230", "9120", "9220L", "9639",
]

DRUG_CODE_14 = DRUG_CODE_12 + ["9064", "9250B"]


@dataclass
class FilterProfile:
    name: str
    drug_list: list = field(default_factory=list)
    drug_list_role: str = "drug_name"   # "drug_name" or "drug_code" — which Dataset Profile role this list matches
    region_values: list = field(default_factory=list)  # matches the "region" role
    date_start: str = ""                # matches the "fill_date" role
    date_end: str = ""
    description: str = ""


def _ensure_dir():
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def save_profile(profile: FilterProfile) -> None:
    _ensure_dir()
    path = PROFILE_DIR / f"{_safe_filename(profile.name)}.json"
    path.write_text(json.dumps(asdict(profile), indent=2))


def load_profile(name: str) -> FilterProfile:
    path = PROFILE_DIR / f"{_safe_filename(name)}.json"
    data = json.loads(path.read_text())
    return FilterProfile(**data)


def list_profiles() -> list[str]:
    _ensure_dir()
    names = sorted(p.stem for p in PROFILE_DIR.glob("*.json"))
    return names


def delete_profile(name: str) -> None:
    path = PROFILE_DIR / f"{_safe_filename(name)}.json"
    path.unlink(missing_ok=True)


def seed_default_if_missing() -> None:
    """Create the standard drug filter profiles on first run."""
    _ensure_dir()
    existing = list_profiles()

    if "opioid_8_drug" not in existing:
        save_profile(FilterProfile(
            name="opioid_8_drug",
            drug_list=DEFAULT_OPIOID_DRUGS,
            drug_list_role="drug_name",
            description="Standard 8-drug opioid list, by drug name (Rochester pipeline default).",
        ))
    if "drug_code_12" not in existing:
        save_profile(FilterProfile(
            name="drug_code_12",
            drug_list=DRUG_CODE_12,
            drug_list_role="drug_code",
            description="12-code opioid drug list, by drug code — matches a drug_code column.",
        ))
    if "drug_code_14" not in existing:
        save_profile(FilterProfile(
            name="drug_code_14",
            drug_list=DRUG_CODE_14,
            drug_list_role="drug_code",
            description="14-code opioid drug list (superset of the 12-code list), by drug code.",
        ))


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "unnamed_profile"