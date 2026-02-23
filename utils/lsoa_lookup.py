"""
Shared helper for bridging 2011 → 2021 LSOA code changes.

The 2025 IMD uses 2021 LSOA boundaries. Police.uk street crime data
uses 2011 LSOA codes. Without a lookup table, the join between crime
data and IMD data will fail silently for any LSOA whose code changed
between the two boundary revisions.

The ONS published a lookup file alongside the 2021 boundary changes:
  "LSOA (2011) to LSOA (2021) to Local Authority District (2022)
   Lookup for England and Wales"

Download it from:
  https://geoportal.statistics.gov.uk
  Search: "LSOA 2011 to LSOA 2021 lookup"
  File: LSOA11_LSOA21_LAD22_EW_LU.csv (or similar name)
  Save to: data/raw/lsoa_2011_to_2021_lookup.csv

If the file is not present, load_lsoa_lookup() returns None and the
calling script falls back to a direct join on lsoa_code (same behaviour
as with the 2019 IMD). This will silently drop LSOAs whose codes
changed, but is still substantially better than using the 2019 data.

The 2021 boundary revision affected approximately 10% of LSOAs in
England — mostly splits of large LSOAs into two smaller ones. In
London, the affected LSOAs are concentrated in high-growth areas.

Column names in the lookup file (may vary by vintage):
  LSOA11CD  — 2011 LSOA code (matches police.uk data)
  LSOA21CD  — 2021 LSOA code (matches IMD 2025)
"""

import os
import pandas as pd

LOOKUP_PATH = os.path.join("data", "raw", "lsoa_2011_to_2021_lookup.csv")

# Alternative filenames ONS uses for this file
_LOOKUP_ALIASES = [
    "lsoa_2011_to_2021_lookup.csv",
    "LSOA11_LSOA21_LAD22_EW_LU.csv",
    "LSOA11_LSOA21_LAD22_EW_LU_V2.csv",
    "lsoa11_lsoa21_lad22_ew_lu.csv",
]

# Expected column names — ONS sometimes capitalises differently
_CODE_2011_CANDIDATES = ["LSOA11CD", "lsoa11cd", "lsoa_code_2011", "LSOA11"]
_CODE_2021_CANDIDATES = ["LSOA21CD", "lsoa21cd", "lsoa_code_2021", "LSOA21"]


def _find_lookup_file() -> str | None:
    """Return path to the lookup file if it exists under any known alias."""
    raw_dir = os.path.join("data", "raw")
    for name in _LOOKUP_ALIASES:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            return path
    return None


def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """Return the first candidate column name that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_lsoa_lookup() -> pd.DataFrame | None:
    """
    Load the 2011→2021 LSOA correspondence table.

    Returns a DataFrame with columns:
        lsoa_code_2011  — original 2011 code (matches police.uk data)
        lsoa_code_2021  — corresponding 2021 code (matches IMD 2025)

    Returns None if the file is not found, with a printed download hint.

    Notes:
        - Some 2011 LSOAs were split into multiple 2021 LSOAs. The lookup
          contains one row per 2011→2021 pair, so a 2011 code can appear
          more than once if its LSOA was split. map_2011_to_2021() handles
          this by taking the first (most common) 2021 match.
        - LSOAs that were unchanged have identical 2011 and 2021 codes.
    """
    path = _find_lookup_file()

    if path is None:
        print(
            "\n  ── LSOA lookup not found ─────────────────────────────────────\n"
            "  IMD 2025 uses 2021 LSOA boundaries; police.uk data uses 2011\n"
            "  codes. Without the lookup, ~10% of LSOAs won't join correctly.\n"
            "\n"
            "  To fix: download the ONS correspondence file:\n"
            "    1. Go to: https://geoportal.statistics.gov.uk\n"
            "    2. Search: 'LSOA 2011 to LSOA 2021 to Local Authority'\n"
            "    3. Download the CSV (LSOA11_LSOA21_LAD22_EW_LU.csv)\n"
            "    4. Save to: data/raw/lsoa_2011_to_2021_lookup.csv\n"
            "\n"
            "  Falling back to direct join on lsoa_code (2011=2021 assumed).\n"
            "  ─────────────────────────────────────────────────────────────\n"
        )
        return None

    lookup = pd.read_csv(path, low_memory=False)

    col_2011 = _find_col(lookup, _CODE_2011_CANDIDATES)
    col_2021 = _find_col(lookup, _CODE_2021_CANDIDATES)

    if col_2011 is None or col_2021 is None:
        print(
            f"  WARNING: LSOA lookup file found at {path} but expected columns\n"
            f"  not found. Columns in file: {list(lookup.columns)}\n"
            f"  Expected one of {_CODE_2011_CANDIDATES} and {_CODE_2021_CANDIDATES}.\n"
            f"  Falling back to direct join."
        )
        return None

    result = lookup[[col_2011, col_2021]].rename(columns={
        col_2011: "lsoa_code_2011",
        col_2021: "lsoa_code_2021",
    }).drop_duplicates()

    unchanged = (result["lsoa_code_2011"] == result["lsoa_code_2021"]).sum()
    changed   = len(result) - unchanged
    print(f"  LSOA lookup loaded from {os.path.basename(path)}: "
          f"{len(result):,} pairs ({unchanged:,} unchanged, {changed:,} changed)")

    return result


def map_2011_to_2021(
    lsoa_codes_2011: pd.Series,
    lookup: pd.DataFrame | None,
) -> pd.Series:
    """
    Map a Series of 2011 LSOA codes to 2021 codes using the lookup table.

    Where a 2011 LSOA was split into multiple 2021 LSOAs, the first
    matching 2021 code is used (the largest resulting LSOA by population
    in most cases).

    If lookup is None, returns the input series unchanged (fallback mode).

    Args:
        lsoa_codes_2011: Series of 2011 LSOA codes (e.g. from police.uk).
        lookup:          Output of load_lsoa_lookup(), or None for fallback.

    Returns:
        Series of 2021 LSOA codes (or original codes if lookup is None).
    """
    if lookup is None:
        return lsoa_codes_2011

    # One-to-one mapping: take first 2021 match per 2011 code
    mapping = (
        lookup
        .drop_duplicates(subset=["lsoa_code_2011"], keep="first")
        .set_index("lsoa_code_2011")["lsoa_code_2021"]
    )

    mapped   = lsoa_codes_2011.map(mapping)
    n_mapped = mapped.notna().sum()
    n_total  = len(lsoa_codes_2011)
    n_unchanged = (lsoa_codes_2011 == mapped).sum()

    print(f"  LSOA mapping: {n_mapped:,}/{n_total:,} codes mapped "
          f"({n_unchanged:,} unchanged, "
          f"{n_mapped - n_unchanged:,} updated to 2021 boundary)")

    # For codes not in the lookup (shouldn't happen for London data),
    # fall back to the original code
    return mapped.fillna(lsoa_codes_2011)