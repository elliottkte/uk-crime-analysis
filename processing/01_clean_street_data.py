"""
01_clean_street_data.py
-----------------------
Reads raw police.uk street crime CSVs, standardises, validates,
and writes data/processed/street_clean.csv.

Raw files expected at:
    data/raw/**/*street*.csv   (recursive glob, any subfolder depth)

This matches the police.uk bulk download layout where files are
named e.g. 2023-01-metropolitan-street.csv

Run from project root:
    python processing/01_clean_street_data.py

Fix (critique): extract_force() previously used parts[2] which
misclassifies City of London files ('city-of-london-police-street.csv'
splits to 'city' at index 2 but multi-word force names were fragile).
Now uses explicit substring matching so both forces are reliably
identified regardless of filename variation.
"""

import os
import glob
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────
RAW_DIR     = os.path.join("data", "raw")
OUTPUT_PATH = os.path.join("data", "processed", "street_clean.csv")

EXPECTED_CRIME_TYPES = {
    "Anti-social behaviour",
    "Bicycle theft",
    "Burglary",
    "Criminal damage and arson",
    "Drugs",
    "Possession of weapons",
    "Public order",
    "Robbery",
    "Shoplifting",
    "Theft from the person",
    "Vehicle crime",
    "Violence and sexual offences",
    "Other theft",
    "Other crime",
}

DATE_RANGE = ("2023-01-01", "2025-12-31")

# ── Known force name tokens in police.uk filenames ────────────────
# Maps substring found in basename → canonical force label.
# Checked in order: more-specific patterns first.
_FORCE_PATTERNS = [
    ("city-of-london", "city-of-london"),
    ("metropolitan",   "metropolitan"),
]


# ── Helpers ───────────────────────────────────────────────────────

def find_street_files(raw_dir: str) -> list:
    pattern = os.path.join(raw_dir, "**", "*street*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No street crime CSV files found under {raw_dir}.\n"
            "Expected files matching: data/raw/**/*street*.csv\n"
            "Download from: https://data.police.uk/data/"
        )
    return files


def extract_force(filepath: str) -> str:
    """
    Extract force name from filename using substring matching.

    Previous implementation used parts[2] (positional split on '-'),
    which was fragile for 'city-of-london' filenames where parts[2]
    is 'london' rather than 'city'. Explicit substring matching is
    unambiguous regardless of filename variation.

    Returns one of: 'metropolitan', 'city-of-london', or 'unknown'.
    """
    basename = os.path.basename(filepath).lower()
    for token, label in _FORCE_PATTERNS:
        if token in basename:
            return label
    return "unknown"


def load_all(files: list) -> pd.DataFrame:
    frames = []
    for fp in sorted(files):
        try:
            df = pd.read_csv(fp, low_memory=False)
            df["force"] = extract_force(fp)
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: could not read {fp}: {e}")

    if not frames:
        raise RuntimeError("No files loaded successfully.")

    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop columns the notebooks confirmed aren't needed
    df = df.drop(columns=["Context", "Falls within", "Reported by"], errors="ignore")

    # Lowercase snake_case — matches notebook convention exactly
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Drop rows with no location
    df = df.dropna(subset=["latitude", "longitude"])

    # Parse month
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month"])

    # Filter to date range
    df = df[
        (df["month"] >= DATE_RANGE[0]) &
        (df["month"] <= DATE_RANGE[1])
    ].copy()

    # Strip whitespace from crime_type
    if "crime_type" in df.columns:
        df["crime_type"] = df["crime_type"].str.strip()

    return df


def validate(df: pd.DataFrame):
    print(f"\n── Validation ───────────────────────────────")
    print(f"  Total rows:       {len(df):,}")
    print(f"  Date range:       {df['month'].min().strftime('%Y-%m')} to {df['month'].max().strftime('%Y-%m')}")
    print(f"  Forces:           {sorted(df['force'].unique())}")
    print(f"  Missing lsoa:     {df['lsoa_code'].isna().sum():,}")

    # Warn on any 'unknown' force rows — indicates unrecognised filenames
    unknown_force_rows = (df["force"] == "unknown").sum()
    if unknown_force_rows:
        print(f"  WARNING - {unknown_force_rows:,} rows have force='unknown'. "
              "Check that all raw filenames contain 'metropolitan' or "
              "'city-of-london'.")

    unknown_crime = set(df["crime_type"].unique()) - EXPECTED_CRIME_TYPES
    if unknown_crime:
        print(f"  WARNING - unexpected crime types: {unknown_crime}")
    else:
        print(f"  Crime types:      all expected")

    print(f"\n  Rows per force:")
    for force, count in df["force"].value_counts().items():
        print(f"    {force}: {count:,}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("01_clean_street_data.py")
    print("=" * 50)

    print(f"Searching for street crime files in {RAW_DIR}...")
    files = find_street_files(RAW_DIR)
    print(f"Found {len(files)} files")

    print("Loading and concatenating...")
    raw = load_all(files)
    print(f"  {len(raw):,} raw rows loaded")

    print("Cleaning...")
    clean_df = clean(raw)
    print(f"  {len(clean_df):,} rows after cleaning")

    validate(clean_df)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()