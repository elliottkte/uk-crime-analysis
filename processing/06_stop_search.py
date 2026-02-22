"""
06_process_stop_search.py
-------------------------
Reads raw police.uk stop and search CSVs (data/raw/**/*stop*search*.csv),
processes them, and writes all ss_*.csv files consumed by the
Policing Response section.

Key column note: ethnicity column is 'officer-defined_ethnicity'
(hyphenated, as it appears in the raw police.uk data).

Outputs:
    data/processed/ss_outcomes_summary.csv
    data/processed/ss_ethnicity_comparison.csv
    data/processed/ss_outcomes_by_search.csv
    data/processed/ss_borough_full.csv
    data/processed/ss_drugs_comparison.csv
    data/processed/ss_monthly_search_type.csv

Run from project root:
    python processing/06_process_stop_search.py
"""

import os
import glob
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# ── Paths ─────────────────────────────────────────────────────────
RAW_DIR     = os.path.join("data", "raw")
STREET_PATH = os.path.join("data", "processed", "street_clean.csv")
OUT_DIR     = os.path.join("data", "processed")

# ── ONS Census 2021 London population by broad ethnicity (%) ──────
ETHNICITY_POPULATION = {
    "Asian": 19.7,
    "Black": 13.5,
    "Mixed":  5.7,
    "White": 53.8,
    "Other":  7.3,
}

# ── Borough centroids ─────────────────────────────────────────────
BOROUGH_CENTROIDS = {
    "Barking and Dagenham":   (51.5362,  0.0798),
    "Barnet":                  (51.6252, -0.1517),
    "Bexley":                  (51.4549,  0.1505),
    "Brent":                   (51.5588, -0.2817),
    "Bromley":                 (51.4039,  0.0198),
    "Camden":                  (51.5290, -0.1255),
    "City of London":          (51.5155, -0.0922),
    "Croydon":                 (51.3714, -0.0977),
    "Ealing":                  (51.5130, -0.3089),
    "Enfield":                 (51.6521, -0.0807),
    "Greenwich":               (51.4934,  0.0098),
    "Hackney":                 (51.5450, -0.0553),
    "Hammersmith and Fulham":  (51.4927, -0.2339),
    "Haringey":                (51.5906, -0.1119),
    "Harrow":                  (51.5836, -0.3464),
    "Havering":                (51.5779,  0.2120),
    "Hillingdon":              (51.5441, -0.4760),
    "Hounslow":                (51.4746, -0.3680),
    "Islington":               (51.5416, -0.1022),
    "Kensington and Chelsea":  (51.4991, -0.1938),
    "Kingston upon Thames":    (51.4123, -0.3007),
    "Lambeth":                 (51.4571, -0.1231),
    "Lewisham":                (51.4415, -0.0117),
    "Merton":                  (51.4014, -0.1958),
    "Newham":                  (51.5077,  0.0469),
    "Redbridge":               (51.5590,  0.0741),
    "Richmond upon Thames":    (51.4479, -0.3260),
    "Southwark":               (51.5035, -0.0804),
    "Sutton":                  (51.3618, -0.1945),
    "Tower Hamlets":           (51.5099, -0.0059),
    "Waltham Forest":          (51.5908, -0.0134),
    "Wandsworth":              (51.4567, -0.1919),
    "Westminster":             (51.4975, -0.1357),
}


# ── Helpers ───────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def assign_borough(lat: float, lon: float) -> str:
    return min(BOROUGH_CENTROIDS, key=lambda b: haversine(lat, lon, *BOROUGH_CENTROIDS[b]))


def is_arrest(outcome) -> bool:
    if pd.isna(outcome):
        return False
    return "arrest" in str(outcome).lower()


def load_raw(raw_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(raw_dir, "**", "*stop*search*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No stop and search CSV files found under {raw_dir}.\n"
            "Expected files matching: data/raw/**/*stop*search*.csv"
        )
    print(f"  Found {len(files)} stop and search files")

    frames = []
    for fp in sorted(files):
        try:
            frames.append(pd.read_csv(fp, low_memory=False))
        except Exception as e:
            print(f"  WARNING: could not read {fp}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    # Lowercase snake_case — note: police.uk uses 'officer-defined_ethnicity'
    # with a hyphen; we preserve that exactly as pandas sees it
    combined.columns = [c.lower().replace(" ", "_") for c in combined.columns]
    return combined


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    # Date
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year"]  = df["date"].dt.year
    df = df.dropna(subset=["date"])
    df = df[
        (df["month"] >= "2023-01-01") &
        (df["month"] <= "2025-12-31")
    ].copy()
    return df


def map_ethnicity_broad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map officer-defined ethnicity to five broad groups.
    The raw column is 'officer-defined_ethnicity' (hyphenated).
    Notebook confirmed values: Asian, Black, Other, White (already broad).
    """
    eth_col = "officer-defined_ethnicity"
    if eth_col not in df.columns:
        print(f"  WARNING: ethnicity column '{eth_col}' not found")
        df["ethnicity_broad"] = "Unknown"
        return df

    mapping = {
        "Asian": "Asian",
        "Black": "Black",
        "White": "White",
        "Other": "Other",
        "Mixed": "Mixed",
    }
    df["ethnicity_broad"] = (
        df[eth_col]
        .map(mapping)
        .fillna("Other")
    )
    return df


def assign_boroughs(df: pd.DataFrame) -> pd.DataFrame:
    has_gps = df["latitude"].notna() & df["longitude"].notna()
    no_gps = (~has_gps).sum()
    if no_gps:
        pct = round(no_gps / len(df) * 100, 1)
        print(f"  {no_gps:,} records ({pct}%) have no GPS — excluded from borough analysis")

    df.loc[has_gps, "borough"] = df.loc[has_gps].apply(
        lambda r: assign_borough(r["latitude"], r["longitude"]), axis=1
    )
    return df


# ── Output builders ───────────────────────────────────────────────

def build_outcomes_summary(df: pd.DataFrame) -> pd.DataFrame:
    total     = len(df)
    arrests   = df["outcome"].apply(is_arrest).sum()
    no_action = df["outcome"].str.lower().str.contains("no further action", na=False).sum()
    return pd.DataFrame([{
        "total":          total,
        "arrest_rate":    round(arrests   / total * 100, 1),
        "no_action_rate": round(no_action / total * 100, 1),
    }])


def build_ethnicity_comparison(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows  = []
    for eth, pop_pct in ETHNICITY_POPULATION.items():
        mask       = df["ethnicity_broad"] == eth
        stop_count = mask.sum()
        if stop_count == 0:
            continue
        stop_pct   = round(stop_count / total * 100, 1)
        arr_rate   = round(df.loc[mask, "outcome"].apply(is_arrest).sum() / stop_count * 100, 1)
        rows.append({
            "ethnicity":       eth,
            "stop_count":      stop_count,
            "stop_pct":        stop_pct,
            "population_pct":  pop_pct,
            "stop_rate_ratio": round(stop_pct / pop_pct, 2) if pop_pct else None,
            "arrest_rate":     arr_rate,
        })
    return pd.DataFrame(rows)


def build_outcomes_by_search(df: pd.DataFrame) -> pd.DataFrame:
    col = "object_of_search"
    if col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(col)
        .apply(lambda g: pd.Series({
            "total":       len(g),
            "arrest_rate": round(g["outcome"].apply(is_arrest).sum() / len(g) * 100, 1),
        }), include_groups=False)
        .reset_index()
        .sort_values("arrest_rate", ascending=False)
    )


def build_borough_full(df: pd.DataFrame) -> pd.DataFrame:
    bdf = df.dropna(subset=["borough"]).copy()
    total      = bdf.groupby("borough").size().rename("total_searches")
    arrest_r   = (
        bdf.groupby("borough")["outcome"]
        .apply(lambda s: round(s.apply(is_arrest).sum() / len(s) * 100, 1))
        .rename("arrest_rate")
    )
    black_pct  = (
        bdf.groupby("borough")["ethnicity_broad"]
        .apply(lambda s: round((s == "Black").sum() / len(s) * 100, 1))
        .rename("black_pct")
    )
    result = pd.concat([total, arrest_r, black_pct], axis=1).reset_index()
    result["lat"] = result["borough"].map(lambda b: BOROUGH_CENTROIDS.get(b, (None, None))[0])
    result["lon"] = result["borough"].map(lambda b: BOROUGH_CENTROIDS.get(b, (None, None))[1])
    return result


def build_drugs_comparison(df: pd.DataFrame, street_path: str) -> pd.DataFrame:
    col = "object_of_search"
    if col in df.columns:
        drug_ss = (
            df[df[col].str.lower().str.contains("drug", na=False)]
            .groupby("month").size().rename("drug_searches")
            .reset_index()
        )
    else:
        drug_ss = pd.DataFrame(columns=["month", "drug_searches"])

    street = pd.read_csv(street_path)
    street["month"] = pd.to_datetime(street["month"])
    drug_crimes = (
        street[street["crime_type"] == "Drugs"]
        .groupby("month").size().rename("drug_crimes")
        .reset_index()
    )

    merged = drug_ss.merge(drug_crimes, on="month", how="outer").fillna(0)
    merged["month"] = pd.to_datetime(merged["month"])
    return merged.sort_values("month")


def build_monthly_search_type(df: pd.DataFrame) -> pd.DataFrame:
    col = "object_of_search"
    if col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(["month", col])
        .size()
        .reset_index(name="count")
        .assign(month=lambda d: pd.to_datetime(d["month"]))
    )


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("06_process_stop_search.py")
    print("=" * 50)

    print("Loading raw stop and search files...")
    raw = load_raw(RAW_DIR)
    print(f"  {len(raw):,} raw records")

    print("Standardising dates...")
    ss = standardise(raw)
    print(f"  {len(ss):,} records in date range")

    print("Mapping ethnicities...")
    ss = map_ethnicity_broad(ss)
    print(f"  Ethnicity distribution:\n{ss['ethnicity_broad'].value_counts().to_string()}")

    print("Assigning boroughs from GPS...")
    ss = assign_boroughs(ss)

    os.makedirs(OUT_DIR, exist_ok=True)

    outputs = {
        "ss_outcomes_summary.csv":    build_outcomes_summary(ss),
        "ss_ethnicity_comparison.csv": build_ethnicity_comparison(ss),
        "ss_outcomes_by_search.csv":  build_outcomes_by_search(ss),
        "ss_borough_full.csv":        build_borough_full(ss),
        "ss_drugs_comparison.csv":    build_drugs_comparison(ss, STREET_PATH),
        "ss_monthly_search_type.csv": build_monthly_search_type(ss),
    }

    for filename, df_out in outputs.items():
        path = os.path.join(OUT_DIR, filename)
        df_out.to_csv(path, index=False)
        print(f"  ✓ {filename}  ({len(df_out):,} rows)")

    print(f"\n✓ All stop and search outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()