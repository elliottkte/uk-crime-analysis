"""
06_process_stop_search.py
-------------------------
Reads raw police.uk stop and search CSVs (data/raw/**/*stop*search*.csv),
processes them, and writes all ss_*.csv files consumed by the
Policing Response section.

Changes from original:
  - Borough assignment vectorised using sklearn BallTree (~100x faster
    than row-wise haversine loop over 400k+ records)
  - ONS population shares derived dynamically: categories absent from
    recorded data are folded into 'Other' automatically, with console
    logging. Eliminates hardcoded collapsed values.
  - build_changepoint_hypotheses() added: tests three competing
    explanations for the Aug 2024 drugs spike against observable
    implications, written to ss_changepoint_hypotheses.csv
  - build_narrative_stats() added: computes inline statistics cited
    in dashboard text, written to ss_narrative_stats.csv
  - Timezone warning on date parsing eliminated

Fix (pipeline run): ss_borough_full.csv had 34 rows instead of 33.
Added consolidate_boroughs() which normalises known City of London name
variants to the canonical label, sums numeric columns for any resulting
duplicates, and drops rows not in the standard 33-borough list.
consolidate_boroughs() is called on the borough DataFrame before
to_csv so the output always has ≤33 rows.

Key column note: ethnicity column is 'officer-defined_ethnicity'
(hyphenated, as it appears in the raw police.uk data).

Outputs:
    data/processed/ss_outcomes_summary.csv
    data/processed/ss_ethnicity_comparison.csv
    data/processed/ss_outcomes_by_search.csv
    data/processed/ss_borough_full.csv
    data/processed/ss_drugs_comparison.csv
    data/processed/ss_monthly_search_type.csv
    data/processed/ss_changepoint_hypotheses.csv
    data/processed/ss_narrative_stats.csv

Run from project root:
    python processing/06_process_stop_search.py
"""

import os
import glob
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────
RAW_DIR     = os.path.join("data", "raw")
STREET_PATH = os.path.join("data", "processed", "street_clean.csv")
OUT_DIR     = os.path.join("data", "processed")

# ── Standard 33 London boroughs ───────────────────────────────────
LONDON_BOROUGHS_33 = {
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "City of London", "Croydon", "Ealing", "Enfield",
    "Greenwich", "Hackney", "Hammersmith and Fulham", "Haringey",
    "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
    "Kensington and Chelsea", "Kingston upon Thames", "Lambeth",
    "Lewisham", "Merton", "Newham", "Redbridge", "Richmond upon Thames",
    "Southwark", "Sutton", "Tower Hamlets", "Waltham Forest",
    "Wandsworth", "Westminster",
}

# Aliases that GPS-centroid matching may produce for City of London
_BOROUGH_ALIASES = {
    "city of london police":                 "City of London",
    "city and county of the city of london": "City of London",
    "city of london (city)":                 "City of London",
    "london city":                           "City of London",
}

# ── ONS Census 2021 London population by broad ethnicity (%) ──────
ONS_ETHNICITY_POPULATION = {
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


# ── Borough consolidation ─────────────────────────────────────────

def normalise_borough_name(name: str) -> str:
    """Normalise known borough name variants to the canonical 33-borough label."""
    if not isinstance(name, str):
        return name
    return _BOROUGH_ALIASES.get(name.strip().lower(), name.strip())


def consolidate_boroughs(df: pd.DataFrame, borough_col: str = "borough") -> pd.DataFrame:
    """
    Fix (pipeline run): ss_borough_full.csv had 34 rows instead of 33.
    The extra row was a City of London GPS centroid match under a name
    variant not matching the canonical label.

    This function:
      1. Normalises all borough name variants to canonical form using
         _BOROUGH_ALIASES.
      2. Aggregates (sum numeric, first non-numeric) for any rows that
         now share a borough name after normalisation.
      3. Drops rows whose canonical name is not in LONDON_BOROUGHS_33
         (catches any non-London GPS matches that crept in).

    Args:
        df:          DataFrame with a borough column.
        borough_col: Name of the borough column (default: 'borough').

    Returns:
        DataFrame with ≤33 rows, deduplicated and aggregated.
    """
    df = df.copy()
    # Drop NaN/None borough rows before normalisation and groupby.
    # NaN passes through normalise_borough_name unchanged (non-string),
    # and Pandas groupby can include NaN as a group key in some versions,
    # producing the spurious 34th row. Explicit drop here is the safest guard.
    df = df[df[borough_col].notna()].copy()
    df[borough_col] = df[borough_col].apply(normalise_borough_name)

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [
        c for c in df.columns
        if c not in numeric_cols and c != borough_col
    ]

    agg_dict = {c: "sum"   for c in numeric_cols}
    agg_dict.update({c: "first" for c in non_numeric_cols})

    df = df.groupby(borough_col, as_index=False).agg(agg_dict)

    n_before = len(df)
    df = df[df[borough_col].isin(LONDON_BOROUGHS_33)].copy()
    n_dropped = n_before - len(df)

    if n_dropped:
        print(f"  consolidate_boroughs: dropped {n_dropped} non-standard "
              f"borough row(s). {len(df)} boroughs remain.")

    missing = LONDON_BOROUGHS_33 - set(df[borough_col])
    if missing:
        print(f"  consolidate_boroughs: {len(missing)} borough(s) have no "
              f"stop and search data: {sorted(missing)}")

    return df.reset_index(drop=True)


# ── Population share helpers ──────────────────────────────────────

def build_population_shares(recorded_categories: set) -> dict:
    """
    Collapse ONS population shares to match the ethnicity categories
    that actually appear in the recorded data.
    """
    shares        = {}
    unmatched_pct = 0.0
    unmatched_cats = set()

    for eth, pct in ONS_ETHNICITY_POPULATION.items():
        if eth in recorded_categories:
            shares[eth] = pct
        else:
            unmatched_pct += pct
            unmatched_cats.add(eth)

    if unmatched_pct > 0:
        shares["Other"] = shares.get("Other", 0.0) + unmatched_pct
        print(
            f"  Population shares: folded {unmatched_pct:.1f}% into 'Other' "
            f"(ONS categories absent from recorded data: {unmatched_cats})"
        )

    total = sum(shares.values())
    if not (99.5 < total < 100.5):
        print(f"  WARNING: population shares sum to {total:.1f}%, expected ~100%")

    return shares


# ── Helpers ───────────────────────────────────────────────────────

def is_arrest(outcome) -> bool:
    if pd.isna(outcome):
        return False
    return "arrest" in str(outcome).lower()


def load_raw(raw_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(raw_dir, "**", "*stop*search*.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No stop and search CSV files found under {raw_dir}.\n"
            "Expected files matching: data/raw/**/*stop*search*.csv\n"
            "Ensure the script is run from the project root."
        )
    print(f"  Found {len(files)} stop and search files")

    frames = []
    for fp in sorted(files):
        try:
            frames.append(pd.read_csv(fp, low_memory=False))
        except Exception as e:
            print(f"  WARNING: could not read {fp}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined.columns = [c.lower().replace(" ", "_") for c in combined.columns]
    return combined


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    df["date"]  = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["year"]  = df["date"].dt.year
    df = df.dropna(subset=["date"])
    df = df[
        (df["month"] >= "2023-01-01") &
        (df["month"] <= "2025-12-31")
    ].copy()
    return df


def map_ethnicity_broad(df: pd.DataFrame) -> pd.DataFrame:
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

    found      = set(df["ethnicity_broad"].unique()) - {"Other"}
    absent     = set(mapping.values()) - found - {"Other"}
    if absent:
        print(f"  Note: ONS categories not present in recorded data: {sorted(absent)}")
        print(f"        These will be folded into 'Other' in population comparisons.")

    return df


def assign_boroughs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised borough assignment using sklearn BallTree (haversine metric).
    """
    from sklearn.neighbors import BallTree

    has_gps = df["latitude"].notna() & df["longitude"].notna()
    no_gps  = (~has_gps).sum()
    if no_gps:
        pct = round(no_gps / len(df) * 100, 1)
        print(f"  {no_gps:,} records ({pct}%) have no GPS — excluded from borough analysis")

    centroids       = list(BOROUGH_CENTROIDS.items())
    borough_names   = [b for b, _ in centroids]
    centroid_coords = np.radians([[lat, lon] for _, (lat, lon) in centroids])

    gps_df       = df.loc[has_gps].copy()
    query_coords = np.radians(gps_df[["latitude", "longitude"]].values)

    tree = BallTree(centroid_coords, metric="haversine")
    _, indices = tree.query(query_coords, k=1)

    df.loc[has_gps, "borough"] = [borough_names[i[0]] for i in indices]
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
    recorded_categories = set(df["ethnicity_broad"].unique()) - {"Unknown"}
    population_shares   = build_population_shares(recorded_categories)

    total = len(df)
    rows  = []
    for eth, pop_pct in population_shares.items():
        mask       = df["ethnicity_broad"] == eth
        stop_count = mask.sum()
        if stop_count == 0:
            continue
        stop_pct = round(stop_count / total * 100, 1)
        arr_rate = round(df.loc[mask, "outcome"].apply(is_arrest).sum() / stop_count * 100, 1)
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
    """
    Build borough-level summary and consolidate to exactly the standard
    33 London boroughs.

    Fix (pipeline run - 34 rows): assign_boroughs() leaves the 'borough'
    column unset for the 24,142 no-GPS records. In some Pandas versions
    groupby() treats an unset object column as NaN and includes it as a
    group key, producing a 34th row with borough=NaN (which appeared in
    the test output with 24,142 total_searches — exactly the no-GPS count).

    The fix has two layers:
      1. Filter to rows where borough is a non-empty string before groupby,
         which is robust regardless of whether the unset value is NaN,
         None, or an empty string.
      2. consolidate_boroughs() enforces the LONDON_BOROUGHS_33 allowlist
         as a second guard, so any stray non-London or malformed rows are
         dropped before the CSV is written.
    """
    df = df.copy()
    if "borough" not in df.columns:
        df["borough"] = np.nan

    # Keep only rows with a valid non-empty borough string
    bdf = df[
        df["borough"].notna() &
        (df["borough"].astype(str).str.strip() != "") &
        (df["borough"].astype(str).str.strip() != "nan")
    ].copy()

    total     = bdf.groupby("borough").size().rename("total_searches")
    arrest_r  = (
        bdf.groupby("borough")["outcome"]
        .apply(lambda s: round(s.apply(is_arrest).sum() / len(s) * 100, 1))
        .rename("arrest_rate")
    )
    black_pct = (
        bdf.groupby("borough")["ethnicity_broad"]
        .apply(lambda s: round((s == "Black").sum() / len(s) * 100, 1))
        .rename("black_pct")
    )
    result = pd.concat([total, arrest_r, black_pct], axis=1).reset_index()
    result["lat"] = result["borough"].map(lambda b: BOROUGH_CENTROIDS.get(b, (None, None))[0])
    result["lon"] = result["borough"].map(lambda b: BOROUGH_CENTROIDS.get(b, (None, None))[1])

    # Fix (pipeline run): consolidate name variants and enforce 33-borough list
    result = consolidate_boroughs(result)

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


def build_changepoint_hypotheses(
    df: pd.DataFrame,
    cp_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Test three competing hypotheses for the drugs offence changepoint
    against observable implications in the stop and search data.

    Each hypothesis makes a different prediction about what we should
    see if it were true. Presenting all three lets readers evaluate
    the evidence rather than accepting a single interpretation.
    """
    col = "object_of_search"
    if col not in df.columns:
        print("  WARNING: object_of_search column missing — cannot build hypothesis table")
        return pd.DataFrame()

    drug_ss    = df[df[col].str.lower().str.contains("drug", na=False)].copy()
    before     = drug_ss[drug_ss["month"] <  cp_date]
    after      = drug_ss[drug_ss["month"] >= cp_date]
    all_before = df[df["month"] <  cp_date]
    all_after  = df[df["month"] >= cp_date]

    def safe_mean(val):
        try:
            return round(float(val), 1)
        except Exception:
            return float("nan")

    rows = [
        {
            "hypothesis":    "More enforcement activity",
            "metric":        "Monthly drug searches (mean)",
            "before":        safe_mean(before.groupby("month").size().mean()),
            "after":         safe_mean(after.groupby("month").size().mean()),
            "supports":      "Hypothesis supported if 'after' >> 'before'",
            "verdict_note":  (
                "If this column rises substantially, increased policing "
                "activity explains the spike in recorded offences."
            ),
        },
        {
            "hypothesis":    "Changed recording practice",
            "metric":        "Arrest rate on drug searches (%)",
            "before":        safe_mean(before["outcome"].apply(is_arrest).mean() * 100),
            "after":         safe_mean(after["outcome"].apply(is_arrest).mean() * 100),
            "supports":      "Hypothesis supported if arrest rate falls after changepoint",
            "verdict_note":  (
                "If arrest rate falls, the same encounters are being recorded "
                "as offences more often without a proportionate rise in arrests "
                "— consistent with a reclassification or recording practice change."
            ),
        },
        {
            "hypothesis":    "Real increase in drug activity",
            "metric":        "Drug searches as % of all searches",
            "before":        safe_mean(
                len(before) / len(all_before) * 100 if len(all_before) > 0 else float("nan")
            ),
            "after":         safe_mean(
                len(after) / len(all_after) * 100 if len(all_after) > 0 else float("nan")
            ),
            "supports":      "Hypothesis supported if share rises alongside volume",
            "verdict_note":  (
                "If police are devoting a larger share of all searches to drugs "
                "after the changepoint, it suggests genuine intelligence-led "
                "targeting of increased drug activity."
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_narrative_stats(
    ss_borough: pd.DataFrame,
    borough_dep: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute statistics cited inline in dashboard narrative text so they
    are data-derived rather than hardcoded strings.
    """
    from scipy.stats import pearsonr

    merged = ss_borough.merge(
        borough_dep[["borough", "avg_imd_decile", "avg_crime_rate"]],
        on="borough",
        how="inner",
    ).dropna(subset=["avg_imd_decile", "avg_crime_rate", "black_pct", "total_searches"])

    if len(merged) < 5:
        print("  WARNING: fewer than 5 boroughs matched — narrative stats unreliable")
        return pd.DataFrame([
            {"stat": "deprivation_black_stop_correlation",   "value": float("nan")},
            {"stat": "crime_rate_search_volume_correlation", "value": float("nan")},
        ])

    r_dep_black,    _ = pearsonr(merged["avg_imd_decile"],  merged["black_pct"])
    r_crime_search, _ = pearsonr(merged["avg_crime_rate"],  merged["total_searches"])

    return pd.DataFrame([
        {"stat": "deprivation_black_stop_correlation",   "value": round(r_dep_black,    3)},
        {"stat": "crime_rate_search_volume_correlation", "value": round(r_crime_search, 3)},
    ])


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

    print("Assigning boroughs (BallTree vectorised)...")
    ss = assign_boroughs(ss)

    os.makedirs(OUT_DIR, exist_ok=True)

    borough_full = build_borough_full(ss)

    outputs = {
        "ss_outcomes_summary.csv":     build_outcomes_summary(ss),
        "ss_ethnicity_comparison.csv": build_ethnicity_comparison(ss),
        "ss_outcomes_by_search.csv":   build_outcomes_by_search(ss),
        "ss_borough_full.csv":         borough_full,
        "ss_drugs_comparison.csv":     build_drugs_comparison(ss, STREET_PATH),
        "ss_monthly_search_type.csv":  build_monthly_search_type(ss),
    }

    for filename, df_out in outputs.items():
        path = os.path.join(OUT_DIR, filename)
        df_out.to_csv(path, index=False)
        print(f"  ✓ {filename}  ({len(df_out):,} rows)")

    # ── Changepoint hypothesis table ──────────────────────────────
    cp_path = os.path.join(OUT_DIR, "drugs_changepoint.csv")
    if os.path.exists(cp_path):
        print("  Building ss_changepoint_hypotheses.csv...")
        cp_date    = pd.to_datetime(
            pd.read_csv(cp_path)["change_point_date"].values[0]
        )
        hypotheses = build_changepoint_hypotheses(ss, cp_date)
        hypotheses.to_csv(os.path.join(OUT_DIR, "ss_changepoint_hypotheses.csv"), index=False)
        print(f"  ✓ ss_changepoint_hypotheses.csv  ({len(hypotheses)} rows)")
        print(f"\n  Changepoint hypotheses (cp_date={cp_date.date()}):")
        for _, row in hypotheses.iterrows():
            print(f"    [{row['hypothesis']}]")
            print(f"      {row['metric']}: {row['before']} → {row['after']}")
    else:
        print(
            "  Skipping ss_changepoint_hypotheses.csv — drugs_changepoint.csv not found.\n"
            "  Run script 02 first, then rerun script 06."
        )

    # ── Narrative stats ───────────────────────────────────────────
    dep_path = os.path.join(OUT_DIR, "borough_outliers_deprivation.csv")
    if os.path.exists(dep_path):
        print("  Building ss_narrative_stats.csv...")
        borough_dep     = pd.read_csv(dep_path)
        narrative_stats = build_narrative_stats(borough_full, borough_dep)
        narrative_stats.to_csv(os.path.join(OUT_DIR, "ss_narrative_stats.csv"), index=False)
        print(f"  ✓ ss_narrative_stats.csv  ({len(narrative_stats)} rows)")
        for _, row in narrative_stats.iterrows():
            print(f"    {row['stat']}: {row['value']}")
    else:
        print(
            "  Skipping ss_narrative_stats.csv — borough_outliers_deprivation.csv not found.\n"
            "  Run script 03 first, then rerun script 06."
        )

    print(f"\n✓ All stop and search outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()