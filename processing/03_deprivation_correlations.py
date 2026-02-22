"""
03_deprivation_correlations.py
------------------------------
Joins crime data to the IMD 2019 dataset, calculates borough-level
correlations between deprivation domains and crime types, and
identifies outlier boroughs.

Key file details from notebooks:
  - Street file: street_clean.csv
  - IMD file: 2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv
    Columns: 'LSOA code (2011)', 'Local Authority District name (2019)'
  - Population: sapelsoasyoa20222024.xlsx, sheet 'Mid-2022 LSOA 2021',
    skiprows=3, cols [2,3,4] -> lsoa_code, lsoa_name, population

Outputs:
    data/processed/domain_crime_correlations.csv
    data/processed/borough_outliers_deprivation.csv

Run from project root:
    python processing/03_deprivation_correlations.py
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────
STREET_PATH = os.path.join("data", "processed", "street_clean.csv")
IMD_PATH    = os.path.join("data", "raw", "2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv")
POP_PATH    = os.path.join("data", "raw", "sapelsoasyoa20222024.xlsx")
OUT_DIR     = os.path.join("data", "processed")

# ── IMD column names as they appear in the raw file ───────────────
IMD_LSOA_COL    = "LSOA code (2011)"
IMD_BOROUGH_COL = "Local Authority District name (2019)"

# ── Deprivation domain columns to look for in the IMD file ────────
# The notebook only used LSOA code and borough, so we search for
# score columns by keyword. Adjust if your file uses different names.
DOMAIN_KEYWORDS = {
    "IMD":         ["index of multiple deprivation (imd) score"],
    "Income":      ["income score"],
    "Employment":  ["employment score"],
    "Education":   ["education, skills and training score"],
    "Health":      ["health deprivation and disability score"],
    "Barriers":    ["barriers to housing and services score"],
    "Living Env":  ["living environment score"],
}

# ── Borough centroids for map ─────────────────────────────────────
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

# ── Filter to London boroughs only ───────────────────────────────
LONDON_BOROUGHS = set(BOROUGH_CENTROIDS.keys())


# ── Loaders ───────────────────────────────────────────────────────

def load_street() -> pd.DataFrame:
    df = pd.read_csv(STREET_PATH)
    df["month"] = pd.to_datetime(df["month"])
    df["year"]  = df["month"].dt.year
    return df


def load_imd() -> pd.DataFrame:
    """
    Load IMD and normalise column names.
    Searches for domain score columns by keyword (case-insensitive).
    """
    imd = pd.read_csv(IMD_PATH)

    # Build mapping from actual column names to short labels
    col_map = {
        IMD_LSOA_COL:    "lsoa_code",
        IMD_BOROUGH_COL: "borough",
    }

    found_domains = {}
    for label, keywords in DOMAIN_KEYWORDS.items():
        for col in imd.columns:
            if any(kw in col.lower() for kw in keywords):
                col_map[col] = f"score_{label.lower().replace(' ', '_')}"
                found_domains[label] = col
                break

    if not found_domains:
        print("  WARNING: no domain score columns found in IMD file")
        print(f"  Available columns: {imd.columns.tolist()}")

    # Also find decile column
    for col in imd.columns:
        if "decile" in col.lower() and "imd" in col.lower():
            col_map[col] = "imd_decile"
            break

    keep = [c for c in col_map if c in imd.columns]
    imd  = imd[keep].rename(columns=col_map)

    # Filter to London
    imd = imd[imd["borough"].isin(LONDON_BOROUGHS)]

    print(f"  Domain columns found: {list(found_domains.keys())}")
    print(f"  {len(imd):,} London LSOAs loaded")
    return imd


def load_population() -> pd.DataFrame:
    """
    Load ONS LSOA population estimates.
    Matches notebook: sheet 'Mid-2022 LSOA 2021', skiprows=3, cols [2,3,4].
    """
    pop = pd.read_excel(
        POP_PATH,
        sheet_name="Mid-2022 LSOA 2021",
        skiprows=3,
        usecols=[2, 3, 4],
        header=0,
    )
    pop.columns = ["lsoa_code", "lsoa_name", "population"]
    pop = pop.dropna()
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop = pop.dropna(subset=["population"])
    return pop[["lsoa_code", "population"]]


def attach_borough(street: pd.DataFrame, imd: pd.DataFrame) -> pd.DataFrame:
    lookup = imd[["lsoa_code", "borough"]].drop_duplicates()
    return street.merge(lookup, on="lsoa_code", how="left")


# ── Borough-level aggregation ─────────────────────────────────────

def build_borough_imd(imd: pd.DataFrame) -> pd.DataFrame:
    score_cols = [c for c in imd.columns if c.startswith("score_")]
    agg_cols   = score_cols + (["imd_decile"] if "imd_decile" in imd.columns else [])
    return (
        imd.groupby("borough")[agg_cols]
        .mean()
        .round(3)
        .reset_index()
    )


def build_borough_crime_rates(
    street: pd.DataFrame, pop: pd.DataFrame, imd: pd.DataFrame
) -> pd.DataFrame:
    lookup = imd[["lsoa_code", "borough"]].drop_duplicates()
    street_b = street.merge(lookup, on="lsoa_code", how="left")
    street_b = street_b[street_b["borough"].isin(LONDON_BOROUGHS)]

    counts = street_b.groupby("borough").size().reset_index(name="total_crimes")
    years  = street_b["year"].nunique()
    counts["annual_crimes"] = counts["total_crimes"] / years

    # Borough-level population from LSOA sums
    borough_pop = (
        imd[["lsoa_code", "borough"]]
        .merge(pop, on="lsoa_code", how="left")
        .groupby("borough")["population"]
        .sum()
        .reset_index()
    )
    counts = counts.merge(borough_pop, on="borough", how="left")
    counts["population"] = counts["population"].replace(0, np.nan)
    counts["avg_crime_rate"] = (
        counts["annual_crimes"] / counts["population"] * 1000
    ).round(2)
    return counts[["borough", "avg_crime_rate", "population"]]


# ── 1. Domain correlations ────────────────────────────────────────

def build_domain_correlations(
    street: pd.DataFrame,
    borough_imd: pd.DataFrame,
    borough_crime: pd.DataFrame,
    imd: pd.DataFrame,
) -> pd.DataFrame:
    lookup = imd[["lsoa_code", "borough"]].drop_duplicates()
    street_b = street.merge(lookup, on="lsoa_code", how="left")
    type_rates = (
        street_b.groupby(["borough", "crime_type"])
        .size()
        .reset_index(name="count")
        .merge(borough_crime[["borough", "population"]], on="borough", how="left")
    )
    type_rates["rate"] = type_rates["count"] / type_rates["population"] * 1000

    score_cols = [c for c in borough_imd.columns if c.startswith("score_")]
    domain_labels = {
        "score_imd":         "IMD",
        "score_income":      "Income",
        "score_employment":  "Employment",
        "score_education":   "Education",
        "score_health":      "Health",
        "score_barriers":    "Barriers",
        "score_living_env":  "Living Env",
    }

    rows = []
    for crime in type_rates["crime_type"].unique():
        crime_rates = type_rates[type_rates["crime_type"] == crime][["borough", "rate"]]
        merged = borough_imd.merge(crime_rates, on="borough", how="inner")
        if len(merged) < 10:
            continue
        for col in score_cols:
            if col not in merged.columns:
                continue
            r, p = stats.pearsonr(merged[col], merged["rate"])
            rows.append({
                "crime_type":         crime,
                "deprivation_domain": domain_labels.get(col, col),
                "correlation":        round(r, 3),
                "p_value":            round(p, 4),
            })
    return pd.DataFrame(rows)


# ── 2. Borough outlier classification ────────────────────────────

def build_borough_outliers(
    borough_imd: pd.DataFrame,
    borough_crime: pd.DataFrame,
) -> pd.DataFrame:
    imd_col = "score_imd" if "score_imd" in borough_imd.columns else (
        [c for c in borough_imd.columns if c.startswith("score_")] or [None]
    )[0]

    if imd_col is None:
        print("  WARNING: no IMD score column found for outlier classification")
        return pd.DataFrame()

    merged = borough_imd.merge(borough_crime, on="borough", how="inner")
    merged.dropna(subset=[imd_col, "avg_crime_rate"], inplace=True)

    # Compute regression residuals for map sizing/context
    slope, intercept, r, p, se = stats.linregress(
        merged[imd_col], merged["avg_crime_rate"]
    )
    merged["predicted"] = intercept + slope * merged[imd_col]
    merged["residual"]  = merged["avg_crime_rate"] - merged["predicted"]

    # ── Quadrant classification ───────────────────────────────────
    # In London, high-footfall wealthy boroughs (Westminster, City) dominate
    # the regression line, so residual-based thresholds miss deprived boroughs.
    # Instead we classify by quadrant relative to London medians:
    #   High crime + high deprivation  -> "Deprived and high crime"
    #   High crime + low deprivation   -> "Affluent but high crime"
    #   Otherwise                      -> "As expected"
    #
    # IMD score: higher score = MORE deprived (opposite of decile)
    # We use the London median crime rate and London median IMD score as thresholds.
    median_crime = merged["avg_crime_rate"].median()
    median_imd   = merged[imd_col].median()

    def classify(row):
        high_crime = row["avg_crime_rate"] > median_crime
        if not high_crime:
            return "As expected"
        high_deprivation = row[imd_col] > median_imd  # higher IMD score = more deprived
        if high_deprivation:
            return "Deprived and high crime"
        return "Affluent but high crime"

    merged["dominant_outlier"] = merged.apply(classify, axis=1)
    merged["affluent_high"]    = (merged["dominant_outlier"] == "Affluent but high crime").astype(int)
    merged["avg_residual"]     = merged["residual"]
    merged["avg_imd_decile"]   = merged["imd_decile"] if "imd_decile" in merged.columns else np.nan

    merged["latitude"]  = merged["borough"].map(lambda b: BOROUGH_CENTROIDS.get(b, (None, None))[0])
    merged["longitude"] = merged["borough"].map(lambda b: BOROUGH_CENTROIDS.get(b, (None, None))[1])

    keep = [
        "borough", "avg_imd_decile", "avg_crime_rate", "residual",
        "avg_residual", "dominant_outlier", "affluent_high",
        "latitude", "longitude",
    ]
    return merged[[c for c in keep if c in merged.columns]]


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("03_deprivation_correlations.py")
    print("=" * 50)

    print("Loading street crime data...")
    street = load_street()
    print(f"  {len(street):,} records")

    print("Loading IMD data...")
    if not os.path.exists(IMD_PATH):
        raise FileNotFoundError(f"IMD file not found at {IMD_PATH}")
    imd = load_imd()

    print("Loading population data...")
    if not os.path.exists(POP_PATH):
        raise FileNotFoundError(f"Population file not found at {POP_PATH}")
    pop = load_population()
    print(f"  {len(pop):,} LSOAs with population data")

    print("Building borough-level aggregates...")
    borough_imd   = build_borough_imd(imd)
    borough_crime = build_borough_crime_rates(street, pop, imd)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("  Building domain_crime_correlations.csv...")
    corr = build_domain_correlations(street, borough_imd, borough_crime, imd)
    corr.to_csv(os.path.join(OUT_DIR, "domain_crime_correlations.csv"), index=False)
    print(f"    ✓ {len(corr):,} correlation pairs")

    print("  Building borough_outliers_deprivation.csv...")
    outliers = build_borough_outliers(borough_imd, borough_crime)
    outliers.to_csv(os.path.join(OUT_DIR, "borough_outliers_deprivation.csv"), index=False)
    print(f"    ✓ {len(outliers)} boroughs classified")
    for label in ["Deprived and high crime", "Affluent but high crime", "As expected"]:
        n = (outliers["dominant_outlier"] == label).sum()
        print(f"      {label}: {n}")

    print(f"\n✓ Deprivation outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()