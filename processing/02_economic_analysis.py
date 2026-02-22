"""
02_economic_analysis.py
-----------------------
Reads street_clean.csv and raw ONS CPI food inflation data.
Produces all outputs consumed by the Economic Crime section.

Outputs:
    data/processed/crime_indexed.csv
    data/processed/food_inflation_ons.csv
    data/processed/shoplifting_lag_correlations.csv
    data/processed/food_inflation_correlations.csv
    data/processed/shoplifting_decomposition.csv
    data/processed/drugs_changepoint.csv
    data/processed/borough_shoplifting_trend.csv

Run from project root:
    python processing/02_economic_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────
STREET_PATH  = os.path.join("data", "processed", "street_clean.csv")
ONS_CPI_PATH = os.path.join("data", "raw", "ons_cpi_food_d7g8.csv")
IMD_PATH     = os.path.join("data", "raw", "2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv")
POP_PATH     = os.path.join("data", "raw", "sapelsoasyoa20222024.xlsx")
OUT_DIR      = os.path.join("data", "processed")

# ── IMD column names as they appear in the raw file ───────────────
IMD_LSOA_COL    = "LSOA code (2011)"
IMD_BOROUGH_COL = "Local Authority District name (2019)"


# ── Loaders ───────────────────────────────────────────────────────

def load_street() -> pd.DataFrame:
    df = pd.read_csv(STREET_PATH)
    df["month"] = pd.to_datetime(df["month"])
    df["year"]  = df["month"].dt.year
    return df


def fetch_food_inflation_from_ons() -> pd.DataFrame:
    """
    Fetch D7G8 monthly series from the ONS generator CSV endpoint and cache
    to ONS_CPI_PATH so subsequent runs read from disk.

    The old api.ons.gov.uk endpoint returns 403. The generator URL returns
    the full time series CSV and requires a browser-like User-Agent header.
    """
    import urllib.request

    url = (
        "https://www.ons.gov.uk/generator"
        "?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7g8/mm23"
    )
    print("    Fetching D7G8 from ONS website...")

    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,*/*",
    })

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch D7G8 from ONS: {e}\n\n"
            "To fix: go to https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7g8/mm23\n"
            "Click 'Download full time series as .csv'\n"
            f"Save the file to: {ONS_CPI_PATH}"
        )

    # The ONS CSV has several header rows then annual/quarterly/monthly data.
    # We want only rows where the Period column matches 'YYYY MON' (monthly).
    rows = []
    for line in raw.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        period, value = parts[0].strip().strip('"'), parts[1].strip().strip('"')
        # Monthly rows look like "2023 JAN", "2023 FEB", etc.
        # Annual rows are "2023", quarterly are "2023 Q1" — skip those
        tokens = period.split()
        if len(tokens) == 2 and len(tokens[0]) == 4 and tokens[0].isdigit() and len(tokens[1]) == 3:
            try:
                rows.append({"month_str": period, "food_inflation": float(value)})
            except ValueError:
                continue

    if not rows:
        raise ValueError(
            "Could not parse monthly rows from ONS CSV. "
            f"The file format may have changed. Raw preview:\n{raw[:500]}"
        )

    df = pd.DataFrame(rows)
    df["month"] = pd.to_datetime(df["month_str"], format="%Y %b", errors="coerce")
    df = df.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)

    # Cache to disk
    os.makedirs(os.path.dirname(ONS_CPI_PATH) or ".", exist_ok=True)
    df[["month_str", "food_inflation"]].to_csv(ONS_CPI_PATH, index=False, header=False)
    print(f"    Fetched {len(df)} monthly observations, saved to {ONS_CPI_PATH}")

    return df[["month", "food_inflation"]]


def load_food_inflation() -> pd.DataFrame:
    """
    Load ONS CPI series D7G8 (food and non-alcoholic beverages).
    If the file doesn't exist, fetches it automatically from the ONS API.
    """
    if not os.path.exists(ONS_CPI_PATH):
        print(f"    {ONS_CPI_PATH} not found — fetching from ONS API...")
        df = fetch_food_inflation_from_ons()
    else:
        df = pd.read_csv(ONS_CPI_PATH, header=None, names=["month", "food_inflation"])
        df["month"] = pd.to_datetime(df["month"], format="%Y %b", errors="coerce")
        df.dropna(subset=["month"], inplace=True)
        df["food_inflation"] = pd.to_numeric(df["food_inflation"], errors="coerce")

    df = df[
        (df["month"] >= "2023-01-01") &
        (df["month"] <= "2025-12-31")
    ].copy()
    return df.reset_index(drop=True)


def load_imd_borough_lookup() -> pd.DataFrame:
    """
    Load IMD file and return LSOA->borough lookup.
    Uses the actual column names from the raw file as seen in the notebook.
    """
    imd = pd.read_csv(IMD_PATH)
    lookup = imd[[IMD_LSOA_COL, IMD_BOROUGH_COL]].rename(columns={
        IMD_LSOA_COL:    "lsoa_code",
        IMD_BOROUGH_COL: "borough",
    })
    return lookup.drop_duplicates(subset=["lsoa_code"])


def attach_borough(street: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    return street.merge(lookup, on="lsoa_code", how="left")


# ── 1. Crime indexed to January 2023 ─────────────────────────────

def build_crime_indexed(street: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        street.groupby(["month", "crime_type"])
        .size()
        .reset_index(name="count")
    )
    baseline = (
        monthly[monthly["month"] == "2023-01-01"]
        .set_index("crime_type")["count"]
        .rename("baseline")
    )
    monthly = monthly.join(baseline, on="crime_type")
    monthly.dropna(subset=["baseline"], inplace=True)
    monthly = monthly[monthly["baseline"] > 0].copy()
    monthly["index_value"] = (monthly["count"] / monthly["baseline"] * 100).round(2)
    return monthly[["month", "crime_type", "count", "index_value"]]


# ── 2. Food inflation correlations ───────────────────────────────

def build_food_inflation_correlations(street: pd.DataFrame, food: pd.DataFrame) -> pd.DataFrame:
    monthly_crime = (
        street.groupby(["month", "crime_type"])
        .size()
        .reset_index(name="count")
    )
    rows = []
    for crime in monthly_crime["crime_type"].unique():
        subset = monthly_crime[monthly_crime["crime_type"] == crime]
        merged = subset.merge(food, on="month", how="inner")
        if len(merged) < 6:
            continue
        r, p = stats.pearsonr(merged["food_inflation"], merged["count"])
        rows.append({"crime_type": crime, "correlation": round(r, 3), "p_value": round(p, 4)})
    return pd.DataFrame(rows).sort_values("correlation", ascending=False)


# ── 3. Shoplifting lag correlations ──────────────────────────────

def build_lag_correlations(street: pd.DataFrame, food: pd.DataFrame, max_lag: int = 12) -> pd.DataFrame:
    monthly_shop = (
        street[street["crime_type"] == "Shoplifting"]
        .groupby("month").size()
        .reset_index(name="count")
    )
    merged = monthly_shop.merge(food, on="month", how="inner").sort_values("month")
    rows = []
    for lag in range(0, max_lag + 1):
        shop_lagged = merged["count"].iloc[lag:].reset_index(drop=True)
        food_base   = merged["food_inflation"].iloc[:len(merged) - lag].reset_index(drop=True)
        if len(shop_lagged) < 6:
            continue
        r, p = stats.pearsonr(food_base, shop_lagged)
        rows.append({"lag_months": lag, "r": round(r, 3), "p_value": round(p, 4)})
    return pd.DataFrame(rows)


# ── 4. Seasonal decomposition ─────────────────────────────────────

def build_decomposition(street: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        street[street["crime_type"] == "Shoplifting"]
        .groupby("month").size()
        .reset_index(name="observed")
        .sort_values("month")
        .set_index("month")
    )
    if len(monthly) < 24:
        print("  WARNING: fewer than 24 months — decomposition may be unreliable")
    decomp = seasonal_decompose(monthly["observed"], model="additive", period=12)
    return pd.DataFrame({
        "month":    monthly.index,
        "observed": monthly["observed"].values,
        "trend":    decomp.trend.values,
        "seasonal": decomp.seasonal.values,
        "residual": decomp.resid.values,
    })


# ── 5. Drugs changepoint ──────────────────────────────────────────

def build_drugs_changepoint(street: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        street[street["crime_type"] == "Drugs"]
        .groupby("month").size()
        .reset_index(name="count")
        .sort_values("month")
    )
    counts = monthly["count"].values
    months = monthly["month"].tolist()
    lo = max(3, int(len(counts) * 0.20))
    hi = min(len(counts) - 3, int(len(counts) * 0.80))

    best_score, best_month = float("inf"), None
    for i in range(lo, hi):
        score = np.var(counts[:i]) * i + np.var(counts[i:]) * (len(counts) - i)
        if score < best_score:
            best_score, best_month = score, months[i]

    before = monthly[monthly["month"] <  best_month]["count"]
    after  = monthly[monthly["month"] >= best_month]["count"]
    return pd.DataFrame([{
        "change_point_date": str(best_month)[:10],
        "mean_before":       round(before.mean(), 1),
        "mean_after":        round(after.mean(), 1),
        "variance_score":    round(best_score, 1),
    }])


# ── 6. Borough shoplifting trend ──────────────────────────────────

def build_borough_shoplifting_trend(street: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Uses borough column already attached to street data by attach_borough() in main().
    lookup is only passed for the IMD decile join below.
    """
    shop = street[street["crime_type"] == "Shoplifting"].copy()
    # borough is already on street from attach_borough() in main() — do not re-merge
    # (re-merging creates borough_x/borough_y duplicates and loses the column)
    if "borough" not in shop.columns:
        shop = shop.merge(lookup[["lsoa_code", "borough"]], on="lsoa_code", how="left")
    shop = shop.dropna(subset=["borough"])

    annual = (
        shop.groupby(["borough", "year"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    if 2023 not in annual.columns or 2025 not in annual.columns:
        raise ValueError("Need data for both 2023 and 2025")

    annual["change_pct"] = (
        (annual[2025] - annual[2023]) / annual[2023].replace(0, np.nan) * 100
    ).round(1)
    annual.rename(columns={2023: "count_2023", 2025: "count_2025"}, inplace=True)

    # Attach avg IMD decile from IMD file
    try:
        imd = pd.read_csv(IMD_PATH)
        decile_col = [c for c in imd.columns if "decile" in c.lower() and "imd" in c.lower()]
        if decile_col:
            imd_decile = (
                imd[[IMD_LSOA_COL, IMD_BOROUGH_COL, decile_col[0]]]
                .rename(columns={
                    IMD_LSOA_COL:    "lsoa_code",
                    IMD_BOROUGH_COL: "borough",
                    decile_col[0]:   "imd_decile",
                })
                .groupby("borough")["imd_decile"]
                .mean().round(1)
                .reset_index()
                .rename(columns={"imd_decile": "avg_imd_decile"})
            )
            annual = annual.merge(imd_decile, on="borough", how="left")
        else:
            annual["avg_imd_decile"] = np.nan
    except Exception as e:
        print(f"  WARNING: could not attach IMD deciles: {e}")
        annual["avg_imd_decile"] = np.nan

    return annual[["borough", "count_2023", "count_2025", "change_pct", "avg_imd_decile"]]


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("02_economic_analysis.py")
    print("=" * 50)

    print("Loading street crime data...")
    street = load_street()
    print(f"  {len(street):,} records")

    print("Loading IMD borough lookup...")
    lookup = load_imd_borough_lookup()
    print(f"  {len(lookup):,} LSOAs in lookup")

    print("Attaching boroughs to street data...")
    street = attach_borough(street, lookup)

    print("Loading food inflation data...")
    food = load_food_inflation()
    print(f"  {len(food)} monthly observations")

    os.makedirs(OUT_DIR, exist_ok=True)

    steps = [
        ("crime_indexed.csv",               build_crime_indexed,               (street,)),
        ("food_inflation_ons.csv",           lambda f: f,                       (food,)),
        ("food_inflation_correlations.csv",  build_food_inflation_correlations, (street, food)),
        ("shoplifting_lag_correlations.csv", build_lag_correlations,            (street, food)),
        ("shoplifting_decomposition.csv",    build_decomposition,               (street,)),
        ("drugs_changepoint.csv",            build_drugs_changepoint,           (street,)),
        ("borough_shoplifting_trend.csv",    build_borough_shoplifting_trend,   (street, lookup)),
    ]

    for filename, fn, args in steps:
        print(f"  Building {filename}...")
        try:
            df_out = fn(*args)
            df_out.to_csv(os.path.join(OUT_DIR, filename), index=False)
            print(f"    ✓ {len(df_out):,} rows")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n✓ Economic analysis outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()