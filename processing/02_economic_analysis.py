"""
02_economic_analysis.py
-----------------------
Reads street_clean.csv and raw ONS CPI food inflation data.
Produces all outputs consumed by the Economic Crime section.

Changes from original:
  - Lag correlations bootstrapped (1,000 iterations) with 95% CI.
  - Decomposition uses STL (robust=True).
  - build_borough_shoplifting_trend() has explicit assertions.
  - _format_lag_narrative() dead-code branch removed: column is
    always 'lag', never 'lag_months'. KeyError risk eliminated.
  - STL reliability metadata written to shoplifting_decomposition.csv
    so the dashboard can surface a data-quality caveat when fewer
    than 24 months are available (critique: STL caveat not propagated).
  - build_decomposition() now returns a 'stl_reliable' boolean column
    (True when n_months >= 24) so the section renderer can conditionally
    show a warning without re-running the check at render time.
  - Updated to IMD 2025 (was 2019). LSOA codes bridged from 2011 to 2021
    boundaries via lsoa_2011_to_2021_lookup.csv.

Outputs:
    data/processed/crime_indexed.csv
    data/processed/food_inflation_ons.csv
    data/processed/shoplifting_lag_correlations.csv
    data/processed/food_inflation_correlations.csv
    data/processed/shoplifting_decomposition.csv    ← now includes stl_reliable
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
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────
STREET_PATH  = os.path.join("data", "processed", "street_clean.csv")
ONS_CPI_PATH = os.path.join("data", "raw", "ons_cpi_food_d7g8.csv")
IMD_PATH         = os.path.join("data", "raw", "imd_2025.csv")
POP_PATH         = os.path.join("data", "raw", "sapelsoasyoa20222024.xlsx")
LSOA_LOOKUP_PATH = os.path.join("data", "raw", "lsoa_2011_to_2021_lookup.csv")
OUT_DIR          = os.path.join("data", "processed")

# IMD 2025 uses 2021 LSOA boundaries and 2024 LAD names
IMD_LSOA_COL    = "LSOA code (2021)"
IMD_BOROUGH_COL = "Local Authority District name (2024)"

# Standard 33 London boroughs — used to filter out non-London LSOAs
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

# Minimum months for STL to be considered reliable (2 full annual cycles)
STL_MIN_MONTHS = 24


# ── Loaders ───────────────────────────────────────────────────────

def load_street() -> pd.DataFrame:
    df = pd.read_csv(STREET_PATH)
    df["month"] = pd.to_datetime(df["month"])
    df["year"]  = df["month"].dt.year
    return df


def fetch_food_inflation_from_ons() -> pd.DataFrame:
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

    rows = []
    for line in raw.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        period, value = parts[0].strip().strip('"'), parts[1].strip().strip('"')
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

    os.makedirs(os.path.dirname(ONS_CPI_PATH) or ".", exist_ok=True)
    df[["month_str", "food_inflation"]].to_csv(ONS_CPI_PATH, index=False, header=False)
    print(f"    Fetched {len(df)} monthly observations, saved to {ONS_CPI_PATH}")

    return df[["month", "food_inflation"]]


def load_food_inflation() -> pd.DataFrame:
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


def _load_lsoa_bridge() -> pd.Series | None:
    """
    Load the 2011→2021 LSOA correspondence table and return a Series
    mapping lsoa_code_2011 → lsoa_code_2021.

    Police.uk street crime data uses 2011 LSOA codes; IMD 2025 uses
    2021 boundaries. Without bridging, ~10% of London LSOAs (those
    whose boundaries changed) will fail to join.

    Returns None if the lookup file is not present (falls back to
    direct join, same behaviour as with the 2019 IMD).
    """
    if not os.path.exists(LSOA_LOOKUP_PATH):
        print(f"  WARNING: LSOA lookup not found at {LSOA_LOOKUP_PATH}.")
        print("  IMD 2025 uses 2021 LSOA codes; police.uk data uses 2011 codes.")
        print("  Download the ONS LSOA 2011 to LSOA 2021 lookup from geoportal.statistics.gov.uk")
        print("  and save to data/raw/lsoa_2011_to_2021_lookup.csv")
        print("  Falling back to direct join (changed-boundary LSOAs will not match).")
        return None

    lut = pd.read_csv(LSOA_LOOKUP_PATH, low_memory=False)

    # ONS column names vary slightly by vintage — find them defensively
    col_2011 = next((c for c in lut.columns if "LSOA11CD" in c or c == "lsoa_code_2011"), None)
    col_2021 = next((c for c in lut.columns if "LSOA21CD" in c or c == "lsoa_code_2021"), None)

    if col_2011 is None or col_2021 is None:
        print(f"  WARNING: LSOA lookup columns not recognised. Found: {list(lut.columns)}")
        print("  Expected columns containing 'LSOA11CD' and 'LSOA21CD'. Falling back.")
        return None

    bridge = (
        lut[[col_2011, col_2021]]
        .drop_duplicates(subset=[col_2011], keep="first")
        .set_index(col_2011)[col_2021]
    )
    n_changed = (bridge.index != bridge.values).sum()
    print(f"  LSOA bridge loaded: {len(bridge):,} codes ({n_changed:,} changed boundary)")
    return bridge


def load_imd_borough_lookup() -> pd.DataFrame:
    """
    Build an lsoa_code → borough lookup using IMD 2025.

    IMD 2025 uses 2021 LSOA codes. The street crime data uses 2011 codes.
    We bridge them via the ONS lookup file so that LSOAs whose boundaries
    changed between 2011 and 2021 still join correctly.
    """
    imd = pd.read_csv(IMD_PATH)
    lookup = imd[[IMD_LSOA_COL, IMD_BOROUGH_COL]].rename(columns={
        IMD_LSOA_COL:    "lsoa_code_2021",
        IMD_BOROUGH_COL: "borough",
    }).drop_duplicates(subset=["lsoa_code_2021"])

    bridge = _load_lsoa_bridge()
    if bridge is not None:
        # bridge is a Series: index=2011 code, values=2021 code.
        # Some 2011 LSOAs were split into multiple 2021 LSOAs, so
        # inverting (2021->2011) can produce duplicate 2021 keys.
        # Keep only the first 2011 match per 2021 code.
        inv_df = (
            bridge.reset_index()
            .set_axis(["lsoa_2011", "lsoa_2021"], axis=1)
            .drop_duplicates(subset=["lsoa_2021"], keep="first")
        )
        inverse = inv_df.set_index("lsoa_2021")["lsoa_2011"]
        lookup["lsoa_code"] = (
            lookup["lsoa_code_2021"].map(inverse)
            .fillna(lookup["lsoa_code_2021"])
        )
    else:
        lookup["lsoa_code"] = lookup["lsoa_code_2021"]

    return lookup[["lsoa_code", "borough"]].drop_duplicates(subset=["lsoa_code"])


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


# ── 3. Shoplifting lag correlations (bootstrapped) ───────────────

def bootstrap_lag_correlation(
    food: pd.DataFrame,
    shop: pd.DataFrame,
    lag: int,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)

    merged = shop.merge(food, on="month", how="inner").sort_values("month")
    if len(merged) <= lag:
        return {"lag": lag, "r": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0}

    shop_lagged = merged["count"].iloc[lag:].reset_index(drop=True)
    food_base   = merged["food_inflation"].iloc[:len(merged) - lag].reset_index(drop=True)

    if len(shop_lagged) < 6:
        return {"lag": lag, "r": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": len(shop_lagged)}

    observed_r, _ = pearsonr(food_base, shop_lagged)

    n = len(shop_lagged)
    bootstrap_rs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        r, _ = pearsonr(food_base.iloc[idx], shop_lagged.iloc[idx])
        bootstrap_rs.append(r)

    return {
        "lag":      lag,
        "r":        round(observed_r, 3),
        "ci_lower": round(float(np.percentile(bootstrap_rs, 2.5)),  3),
        "ci_upper": round(float(np.percentile(bootstrap_rs, 97.5)), 3),
        "n":        n,
    }


def build_lag_correlations(
    street: pd.DataFrame,
    food: pd.DataFrame,
    max_lag: int = 12,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    monthly_shop = (
        street[street["crime_type"] == "Shoplifting"]
        .groupby("month").size()
        .reset_index(name="count")
    )

    rows = []
    for lag in range(0, max_lag + 1):
        result = bootstrap_lag_correlation(food, monthly_shop, lag, n_bootstrap)
        rows.append(result)

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["r"])
    if not valid.empty:
        best_idx = valid["r"].abs().idxmax()
        df["best_lag"] = False
        df.loc[best_idx, "best_lag"] = True

    return df


# ── 4. Seasonal decomposition (STL) ──────────────────────────────

def build_decomposition(street: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose monthly shoplifting using STL (robust=True).

    Fix (critique): now writes a 'stl_reliable' boolean column to the
    output CSV. The dashboard section renderer checks this flag and
    shows a data-quality warning when stl_reliable is False (< 24
    months of data). Previously the caveat was only logged to the
    console and never surfaced to dashboard users.

    Also writes 'n_months' so the dashboard can quote the exact
    series length in the warning message.
    """
    monthly = (
        street[street["crime_type"] == "Shoplifting"]
        .groupby("month").size()
        .reset_index(name="observed")
        .sort_values("month")
        .set_index("month")
    )

    n_months = len(monthly)
    stl_reliable = n_months >= STL_MIN_MONTHS

    if not stl_reliable:
        print(f"  WARNING: only {n_months} months of shoplifting data — "
              f"STL requires at least {STL_MIN_MONTHS} for reliable decomposition. "
              "stl_reliable=False will be written to the output CSV so the "
              "dashboard can surface this caveat to users.")

    stl    = STL(monthly["observed"], period=12, robust=True)
    result = stl.fit()

    df = pd.DataFrame({
        "month":    monthly.index,
        "observed": monthly["observed"].values,
        "trend":    result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
    }).reset_index(drop=True)

    # Reliability metadata — used by the dashboard to conditionally warn
    df["stl_reliable"] = stl_reliable
    df["n_months"]     = n_months

    return df


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
    shop = street[street["crime_type"] == "Shoplifting"].copy()

    assert "borough" in shop.columns, (
        "borough column missing from street data — "
        "ensure attach_borough() was called before build_borough_shoplifting_trend()"
    )
    assert shop["borough"].notna().sum() > 0, (
        "No borough values attached to street data — "
        "check that lsoa_code values match between street_clean.csv and IMD file"
    )

    shop = shop.dropna(subset=["borough"])
    shop = shop[shop["borough"].isin(LONDON_BOROUGHS_33)]

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

    try:
        imd = pd.read_csv(IMD_PATH)
        decile_col = [c for c in imd.columns if "decile" in c.lower() and "imd" in c.lower()]
        if decile_col:
            # IMD 2025 groups by 2021 borough name — maps directly to borough
            imd_decile = (
                imd[[IMD_BOROUGH_COL, decile_col[0]]]
                .rename(columns={
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

    print("  Building crime_indexed.csv...")
    try:
        df_out = build_crime_indexed(street)
        df_out.to_csv(os.path.join(OUT_DIR, "crime_indexed.csv"), index=False)
        print(f"    ✓ {len(df_out):,} rows")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("  Building food_inflation_ons.csv...")
    try:
        food.to_csv(os.path.join(OUT_DIR, "food_inflation_ons.csv"), index=False)
        print(f"    ✓ {len(food):,} rows")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("  Building food_inflation_correlations.csv...")
    try:
        df_out = build_food_inflation_correlations(street, food)
        df_out.to_csv(os.path.join(OUT_DIR, "food_inflation_correlations.csv"), index=False)
        print(f"    ✓ {len(df_out):,} rows")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("  Building shoplifting_lag_correlations.csv (bootstrapped, n=1000)...")
    try:
        df_out = build_lag_correlations(street, food, max_lag=12, n_bootstrap=1000)
        df_out.to_csv(os.path.join(OUT_DIR, "shoplifting_lag_correlations.csv"), index=False)
        best = df_out[df_out["best_lag"] == True].iloc[0]
        print(
            f"    ✓ {len(df_out):,} rows | best lag: {int(best['lag'])} months "
            f"r={best['r']} (95% CI: {best['ci_lower']}–{best['ci_upper']}, n={int(best['n'])})"
        )
    except Exception as e:
        print(f"    ERROR: {e}")

    print("  Building shoplifting_decomposition.csv (STL, robust=True)...")
    try:
        df_out = build_decomposition(street)
        df_out.to_csv(os.path.join(OUT_DIR, "shoplifting_decomposition.csv"), index=False)
        reliable = bool(df_out["stl_reliable"].iloc[0])
        n_m      = int(df_out["n_months"].iloc[0])
        print(f"    ✓ {len(df_out):,} rows | stl_reliable={reliable} (n_months={n_m})")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("  Building drugs_changepoint.csv...")
    try:
        df_out = build_drugs_changepoint(street)
        df_out.to_csv(os.path.join(OUT_DIR, "drugs_changepoint.csv"), index=False)
        print(f"    ✓ {len(df_out):,} rows")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("  Building borough_shoplifting_trend.csv...")
    try:
        df_out = build_borough_shoplifting_trend(street, lookup)
        df_out.to_csv(os.path.join(OUT_DIR, "borough_shoplifting_trend.csv"), index=False)
        print(f"    ✓ {len(df_out):,} rows")
    except Exception as e:
        print(f"    ERROR: {e}")

    print(f"\n✓ Economic analysis outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()