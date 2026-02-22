"""
04_train_model.py
-----------------
Builds a Random Forest model to predict LSOA-level crime rates
from IMD deprivation features.

Outputs:
    data/processed/modelling_data.csv
    models/crime_rate_model.pkl

Run from project root:
    python processing/04_train_model.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ── Paths ─────────────────────────────────────────────────────────
STREET_PATH = os.path.join("data", "processed", "street_clean.csv")
IMD_PATH    = os.path.join("data", "raw", "2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv")
POP_PATH    = os.path.join("data", "raw", "sapelsoasyoa20222024.xlsx")
OUT_CSV     = os.path.join("data", "processed", "modelling_data.csv")
OUT_MODEL   = os.path.join("models", "crime_rate_model.pkl")

IMD_LSOA_COL = "LSOA code (2011)"

RANDOM_STATE = 42
TEST_SIZE    = 0.20
OUTLIER_CAP  = 0.99

# ── Domain score keywords — matched against actual IMD column names ──
DOMAIN_KEYWORDS = [
    "index of multiple deprivation (imd) score",
    "income score",
    "employment score",
    "education, skills and training score",
    "health deprivation and disability score",
    "barriers to housing and services score",
    "living environment score",
]


# ── Loaders ───────────────────────────────────────────────────────

def load_lsoa_crime_rates() -> pd.DataFrame:
    street = pd.read_csv(STREET_PATH)
    counts = street.groupby("lsoa_code").size().reset_index(name="crime_count")

    # Population from ONS Excel — matches notebook approach exactly
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
    pop = pop[["lsoa_code", "population"]]

    counts = counts.merge(pop, on="lsoa_code", how="left")
    counts["population"] = counts["population"].replace(0, np.nan)
    counts.dropna(subset=["population"], inplace=True)
    counts["crime_rate"] = (counts["crime_count"] / counts["population"] * 1000).round(2)
    return counts


def load_imd_features() -> pd.DataFrame:
    imd = pd.read_csv(IMD_PATH)

    # Find score columns by keyword match
    feature_cols = []
    for col in imd.columns:
        if any(kw in col.lower() for kw in DOMAIN_KEYWORDS):
            feature_cols.append(col)

    if not feature_cols:
        print(f"  WARNING: no feature columns found. IMD columns are:")
        for c in imd.columns:
            print(f"    {c}")
        raise ValueError("Cannot train model without feature columns")

    print(f"  Using features: {feature_cols}")
    return imd[[IMD_LSOA_COL] + feature_cols].rename(
        columns={IMD_LSOA_COL: "lsoa_code"}
    ).dropna()


def build_modelling_data(lsoa_rates: pd.DataFrame, imd: pd.DataFrame) -> pd.DataFrame:
    df = imd.merge(lsoa_rates, on="lsoa_code", how="inner")
    df.dropna(subset=["crime_rate"], inplace=True)

    cap_val  = df["crime_rate"].quantile(OUTLIER_CAP)
    n_capped = (df["crime_rate"] > cap_val).sum()
    if n_capped:
        print(f"  Capping {n_capped} outliers above {cap_val:.1f} per 1,000")
    df["crime_rate"]     = df["crime_rate"].clip(upper=cap_val)
    df["log_crime_rate"] = np.log1p(df["crime_rate"])
    return df


# ── Training ──────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, feature_cols: list) -> tuple:
    X = df[feature_cols].values
    y = df["log_crime_rate"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n── Model performance ────────────────────────")
    print(f"  Training rows: {len(X_train):,}")
    print(f"  Test rows:     {len(X_test):,}")
    print(f"  R²:            {r2:.3f}")
    print(f"  MAE (log):     {mae:.3f}")

    print(f"\n── Feature importances ──────────────────────")
    for feat, imp in sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: -x[1]
    ):
        print(f"  {feat[:40]:<40} {imp:.3f}")

    return model, r2, mae


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("04_train_model.py")
    print("=" * 50)

    print("Building LSOA crime rates...")
    lsoa_rates = load_lsoa_crime_rates()
    print(f"  {len(lsoa_rates):,} LSOAs")

    print("Loading IMD features...")
    if not os.path.exists(IMD_PATH):
        raise FileNotFoundError(f"IMD file not found at {IMD_PATH}")
    imd_features = load_imd_features()
    print(f"  {len(imd_features):,} LSOAs with IMD data")

    print("Joining data...")
    df = build_modelling_data(lsoa_rates, imd_features)
    print(f"  {len(df):,} LSOAs in final dataset")

    feature_cols = [c for c in df.columns if c not in
                    ["lsoa_code", "lsoa_name", "crime_count", "population",
                     "crime_rate", "log_crime_rate"]]

    model, r2, mae = train_and_evaluate(df, feature_cols)

    os.makedirs(os.path.dirname(OUT_CSV),   exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Modelling data written to {OUT_CSV}")

    joblib.dump(model, OUT_MODEL)
    print(f"✓ Model saved to {OUT_MODEL}  (R²={r2:.3f}, MAE={mae:.3f})")


if __name__ == "__main__":
    main()