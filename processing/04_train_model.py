"""
04_train_model.py
-----------------
Builds a Random Forest model to predict LSOA-level crime rates
from IMD deprivation features.

Changes from original:
  - Spatial block cross-validation replaces random train/test split.
    LSOAs are clustered geographically; each fold holds out a spatial
    cluster rather than random rows. This avoids inflating R² through
    spatial autocorrelation between adjacent LSOAs.
  - Moran's I reported on model residuals. If significant, narrative
    is updated to note partial inflation from spatial dependence.
  - Causal language corrected: "dominant driver" -> "accounts for the
    majority of predictable variation". Model is predictive, not causal.
  - Spatial CV R² reported in model output rather than random split R².

Additional dependencies:
    libpysal, esda  (pip install libpysal esda)

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
from sklearn.cluster import KMeans

# ── Paths ─────────────────────────────────────────────────────────
STREET_PATH = os.path.join("data", "processed", "street_clean.csv")
IMD_PATH    = os.path.join("data", "raw", "2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv")
POP_PATH    = os.path.join("data", "raw", "sapelsoasyoa20222024.xlsx")
OUT_CSV     = os.path.join("data", "processed", "modelling_data.csv")
OUT_MODEL   = os.path.join("models", "crime_rate_model.pkl")

IMD_LSOA_COL = "LSOA code (2011)"

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
OUTLIER_CAP   = 0.99
N_SPATIAL_FOLDS = 5

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

    feature_cols = []
    for col in imd.columns:
        if any(kw in col.lower() for kw in DOMAIN_KEYWORDS):
            feature_cols.append(col)

    if not feature_cols:
        print(f"  WARNING: no feature columns found. IMD columns are:")
        for c in imd.columns:
            print(f"    {c}")
        raise ValueError("Cannot train model without feature columns")

    # Also load lat/lon for spatial CV — derived from LSOA centroids in the IMD file
    # The IMD file does not contain coordinates; we attach them from the street data
    print(f"  Using features: {feature_cols}")
    return imd[[IMD_LSOA_COL] + feature_cols].rename(
        columns={IMD_LSOA_COL: "lsoa_code"}
    ).dropna()


def attach_lsoa_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach mean lat/lon per LSOA from street_clean.csv.
    Required for spatial cross-validation block assignment.
    """
    street = pd.read_csv(STREET_PATH, usecols=["lsoa_code", "latitude", "longitude"])
    centroids = (
        street.groupby("lsoa_code")[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )
    return df.merge(centroids, on="lsoa_code", how="left")


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


# ── Spatial cross-validation ──────────────────────────────────────

def spatial_cv_score(
    df: pd.DataFrame,
    feature_cols: list,
    n_folds: int = N_SPATIAL_FOLDS,
) -> list:
    """
    Block spatial cross-validation: holds out geographic clusters of LSOAs
    rather than random rows.

    Why this matters: adjacent LSOAs share deprivation characteristics.
    A random split leaks spatial information from training into test set,
    inflating R². Holding out geographic blocks gives a more honest estimate
    of how well deprivation predicts crime in unseen areas.

    Requires latitude and longitude columns on df (attached via
    attach_lsoa_centroids()).

    Args:
        df:           Modelling DataFrame with lat/lon and feature columns.
        feature_cols: List of feature column names.
        n_folds:      Number of spatial folds (geographic clusters).

    Returns:
        List of R² scores, one per fold.
    """
    coords = df[["latitude", "longitude"]].values
    df     = df.copy()
    df["spatial_fold"] = KMeans(
        n_clusters=n_folds, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(coords)

    scores = []
    for fold in range(n_folds):
        train = df[df["spatial_fold"] != fold]
        test  = df[df["spatial_fold"] == fold]
        if len(test) < 10:
            print(f"  WARNING: fold {fold} has only {len(test)} test rows — skipping")
            continue
        m = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(train[feature_cols], train["log_crime_rate"])
        preds = m.predict(test[feature_cols])
        scores.append(r2_score(test["log_crime_rate"], preds))
        print(f"    Fold {fold}: R²={scores[-1]:.3f} ({len(test):,} test LSOAs)")

    return scores


def check_spatial_autocorrelation(df: pd.DataFrame, residuals: np.ndarray) -> dict:
    """
    Report Moran's I on model residuals to test for spatial autocorrelation.

    A significant Moran's I means neighbouring LSOAs have correlated residuals,
    suggesting the model's random-split R² is inflated by spatial leakage.

    Requires libpysal and esda:
        pip install libpysal esda

    Returns:
        Dict with morans_i and p_value. Returns NaN values if dependencies
        are not installed, with a warning.
    """
    try:
        from libpysal.weights import KNN
        from esda.moran import Moran
    except ImportError:
        print(
            "  WARNING: libpysal/esda not installed — skipping Moran's I.\n"
            "  Install with: pip install libpysal esda"
        )
        return {"morans_i": float("nan"), "p_value": float("nan")}

    coords = df[["latitude", "longitude"]].values
    w      = KNN.from_array(coords, k=8)
    w.transform = "R"
    mi = Moran(residuals, w)
    return {"morans_i": round(mi.I, 3), "p_value": round(mi.p_sim, 4)}


# ── Training ──────────────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Train Random Forest and evaluate with both random split (for reference)
    and spatial block CV (the honest estimate).

    Narrative note: R² here quantifies predictive association, not causal
    explanation. Deprivation features account for a substantial proportion
    of predictable variation in crime rates, but causal inference requires
    stronger identification than a cross-sectional observational model.
    """
    X = df[feature_cols].values
    y = df["log_crime_rate"].values

    # Random split — kept for reference but spatial CV is the reported metric
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    r2_random   = r2_score(y_test, y_pred_test)
    mae_random  = mean_absolute_error(y_test, y_pred_test)

    print(f"\n── Random split performance (reference only) ────────")
    print(f"  Training rows: {len(X_train):,}")
    print(f"  Test rows:     {len(X_test):,}")
    print(f"  R²:            {r2_random:.3f}")
    print(f"  MAE (log):     {mae_random:.3f}")
    print(f"  Note: random split R² is likely inflated by spatial")
    print(f"        autocorrelation between adjacent LSOAs.")

    # Spatial CV — reported metric
    print(f"\n── Spatial block CV ({N_SPATIAL_FOLDS} folds) ───────────────────")
    has_coords = "latitude" in df.columns and "longitude" in df.columns
    if has_coords and df[["latitude", "longitude"]].notna().all().all():
        cv_scores    = spatial_cv_score(df, feature_cols, n_folds=N_SPATIAL_FOLDS)
        r2_spatial   = float(np.mean(cv_scores))
        r2_spatial_std = float(np.std(cv_scores))
        print(f"  Spatial CV R²: {r2_spatial:.3f} ± {r2_spatial_std:.3f}")
        print(f"  (Mean ± SD across {len(cv_scores)} folds)")
        print(f"  This is the reported R² — random split inflates the figure.")
    else:
        print("  WARNING: lat/lon not available — spatial CV skipped")
        cv_scores  = []
        r2_spatial = r2_random
        print(f"  Falling back to random split R²: {r2_spatial:.3f}")

    # Moran's I on full-data residuals
    print(f"\n── Spatial autocorrelation in residuals ─────────────")
    y_pred_all = model.predict(X)
    residuals  = y - y_pred_all
    if has_coords:
        moran = check_spatial_autocorrelation(df, residuals)
        print(f"  Moran's I: {moran['morans_i']}  p={moran['p_value']}")
        if not np.isnan(moran["p_value"]) and moran["p_value"] < 0.05:
            print(
                "  Significant spatial autocorrelation in residuals detected.\n"
                "  Model predictions for neighbouring LSOAs are not independent.\n"
                "  Spatial CV R² is a more reliable performance estimate."
            )
    else:
        moran = {"morans_i": float("nan"), "p_value": float("nan")}

    print(f"\n── Feature importances ──────────────────────────────")
    for feat, imp in sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: -x[1]
    ):
        print(f"  {feat[:40]:<40} {imp:.3f}")

    return model, r2_spatial, mae_random, moran


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
    print(f"  {len(df):,} LSOAs in modelling dataset")

    print("Attaching LSOA centroids for spatial CV...")
    df = attach_lsoa_centroids(df)
    n_with_coords = df[["latitude", "longitude"]].notna().all(axis=1).sum()
    print(f"  {n_with_coords:,} LSOAs with coordinates")

    feature_cols = [c for c in df.columns if c not in
                    ["lsoa_code", "lsoa_name", "crime_count", "population",
                     "crime_rate", "log_crime_rate", "latitude", "longitude",
                     "spatial_fold"]]

    model, r2_spatial, mae, moran = train_and_evaluate(df, feature_cols)

    os.makedirs(os.path.dirname(OUT_CSV),   exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Modelling data written to {OUT_CSV}")

    joblib.dump(model, OUT_MODEL)
    print(
        f"✓ Model saved to {OUT_MODEL}\n"
        f"  Spatial CV R²={r2_spatial:.3f}  MAE={mae:.3f}  "
        f"Moran's I={moran['morans_i']} (p={moran['p_value']})"
    )


if __name__ == "__main__":
    main()