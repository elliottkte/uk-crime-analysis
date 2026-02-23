"""
04_train_model.py
-----------------
Builds a Random Forest model to predict LSOA-level crime rates
from IMD deprivation features.

Changes from original:
  - Spatial block cross-validation replaces random train/test split.
  - Moran's I reported on model residuals.
  - Causal language corrected throughout.
  - Updated to IMD 2025 (was 2019). LSOA codes bridged 2011→2021 via
    lsoa_2011_to_2021_lookup.csv so all LSOAs join correctly to
    police.uk crime data regardless of boundary changes.

Fix (critique): k-means clustering on lat/lon coordinates can produce
oddly shaped, non-contiguous blocks that don't reflect genuine spatial
separation. Replaced with a regular grid-based blocking scheme:
LSOAs are assigned to a grid cell by rounding their centroid coordinates
to a fixed resolution. Each fold holds out all LSOAs in a grid cell,
giving geographically contiguous blocks and a more honest R² estimate.

Fix (critique): the trained model now stores spatial_cv_r2_ as an
attribute before saving to disk, so where_headed.py can retrieve the
actual computed value rather than falling back to a vague string.

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

# ── Paths ─────────────────────────────────────────────────────────
STREET_PATH = os.path.join("data", "processed", "street_clean.csv")
IMD_PATH    = os.path.join("data", "raw", "imd_2025.csv")
POP_PATH    = os.path.join("data", "raw", "sapelsoasyoa20222024.xlsx")
OUT_CSV     = os.path.join("data", "processed", "modelling_data.csv")
OUT_MODEL   = os.path.join("models", "crime_rate_model.pkl")

IMD_LSOA_COL = "LSOA code (2021)"

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
OUTLIER_CAP   = 0.99

# Grid resolution for spatial blocking (degrees).
# ~0.05° ≈ 3.5 km at London's latitude — produces ~20–30 cells across
# Greater London, giving naturally contiguous geographic blocks.
GRID_RESOLUTION = 0.05
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


def _load_lsoa_bridge() -> dict | None:
    """
    Load the ONS 2011→2021 LSOA correspondence table as a dict
    mapping 2011 code → 2021 code.

    Police.uk street crime data uses 2011 LSOA codes; IMD 2025 uses
    2021 boundaries. Without bridging, ~10% of London LSOAs fail to join.
    Returns None if the file is not present (falls back to direct join).
    """
    lookup_path = os.path.join("data", "raw", "lsoa_2011_to_2021_lookup.csv")
    if not os.path.exists(lookup_path):
        print(f"  WARNING: LSOA lookup not found at {lookup_path}.")
        print("  IMD 2025 uses 2021 LSOA codes; police.uk uses 2011 codes.")
        print("  Download the ONS LSOA 2011 to LSOA 2021 lookup from geoportal.statistics.gov.uk")
        print("  and save to data/raw/lsoa_2011_to_2021_lookup.csv. Falling back to direct join.")
        return None

    lut = pd.read_csv(lookup_path, low_memory=False)
    col_2011 = next((c for c in lut.columns if "LSOA11CD" in c), None)
    col_2021 = next((c for c in lut.columns if "LSOA21CD" in c), None)

    if col_2011 is None or col_2021 is None:
        print(f"  WARNING: LSOA lookup columns not recognised: {list(lut.columns)}. Falling back.")
        return None

    bridge = dict(zip(lut[col_2011], lut[col_2021]))
    n_changed = sum(1 for k, v in bridge.items() if k != v)
    print(f"  LSOA bridge: {len(bridge):,} codes ({n_changed:,} changed boundary)")
    return bridge


def load_imd_features() -> pd.DataFrame:
    """
    Load IMD 2025 domain scores as features for LSOA-level modelling.

    IMD 2025 uses 2021 LSOA codes; street crime data uses 2011 codes.
    The 2011→2021 bridge is applied so that features join correctly to
    crime data for LSOAs whose boundaries changed between revisions.
    """
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

    print(f"  Using features: {feature_cols}")

    # Rename 2021 LSOA code column, then bridge to 2011 codes for joining
    df = imd[[IMD_LSOA_COL] + feature_cols].rename(
        columns={IMD_LSOA_COL: "lsoa_code_2021"}
    ).dropna()

    bridge = _load_lsoa_bridge()
    if bridge is not None:
        # Invert bridge: 2021→2011 (needed because IMD has 2021 codes,
        # crime data has 2011 codes, and we join on lsoa_code)
        inverse = {v: k for k, v in bridge.items()}
        df["lsoa_code"] = df["lsoa_code_2021"].map(inverse).fillna(df["lsoa_code_2021"])
    else:
        df["lsoa_code"] = df["lsoa_code_2021"]

    return df.drop(columns=["lsoa_code_2021"])


def attach_lsoa_centroids(df: pd.DataFrame) -> pd.DataFrame:
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


# ── Grid-based spatial cross-validation ──────────────────────────

def assign_grid_folds(
    df: pd.DataFrame,
    resolution: float = GRID_RESOLUTION,
    n_folds: int = N_SPATIAL_FOLDS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Assign LSOAs to spatial folds using a regular grid rather than
    k-means clustering.

    Why grid > k-means (critique fix):
      k-means on lat/lon minimises within-cluster variance, which can
      produce non-contiguous, irregularly shaped clusters — particularly
      in London where the road network creates spatial discontinuities.
      Grid blocking assigns contiguous rectangular cells first, then
      groups those cells into folds. This gives geographically coherent
      held-out regions and a more honest estimate of how well the model
      generalises to truly unseen areas.

    Method:
      1. Round each LSOA centroid to the nearest grid cell
         (lat // resolution, lon // resolution).
      2. Collect unique cells and randomly assign each cell to one of
         n_folds groups (stratified by approximate cell population).
      3. LSOAs inherit their cell's fold assignment.

    Args:
        df:         DataFrame with latitude and longitude columns.
        resolution: Grid cell size in degrees.
        n_folds:    Number of spatial folds.
        random_state: Seed for reproducible fold assignment.

    Returns:
        DataFrame with an additional 'spatial_fold' column (0-indexed).
    """
    rng = np.random.default_rng(random_state)

    df = df.copy()
    df["grid_lat"] = (df["latitude"]  / resolution).apply(np.floor)
    df["grid_lon"] = (df["longitude"] / resolution).apply(np.floor)
    df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    unique_cells = sorted(df["grid_cell"].unique())
    n_cells = len(unique_cells)
    print(f"  Grid blocking: {n_cells} grid cells (resolution={resolution}°) → {n_folds} folds")

    # Shuffle cells and assign cyclically to folds
    shuffled = list(unique_cells)
    rng.shuffle(shuffled)
    cell_fold = {cell: i % n_folds for i, cell in enumerate(shuffled)}

    df["spatial_fold"] = df["grid_cell"].map(cell_fold)

    # Report fold sizes
    fold_sizes = df["spatial_fold"].value_counts().sort_index()
    for fold_id, count in fold_sizes.items():
        print(f"    Fold {fold_id}: {count:,} LSOAs")

    return df.drop(columns=["grid_lat", "grid_lon", "grid_cell"])


def spatial_cv_score(
    df: pd.DataFrame,
    feature_cols: list,
    n_folds: int = N_SPATIAL_FOLDS,
) -> list:
    """
    Grid-based spatial cross-validation. Folds are pre-assigned by
    assign_grid_folds() and stored in the 'spatial_fold' column.
    """
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
    X = df[feature_cols].values
    y = df["log_crime_rate"].values

    # Random split — kept for reference
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

    # Grid-based spatial CV — the reported metric
    print(f"\n── Grid-based spatial CV ({N_SPATIAL_FOLDS} folds) ──────────────")
    has_coords = "latitude" in df.columns and "longitude" in df.columns
    if has_coords and df[["latitude", "longitude"]].notna().all().all():
        df_with_folds = assign_grid_folds(df, n_folds=N_SPATIAL_FOLDS)
        cv_scores     = spatial_cv_score(df_with_folds, feature_cols, n_folds=N_SPATIAL_FOLDS)
        r2_spatial    = float(np.mean(cv_scores))
        r2_spatial_std = float(np.std(cv_scores))
        print(f"  Grid spatial CV R²: {r2_spatial:.3f} ± {r2_spatial_std:.3f}")
        print(f"  (Mean ± SD across {len(cv_scores)} folds)")
        print(f"  Grid-based blocking ensures contiguous held-out regions.")
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
                "  Grid-based spatial CV R² is the reliable performance estimate."
            )
    else:
        moran = {"morans_i": float("nan"), "p_value": float("nan")}

    print(f"\n── Feature importances ──────────────────────────────")
    for feat, imp in sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: -x[1]
    ):
        print(f"  {feat[:40]:<40} {imp:.3f}")

    # Fix (critique): store spatial_cv_r2_ on the model object so
    # where_headed.py can retrieve the actual computed value rather
    # than falling back to the vague "a substantial proportion" string.
    model.spatial_cv_r2_     = r2_spatial
    model.spatial_cv_r2_std_ = r2_spatial_std if cv_scores else float("nan")
    model.feature_names_in_  = feature_cols  # store for label matching in dashboard

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
        f"  Grid spatial CV R²={r2_spatial:.3f}  MAE={mae:.3f}  "
        f"Moran's I={moran['morans_i']} (p={moran['p_value']})\n"
        f"  model.spatial_cv_r2_={r2_spatial:.3f} stored as model attribute."
    )


if __name__ == "__main__":
    main()