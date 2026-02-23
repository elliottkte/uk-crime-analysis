"""
tests/test_pipeline.py
----------------------
Schema and sanity tests for all processed data files.

Run with:
    pytest tests/test_pipeline.py -v

The tests are intentionally lightweight — they check that the processed
files exist, have the expected columns, and contain plausible values.
They do not rerun the processing scripts. The intent is to catch silent
failures in the pipeline (wrong column names after a merge, empty
DataFrames, NaN-only columns) before they reach the dashboard.

Coverage:
  - ProcessedFileExistence     : every expected processed file is present
  - TestStreetClean            : street_clean.csv schema
  - TestEconomicProcessed      : 02_economic_analysis.py outputs
  - TestDeprivationProcessed   : 03_deprivation_correlations.py outputs
  - TestModelOutputs           : 04_train_model.py outputs
  - TestVulnerabilityProcessed : 05_vulnerability_index.py outputs
  - TestStopSearchProcessed    : 06_process_stop_search.py outputs
  - TestProcessedSchemas       : new files added in the critique fixes
      - shoplifting_lag_correlations.csv  (now has CI columns)
      - ss_narrative_stats.csv
      - ss_changepoint_hypotheses.csv
      - borough_weight_sensitivity.csv
"""

import os
import pytest
import joblib
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────
PROCESSED = os.path.join("data", "processed")
MODELS    = "models"


def p(filename: str) -> str:
    """Return the full path to a processed file."""
    return os.path.join(PROCESSED, filename)


# ── Helpers ───────────────────────────────────────────────────────

def load(filename: str) -> pd.DataFrame:
    """Load a processed CSV and return a DataFrame."""
    return pd.read_csv(p(filename))


def assert_columns(df: pd.DataFrame, required_cols: list, filename: str):
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, (
        f"{filename} is missing columns: {missing}. "
        f"Found: {list(df.columns)}"
    )


def assert_not_empty(df: pd.DataFrame, filename: str):
    assert not df.empty, f"{filename} is empty."


def assert_no_all_nan_columns(df: pd.DataFrame, filename: str, skip: list = None):
    """Fail if any column (except those in skip) is entirely NaN."""
    skip = skip or []
    for col in df.columns:
        if col in skip:
            continue
        assert not df[col].isna().all(), (
            f"{filename}: column '{col}' is entirely NaN."
        )


def assert_values_in_range(
    df: pd.DataFrame, col: str, lo: float, hi: float, filename: str
):
    vals = df[col].dropna()
    assert not vals.empty, f"{filename}: column '{col}' has no non-NaN values."
    assert vals.min() >= lo and vals.max() <= hi, (
        f"{filename}: column '{col}' has values outside [{lo}, {hi}]. "
        f"Min={vals.min():.4f}, Max={vals.max():.4f}"
    )


# ══════════════════════════════════════════════════════════════════
# File existence
# ══════════════════════════════════════════════════════════════════

EXPECTED_FILES = [
    # 01
    "street_clean.csv",
    # 02
    "crime_indexed.csv",
    "food_inflation_correlations.csv",
    "shoplifting_lag_correlations.csv",
    "shoplifting_decomposition.csv",
    "drugs_changepoint.csv",
    "borough_shoplifting_trend.csv",
    # 03
    "domain_crime_correlations.csv",
    "borough_outliers_deprivation.csv",
    # 04
    "modelling_data.csv",
    # 05
    "borough_vulnerability.csv",
    "borough_weight_sensitivity.csv",
    # 06
    "ss_outcomes_summary.csv",
    "ss_ethnicity_comparison.csv",
    "ss_outcomes_by_search.csv",
    "ss_borough_full.csv",
    "ss_drugs_comparison.csv",
    "ss_monthly_search_type.csv",
    "ss_narrative_stats.csv",
    "ss_changepoint_hypotheses.csv",
]

EXPECTED_MODELS = [
    "crime_rate_model.pkl",
]


@pytest.mark.parametrize("filename", EXPECTED_FILES)
def test_processed_file_exists(filename):
    assert os.path.exists(p(filename)), (
        f"{filename} not found in {PROCESSED}. "
        "Has the relevant processing script been run?"
    )


@pytest.mark.parametrize("filename", EXPECTED_MODELS)
def test_model_file_exists(filename):
    path = os.path.join(MODELS, filename)
    assert os.path.exists(path), (
        f"{filename} not found in {MODELS}. "
        "Run processing/04_train_model.py first."
    )


# ══════════════════════════════════════════════════════════════════
# 01 — street_clean.csv
# ══════════════════════════════════════════════════════════════════

class TestStreetClean:
    # borough is NOT written to street_clean.csv — it is attached downstream
    # in scripts 02/03 via the IMD LSOA lookup. force is the police force field.
    REQUIRED = [
        "month", "crime_type", "lsoa_code", "lsoa_name",
        "latitude", "longitude", "force",
    ]

    @pytest.fixture(scope="class")
    def df(self):
        return load("street_clean.csv")

    def test_columns(self, df):
        assert_columns(df, self.REQUIRED, "street_clean.csv")

    def test_not_empty(self, df):
        assert_not_empty(df, "street_clean.csv")

    def test_month_parseable(self, df):
        parsed = pd.to_datetime(df["month"], errors="coerce")
        assert parsed.notna().all(), "street_clean.csv: month column has unparseable values."

    def test_force_values(self, df):
        valid_forces = {"metropolitan", "city"}
        found = set(df["force"].dropna().unique())
        unexpected = found - valid_forces
        assert not unexpected, (
            f"street_clean.csv: unexpected force values: {unexpected}"
        )

    def test_coordinates_plausible(self, df):
        # street_clean.csv includes all raw records before borough filtering.
        # Some records have GPS coordinates slightly outside Greater London
        # (e.g. border areas, or data quality issues in raw files).
        # Use a broad UK bounding box — borough filtering happens downstream.
        assert_values_in_range(df.dropna(subset=["latitude"]),  "latitude",  49.0, 61.0, "street_clean.csv")
        assert_values_in_range(df.dropna(subset=["longitude"]), "longitude", -8.0,  2.0, "street_clean.csv")

    def test_minimum_row_count(self, df):
        assert len(df) >= 10_000, (
            f"street_clean.csv has only {len(df):,} rows. "
            "Pipeline may have failed silently."
        )


# ══════════════════════════════════════════════════════════════════
# 02 — economic analysis outputs
# ══════════════════════════════════════════════════════════════════

class TestEconomicProcessed:

    def test_crime_indexed_columns(self):
        df = load("crime_indexed.csv")
        assert_columns(df, ["month", "crime_type", "index_value"], "crime_indexed.csv")
        assert_not_empty(df, "crime_indexed.csv")

    def test_food_inflation_correlations_columns(self):
        df = load("food_inflation_correlations.csv")
        assert_columns(df, ["crime_type", "correlation", "p_value"], "food_inflation_correlations.csv")
        assert_not_empty(df, "food_inflation_correlations.csv")

    def test_shoplifting_decomposition_columns(self):
        df = load("shoplifting_decomposition.csv")
        assert_columns(df, ["month", "observed", "trend", "seasonal", "residual"],
                       "shoplifting_decomposition.csv")
        assert_not_empty(df, "shoplifting_decomposition.csv")

    def test_drugs_changepoint_columns(self):
        df = load("drugs_changepoint.csv")
        assert_columns(
            df,
            ["change_point_date", "mean_before", "mean_after"],
            "drugs_changepoint.csv",
        )
        assert_not_empty(df, "drugs_changepoint.csv")
        assert float(df["mean_after"].values[0]) > float(df["mean_before"].values[0]), (
            "drugs_changepoint.csv: mean_after should be larger than mean_before."
        )

    def test_borough_shoplifting_trend_columns(self):
        df = load("borough_shoplifting_trend.csv")
        assert_columns(
            df,
            ["borough", "change_pct", "count_2023", "count_2025"],
            "borough_shoplifting_trend.csv",
        )
        assert_not_empty(df, "borough_shoplifting_trend.csv")

    def test_no_duplicate_boroughs_in_trend(self):
        df = load("borough_shoplifting_trend.csv")
        assert not any(c.endswith("_x") or c.endswith("_y") for c in df.columns), (
            "borough_shoplifting_trend.csv has _x/_y suffixed columns, "
            "indicating a duplicate-key merge. Check the borough merge in 02_economic_analysis.py."
        )


# ══════════════════════════════════════════════════════════════════
# 03 — deprivation correlations outputs
# ══════════════════════════════════════════════════════════════════

class TestDeprivationProcessed:

    def test_domain_crime_correlations_columns(self):
        df = load("domain_crime_correlations.csv")
        assert_columns(df, ["crime_type", "deprivation_domain", "correlation", "p_value"], "domain_crime_correlations.csv")
        assert_not_empty(df, "domain_crime_correlations.csv")

    def test_borough_outliers_deprivation_columns(self):
        df = load("borough_outliers_deprivation.csv")
        assert_columns(
            df,
            ["borough", "avg_imd_decile", "avg_crime_rate", "residual"],
            "borough_outliers_deprivation.csv",
        )
        assert_not_empty(df, "borough_outliers_deprivation.csv")

    def test_borough_outliers_no_duplicate_borough_column(self):
        df = load("borough_outliers_deprivation.csv")
        assert not any(c.endswith("_x") or c.endswith("_y") for c in df.columns), (
            "borough_outliers_deprivation.csv has _x/_y columns from a duplicate-key merge."
        )


# ══════════════════════════════════════════════════════════════════
# 04 — model outputs
# ══════════════════════════════════════════════════════════════════

class TestModelOutputs:

    @pytest.fixture(scope="class")
    def model(self):
        return joblib.load(os.path.join(MODELS, "crime_rate_model.pkl"))

    def test_model_has_feature_importances(self, model):
        assert hasattr(model, "feature_importances_"), (
            "crime_rate_model.pkl does not have feature_importances_. "
            "Is it a RandomForest model?"
        )

    def test_feature_importances_sum_to_one(self, model):
        total = model.feature_importances_.sum()
        assert abs(total - 1.0) < 0.001, (
            f"Feature importances sum to {total:.4f}, expected ~1.0."
        )

    def test_modelling_data_columns(self):
        df = load("modelling_data.csv")
        assert_columns(df, ["lsoa_code", "crime_rate"], "modelling_data.csv")
        assert_not_empty(df, "modelling_data.csv")

    def test_modelling_data_minimum_rows(self):
        df = load("modelling_data.csv")
        assert len(df) >= 4_000, (
            f"modelling_data.csv has only {len(df):,} rows. "
            "London has ~5,000 LSOAs; pipeline may have failed."
        )


# ══════════════════════════════════════════════════════════════════
# 05 — vulnerability index outputs
# ══════════════════════════════════════════════════════════════════

class TestVulnerabilityProcessed:

    def test_borough_vulnerability_columns(self):
        df = load("borough_vulnerability.csv")
        assert_columns(
            df,
            [
                "borough", "vulnerability_score", "risk_tier",
                "latitude", "longitude",
            ],
            "borough_vulnerability.csv",
        )
        assert_not_empty(df, "borough_vulnerability.csv")

    def test_vulnerability_scores_in_range(self):
        df = load("borough_vulnerability.csv")
        assert_values_in_range(df, "vulnerability_score", 0, 100, "borough_vulnerability.csv")

    def test_33_boroughs(self):
        df = load("borough_vulnerability.csv")
        n = len(df)
        assert n == 33, (
            f"borough_vulnerability.csv has {n} rows. Expected 33 London boroughs."
        )

    def test_risk_tier_values(self):
        df = load("borough_vulnerability.csv")
        valid_tiers = {"Higher risk", "Medium risk", "Lower risk"}
        found_tiers = set(df["risk_tier"].dropna().unique())
        unexpected = found_tiers - valid_tiers
        assert not unexpected, (
            f"borough_vulnerability.csv has unexpected risk_tier values: {unexpected}"
        )


# ══════════════════════════════════════════════════════════════════
# 06 — stop and search outputs
# ══════════════════════════════════════════════════════════════════

class TestStopSearchProcessed:

    def test_outcomes_summary_columns(self):
        df = load("ss_outcomes_summary.csv")
        assert_columns(
            df, ["total", "arrest_rate", "no_action_rate"],
            "ss_outcomes_summary.csv",
        )

    def test_arrest_rate_plausible(self):
        df = load("ss_outcomes_summary.csv")
        rate = float(df["arrest_rate"].values[0])
        assert 1 <= rate <= 50, (
            f"ss_outcomes_summary.csv: arrest_rate={rate} is implausible. "
            "Expected roughly 10–20% for London stop and search."
        )

    def test_ethnicity_comparison_columns(self):
        df = load("ss_ethnicity_comparison.csv")
        assert_columns(
            df,
            ["ethnicity", "stop_pct", "population_pct", "stop_rate_ratio", "arrest_rate"],
            "ss_ethnicity_comparison.csv",
        )
        assert_not_empty(df, "ss_ethnicity_comparison.csv")

    def test_ethnicity_stops_sum_to_100(self):
        df = load("ss_ethnicity_comparison.csv")
        total = df["stop_pct"].sum()
        assert abs(total - 100) < 1, (
            f"ss_ethnicity_comparison.csv: stop_pct sums to {total:.1f}, expected ~100."
        )

    def test_borough_full_columns(self):
        df = load("ss_borough_full.csv")
        assert_columns(
            df,
            ["borough", "total_searches", "arrest_rate", "black_pct", "lat", "lon"],
            "ss_borough_full.csv",
        )
        assert_not_empty(df, "ss_borough_full.csv")

    def test_no_hardcoded_ethnicity_columns(self):
        """
        Check that ss_ethnicity_comparison.csv does not contain a
        'Mixed' category — the Met Police data has only four categories
        and Mixed should be folded into Other at pipeline time.
        """
        df = load("ss_ethnicity_comparison.csv")
        ethnicities = df["ethnicity"].str.lower().tolist()
        assert "mixed" not in ethnicities, (
            "ss_ethnicity_comparison.csv contains a 'Mixed' category. "
            "This should be folded into Other by build_population_shares() "
            "in 06_process_stop_search.py."
        )


# ══════════════════════════════════════════════════════════════════
# New files — critique fixes (scripts 02, 05, 06)
# ══════════════════════════════════════════════════════════════════

class TestProcessedSchemas:
    """
    Schema and value tests for the three new processed files added
    during the methodology critique and the updated lag correlations.
    """

    # ── shoplifting_lag_correlations.csv ──────────────────────────

    def test_lag_correlations_required_columns(self):
        """
        The updated file must include bootstrapped CI columns.
        If ci_lower / ci_upper are missing the dashboard narrative
        will silently fall back to the old single-point estimate.
        """
        df = load("shoplifting_lag_correlations.csv")
        assert_columns(
            df,
            ["r", "ci_lower", "ci_upper", "n"],
            "shoplifting_lag_correlations.csv",
        )

    def test_lag_correlations_ci_direction(self):
        """ci_lower must not exceed ci_upper for any row."""
        df = load("shoplifting_lag_correlations.csv")
        invalid = df[df["ci_lower"] > df["ci_upper"]]
        assert invalid.empty, (
            f"shoplifting_lag_correlations.csv: {len(invalid)} rows where "
            f"ci_lower > ci_upper:\n{invalid}"
        )

    def test_lag_correlations_r_in_range(self):
        df = load("shoplifting_lag_correlations.csv")
        assert_values_in_range(df.dropna(subset=["r"]), "r", -1, 1,
                               "shoplifting_lag_correlations.csv")

    def test_lag_correlations_n_positive(self):
        df = load("shoplifting_lag_correlations.csv")
        assert (df["n"].dropna() > 0).all(), (
            "shoplifting_lag_correlations.csv: n column contains non-positive values."
        )

    def test_lag_correlations_has_best_lag_flag(self):
        """
        The pipeline should mark exactly one row as best_lag=True.
        The dashboard uses this to select the canonical result.
        """
        df = load("shoplifting_lag_correlations.csv")
        if "best_lag" in df.columns:
            n_best = int(df["best_lag"].sum())
            assert n_best == 1, (
                f"shoplifting_lag_correlations.csv: expected exactly 1 row with "
                f"best_lag=True, found {n_best}."
            )

    # ── ss_narrative_stats.csv ────────────────────────────────────

    def test_narrative_stats_columns(self):
        df = load("ss_narrative_stats.csv")
        assert_columns(df, ["stat", "value"], "ss_narrative_stats.csv")
        assert_not_empty(df, "ss_narrative_stats.csv")

    def test_narrative_stats_expected_keys(self):
        df = load("ss_narrative_stats.csv")
        expected_stats = {
            "deprivation_black_stop_correlation",
            "crime_rate_search_volume_correlation",
        }
        found_stats = set(df["stat"].tolist())
        missing = expected_stats - found_stats
        assert not missing, (
            f"ss_narrative_stats.csv is missing stats: {missing}. "
            "Check build_narrative_stats() in 06_process_stop_search.py."
        )

    def test_narrative_stats_values_are_correlations(self):
        """Correlation values must lie in [-1, 1]."""
        df = load("ss_narrative_stats.csv")
        corr_rows = df[df["stat"].str.contains("correlation", na=False)]
        assert_values_in_range(
            corr_rows, "value", -1, 1, "ss_narrative_stats.csv"
        )

    def test_narrative_stats_no_hardcoded_values(self):
        """
        Guard against the old hardcoded r=-0.62 and r=0.79 being
        inadvertently reproduced if the pipeline re-uses stale data.
        This is a soft check: warn if the values are suspiciously
        round numbers that match the original hardcoded figures.
        """
        df = load("ss_narrative_stats.csv")
        vals = df["value"].tolist()
        # Exact match on the old hardcoded values would be suspicious
        # (real computed values are rarely that round).
        # We do not fail the test — the values could legitimately be
        # close — but we flag it for review.
        for val in vals:
            if round(val, 2) in (-0.62, 0.79):
                import warnings
                warnings.warn(
                    f"ss_narrative_stats.csv contains value {val:.2f} which "
                    "matches an old hardcoded figure. Verify this is a genuine "
                    "computed result and not a pipeline artefact.",
                    UserWarning,
                )

    # ── ss_changepoint_hypotheses.csv ─────────────────────────────

    def test_changepoint_hypotheses_columns(self):
        df = load("ss_changepoint_hypotheses.csv")
        assert_columns(
            df,
            ["hypothesis", "metric", "before", "after"],
            "ss_changepoint_hypotheses.csv",
        )
        assert_not_empty(df, "ss_changepoint_hypotheses.csv")

    def test_changepoint_hypotheses_three_rows(self):
        """
        build_changepoint_hypotheses() produces exactly three rows —
        one per competing explanation.
        """
        df = load("ss_changepoint_hypotheses.csv")
        n = len(df)
        assert n == 3, (
            f"ss_changepoint_hypotheses.csv has {n} rows. Expected 3 "
            "(one per hypothesis). Check build_changepoint_hypotheses() "
            "in 06_process_stop_search.py."
        )

    def test_changepoint_hypotheses_expected_names(self):
        df = load("ss_changepoint_hypotheses.csv")
        expected = {
            "More enforcement activity",
            "Changed recording practice",
            "Real increase in drug activity",
        }
        found = set(df["hypothesis"].tolist())
        missing = expected - found
        assert not missing, (
            f"ss_changepoint_hypotheses.csv is missing hypotheses: {missing}. "
            "Found: {found}"
        )

    def test_changepoint_hypotheses_before_after_numeric(self):
        df = load("ss_changepoint_hypotheses.csv")
        for col in ["before", "after"]:
            non_numeric = df[col].apply(
                lambda v: not isinstance(v, (int, float))
                          and not str(v).replace(".", "").replace("-", "").isdigit()
            )
            assert not non_numeric.any(), (
                f"ss_changepoint_hypotheses.csv: column '{col}' contains "
                f"non-numeric values: {df.loc[non_numeric, col].tolist()}"
            )

    def test_changepoint_hypotheses_supports_column_present(self):
        """
        The 'supports' column is displayed in the dashboard table.
        Not strictly required but warn if absent.
        """
        df = load("ss_changepoint_hypotheses.csv")
        if "supports" not in df.columns:
            import warnings
            warnings.warn(
                "ss_changepoint_hypotheses.csv has no 'supports' column. "
                "The dashboard table will display without an evidence "
                "assessment column.",
                UserWarning,
            )

    # ── borough_weight_sensitivity.csv ────────────────────────────

    def test_weight_sensitivity_columns(self):
        df = load("borough_weight_sensitivity.csv")
        assert_columns(
            df,
            ["borough", "scenario", "score", "rank"],
            "borough_weight_sensitivity.csv",
        )
        assert_not_empty(df, "borough_weight_sensitivity.csv")

    def test_weight_sensitivity_four_scenarios(self):
        """
        build_weight_sensitivity() tests four weighting schemes.
        """
        df = load("borough_weight_sensitivity.csv")
        n_scenarios = df["scenario"].nunique()
        assert n_scenarios == 4, (
            f"borough_weight_sensitivity.csv has {n_scenarios} scenarios. "
            "Expected 4 (Base, Deprivation-heavy, Crime-trend-heavy, Equal)."
        )

    def test_weight_sensitivity_expected_scenario_names(self):
        df = load("borough_weight_sensitivity.csv")
        expected = {"Base", "Deprivation-heavy", "Crime-trend-heavy", "Equal"}
        found = set(df["scenario"].unique())
        missing = expected - found
        assert not missing, (
            f"borough_weight_sensitivity.csv is missing scenarios: {missing}. "
            f"Found: {found}"
        )

    def test_weight_sensitivity_33_boroughs_per_scenario(self):
        """Each scenario must cover all 33 boroughs."""
        df = load("borough_weight_sensitivity.csv")
        counts = df.groupby("scenario")["borough"].count()
        wrong = counts[counts != 33]
        assert wrong.empty, (
            f"borough_weight_sensitivity.csv: some scenarios do not have "
            f"exactly 33 boroughs:\n{wrong}"
        )

    def test_weight_sensitivity_ranks_valid(self):
        """
        Ranks must cover 1–33. Tied scores legitimately produce tied ranks
        (e.g. rank 7 appearing twice, rank 8 skipped) when using min-rank.
        We check coverage rather than strict sequence.
        """
        df = load("borough_weight_sensitivity.csv")
        for scenario, group in df.groupby("scenario"):
            ranks = group["rank"].tolist()
            assert len(ranks) == 33, (
                f"borough_weight_sensitivity.csv: '{scenario}' has {len(ranks)} rows, expected 33."
            )
            assert min(ranks) == 1, (
                f"borough_weight_sensitivity.csv: '{scenario}' minimum rank is {min(ranks)}, expected 1."
            )
            assert max(ranks) <= 33, (
                f"borough_weight_sensitivity.csv: '{scenario}' has rank > 33: {max(ranks)}."
            )

    def test_weight_sensitivity_scores_non_negative(self):
        df = load("borough_weight_sensitivity.csv")
        assert (df["score"] >= 0).all(), (
            "borough_weight_sensitivity.csv: score has negative values."
        )