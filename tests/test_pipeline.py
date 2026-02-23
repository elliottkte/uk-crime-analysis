"""

Run from the project root:
    python -m pytest tests/ -v
"""

import sys
import os
import importlib.util
import pytest
import pandas as pd
import numpy as np

# ── Import helpers ────────────────────────────────────────────────
# Processing filenames start with digits so can't be imported directly.
# We use importlib to load them by file path.

def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

script01 = _load_module("clean_street",    os.path.join(_ROOT, "processing", "01_clean_street_data.py"))
script05 = _load_module("vulnerability",   os.path.join(_ROOT, "processing", "05_vulnerability_index.py"))


# ── 01: extract_force ─────────────────────────────────────────────

class TestExtractForce:
    def test_metropolitan(self):
        assert script01.extract_force("2023-01-metropolitan-street.csv") == "metropolitan"

    def test_city_of_london(self):
        # split('-')[2] gives 'city' from '2023-01-city-of-london-...'
        assert script01.extract_force("2023-01-city-of-london-street.csv") == "city"

    def test_full_path(self):
        path = os.path.join("data", "raw", "2024-06-metropolitan-street.csv")
        assert script01.extract_force(path) == "metropolitan"

    def test_malformed_returns_unknown(self):
        assert script01.extract_force("notavalidfilename.csv") == "unknown"

    def test_case_insensitive(self):
        assert script01.extract_force("2023-01-Metropolitan-street.csv") == "metropolitan"


# ── 01: clean() ───────────────────────────────────────────────────

class TestClean:
    def _make_df(self, **overrides) -> pd.DataFrame:
        """Return a minimal valid raw street crime DataFrame."""
        base = {
            "Crime ID":               ["abc123", "def456", "ghi789"],
            "Month":                  ["2023-01", "2024-06", "2025-12"],
            "Reported by":            ["Met", "Met", "Met"],
            "Falls within":           ["Met", "Met", "Met"],
            "Longitude":              [-0.1, -0.2, -0.3],
            "Latitude":               [51.5, 51.6, 51.4],
            "Location":               ["On or near X", "On or near Y", "On or near Z"],
            "LSOA code":              ["E01000001", "E01000002", "E01000003"],
            "LSOA name":              ["Camden 001A", "Hackney 002B", "Lambeth 003C"],
            "Crime type":             ["Shoplifting", "Drugs", "Burglary"],
            "Last outcome category":  ["Under investigation"] * 3,
            "Context":                [None, None, None],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_returns_dataframe(self):
        assert isinstance(script01.clean(self._make_df()), pd.DataFrame)

    def test_drops_unwanted_columns(self):
        result = script01.clean(self._make_df())
        for col in ["Context", "Falls within", "Reported by"]:
            assert col not in result.columns

    def test_snake_case_columns(self):
        result = script01.clean(self._make_df())
        for col in result.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"
            assert " " not in col,     f"Column '{col}' contains spaces"

    def test_month_is_datetime(self):
        result = script01.clean(self._make_df())
        assert pd.api.types.is_datetime64_any_dtype(result["month"])

    def test_drops_missing_coordinates(self):
        df = self._make_df()
        df.loc[0, "Longitude"] = None
        assert len(script01.clean(df)) == 2

    def test_drops_rows_outside_date_range(self):
        df = self._make_df(Month=["2022-12", "2023-01", "2026-01"])
        result = script01.clean(df)
        assert len(result) == 1
        assert result.iloc[0]["month"].year == 2023

    def test_strips_crime_type_whitespace(self):
        df = self._make_df(**{"Crime type": ["  Shoplifting  ", "Drugs", "Burglary"]})
        result = script01.clean(df)
        assert result["crime_type"].iloc[0] == "Shoplifting"

    def test_expected_crime_types_non_empty(self):
        assert len(script01.EXPECTED_CRIME_TYPES) > 0
        assert all(isinstance(t, str) for t in script01.EXPECTED_CRIME_TYPES)

    def test_date_range_constants(self):
        lo, hi = script01.DATE_RANGE
        assert lo < hi


# ── 02: ONS monthly row parser ────────────────────────────────────

class TestFoodInflationParser:
    """
    Tests the logic that filters monthly rows from the ONS CSV.
    Annual rows: '2023', quarterly: '2023 Q1', monthly: '2023 JAN'.
    Only monthly rows should be kept.
    """

    def _is_monthly(self, period: str) -> bool:
        tokens = period.strip().split()
        return (
            len(tokens) == 2
            and len(tokens[0]) == 4
            and tokens[0].isdigit()
            and len(tokens[1]) == 3
            and tokens[1].isalpha()
        )

    def test_monthly_rows_accepted(self):
        for period in ["2023 JAN", "2024 DEC", "2025 SEP", "2023 FEB"]:
            assert self._is_monthly(period), f"Should accept: {period}"

    def test_annual_rows_rejected(self):
        for period in ["2023", "2024", "2025"]:
            assert not self._is_monthly(period), f"Should reject: {period}"

    def test_quarterly_rows_rejected(self):
        for period in ["2023 Q1", "2024 Q4", "2025 Q2"]:
            assert not self._is_monthly(period), f"Should reject: {period}"

    def test_header_and_junk_rejected(self):
        for period in ["", "Title", "Important notes", "Period"]:
            assert not self._is_monthly(period), f"Should reject: {period}"


# ── 05: minmax normalisation ──────────────────────────────────────

class TestMinmax:
    def test_output_range(self):
        result = script05.minmax(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_flat_series_returns_half(self):
        result = script05.minmax(pd.Series([7.0, 7.0, 7.0]))
        assert (result == 0.5).all()

    def test_preserves_index(self):
        s = pd.Series([10.0, 20.0, 30.0], index=[5, 6, 7])
        assert list(script05.minmax(s).index) == [5, 6, 7]

    def test_single_value(self):
        assert script05.minmax(pd.Series([42.0])).iloc[0] == pytest.approx(0.5)


# ── 05: risk tier thresholds ─────────────────────────────────────

class TestRiskThresholds:
    def _tier(self, score: float) -> str:
        if score >= script05.RISK_THRESHOLDS["higher"]:
            return "Higher risk"
        if score <= script05.RISK_THRESHOLDS["lower"]:
            return "Lower risk"
        return "Medium risk"

    def test_higher_boundary(self):
        assert self._tier(script05.RISK_THRESHOLDS["higher"]) == "Higher risk"
        assert self._tier(100) == "Higher risk"

    def test_lower_boundary(self):
        assert self._tier(script05.RISK_THRESHOLDS["lower"]) == "Lower risk"
        assert self._tier(0) == "Lower risk"

    def test_medium_band(self):
        lo = script05.RISK_THRESHOLDS["lower"]
        hi = script05.RISK_THRESHOLDS["higher"]
        assert self._tier(lo + 1) == "Medium risk"
        assert self._tier(hi - 1) == "Medium risk"


# ── 06: stop and search helpers ───────────────────────────────────

class TestIsArrest:
    def _is_arrest(self, outcome) -> bool:
        if pd.isna(outcome):
            return False
        return "arrest" in str(outcome).lower()

    def test_arrest_detected(self):
        assert self._is_arrest("Arrest")
        assert self._is_arrest("arrest")
        assert self._is_arrest("Local arrest")

    def test_non_arrest_outcomes(self):
        assert not self._is_arrest("A no further action disposal")
        assert not self._is_arrest("Penalty Notice for Disorder")
        assert not self._is_arrest("Community resolution")

    def test_null_outcome(self):
        assert not self._is_arrest(None)
        assert not self._is_arrest(float("nan"))


# ── Data contract: processed CSV schemas ─────────────────────────

PROCESSED = os.path.join(_ROOT, "data", "processed")


def _load_processed(filename: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED, filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found — run pipeline first")
    return pd.read_csv(path)


class TestProcessedSchemas:
    def test_street_clean_required_columns(self):
        df = _load_processed("street_clean.csv")
        required = {"crime_id", "month", "longitude", "latitude",
                    "lsoa_code", "crime_type", "last_outcome_category", "force"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_street_clean_forces(self):
        df = _load_processed("street_clean.csv")
        unexpected = set(df["force"].unique()) - {"metropolitan", "city"}
        assert not unexpected, f"Unexpected force values: {unexpected}"

    def test_crime_indexed_index_value_complete(self):
        df = _load_processed("crime_indexed.csv")
        assert "index_value" in df.columns
        assert df["index_value"].notna().all()

    def test_ss_ethnicity_has_ratio_column(self):
        df = _load_processed("ss_ethnicity_comparison.csv")
        assert "stop_rate_ratio" in df.columns
        assert len(df) >= 4

    def test_vulnerability_scores_in_range(self):
        df = _load_processed("borough_vulnerability.csv")
        assert df["vulnerability_score"].between(0, 100).all(), \
            "Vulnerability scores outside 0–100"
        valid_tiers = {"Higher risk", "Medium risk", "Lower risk"}
        unexpected = set(df["risk_tier"].unique()) - valid_tiers
        assert not unexpected, f"Unexpected risk tiers: {unexpected}"

    def test_borough_outliers_all_three_categories_present(self):
        df = _load_processed("borough_outliers_deprivation.csv")
        categories = set(df["dominant_outlier"].unique())
        assert "Deprived and high crime" in categories, \
            "No deprived boroughs — check quadrant classification logic"
        assert "Affluent but high crime" in categories

    def test_shoplifting_scenarios_ordering(self):
        df = _load_processed("shoplifting_scenarios.csv")
        for col in ["optimistic", "central", "pessimistic"]:
            assert col in df.columns, f"Missing column: {col}"
        assert (df["pessimistic"] >= df["central"]).all(), \
            "Pessimistic scenario should be >= central"
        assert (df["central"] >= df["optimistic"]).all(), \
            "Central scenario should be >= optimistic"

    def test_food_inflation_date_range(self):
        df = _load_processed("food_inflation_ons.csv")
        df["month"] = pd.to_datetime(df["month"])
        assert df["month"].min().year >= 2023
        assert df["month"].max().year <= 2025
        assert len(df) == 36, f"Expected 36 monthly rows, got {len(df)}"