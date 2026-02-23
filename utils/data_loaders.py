"""
utils/data_loaders.py
---------------------
All data loading functions for the dashboard.
Every function is decorated with @st.cache_data or @st.cache_resource
so that data is only read from disk once per session.

Import example:
    from utils.data_loaders import load_full_street, load_model
"""

import os
import joblib
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────
_PROCESSED = os.path.join("data", "processed")
_MODELS    = "models"


def _path(filename: str) -> str:
    return os.path.join(_PROCESSED, filename)


# ── Shared / multi-section ────────────────────────────────────────

@st.cache_data
def load_full_street() -> pd.DataFrame:
    """
    Main street crime dataset. Used by multiple sections.
    Returns a DataFrame with columns including month (datetime),
    crime_type, lsoa_code, borough, year.
    """
    try:
        df = pd.read_csv(_path("street_clean.csv"))
        df["month"] = pd.to_datetime(df["month"])
        df["year"]  = df["month"].dt.year
        return df
    except FileNotFoundError:
        st.error(
            "street_clean.csv not found. "
            "Run processing/01_clean_street_data.py first."
        )
        st.stop()
    except Exception as e:
        st.error(f"Could not load street crime data: {e}")
        st.stop()


@st.cache_resource
def load_model():
    """
    Trained Random Forest model. Cached as a resource so it is not
    reloaded on every user interaction.
    """
    model_path = os.path.join(_MODELS, "crime_rate_model.pkl")
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(
            "crime_rate_model.pkl not found. "
            "Run processing/04_train_model.py first."
        )
        st.stop()
    except Exception as e:
        st.error(f"Could not load predictive model: {e}")
        st.stop()


@st.cache_data
def load_modelling_data() -> pd.DataFrame:
    try:
        return pd.read_csv(_path("modelling_data.csv"))
    except FileNotFoundError:
        st.error(
            "modelling_data.csv not found. "
            "Run processing/04_train_model.py first."
        )
        st.stop()
    except Exception as e:
        st.error(f"Could not load modelling data: {e}")
        st.stop()


# ── Street summary (replaces load_full_street for deployment) ─────

@st.cache_data
def load_street_summary() -> dict:
    """
    Returns pre-computed monthly aggregations that replace load_full_street()
    for dashboard rendering. These are small files suitable for git/deployment.

    Keys:
        monthly_by_crime    – monthly counts per crime type
        monthly_totals      – monthly total across all crimes
        crime_annual_change – crime type counts by year + % change
        headline_totals     – scalar totals (total_crimes, date range)
    """
    files = {
        "monthly_by_crime":    "monthly_by_crime.csv",
        "monthly_totals":      "monthly_totals.csv",
        "crime_annual_change": "crime_annual_change.csv",
        "headline_totals":     "headline_totals.csv",
    }

    result  = {}
    missing = []

    for key, filename in files.items():
        try:
            df = pd.read_csv(_path(filename))
            if "month" in df.columns:
                df["month"] = pd.to_datetime(df["month"])
            result[key] = df
        except FileNotFoundError:
            missing.append(filename)
        except Exception as e:
            st.error(f"Could not load {filename}: {e}")
            st.stop()

    if missing:
        st.error(
            f"Street summary files not found: {', '.join(missing)}. "
            "Run processing/07_precompute_summary.py first."
        )
        st.stop()

    return result


# ── Economic Crime section ────────────────────────────────────────

@st.cache_data
def load_economic_crime_data() -> dict:
    """
    Returns a dict of DataFrames used by the Economic Crime section.

    Keys:
        indexed     – crime_indexed.csv
        corr_df     – food_inflation_correlations.csv
        lag_df      – shoplifting_lag_correlations.csv
                      Now includes ci_lower, ci_upper, n, best_lag columns
                      from bootstrapped correlation (script 02).
        decomp      – shoplifting_decomposition.csv
        changepoint – drugs_changepoint.csv
        borough     – borough_shoplifting_trend.csv
        food        – food_inflation_ons.csv
    """
    files = {
        "indexed":     "crime_indexed.csv",
        "corr_df":     "food_inflation_correlations.csv",
        "lag_df":      "shoplifting_lag_correlations.csv",
        "decomp":      "shoplifting_decomposition.csv",
        "changepoint": "drugs_changepoint.csv",
        "borough":     "borough_shoplifting_trend.csv",
        "food":        "food_inflation_ons.csv",
    }

    date_cols = {
        "indexed": "month",
        "decomp":  "month",
        "food":    "month",
    }

    result  = {}
    missing = []

    for key, filename in files.items():
        try:
            df = pd.read_csv(_path(filename))
            if key in date_cols:
                df[date_cols[key]] = pd.to_datetime(df[date_cols[key]])
            result[key] = df
        except FileNotFoundError:
            missing.append(filename)
        except Exception as e:
            st.error(f"Could not load {filename}: {e}")
            st.stop()

    if missing:
        st.error(
            f"Economic crime data files not found: {', '.join(missing)}. "
            "Run processing/02_economic_analysis.py first."
        )
        st.stop()

    return result


# ── Crime & Deprivation section ───────────────────────────────────

@st.cache_data
def load_deprivation_data() -> dict:
    """
    Returns a dict of DataFrames used by the Crime & Deprivation section.

    Keys:
        domain_corr – domain_crime_correlations.csv
        borough_dep – borough_outliers_deprivation.csv
    """
    files = {
        "domain_corr": "domain_crime_correlations.csv",
        "borough_dep": "borough_outliers_deprivation.csv",
    }

    result  = {}
    missing = []

    for key, filename in files.items():
        try:
            result[key] = pd.read_csv(_path(filename))
        except FileNotFoundError:
            missing.append(filename)
        except Exception as e:
            st.error(f"Could not load {filename}: {e}")
            st.stop()

    if missing:
        st.error(
            f"Deprivation data files not found: {', '.join(missing)}. "
            "Run processing/03_deprivation_correlations.py first."
        )
        st.stop()

    return result


# ── Policing Response section ─────────────────────────────────────

@st.cache_data
def load_policing_data() -> dict:
    """
    Returns a dict of DataFrames used by the Policing Response section.

    Keys:
        outcomes_summary       – ss_outcomes_summary.csv
        ethnicity              – ss_ethnicity_comparison.csv
        outcomes_by_search     – ss_outcomes_by_search.csv
        ss_borough             – ss_borough_full.csv
        drugs_comparison       – ss_drugs_comparison.csv
        monthly_search_type    – ss_monthly_search_type.csv
        narrative_stats        – ss_narrative_stats.csv
                                 Contains data-derived inline statistics:
                                 deprivation_black_stop_correlation,
                                 crime_rate_search_volume_correlation
        changepoint_hypotheses – ss_changepoint_hypotheses.csv
                                 Three competing hypotheses for the Aug 2024
                                 drugs spike with before/after metrics.
    """
    files = {
        "outcomes_summary":       "ss_outcomes_summary.csv",
        "ethnicity":              "ss_ethnicity_comparison.csv",
        "outcomes_by_search":     "ss_outcomes_by_search.csv",
        "ss_borough":             "ss_borough_full.csv",
        "drugs_comparison":       "ss_drugs_comparison.csv",
        "monthly_search_type":    "ss_monthly_search_type.csv",
        "narrative_stats":        "ss_narrative_stats.csv",
        "changepoint_hypotheses": "ss_changepoint_hypotheses.csv",
    }

    date_cols = {
        "drugs_comparison":    "month",
        "monthly_search_type": "month",
    }

    # Files that are optional — generated only if upstream dependencies exist.
    # If absent, the section uses fallback values rather than stopping.
    optional = {"narrative_stats", "changepoint_hypotheses"}

    result  = {}
    missing = []

    for key, filename in files.items():
        try:
            df = pd.read_csv(_path(filename))
            if key in date_cols:
                df[date_cols[key]] = pd.to_datetime(df[date_cols[key]])
            result[key] = df
        except FileNotFoundError:
            if key in optional:
                result[key] = pd.DataFrame()
            else:
                missing.append(filename)
        except Exception as e:
            st.error(f"Could not load {filename}: {e}")
            st.stop()

    if missing:
        st.error(
            f"Policing data files not found: {', '.join(missing)}. "
            "Run processing/06_process_stop_search.py first."
        )
        st.stop()

    return result


# ── Where is London Headed section ───────────────────────────────

@st.cache_data
def load_outlook_data() -> dict:
    """
    Returns a dict of DataFrames used by the outlook section.

    Keys:
        vulnerability      – borough_vulnerability.csv
        trajectory         – crime_trajectory.csv
        scenarios          – shoplifting_scenarios.csv
        weight_sensitivity – borough_weight_sensitivity.csv
                             Long DataFrame showing borough rankings across
                             four alternative weighting schemes. Used in
                             methodology expander to show which boroughs
                             are robustly high-risk regardless of weights.
    """
    files = {
        "vulnerability":      "borough_vulnerability.csv",
        "trajectory":         "crime_trajectory.csv",
        "scenarios":          "shoplifting_scenarios.csv",
        "weight_sensitivity": "borough_weight_sensitivity.csv",
    }

    date_cols = {
        "scenarios": "month",
    }

    # weight_sensitivity is optional — generated by script 05 but not
    # strictly required for the section to render.
    optional = {"weight_sensitivity"}

    result  = {}
    missing = []

    for key, filename in files.items():
        try:
            df = pd.read_csv(_path(filename))
            if key in date_cols:
                df[date_cols[key]] = pd.to_datetime(df[date_cols[key]])
            result[key] = df
        except FileNotFoundError:
            if key in optional:
                result[key] = pd.DataFrame()
            else:
                missing.append(filename)
        except Exception as e:
            st.error(f"Could not load {filename}: {e}")
            st.stop()

    if missing:
        st.error(
            f"Outlook data files not found: {', '.join(missing)}. "
            "Run processing/05_vulnerability_index.py first."
        )
        st.stop()

    return result


# ── Narrative stats helper ────────────────────────────────────────

def get_narrative_stat(narrative_stats: pd.DataFrame, stat_name: str) -> float:
    """
    Retrieve a single value from the narrative_stats DataFrame by stat name.

    Used by section renderers to inject data-derived statistics into
    narrative text rather than hardcoding them.

    Args:
        narrative_stats: DataFrame with 'stat' and 'value' columns,
                         loaded from ss_narrative_stats.csv.
        stat_name:       Name of the stat to retrieve.

    Returns:
        Float value, or float('nan') if not found or DataFrame is empty.
    """
    if narrative_stats.empty or "stat" not in narrative_stats.columns:
        return float("nan")
    mask = narrative_stats["stat"] == stat_name
    vals = narrative_stats.loc[mask, "value"].values
    return float(vals[0]) if len(vals) else float("nan")