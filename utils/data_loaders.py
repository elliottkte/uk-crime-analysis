"""
utils/data_loaders.py
---------------------
All data loading functions for the dashboard.
Every function is decorated with @st.cache_data or @st.cache_resource
so that data is only read from disk once per session.

Fix (critique): added build_imd_label_map() which constructs a
{actual_feature_name: display_label} mapping at runtime from the
model's stored feature_names_in_ attribute. This resolves the silent
mismatch between IMD_DOMAIN_LABELS keys (which used short strings)
and the actual long column names produced by the IMD CSV, which caused
the feature importance chart to silently display wrong or empty labels.

Fix (critique): script 06 requires drugs_changepoint.csv (from script
02) to build the hypothesis table. The dependency note is now surfaced
in the returned dict so section renderers can show a clear message
rather than a silent empty DataFrame.
"""

import os
import joblib
import pandas as pd
import streamlit as st

from utils.constants import IMD_DOMAIN_LABELS, IMD_DOMAIN_SHORT_LABELS

# ── Paths ─────────────────────────────────────────────────────────
_PROCESSED = os.path.join("data", "processed")
_MODELS    = "models"


def _path(filename: str) -> str:
    return os.path.join(_PROCESSED, filename)


# ── IMD label mapping ─────────────────────────────────────────────

def build_imd_label_map(model) -> dict:
    """
    Build a {actual_feature_name: display_label} mapping from the
    model's stored feature_names_in_ attribute.

    Fix (critique): IMD_DOMAIN_LABELS in constants.py previously used
    short keys that did not match the actual long column names from the
    IMD CSV. This function resolves that mismatch at runtime by doing
    substring matching against the full and short label dicts, falling
    back to the raw feature name if no match is found.

    Args:
        model: Trained RandomForest model. Must have feature_names_in_
               attribute set by 04_train_model.py.

    Returns:
        dict mapping actual feature column name → readable display label.
    """
    if not hasattr(model, "feature_names_in_"):
        # Old model without stored feature names — return identity map
        return {}

    label_map = {}
    for feat in model.feature_names_in_:
        feat_lower = feat.lower()

        # Try exact match against full IMD_DOMAIN_LABELS keys first
        matched = False
        for key, label in IMD_DOMAIN_LABELS.items():
            if key in feat_lower or feat_lower in key:
                label_map[feat] = label
                matched = True
                break

        if not matched:
            # Try partial match against short labels
            for keyword, label in IMD_DOMAIN_SHORT_LABELS.items():
                if keyword in feat_lower:
                    label_map[feat] = label
                    matched = True
                    break

        if not matched:
            # Fallback: use the raw feature name, cleaned up
            label_map[feat] = feat.replace("_", " ").title()

    return label_map


# ── Shared / multi-section ────────────────────────────────────────

@st.cache_data
def load_full_street() -> pd.DataFrame:
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


# ── Street summary ────────────────────────────────────────────────

@st.cache_data
def load_street_summary() -> dict:
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
                      Includes ci_lower, ci_upper, n, best_lag columns.
        decomp      – shoplifting_decomposition.csv
                      Now includes stl_reliable (bool) and n_months (int)
                      columns written by build_decomposition() in script 02.
                      Renderers should check stl_reliable and show a caveat
                      when False.
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

    The 'changepoint_hypotheses' key returns an empty DataFrame with a
    '_missing_reason' attribute if the file is absent, so the section
    renderer can show a clear informative message explaining which
    upstream script needs to be rerun, rather than a generic empty state.
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
                empty = pd.DataFrame()
                if key == "changepoint_hypotheses":
                    # Carry a reason so the renderer can be specific
                    empty._missing_reason = (
                        "ss_changepoint_hypotheses.csv not found. "
                        "Run scripts 02 and 06 in sequence: "
                        "drugs_changepoint.csv (script 02) must exist "
                        "before script 06 can build the hypothesis table."
                    )
                result[key] = empty
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


# ── Where is London Headed section ────────────────────────────────

@st.cache_data
def load_outlook_data() -> dict:
    files = {
        "vulnerability":      "borough_vulnerability.csv",
        "trajectory":         "crime_trajectory.csv",
        "scenarios":          "shoplifting_scenarios.csv",
        "weight_sensitivity": "borough_weight_sensitivity.csv",
    }

    date_cols = {
        "scenarios": "month",
    }

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


# ── Narrative stats helper ─────────────────────────────────────────

def get_narrative_stat(narrative_stats: pd.DataFrame, stat_name: str) -> float:
    if narrative_stats.empty or "stat" not in narrative_stats.columns:
        return float("nan")
    mask = narrative_stats["stat"] == stat_name
    vals = narrative_stats.loc[mask, "value"].values
    return float(vals[0]) if len(vals) else float("nan")