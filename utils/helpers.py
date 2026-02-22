"""
utils/helpers.py
----------------
Small general-purpose helper functions used across sections.
These are pure Python with no Streamlit or Plotly dependencies
so they can also be used safely inside processing scripts.

Import example:
    from utils.helpers import safe_get_borough, get_corr, fmt_pct
"""

import pandas as pd


# ── DataFrame helpers ─────────────────────────────────────────────

def safe_get_borough(df: pd.DataFrame, borough_name: str) -> pd.Series | None:
    """
    Return the first row for *borough_name* from a DataFrame that
    has a 'borough' column.

    Returns None rather than raising if the borough is absent or
    the column doesn't exist, so callers can guard with a simple
    `if row is not None` check.

    Args:
        df:           DataFrame with a 'borough' column.
        borough_name: Exact borough name to look up.

    Returns:
        A pandas Series (row), or None if not found.
    """
    if "borough" not in df.columns:
        return None
    mask = df["borough"] == borough_name
    if not mask.any():
        return None
    return df.loc[mask].iloc[0]


def get_corr(
    domain_corr: pd.DataFrame,
    crime: str,
    domain: str,
) -> float:
    """
    Retrieve a single correlation value from the domain_crime_correlations
    DataFrame by crime type and deprivation domain.

    Returns float('nan') rather than raising if the combination is
    not found, so callers can format it safely without a try/except.

    Args:
        domain_corr: DataFrame with columns crime_type,
                     deprivation_domain, correlation.
        crime:       Crime type label (e.g. 'Burglary').
        domain:      Deprivation domain label (e.g. 'Living Env').

    Returns:
        Correlation value as float, or nan if not found.
    """
    mask = (
        (domain_corr["crime_type"]         == crime) &
        (domain_corr["deprivation_domain"] == domain)
    )
    vals = domain_corr.loc[mask, "correlation"].values
    return float(vals[0]) if len(vals) else float("nan")


def get_ethnicity_val(
    ethnicity_df: pd.DataFrame,
    ethnicity: str,
    column: str,
) -> float:
    """
    Retrieve a single value from the stop and search ethnicity
    comparison DataFrame.

    Returns float('nan') if the ethnicity or column is not found.

    Args:
        ethnicity_df: DataFrame with an 'ethnicity' column.
        ethnicity:    Broad ethnicity group (e.g. 'Black', 'White').
        column:       Column to retrieve (e.g. 'stop_pct', 'arrest_rate').

    Returns:
        Value as float, or nan if not found.
    """
    if column not in ethnicity_df.columns:
        return float("nan")
    mask = ethnicity_df["ethnicity"] == ethnicity
    vals = ethnicity_df.loc[mask, column].values
    return float(vals[0]) if len(vals) else float("nan")


def pct_change(df: pd.DataFrame, col_before: str, col_after: str) -> pd.Series:
    """
    Compute percentage change between two columns, returning NaN
    where the before value is zero.

    Args:
        df:          Source DataFrame.
        col_before:  Column name for the starting value.
        col_after:   Column name for the ending value.

    Returns:
        A pandas Series of percentage change values, rounded to 1 dp.
    """
    return (
        (df[col_after] - df[col_before])
        / df[col_before].replace(0, float("nan"))
        * 100
    ).round(1)


# ── Formatting helpers ────────────────────────────────────────────

def fmt_pct(value: float, sign: bool = True, decimals: int = 0) -> str:
    """
    Format a float as a percentage string.

    Args:
        value:    Numeric value (e.g. 53.5 for 53.5%).
        sign:     If True, prepend '+' for positive values.
        decimals: Number of decimal places.

    Returns:
        Formatted string e.g. '+53%', '-18.5%', '7.1%'.
    """
    fmt = f"+.{decimals}f" if sign else f".{decimals}f"
    return f"{value:{fmt}}%"


def fmt_count(value: float | int) -> str:
    """Format a number with thousands separator."""
    return f"{int(value):,}"


def fmt_rate(value: float, decimals: int = 1) -> str:
    """Format a rate (e.g. per 1,000) to a given number of decimal places."""
    return f"{value:.{decimals}f}"


# ── Validation helpers ────────────────────────────────────────────

def check_required_columns(
    df: pd.DataFrame,
    required: list[str],
    label: str = "DataFrame",
) -> list[str]:
    """
    Check that all required columns are present.

    Returns a list of missing column names (empty list if all present).
    Useful for giving clear error messages in processing scripts.

    Args:
        df:       DataFrame to check.
        required: List of expected column names.
        label:    Human-readable name for the DataFrame, used in messages.

    Returns:
        List of missing column names.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING [{label}]: missing columns: {missing}")
    return missing


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill: float = 0.0,
) -> pd.Series:
    """
    Divide two Series, replacing division-by-zero results with `fill`.

    Args:
        numerator:   Numerator Series.
        denominator: Denominator Series.
        fill:        Value to use where denominator is zero.

    Returns:
        Result Series with zero-division values replaced by `fill`.
    """
    return numerator.div(denominator.replace(0, float("nan"))).fillna(fill)