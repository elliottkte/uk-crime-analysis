"""
07_precompute_summary.py
------------------------
Pre-computes monthly aggregations from street_clean.csv so that
the dashboard does not need to load the full 641MB file at runtime.

This enables deployment on Streamlit Community Cloud where the large
street_clean.csv cannot be committed to GitHub.

Outputs (all small, suitable for git):
    data/processed/monthly_by_crime.csv   — monthly counts per crime type
    data/processed/monthly_totals.csv     — monthly total across all crimes
    data/processed/crime_annual_change.csv — crime type counts by year + % change

Run from project root:
    python processing/07_precompute_summary.py
"""

import os
import pandas as pd

STREET_PATH = os.path.join("data", "processed", "street_clean.csv")
OUT_DIR     = os.path.join("data", "processed")


def main():
    print("07_precompute_summary.py")
    print("=" * 50)

    if not os.path.exists(STREET_PATH):
        raise FileNotFoundError(
            f"{STREET_PATH} not found. Run 01_clean_street_data.py first."
        )

    print("Loading street crime data...")
    street = pd.read_csv(STREET_PATH)
    street["month"] = pd.to_datetime(street["month"])
    street["year"]  = street["month"].dt.year
    print(f"  {len(street):,} records")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Monthly counts per crime type ─────────────────────────
    print("  Building monthly_by_crime.csv...")
    monthly_by_crime = (
        street.groupby(["crime_type", "month"])
        .size()
        .reset_index(name="count")
    )
    monthly_by_crime.to_csv(
        os.path.join(OUT_DIR, "monthly_by_crime.csv"), index=False
    )
    print(f"    ✓ {len(monthly_by_crime):,} rows")

    # ── 2. Monthly totals (all crime types combined) ──────────────
    print("  Building monthly_totals.csv...")
    monthly_totals = (
        street.groupby("month")
        .size()
        .reset_index(name="count")
    )
    monthly_totals.to_csv(
        os.path.join(OUT_DIR, "monthly_totals.csv"), index=False
    )
    print(f"    ✓ {len(monthly_totals):,} rows")

    # ── 3. Annual counts per crime type + % change 2023 vs 2025 ──
    print("  Building crime_annual_change.csv...")
    annual = (
        street.groupby(["crime_type", "year"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    if 2023 in annual.columns and 2025 in annual.columns:
        annual["change_pct"] = (
            (annual[2025] - annual[2023])
            / annual[2023].replace(0, float("nan")) * 100
        ).round(1)
    annual.columns = [str(c) for c in annual.columns]
    annual.to_csv(
        os.path.join(OUT_DIR, "crime_annual_change.csv"), index=False
    )
    print(f"    ✓ {len(annual)} crime types")

    # ── 4. Headline totals scalar ─────────────────────────────────
    print("  Building headline_totals.csv...")
    headline = pd.DataFrame([{
        "total_crimes": len(street),
        "date_from":    street["month"].min().strftime("%Y-%m"),
        "date_to":      street["month"].max().strftime("%Y-%m"),
    }])
    headline.to_csv(
        os.path.join(OUT_DIR, "headline_totals.csv"), index=False
    )
    print(f"    ✓ 1 row")

    print(f"\n✓ Summary outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()