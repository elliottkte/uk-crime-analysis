"""
sections/the_story.py
---------------------
'The Story' section — overview of the dashboard's five key findings,
a crime-type % change chart, and the economic backdrop timeline.
"""

import plotly.graph_objects as go
import streamlit as st

from utils.charts import (
    apply_base_layout,
    crime_change_overview_chart,
    monthly_total_chart,
    style_xaxis,
    style_yaxis,
)
from utils.constants import CHART_CONFIG
from utils.data_loaders import load_street_summary


def render():
    st.title("London Crime & the Cost of Living")
    st.markdown("""
    Between 2023 and 2025, London's recorded crime data reflects one of the
    most difficult economic periods in a generation. As household finances
    were squeezed by food inflation, rising energy bills, and stagnant wages,
    certain crimes surged in ways the data links directly to financial pressure.

    This dashboard examines three years of Metropolitan and City of London
    Police data: which crimes rose, which fell, which areas were hardest hit,
    and what the evidence suggests about where London is headed.
    """)

    summary = load_street_summary()

    headline      = summary["headline_totals"].iloc[0]
    annual        = summary["crime_annual_change"].copy()
    monthly_all   = summary["monthly_totals"]

    # ── Derive headline figures ───────────────────────────────────
    total_crimes = int(headline["total_crimes"])

    annual = annual.set_index("crime_type")

    def change(crime_type):
        if crime_type in annual.index and "change_pct" in annual.columns:
            return float(annual.loc[crime_type, "change_pct"])
        return 0.0

    shop_change     = change("Shoplifting")
    drug_change     = change("Drugs")
    vehicle_change  = change("Vehicle crime")
    burglary_change = change("Burglary")

    # ── Headline metrics ──────────────────────────────────────────
    st.caption(
        "Metropolitan Police and City of London Police recorded crime, "
        "January 2023 to December 2025."
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total crimes recorded", f"{total_crimes:,}")
    col2.metric("Shoplifting 2023–2025",  f"{shop_change:+.0f}%")
    col3.metric("Drug offences 2023–2025", f"{drug_change:+.0f}%")
    col4.metric("Vehicle crime 2023–2025", f"{vehicle_change:+.0f}%")

    st.divider()

    # ── Five key findings ─────────────────────────────────────────
    st.subheader("Five things the data shows")
    st.markdown("""
    This is not a story about crime rising across the board. Crimes linked to
    financial pressure surged while crimes driven by technology and physical
    environment fell. Each section of this dashboard examines a different part
    of that picture.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**1. Economic crime rose sharply and kept rising**")
        st.markdown(f"""
        Shoplifting increased {shop_change:.0f}% between 2023 and 2025. It
        kept rising even after food inflation fell. The data shows shoplifting
        responds to food price changes with a five month delay. By the time
        prices eased, many households had already exhausted their savings and
        credit. The financial damage outlasted the acute crisis.
        """)

        st.markdown("**2. Deprivation predicts crime, but not always how you expect**")
        st.markdown("""
        A model trained purely on deprivation data explains 66% of the variation
        in crime rates across London's neighbourhoods. The type of deprivation
        matters though: bad housing predicts burglary, unemployment predicts
        violence, and shoplifting is predicted by almost nothing. It has become
        a cross-demographic phenomenon, occurring in wealthy high streets as
        much as deprived estates.
        """)

        st.markdown("**3. Drug offences jumped 50%, but not because of more drug activity**")
        st.markdown(f"""
        Drug offences rose {drug_change:.0f}% over the period, but the shape
        of the increase is telling. A sudden structural shift in August 2024
        saw monthly offences jump from around 2,900 to 4,400 and stay there.
        Stop and search volumes did not change. Operation Yamata and the
        Metropolitan Police Drugs Action Plan changed how encounters were
        classified, not how many occurred.
        """)

    with col2:
        st.markdown("**4. Stop and search raises serious questions**")
        st.markdown("""
        Black Londoners make up 13.5% of the population but 40.9% of all stop
        and searches, a ratio of 3x. The arrest rate for Black people is 16.9%
        compared to 17.6% for White people, a difference of less than one
        percentage point. The additional stops are not producing proportionally
        more arrests. This is a finding the Metropolitan Police's own scrutiny
        panels have consistently raised.
        """)

        st.markdown("**5. Some crimes fell, and the reasons matter**")
        st.markdown(f"""
        Vehicle crime fell {abs(vehicle_change):.0f}% and burglary fell
        {abs(burglary_change):.0f}%. These are not economic crimes. They are
        driven by technology and physical environment. Improved vehicle security,
        smart home systems, and better street lighting have had measurable
        effects. The contrast with economically-driven crimes is clear: you
        cannot engineer your way out of financial desperation the way you can
        out of a poorly secured car.
        """)

    st.divider()

    # ── Overview chart ────────────────────────────────────────────
    st.subheader("Every crime type: 2023 to 2025")
    st.markdown("""
    The chart shows the percentage change for every recorded crime type.
    Crimes linked to economic pressure sit at the top. Crimes linked to
    technology and physical environment sit at the bottom.
    """)

    annual_reset = summary["crime_annual_change"].copy()
    annual_reset.columns = [
        int(c) if c.isdigit() else c for c in annual_reset.columns
    ]
    if "change_pct" in annual_reset.columns:
        annual_reset = annual_reset.rename(columns={"change_pct": "change"})
    annual_reset = annual_reset.set_index("crime_type")

    fig1 = crime_change_overview_chart(annual_reset)
    st.plotly_chart(fig1, use_container_width=True, config=CHART_CONFIG)

    st.divider()

    # ── Economic timeline ─────────────────────────────────────────
    st.subheader("The economic backdrop")
    st.markdown("""
    The chart shows total monthly recorded crime alongside key moments in the
    UK cost of living crisis. The relationship between economic events and
    crime patterns is examined in detail in the Economic Crime section.
    """)

    fig2 = monthly_total_chart(monthly_all, annotate_events=True)
    st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)

    st.markdown("""
    Use the navigation on the left to explore each finding. Each section
    builds on the last: economic analysis, deprivation patterns, policing
    response, and the outlook for 2026.
    """)

    st.caption("""
    Source: Metropolitan Police & City of London Police via police.uk |
    ONS Population Estimates 2022 |
    Index of Multiple Deprivation 2025 |
    ONS Census 2021
    """)