"""
sections/policing_response.py
-----------------------------
'Policing Response' section — stop and search ethnicity breakdown,
effectiveness by search type, drugs comparison, and borough map.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.charts import apply_base_layout, style_xaxis, style_yaxis
from utils.constants import (
    CHART_CONFIG,
    LONDON_MAP_CENTRE,
    LONDON_MAP_ZOOM,
    MAPBOX_STYLE,
)
from utils.data_loaders import load_policing_data
from utils.helpers import get_ethnicity_val


def render():
    st.title("Policing Response")
    st.markdown("""
    Stop and search is one of the most visible and most contested tools
    available to the Metropolitan Police. The data raises questions about
    who is being stopped, why, and whether it is working.
    """)

    data = load_policing_data()

    outcomes_summary   = data["outcomes_summary"]
    ethnicity          = data["ethnicity"]
    outcomes_by_search = data["outcomes_by_search"]
    ss_borough         = data["ss_borough"]
    drugs_comparison   = data["drugs_comparison"]

    # ── Derived values ────────────────────────────────────────────
    total_searches  = int(outcomes_summary["total"].values[0])
    arrest_rate     = float(outcomes_summary["arrest_rate"].values[0])
    no_action_rate  = float(outcomes_summary["no_action_rate"].values[0])

    black_ratio       = get_ethnicity_val(ethnicity, "Black", "stop_rate_ratio")
    black_stop_pct    = get_ethnicity_val(ethnicity, "Black", "stop_pct")
    black_pop_pct     = get_ethnicity_val(ethnicity, "Black", "population_pct")
    black_arrest_rate = get_ethnicity_val(ethnicity, "Black", "arrest_rate")
    white_arrest_rate = get_ethnicity_val(ethnicity, "White", "arrest_rate")

    cp_date         = pd.to_datetime("2024-08-01")
    before_searches = drugs_comparison[drugs_comparison["month"] <  cp_date]["drug_searches"].mean()
    after_searches  = drugs_comparison[drugs_comparison["month"] >= cp_date]["drug_searches"].mean()

    top_borough            = ss_borough.nlargest(1, "total_searches").iloc[0]
    highest_arrest_borough = ss_borough.nlargest(1, "arrest_rate").iloc[0]

    # Drug searches as % of total — derived from data
    drug_rows = outcomes_by_search[
        outcomes_by_search["object_of_search"]
        .str.lower().str.contains("drug", na=False)
    ]
    drug_search_pct = round(drug_rows["total"].sum() / total_searches * 100)

    # ── Headline metrics ──────────────────────────────────────────
    st.caption(
        "2023 to 2025 stop and search records from the Metropolitan "
        "Police and City of London Police. Population proportions "
        "from ONS Census 2021."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Total stop and searches",     f"{total_searches:,}",  delta_color="off")
    col2.metric("Result in no further action", f"{no_action_rate}%",   delta_color="off")
    col3.metric(
        "Black people as % of stops",
        f"{black_stop_pct}%",
        f"vs {black_pop_pct}% of London population",
        delta_color="off",
    )

    st.divider()

    # ── 1. Who is being stopped ───────────────────────────────────
    _render_ethnicity_chart(
        ethnicity, black_pop_pct, black_stop_pct,
        black_ratio, black_arrest_rate, white_arrest_rate,
    )

    st.divider()

    # ── 2. Effectiveness ──────────────────────────────────────────
    _render_effectiveness_chart(
        outcomes_by_search, arrest_rate, total_searches, drug_search_pct
    )

    st.divider()

    # ── 3. Drug searches vs drug crimes ──────────────────────────
    _render_drugs_comparison_chart(
        drugs_comparison, cp_date, before_searches, after_searches
    )

    st.divider()

    # ── 4. Borough map ────────────────────────────────────────────
    _render_borough_map(ss_borough, top_borough, highest_arrest_borough)

    st.caption("""
    Source: Metropolitan Police & City of London Police stop and search data
    via police.uk, 2023 to 2025 |
    Population figures: ONS Census 2021 |
    Borough assignment based on nearest centroid from GPS coordinates |
    24,142 records (5.9%) had no GPS coordinates and could not be assigned
    to a borough
    """)


# ── Sub-renderers ─────────────────────────────────────────────────

def _render_ethnicity_chart(
    ethnicity, black_pop_pct, black_stop_pct,
    black_ratio, black_arrest_rate, white_arrest_rate,
):
    st.subheader("1. Who is being stopped?")
    st.markdown("""
    The ethnic breakdown of stop and search in London is significantly skewed
    relative to the population. The chart compares the percentage of stops
    with each group's share of London's population, based on the 2021 Census.
    """)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="% of stops",
        x=ethnicity["ethnicity"],
        y=ethnicity["stop_pct"],
        marker_color="#e74c3c",
        hovertemplate="%{x}: %{y:.1f}% of stops<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="% of London population (Census 2021)",
        x=ethnicity["ethnicity"],
        y=ethnicity["population_pct"],
        marker_color="rgba(149,165,166,0.5)",
        hovertemplate="%{x}: %{y:.1f}% of population<extra></extra>",
    ))
    fig = apply_base_layout(
        fig, height=380, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig = style_xaxis(fig, show_labels=True)
    fig = style_yaxis(fig, title="%", ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**The disparity in numbers**")
        st.markdown(f"""
        Black people make up {black_pop_pct}% of London's population but
        {black_stop_pct}% of all stop and searches, a ratio of {black_ratio:.2f}x.
        White people are stopped at 0.74x their population share and are
        under-represented in stop and search relative to their numbers.

        The disparity is consistent across the three year period and is not
        explained by deprivation. Boroughs with higher Black stop percentages
        are spread across all deprivation levels. The correlation between
        deprivation and Black stop percentage is r=-0.62, meaning the disparity
        is actually higher in wealthier boroughs.
        """)
    with col2:
        st.markdown("**Does the disparity produce results?**")
        st.markdown(f"""
        The key question is whether the additional stops of Black people
        produce proportionally more arrests. They do not.

        The arrest rate for Black people is {black_arrest_rate}% compared to
        {white_arrest_rate}% for White people, a difference of less than one
        percentage point. The additional searches are not uncovering
        proportionally more crime. This is a finding the Metropolitan Police's
        own scrutiny panels have consistently raised and is reflected in the
        Home Office's annual stop and search statistics.
        """)


def _render_effectiveness_chart(
    outcomes_by_search, arrest_rate, total_searches, drug_search_pct
):
    st.subheader("2. Is stop and search an effective tool?")
    st.markdown(f"""
    Across all {total_searches:,} searches, only {arrest_rate}% resulted in
    an arrest. Effectiveness varies significantly by what police are
    searching for.
    """)

    outcomes_plot = outcomes_by_search[
        outcomes_by_search["object_of_search"] != "Detailed object of search unavailable"
    ].sort_values("arrest_rate", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=outcomes_plot["arrest_rate"],
        y=outcomes_plot["object_of_search"],
        orientation="h",
        marker=dict(
            color=outcomes_plot["arrest_rate"],
            colorscale="RdYlGn",
            showscale=False,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Arrest rate: %{x:.1f}%<br>"
            "Total searches: %{customdata:,}"
            "<extra></extra>"
        ),
        customdata=outcomes_plot["total"],
    ))
    fig.add_vline(x=arrest_rate, line_dash="dot", line_color="white", opacity=0.5)
    fig.add_annotation(
        x=arrest_rate, y=1.05, yref="paper",
        text=f"Overall average: {arrest_rate}%",
        showarrow=False,
        font=dict(color="white", size=10),
        xanchor="left", xshift=5,
    )
    fig = apply_base_layout(fig, height=380, hovermode="y")
    fig = style_xaxis(fig, show_labels=True, title="Arrest rate (%)", ticksuffix="%")
    fig = style_yaxis(fig)

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    st.markdown(f"""
    Searches for stolen goods (24.3%) and evidence of offences (21.6%) have
    the highest arrest rates. These are targeted, intelligence-led searches
    more likely to find what they are looking for. Drug searches account for
    approximately {drug_search_pct}% of all stops but result in arrest only
    13.2% of the time, below the overall average. Fireworks searches at 4.4%
    are the least productive.
    """)


def _render_drugs_comparison_chart(
    drugs_comparison, cp_date, before_searches, after_searches
):
    st.subheader("3. Drug searches did not drive the August 2024 crime spike")
    st.markdown("""
    The Economic Crime section identified a sudden and sustained increase in
    recorded drug offences from August 2024. The stop and search data helps
    clarify what happened.
    """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drugs_comparison["month"],
        y=drugs_comparison["drug_searches"],
        name="Drug stop and searches",
        line=dict(color="#9b59b6", width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:,} searches<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=drugs_comparison["month"],
        y=drugs_comparison["drug_crimes"],
        name="Recorded drug offences",
        line=dict(color="#e74c3c", width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:,} offences<extra></extra>",
    ))
    fig.add_vline(
        x=cp_date, line_dash="dash",
        line_color="white", opacity=0.6, line_width=2,
    )
    fig.add_annotation(
        x=cp_date, y=1.05, yref="paper",
        text="August 2024: drug offences spike",
        showarrow=False,
        font=dict(color="white", size=10),
        xanchor="left", xshift=5,
        bgcolor="rgba(0,0,0,0.5)",
    )
    fig = apply_base_layout(
        fig, height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig = style_xaxis(fig)
    fig = style_yaxis(fig, title="Monthly count")

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Search volumes stayed flat**")
        st.markdown(f"""
        Drug stop and searches averaged {before_searches:,.0f} per month
        before August 2024 and {after_searches:,.0f} per month afterwards,
        a change of less than 0.3%. If the recorded crime spike had been
        driven by more searches, the purple line would rise alongside the
        red one. It does not.
        """)
    with col2:
        st.markdown("**The most likely explanation**")
        st.markdown("""
        The evidence points to a change in how drug encounters were recorded,
        not how many occurred. Operation Yamata and the Metropolitan Police's
        Drugs Action Plan explicitly targeted higher recording rates as part
        of a zero tolerance approach to drug supply networks.

        Drug activity almost certainly did not increase by 50% overnight.
        The recording of it did. This is an important caveat when interpreting
        drug offence figures elsewhere in this dashboard.
        """)


def _render_borough_map(ss_borough, top_borough, highest_arrest_borough):
    st.subheader("4. Where is stop and search concentrated?")
    st.markdown("""
    Stop and search is not evenly distributed across London. The map shows
    total searches per borough with colour indicating arrest rate. A large
    bubble in red indicates high-volume, low-effectiveness searching.
    A large bubble in green indicates high-volume searching that is producing
    results.
    """)

    fig = px.scatter_mapbox(
        ss_borough,
        lat="lat", lon="lon",
        size="total_searches",
        color="arrest_rate",
        color_continuous_scale="RdYlGn",
        size_max=40,
        hover_name="borough",
        hover_data={
            "total_searches": ":,",
            "arrest_rate":    ":.1f",
            "black_pct":      ":.1f",
            "lat":            False,
            "lon":            False,
        },
        labels={
            "total_searches": "Total searches",
            "arrest_rate":    "Arrest rate (%)",
            "black_pct":      "Black stops (%)",
        },
        zoom=LONDON_MAP_ZOOM,
        center=LONDON_MAP_CENTRE,
        mapbox_style=MAPBOX_STYLE,
        height=500,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode=False,
        coloraxis_colorbar=dict(title="Arrest rate (%)"),
    )
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{top_borough['borough']}: most searched borough**")
        st.markdown(f"""
        {top_borough['borough']} has the highest stop and search volume with
        {int(top_borough['total_searches']):,} searches over the period. This
        is consistent with its position as London's highest crime rate borough.
        Search volume correlates with crime rate across all boroughs at r=0.79,
        meaning police are broadly searching where crime is highest.
        """)
    with col2:
        st.markdown(f"**{highest_arrest_borough['borough']}: most effective searches**")
        st.markdown(f"""
        {highest_arrest_borough['borough']} has the highest arrest rate at
        {highest_arrest_borough['arrest_rate']:.1f}%, meaning its searches
        are most likely to produce a meaningful outcome. High arrest rates
        relative to search volume suggest more intelligence-led, targeted
        activity rather than high-volume general stops.
        """)