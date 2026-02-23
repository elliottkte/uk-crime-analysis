"""
sections/crime_deprivation.py
-----------------------------
'Crime & Deprivation' section — borough map, deprivation domain
heatmap, and the cross-deprivation correlation findings.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.charts import apply_base_layout
from utils.constants import (
    CHART_CONFIG,
    DEPRIVATION_OUTLIER_COLOURS,
    LONDON_MAP_CENTRE,
    LONDON_MAP_ZOOM,
    MAPBOX_STYLE,
)
from utils.data_loaders import load_deprivation_data
from utils.helpers import get_corr


def render():
    st.title("Crime & Deprivation")
    st.markdown("""
    Poverty and crime are linked in London, but not in a simple or uniform
    way. Some of the highest-crime areas are among the city's wealthiest.
    And the type of deprivation matters: bad housing predicts different
    crimes than unemployment does.
    """)

    data        = load_deprivation_data()
    domain_corr = data["domain_corr"]
    borough_dep = data["borough_dep"]

    # ── Derived values ────────────────────────────────────────────
    total_boroughs       = len(borough_dep)
    deprived_high        = borough_dep[borough_dep["dominant_outlier"] == "Deprived and high crime"]
    affluent_high        = borough_dep[borough_dep["dominant_outlier"] == "Affluent but high crime"]
    top_affluent_borough = borough_dep.sort_values("affluent_high",  ascending=False).iloc[0]
    top_deprived_borough = borough_dep.sort_values("avg_residual",   ascending=False).iloc[0]

    living_burg      = get_corr(domain_corr, "Burglary",                      "Living Env")
    income_violence  = get_corr(domain_corr, "Violence And Sexual Offences",  "Income")
    shop_income_corr = get_corr(domain_corr, "Shoplifting",                   "Income")
    shop_living_corr = get_corr(domain_corr, "Shoplifting",                   "Living Env")

    # ── Headline metrics ──────────────────────────────────────────
    st.caption(
        "All 33 London boroughs. 2023 to 2025 recorded crime data "
        "and 2025 English Indices of Deprivation."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Boroughs analysed",                f"{total_boroughs}", delta_color="off")
    col2.metric("Deprived boroughs with high crime", f"{len(deprived_high)}", delta_color="off")
    col3.metric("Wealthy boroughs with high crime",  f"{len(affluent_high)}", delta_color="off")

    st.divider()

    # ── 1. Map ────────────────────────────────────────────────────
    _render_map(borough_dep, top_affluent_borough, top_deprived_borough)

    st.divider()

    # ── 2. Heatmap ────────────────────────────────────────────────
    _render_heatmap(
        domain_corr,
        living_burg,
        income_violence,
        shop_income_corr,
        shop_living_corr,
    )

    st.caption("""
    Source: Metropolitan Police & City of London Police via police.uk |
    Deprivation: Ministry of Housing, Communities & Local Government,
    English Indices of Deprivation 2025 |
    Correlations calculated at borough level across all 33 London boroughs.
    Association does not imply causation.
    """)


# ── Sub-renderers ─────────────────────────────────────────────────

def _render_map(borough_dep, top_affluent_borough, top_deprived_borough):
    st.subheader("1. The geography of crime and deprivation")
    st.markdown("""
    The map shows boroughs with significantly more crime than their wealth
    level would suggest. Larger bubbles indicate higher crime rates per
    1,000 residents. Orange boroughs are wealthy but high-crime, driven by
    tourism and footfall. Red boroughs are deprived and high-crime, driven
    by hardship.
    """)

    map_data = borough_dep[borough_dep["dominant_outlier"] != "As expected"].copy()
    map_data["size"] = (map_data["avg_crime_rate"] / 20).clip(lower=5)

    fig = px.scatter_mapbox(
        map_data,
        lat="latitude", lon="longitude",
        color="dominant_outlier",
        color_discrete_map=DEPRIVATION_OUTLIER_COLOURS,
        size="size", size_max=40,
        hover_name="borough",
        hover_data={
            "avg_imd_decile":   ":.1f",
            "avg_crime_rate":   ":.0f",
            "latitude":         False,
            "longitude":        False,
            "dominant_outlier": False,
            "size":             False,
        },
        labels={
            "avg_imd_decile": "Avg deprivation decile",
            "avg_crime_rate": "Crime rate per 1,000 residents",
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, title=""),
    )
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Why is {top_affluent_borough['borough']} so high?**")
        st.markdown(f"""
        {top_affluent_borough['borough']} is one of London's wealthiest areas
        yet has the highest crime rate of any borough. This is not a deprivation
        story. Millions of tourists and commuters pass through every day, and
        theft, pickpocketing and anti-social behaviour follow people rather than
        poverty. The crimes here are overwhelmingly opportunistic, targeting
        visitors rather than reflecting conditions for local residents.
        """)
    with col2:
        st.markdown(f"**What is happening in {top_deprived_borough['borough']}?**")
        st.markdown(f"""
        {top_deprived_borough['borough']} sits at the other end: genuinely
        deprived, with crime rates to match. Unlike {top_affluent_borough['borough']},
        the crimes here are more likely to affect local residents directly.
        Violence, drug offences, and robbery reflect financial hardship rather
        than tourist pickpocketing. The cost of living crisis has hit these
        communities hardest.
        """)


def _render_heatmap(
    domain_corr,
    living_burg,
    income_violence,
    shop_income_corr,
    shop_living_corr,
):
    st.subheader("2. It is not just about being poor. It is about how you are poor.")
    st.markdown("""
    Deprivation is not a single thing. Someone may have a low income but
    live in a well-maintained area. Someone else may be employed but live in
    overcrowded, poorly-lit housing. Different types of deprivation are
    associated with very different crimes. In some cases there is almost no
    relationship at all.

    The table shows how strongly each type of deprivation is associated with
    each crime type. Deeper green indicates a stronger positive association.
    These are correlations across London boroughs and indicate association,
    not causation.
    """)

    heatmap_pivot = domain_corr.pivot(
        index="deprivation_domain",
        columns="crime_type",
        values="correlation",
    )

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0, zmin=-0.2, zmax=0.5,
        text=heatmap_pivot.round(2).values,
        texttemplate="%{text}",
        hovertemplate=(
            "<b>%{y}</b> deprivation, <b>%{x}</b><br>"
            "Correlation: %{z:.3f}"
            "<extra></extra>"
        ),
        showscale=True,
        colorbar=dict(
            title="Strength of<br>association",
            tickvals=[-0.2, 0, 0.2, 0.4],
            ticktext=["Weak/negative", "None", "Moderate", "Stronger"],
        ),
    ))
    fig = apply_base_layout(
        fig, height=380,
        hovermode="closest",
        margin=dict(l=140, b=100),
    )
    fig.update_xaxes(title="", side="bottom", tickangle=-20, showspikes=False)
    fig.update_yaxes(title="Type of deprivation", showspikes=False)

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Bad housing is associated with burglary and robbery**")
        st.markdown(f"""
        Poor physical environments, including run-down housing, bad lighting,
        and neglected streets, show one of the stronger associations with
        burglary and robbery in this analysis (r={living_burg:.2f} with
        burglary). These are crimes of opportunity. Poor physical conditions
        create them regardless of income levels in the area.
        """)
    with col2:
        st.markdown("**Low income and unemployment are associated with violence**")
        st.markdown(f"""
        Violence and drug offences show a moderate association with income
        and employment deprivation (r={income_violence:.2f} for violence and
        income deprivation). These crimes are linked to financial desperation
        and the street economy, concentrated in areas where the cost of living
        crisis has had the most structural impact.
        """)
    with col3:
        st.markdown("**Shoplifting has almost no relationship with deprivation**")
        st.markdown(f"""
        Shoplifting shows near-zero association with income deprivation
        (r={shop_income_corr:.3f}) and any other deprivation measure. The
        only weak signal is housing quality (r={shop_living_corr:.2f}). This
        supports the argument in the Economic Crime section: shoplifting is no
        longer concentrated in deprived areas. The cost of living crisis has
        made it a cross-community phenomenon.
        """)