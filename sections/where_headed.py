"""
sections/where_headed.py
------------------------
'Where is London Headed?' section — vulnerability index map,
crime trajectory chart, shoplifting scenarios, and the structural
argument from the Random Forest model.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.charts import apply_base_layout, horizontal_bar_chart, style_xaxis, style_yaxis
from utils.constants import (
    CHART_CONFIG,
    IMD_DOMAIN_LABELS,
    LONDON_MAP_CENTRE,
    LONDON_MAP_ZOOM,
    MAPBOX_STYLE,
    RISK_COLOURS,
    VULNERABILITY_RENAME,
)
from utils.data_loaders import load_street_summary, load_model, load_outlook_data


def render():
    st.title("Where is London Headed?")
    st.markdown("""
    Drawing together findings from across this dashboard, this section asks
    what London's crime landscape is likely to look like in 2026 and beyond.
    The data points to a city under sustained structural pressure, with some
    crimes responding to economic conditions and others to policing decisions.
    """)

    data          = load_outlook_data()
    vulnerability = data["vulnerability"]
    trajectory    = data["trajectory"]
    scenarios     = data["scenarios"]
    summary       = load_street_summary()
    model         = load_model()

    monthly_shoplifting = (
        summary["monthly_by_crime"][
            summary["monthly_by_crime"]["crime_type"] == "Shoplifting"
        ][["month", "count"]].copy()
    )

    # ── Derived values ────────────────────────────────────────────
    higher_risk_count = len(vulnerability[vulnerability["risk_tier"] == "Higher risk"])
    lower_risk_count  = len(vulnerability[vulnerability["risk_tier"] == "Lower risk"])
    top1 = vulnerability.nlargest(1, "vulnerability_score").iloc[0]
    top2 = vulnerability.nlargest(2, "vulnerability_score").iloc[1]
    bot1 = vulnerability.nsmallest(1, "vulnerability_score").iloc[0]
    bot2 = vulnerability.nsmallest(2, "vulnerability_score").iloc[1]

    top1_shop_change = vulnerability.loc[
        vulnerability["borough"] == top1["borough"], "change_pct"
    ].values[0]
    top2_shop_change = vulnerability.loc[
        vulnerability["borough"] == top2["borough"], "change_pct"
    ].values[0]

    # ── Headline metrics ──────────────────────────────────────────
    st.caption(
        "Vulnerability index combines deprivation, shoplifting trend, "
        "crime-deprivation mismatch, and policing intensity across all "
        "33 London boroughs. Weights reflect analytical judgment and "
        "are not statistically derived."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Higher risk boroughs", f"{higher_risk_count}",
        "Elevated on multiple indicators", delta_color="off",
    )
    col2.metric(
        "Most vulnerable borough", f"{top1['borough']}",
        "Highest composite risk score", delta_color="off",
    )
    col3.metric(
        "Lower risk boroughs", f"{lower_risk_count}",
        "Below average on all indicators", delta_color="off",
    )

    st.divider()

    # ── 1. Vulnerability map ──────────────────────────────────────
    _render_vulnerability_map(
        vulnerability, top1, top2, bot1, bot2,
        top1_shop_change, top2_shop_change,
    )

    st.divider()

    # ── 2. Crime trajectory ───────────────────────────────────────
    _render_trajectory_chart(trajectory)

    st.divider()

    # ── 3. Shoplifting scenarios ──────────────────────────────────
    _render_scenarios_chart(monthly_shoplifting, scenarios)

    st.divider()

    # ── 4. Structural argument ────────────────────────────────────
    _render_model_importance(model)

    with st.expander("Methodology: vulnerability index and model detail"):
        st.markdown("""
        **Vulnerability index** combines four borough-level indicators,
        each normalised to 0 to 1 using min-max scaling and weighted as
        follows. Weights reflect analytical judgment about relative importance
        and are not statistically derived:

        - Deprivation (35%): average IMD decile, inverted so higher equals
          more deprived. Uses 2019 IMD, the most recent available.
        - Shoplifting trend (30%): % change in shoplifting 2023 to 2025.
        - Crime-deprivation mismatch (20%): residual from borough-level
          regression of crime rate on deprivation. A positive residual means
          more crime than deprivation alone predicts.
        - Policing intensity (15%): stop and search volume weighted by inverse
          arrest rate. High volume with low effectiveness produces a higher
          risk score.

        **Predictive model:** Random Forest Regressor (100 estimators,
        random_state=42). Target: crime rate per 1,000 residents per LSOA,
        log-transformed to address right skew. Extreme outliers capped at
        the 99th percentile. Features: seven 2019 IMD domain scores.
        Train/test split: 80/20. R² = 0.659 on held-out test set.
        MAE = 0.952 on log-transformed scale.

        **Limitations:** IMD data is from 2019 and does not reflect
        post-pandemic changes to deprivation. Shoplifting scenarios are
        assumption-based projections, not statistical forecasts. The
        vulnerability index should be treated as indicative: a framework
        for thinking about relative risk, not a definitive ranking.
        Borough assignment of stop and search records uses nearest centroid
        approximation. 5.9% of records had no GPS coordinates and are
        excluded. All crime data is recorded crime, which reflects
        enforcement activity as well as actual crime levels.
        """)

    st.caption("""
    Source: Metropolitan Police & City of London Police via police.uk |
    Deprivation: MHCLG English Indices of Deprivation 2019 |
    Food inflation: ONS CPI series D7G8 |
    Population: ONS Census 2021 |
    Vulnerability index: original analysis combining deprivation, crime
    trends, and policing data
    """)


# ── Sub-renderers ─────────────────────────────────────────────────

def _render_vulnerability_map(
    vulnerability, top1, top2, bot1, bot2,
    top1_shop_change, top2_shop_change,
):
    st.subheader("1. Which boroughs are most at risk heading into 2026?")
    st.markdown("""
    The vulnerability index combines four factors: how deprived a borough is,
    how fast shoplifting has risen there, whether crime is higher than
    deprivation alone would predict, and how intensively but how ineffectively
    it is being policed. A high score means a borough is under pressure on
    multiple fronts.

    Weights applied: deprivation (35%), shoplifting trend (30%),
    crime-deprivation mismatch (20%), policing intensity adjusted for
    effectiveness (15%). All components normalised to 0 to 1 before weighting.
    """)

    fig = px.scatter_mapbox(
        vulnerability,
        lat="latitude", lon="longitude",
        color="risk_tier",
        color_discrete_map=RISK_COLOURS,
        size="vulnerability_score", size_max=35,
        hover_name="borough",
        hover_data={
            "vulnerability_score": ":.1f",
            "risk_tier":           False,
            "latitude":            False,
            "longitude":           False,
        },
        labels={"vulnerability_score": "Vulnerability score (0 to 100)"},
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
        st.markdown("**Highest vulnerability boroughs**")
        st.dataframe(
            vulnerability.nlargest(5, "vulnerability_score")[
                ["borough", "vulnerability_score", "risk_tier"]
            ].rename(columns=VULNERABILITY_RENAME).round(1),
            hide_index=True,
        )
        st.markdown(f"""
        **{top1['borough']}** scores highest, combining very high deprivation
        with a shoplifting increase of {top1_shop_change:+.1f}% and sustained
        high policing intensity. **{top2['borough']}** follows closely with
        near-maximum deprivation scores and a shoplifting increase of
        {top2_shop_change:+.1f}%.
        """)
    with col2:
        st.markdown("**Lowest vulnerability boroughs**")
        st.dataframe(
            vulnerability.nsmallest(5, "vulnerability_score")[
                ["borough", "vulnerability_score", "risk_tier"]
            ].rename(columns=VULNERABILITY_RENAME).round(1),
            hide_index=True,
        )
        st.markdown(f"""
        **{bot1['borough']}** and **{bot2['borough']}** score lowest. Both
        are less deprived, have lower crime rates relative to their deprivation
        level, and show more targeted policing. These are boroughs where the
        structural conditions for elevated crime are weakest.
        """)


def _render_trajectory_chart(trajectory: pd.DataFrame):
    st.subheader("2. Where is each crime type headed?")
    st.markdown("""
    The chart shows the actual change recorded between 2023 and 2025 for
    each major crime type. Hover over each bar for the primary driver and
    main policy lever. The directional assessment reflects analytical
    judgment informed by the data and is not a statistical forecast.
    """)

    trajectory_sorted = trajectory.sort_values("trend_pct", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=trajectory_sorted["trend_pct"],
        y=trajectory_sorted["crime_type"],
        orientation="h",
        marker=dict(
            color=trajectory_sorted["trend_pct"],
            colorscale=[[0, "#2ecc71"], [0.5, "#f39c12"], [1, "#e74c3c"]],
            showscale=False,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Change 2023 to 2025: %{x:+.1f}%<br>"
            "Primary driver: %{customdata[0]}<br>"
            "Main policy lever: %{customdata[1]}"
            "<extra></extra>"
        ),
        customdata=trajectory_sorted[["key_driver", "policy_lever"]].values,
    ))
    fig.add_vline(x=0, line_color="white", opacity=0.4)
    fig = apply_base_layout(fig, height=400, hovermode="y")
    fig = style_xaxis(fig, show_labels=True, title="% change 2023 to 2025", ticksuffix="%")
    fig = style_yaxis(fig)

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Crimes likely to remain elevated**")
        st.markdown("""
        Shoplifting (+53.5%) and theft from the person (+21.2%) are
        structurally linked to financial pressure and tourist footfall
        respectively. Neither driver is easing in 2026. Food inflation is
        rising again and London's footfall continues to recover. These crimes
        are unlikely to fall without direct intervention.

        Weapons possession (+15.4%) has risen sharply since August 2024,
        closely tracking the drug enforcement shift. This suggests it reflects
        increased policing activity rather than a genuine surge in weapons
        carrying.
        """)
    with col2:
        st.markdown("**Crimes likely to stabilise**")
        st.markdown("""
        Violence (+4.0%) and robbery (-1.7%) are broadly flat. Both are driven
        primarily by income and living environment deprivation, neither of which
        is improving rapidly. These crimes reflect long-term structural
        conditions rather than short-term economic shocks and are unlikely to
        move significantly in either direction without substantial economic
        change.
        """)
    with col3:
        st.markdown("**Crimes continuing to fall**")
        st.markdown("""
        Burglary (-13.0%) and vehicle crime (-18.5%) have fallen consistently
        and the trajectory looks set to continue. Burglary is associated with
        living environment deprivation: improved housing and street lighting
        investment in several boroughs has had a measurable effect. Vehicle
        crime reflects improving anti-theft technology rather than economic
        conditions and is relatively insensitive to the cost of living crisis.
        """)


def _render_scenarios_chart(monthly_shop: pd.DataFrame, scenarios: pd.DataFrame):
    st.subheader("3. Shoplifting in 2026: three scenarios")
    st.markdown("""
    Shoplifting is the crime most directly responsive to economic conditions
    and the one with the most available policy levers. The scenarios below
    reflect three plausible futures depending on how economic and policy
    conditions develop. These are assumption-based projections, not
    statistical forecasts.
    """)

    historical = monthly_shop.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical["month"],
        y=historical["count"],
        name="Recorded 2023 to 2025",
        line=dict(color="#e74c3c", width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:,} incidents<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=scenarios["month"], y=scenarios["pessimistic"],
        name="Pessimistic",
        line=dict(color="#e74c3c", width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} projected<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=scenarios["month"], y=scenarios["optimistic"],
        name="Optimistic",
        line=dict(color="#2ecc71", width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} projected<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=scenarios["month"], y=scenarios["central"],
        name="Central",
        line=dict(color="#f39c12", width=2, dash="dash"),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} projected<extra></extra>",
    ))
    # Shaded range between pessimistic and optimistic
    fig.add_trace(go.Scatter(
        x=pd.concat([scenarios["month"], scenarios["month"][::-1]]),
        y=pd.concat([scenarios["pessimistic"], scenarios["optimistic"][::-1]]),
        fill="toself",
        fillcolor="rgba(231,76,60,0.08)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_vline(x="2025-12-01", line_dash="dot", line_color="white", opacity=0.4)
    fig.add_annotation(
        x="2025-12-01", y=1.05, yref="paper",
        text="2026 projections",
        showarrow=False,
        font=dict(color="white", size=10),
        xanchor="left", xshift=5,
    )
    fig = apply_base_layout(
        fig, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig = style_xaxis(fig)
    fig = style_yaxis(fig, title="Monthly shoplifting incidents")

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Optimistic: approximately 6,000 per month by December 2026**")
        st.markdown("""
        Food inflation falls back toward 2%. Labour's Crime and Policing Bill
        passes with effective retail crime powers. Wage growth continues to
        outpace inflation, gradually rebuilding household finances.
        Neighbourhood policing guarantee delivers visible presence on high
        streets.
        """)
    with col2:
        st.markdown("**Central: approximately 7,300 per month through 2026**")
        st.markdown("""
        Current conditions persist. Food inflation stays in the 3 to 5% range
        and household finances remain stretched but stable. Policing bill passes
        but implementation is slow. Shoplifting plateaus at its current elevated
        level, high but no longer rising.
        """)
    with col3:
        st.markdown("**Pessimistic: approximately 8,500 per month by December 2026**")
        st.markdown("""
        Food inflation continues rising, already back above 4% in late 2025.
        April 2026 brings further bill increases. Retail crime bill is delayed
        in Parliament. The five month delay pattern means pressure building now
        feeds into crime through spring 2026.
        """)


def _render_model_importance(model):
    st.subheader("4. Why policing alone cannot resolve this")
    st.markdown("""
    A Random Forest model trained on deprivation indicators alone accounts for
    66% of the variation in crime rates across London's neighbourhoods. Without
    any knowledge of policing levels, enforcement activity, or local events,
    purely from knowing how deprived an area is, the model accounts for two
    thirds of what we actually observe.
    """)

    features   = list(IMD_DOMAIN_LABELS.keys())
    importance = pd.DataFrame({
        "feature":    [IMD_DOMAIN_LABELS[f] for f in features],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance["importance"],
        y=importance["feature"],
        orientation="h",
        marker=dict(
            color=importance["importance"],
            colorscale="Blues",
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Relative importance: %{x:.3f}<extra></extra>",
    ))
    fig = apply_base_layout(fig, height=380, hovermode="y")
    fig = style_xaxis(
        fig, show_labels=True,
        title="Relative importance in predicting crime rate",
    )
    fig = style_yaxis(fig)

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**What 66% predictability means**")
        st.markdown("""
        Knowing only an area's deprivation scores, you can predict its crime
        rate with 66% accuracy. For a social phenomenon as complex as crime,
        that is a high figure. It means structural deprivation, not policing
        decisions or individual behaviour, is the dominant driver of crime
        geography in London.

        The remaining 34% reflects factors the model cannot see: tourist
        footfall in Westminster, policing operations in Hackney, local
        community infrastructure in Haringey. These matter, but they are
        secondary to the structural picture. Note that IMD data is from 2019
        and deprivation has likely worsened in some boroughs since then,
        meaning structural risk may be understated in the most affected areas.
        """)
    with col2:
        st.markdown("**The implication for 2026**")
        st.markdown("""
        London's policing response, including stop and search, targeted
        operations, and the drugs enforcement shift, operates on the 34%.
        The cost of living crisis and its aftermath operates on the 66%.

        The data makes a case that sustained reductions in London crime
        require economic intervention in household incomes, housing quality,
        and employment, not policing alone. Areas like Newham and Hackney
        will continue to score highly on vulnerability indices until their
        underlying deprivation improves, regardless of how many stop and
        searches are conducted there.

        The neighbourhood policing guarantee and the retail crime bill are
        meaningful policy responses, but they are operating on the margins
        of a structural problem that the data suggests can only be resolved
        through economic change.
        """)