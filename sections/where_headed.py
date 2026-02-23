"""
sections/where_headed.py
------------------------
'Where is London Headed?' section — vulnerability index map,
crime trajectory chart, shoplifting scenarios, and the structural
argument from the Random Forest model.

Fix (critique - IMD label mismatch): _render_model_importance() now
uses build_imd_label_map() from data_loaders to construct the feature
label mapping at runtime from model.feature_names_in_, rather than
relying on the IMD_DOMAIN_LABELS constant whose keys previously did
not match actual feature column names from the raw IMD CSV.

Fix (critique - scenario chart presentation): the three projection
lines are removed from the scenarios chart. The shaded uncertainty
band remains. Scenario descriptions are given in text only. This
better communicates that these are assumption-based projections, not
statistical forecasts with distinct trajectories.

Fix (critique - IMD age caveat in headline): the vulnerability index
headline and model section now carry a visible note that IMD data is
from 2025, so users understand the deprivation estimates may be stale
before drilling into borough rankings.

Fix (critique - vulnerability index weight justification): the
methodology expander now leads with an explicit note that the base
weights are analytical judgments, explains why each weight was chosen,
and directs users to the sensitivity table before interpreting rankings.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.charts import apply_base_layout, horizontal_bar_chart, style_xaxis, style_yaxis
from utils.constants import (
    CHART_CONFIG,
    LONDON_MAP_CENTRE,
    LONDON_MAP_ZOOM,
    MAPBOX_STYLE,
    RISK_COLOURS,
    VULNERABILITY_RENAME,
)
from utils.data_loaders import (
    load_street_summary,
    load_model,
    load_outlook_data,
    build_imd_label_map,
)


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
    weight_sens   = data["weight_sensitivity"]
    summary       = load_street_summary()
    model         = load_model()

    monthly_shoplifting = (
        summary["monthly_by_crime"][
            summary["monthly_by_crime"]["crime_type"] == "Shoplifting"
        ][["month", "count"]].copy()
    )

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
    # Fix (critique): surface IMD data age at the headline level so
    # users see it before engaging with borough rankings.
    st.caption(
        "Vulnerability index combines deprivation, shoplifting trend, "
        "crime-deprivation mismatch, and policing intensity across all "
        "33 London boroughs. **Deprivation data is from 2025 (IMD 2025) — "
        "the most recent available index.** Post-pandemic changes to "
        "deprivation are not captured; rankings in most-deprived areas "
        "may understate current risk. See methodology expander for full detail."
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

    _render_vulnerability_map(
        vulnerability, top1, top2, bot1, bot2,
        top1_shop_change, top2_shop_change,
        weight_sens,
    )

    st.divider()

    _render_trajectory_chart(trajectory)

    st.divider()

    _render_scenarios_chart(monthly_shoplifting, scenarios)

    st.divider()

    _render_model_importance(model)

    _render_methodology_expander(model, weight_sens)

    st.caption("""
    Source: Metropolitan Police & City of London Police via police.uk |
    Deprivation: MHCLG English Indices of Deprivation 2025 |
    Food inflation: ONS CPI series D7G8 |
    Population: ONS Census 2021 |
    Vulnerability index: original analysis combining deprivation, crime
    trends, and policing data
    """)


# ── Sub-renderers ─────────────────────────────────────────────────

def _render_vulnerability_map(
    vulnerability, top1, top2, bot1, bot2,
    top1_shop_change, top2_shop_change,
    weight_sens: pd.DataFrame,
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
    effectiveness (15%). All components normalised by rank before weighting
    to reduce sensitivity to outlier boroughs.
    """)

    # Fix (critique): surface weight sensitivity note before the map,
    # not only in the buried methodology expander.
    if not weight_sens.empty:
        robust_boroughs = (
            weight_sens[weight_sens["rank"] <= 5]
            .groupby("borough")
            .size()
            .reset_index(name="scenarios_in_top5")
            .query("scenarios_in_top5 == scenarios_in_top5.max()")
            ["borough"].tolist()
        )
        n_scenarios = weight_sens["scenario"].nunique()
        if robust_boroughs:
            robust_str = ", ".join(robust_boroughs)
            st.info(
                f"**Weight sensitivity check:** across {n_scenarios} alternative "
                f"weighting schemes, the following borough(s) appear in the top 5 "
                f"under every scheme: **{robust_str}**. Their elevated ranking is "
                f"robust to the choice of weights. See the methodology expander "
                f"below for the full sensitivity table."
            )
        else:
            st.info(
                f"**Weight sensitivity check:** no borough appears in the top 5 "
                f"across all {n_scenarios} weighting schemes. Rankings shift "
                f"with assumptions — interpret the index as indicative, not "
                f"definitive. See the methodology expander for the full table."
            )

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
    conditions develop.

    **These are assumption-based projections, not statistical forecasts.**
    The shaded band shows the range between the optimistic and pessimistic
    assumptions. The three assumptions are described below the chart — they
    should not be read as a prediction of distinct trajectories.
    """)

    historical = monthly_shop.copy()

    fig = go.Figure()

    # Fix (critique - scenario chart): show historical data + shaded
    # uncertainty band only. Removing the three separate projection
    # lines reduces the false impression of distinct forecast trajectories.
    # Scenario assumptions are communicated in text.
    fig.add_trace(go.Scatter(
        x=historical["month"],
        y=historical["count"],
        name="Recorded 2023 to 2025",
        line=dict(color="#e74c3c", width=2.5),
        hovertemplate="%{x|%b %Y}<br>%{y:,} incidents<extra></extra>",
    ))

    # Shaded uncertainty band (pessimistic–optimistic range)
    fig.add_trace(go.Scatter(
        x=pd.concat([scenarios["month"], scenarios["month"][::-1]]),
        y=pd.concat([scenarios["pessimistic"], scenarios["optimistic"][::-1]]),
        fill="toself",
        fillcolor="rgba(231,76,60,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Projection range",
        hoverinfo="skip",
    ))

    # Central projection as a single dashed reference line
    fig.add_trace(go.Scatter(
        x=scenarios["month"],
        y=scenarios["central"],
        name="Central assumption",
        line=dict(color="#f39c12", width=2, dash="dash"),
        hovertemplate="%{x|%b %Y}<br>%{y:,.0f} central assumption<extra></extra>",
    ))

    fig.add_vline(x="2025-12-01", line_dash="dot", line_color="white", opacity=0.4)
    fig.add_annotation(
        x="2025-12-01", y=1.05, yref="paper",
        text="2026 projection range",
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
        st.markdown("**Optimistic assumption (~6,000/month by Dec 2026)**")
        st.markdown("""
        Food inflation falls back toward 2%. Labour's Crime and Policing Bill
        passes with effective retail crime powers. Wage growth continues to
        outpace inflation, gradually rebuilding household finances.
        Neighbourhood policing guarantee delivers visible presence on high
        streets.
        """)
    with col2:
        st.markdown("**Central assumption (~7,300/month through 2026)**")
        st.markdown("""
        Current conditions persist. Food inflation stays in the 3 to 5% range
        and household finances remain stretched but stable. Policing bill passes
        but implementation is slow. Shoplifting plateaus at its current elevated
        level, high but no longer rising.
        """)
    with col3:
        st.markdown("**Pessimistic assumption (~8,500/month by Dec 2026)**")
        st.markdown("""
        Food inflation continues rising, already back above 4% in late 2025.
        April 2026 brings further bill increases. Retail crime bill is delayed
        in Parliament. The five month delay pattern means pressure building now
        feeds into crime through spring 2026.
        """)


def _render_model_importance(model):
    st.subheader("4. Why policing alone cannot resolve this")

    r2_display = _get_model_r2(model)

    # Fix (critique): surface IMD data age caveat in the model section
    st.markdown(f"""
    A Random Forest model trained on deprivation indicators alone accounts for
    approximately **{r2_display}** of the variation in crime rates across
    London's neighbourhoods when evaluated using spatial block cross-validation.
    Without any knowledge of policing levels, enforcement activity, or local
    events, purely from knowing how deprived an area is, the model accounts
    for the majority of what we actually observe.

    This figure comes from holding out geographic clusters of LSOAs rather
    than random rows, which gives a more honest estimate of predictive power
    by preventing spatial leakage between training and test data.

    **IMD data caveat:** features are drawn from the 2025 Index of Multiple
    Deprivation, the most recent available. Five years of post-pandemic
    economic pressure are not reflected. In areas that have deteriorated most
    since 2025 — particularly those affected by sustained cost-of-living
    pressure — the model's predictive accuracy and the feature importances
    below may understate the current contribution of deprivation.
    """)

    # Fix (critique): build label map from model's actual feature names
    # rather than relying on IMD_DOMAIN_LABELS keys which previously
    # did not match the long column names from the raw IMD CSV.
    label_map = build_imd_label_map(model)

    if hasattr(model, "feature_names_in_") and label_map:
        feature_names = model.feature_names_in_
        display_labels = [label_map.get(f, f) for f in feature_names]
    else:
        # Fallback for old models without stored feature names
        feature_names  = list(range(len(model.feature_importances_)))
        display_labels = [f"Feature {i}" for i in feature_names]

    importance = pd.DataFrame({
        "feature":    display_labels,
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
        st.markdown("**What this predictability tells us**")
        st.markdown(f"""
        Knowing only an area's deprivation scores, you can predict its crime
        rate with reasonable accuracy. For a social phenomenon as complex as
        crime, that is a substantial figure.

        Notably, the two most predictive features are **Living Environment**
        and **Barriers to Housing and Services**, together accounting for
        roughly half of the model's predictive power. Income and employment
        deprivation, the factors most commonly discussed in relation to crime,
        are considerably less predictive in isolation. This suggests that
        physical environment — overcrowding, housing quality, access to green
        space, and barriers to services — is a stronger spatial predictor of
        crime rates than income poverty alone.

        This is a predictive association, not a causal claim. The remaining
        variation reflects factors the model cannot see: tourist footfall in
        Westminster, policing operations in Hackney, local community
        infrastructure in Haringey.
        """)
    with col2:
        st.markdown("**The implication for 2026**")
        st.markdown("""
        London's policing response, including stop and search, targeted
        operations, and the drugs enforcement shift, operates on the portion
        of variation the model cannot explain. The cost of living crisis and
        its aftermath operates on the portion it can.

        The dominance of housing environment and service access in the model
        points toward a specific policy implication: investment in housing
        quality, overcrowding reduction, and local service infrastructure is
        likely to have a larger effect on crime geography than income transfers
        alone. Areas like Newham and Hackney score highly on both housing
        environment and barriers to services, not just income deprivation.

        The neighbourhood policing guarantee and the retail crime bill are
        meaningful responses, but they operate on the margins of a structural
        problem that the data suggests requires investment in the physical
        and service environment, not just income support or enforcement.
        """)


def _render_methodology_expander(model, weight_sens: pd.DataFrame):
    with st.expander("Methodology: vulnerability index, weight sensitivity, and model detail"):

        # Fix (critique): lead with weight justification note and direct
        # users to the sensitivity table before they interpret rankings.
        st.markdown("""
        **A note on the vulnerability index weights**

        The base weights (deprivation 35%, shoplifting trend 30%,
        crime-deprivation mismatch 20%, policing intensity 15%) reflect
        analytical judgment, not statistical derivation. They encode the view
        that structural deprivation is the most important single driver of
        long-run crime risk, that shoplifting trend is the most timely
        available signal of current economic pressure, and that mismatch and
        policing intensity are important but secondary factors.

        These weights are contestable. The sensitivity table below shows how
        borough rankings shift across four alternative schemes. **Check the
        sensitivity table before treating any specific borough's rank as a
        reliable finding.** Boroughs consistently in the top 5 regardless of
        weights are the most robust.
        """)

        st.markdown("""
        **Vulnerability index** combines four borough-level indicators,
        each normalised by rank (0–1) using rank-based normalisation rather
        than min-max scaling. Rank normalisation is used because min-max
        scaling is sensitive to outlier boroughs: a single extreme value
        (e.g. City of London on crime rate) compresses all other scores
        toward zero and strips meaningful variation from the composite.

        - Deprivation (35%): average IMD decile, inverted so higher equals
          more deprived. Uses 2025 IMD, the most recent available.
          Post-2025 changes are not captured.
        - Shoplifting trend (30%): % change in shoplifting 2023 to 2025.
        - Crime-deprivation mismatch (20%): residual from borough-level
          regression of crime rate on deprivation.
        - Policing intensity (15%): stop and search volume weighted by
          inverse arrest rate.

        Risk tiers are assigned by tertile: the top third of boroughs
        by vulnerability score are "Higher risk", the bottom third are
        "Lower risk", and the middle third are "Medium risk". Tertile
        cut-points are computed from the actual score distribution rather
        than fixed thresholds, ensuring a roughly even three-way split
        (~11 boroughs per tier across 33) regardless of how scores
        cluster. Fixed thresholds were replaced after the first pipeline
        run produced 26 "Higher risk" boroughs and 0 "Lower risk".
        """)

        if not weight_sens.empty:
            st.markdown("**Weight sensitivity table**")

            n_scenarios = weight_sens["scenario"].nunique()

            top5_summary = (
                weight_sens[weight_sens["rank"] <= 5]
                .groupby("borough")
                .size()
                .rename("scenarios_in_top5")
                .reset_index()
                .sort_values("scenarios_in_top5", ascending=False)
            )
            top5_summary["scenarios_in_top5"] = (
                top5_summary["scenarios_in_top5"].astype(str) + f" of {n_scenarios}"
            )
            top5_summary.columns = ["Borough", "Scenarios in top 5"]
            st.dataframe(top5_summary, hide_index=True)

            pivot = weight_sens.pivot(
                index="borough", columns="scenario", values="rank"
            ).reset_index()
            pivot.columns.name = None
            pivot = pivot.sort_values("Base")
            if "Borough" not in pivot.columns:
                pivot = pivot.rename(columns={"borough": "Borough"})
            cols = ["Borough"] + [c for c in pivot.columns if c != "Borough"]
            pivot = pivot[cols]
            st.markdown("**Full ranking table by weighting scenario**")
            st.dataframe(pivot.head(15), hide_index=True)
            st.caption("Showing top 15 boroughs by base-scenario rank.")
        else:
            st.info(
                "Weight sensitivity table not available. "
                "Run processing/05_vulnerability_index.py to generate it."
            )

        st.markdown("""
        **Predictive model:** Random Forest Regressor (100 estimators,
        random_state=42). Target: crime rate per 1,000 residents per LSOA,
        log-transformed to address right skew. Extreme outliers capped at
        the 99th percentile. Features: seven 2025 IMD domain scores.

        **Evaluation:** Grid-based spatial cross-validation (5 folds).
        LSOAs are assigned to rectangular grid cells (~3.5 km at London's
        latitude) which are then grouped into folds. This ensures each
        held-out fold is a geographically contiguous region, preventing
        spatial leakage where neighbouring LSOAs appear in both training
        and test sets. The grid-based approach is preferred over k-means
        blocking because k-means can produce non-contiguous clusters that
        do not reflect genuine spatial separation. The spatial CV R² is the
        reported figure. Moran's I is computed on residuals to test for
        remaining spatial autocorrelation.

        This model quantifies predictive association between deprivation and
        crime rates. It is not a causal model. Causal inference would require
        a stronger identification strategy such as a natural experiment or
        instrumental variable approach.

        **Limitations:** IMD data is from 2025 and does not reflect
        post-pandemic changes to deprivation. Shoplifting scenarios are
        assumption-based projections, not statistical forecasts. The
        vulnerability index should be treated as indicative: a framework
        for thinking about relative risk, not a definitive ranking. The
        weight sensitivity table provides the appropriate uncertainty
        range for borough rankings. Borough assignment of stop and search
        records uses nearest centroid approximation. 5.9% of records had
        no GPS coordinates and are excluded. All crime data is recorded
        crime, which reflects enforcement activity as well as actual crime
        levels.
        """)


# ── Utility ───────────────────────────────────────────────────────

def _get_model_r2(model) -> str:
    if hasattr(model, "spatial_cv_r2_"):
        r2     = model.spatial_cv_r2_
        r2_std = getattr(model, "spatial_cv_r2_std_", float("nan"))
        if not _is_nan(r2_std):
            return f"{r2:.0%} (±{r2_std:.0%} SD across folds)"
        return f"{r2:.0%}"
    return "a substantial proportion"


def _is_nan(value) -> bool:
    try:
        import math
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True