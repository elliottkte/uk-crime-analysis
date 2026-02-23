"""
sections/economic_crime.py
--------------------------
'Economic Crime' section — shoplifting vs food inflation, lag
correlations, seasonal decomposition, drugs changepoint, and
borough-level shoplifting trends.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.charts import (
    add_economic_annotations,
    add_vline_annotation,
    apply_base_layout,
    horizontal_bar_chart,
    style_xaxis,
    style_yaxis,
)
from utils.constants import BOROUGH_RENAME, CHART_CONFIG
from utils.data_loaders import load_economic_crime_data, load_street_summary
from utils.helpers import safe_get_borough


def render():
    st.title("Economic Crime")
    st.markdown("""
    From 2023 onwards, certain crimes responded to financial pressure in ways
    the data makes clear. This section examines which crimes rose, why, and
    what the evidence suggests about the outlook.
    """)

    data    = load_economic_crime_data()
    summary = load_street_summary()

    monthly_by_crime = summary["monthly_by_crime"]

    def monthly_crime(crime_type):
        return (
            monthly_by_crime[monthly_by_crime["crime_type"] == crime_type]
            [["month", "count"]].copy()
        )

    indexed     = data["indexed"]
    lag_df      = data["lag_df"]
    decomp      = data["decomp"]
    changepoint = data["changepoint"]
    borough     = data["borough"]
    food        = data["food"]

    # ── Derive narrative values ───────────────────────────────────
    best_lag    = lag_df.loc[lag_df["r"].abs().idxmax()]
    cp_date     = pd.to_datetime(changepoint["change_point_date"].values[0])
    before_mean = float(changepoint["mean_before"].values[0])
    after_mean  = float(changepoint["mean_after"].values[0])
    pct_increase = round((after_mean - before_mean) / before_mean * 100, 1)

    decomp_clean   = decomp.dropna(subset=["trend"])
    trend_vals     = decomp_clean["trend"]
    trend_increase = round(
        (trend_vals.max() - trend_vals.min()) / trend_vals.min() * 100, 1
    )

    monthly_shop_all = monthly_crime("Shoplifting")
    avg_2025 = monthly_shop_all[monthly_shop_all["month"].dt.year == 2025]["count"].mean()
    avg_2023 = monthly_shop_all[monthly_shop_all["month"].dt.year == 2023]["count"].mean()

    # ── Headline metrics ──────────────────────────────────────────
    st.caption("Metropolitan and City of London Police recorded crime, 2023 to 2025.")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Shoplifting: 2025 monthly average",
        f"{avg_2025:,.0f}",
        f"Up from {avg_2023:,.0f} per month in 2023",
        delta_color="off",
    )
    col2.metric(
        "Drug offences: new monthly average",
        f"{after_mean:.0f}",
        f"Was {before_mean:.0f} before {cp_date.strftime('%b %Y')}",
        delta_color="off",
    )
    col3.metric(
        "Shoplifting underlying trend increase",
        f"+{trend_increase:.1f}%",
        "Seasonal patterns removed",
        delta_color="off",
    )

    st.divider()

    # ── 1. % change chart ─────────────────────────────────────────
    _render_crime_index_chart(indexed)

    st.info("""
    Shoplifting and theft rose sharply and stayed high. Vehicle crime and
    burglary fell consistently. Drug offences were flat for 18 months then
    jumped suddenly in a pattern more consistent with a policing change than
    an economic one.
    """)

    st.divider()

    # ── 2. Shoplifting vs food inflation ──────────────────────────
    _render_shoplifting_inflation_chart(monthly_crime("Shoplifting"), food, best_lag)

    st.divider()

    # ── 3. Trend decomposition ────────────────────────────────────
    _render_decomposition_chart(decomp_clean, trend_increase)

    st.divider()

    # ── 4. Drugs changepoint ──────────────────────────────────────
    _render_drugs_changepoint_chart(
        monthly_crime("Drugs"), cp_date, before_mean, after_mean, pct_increase
    )

    st.divider()

    # ── 5. Borough breakdown ──────────────────────────────────────
    _render_borough_chart(borough)

    st.caption("""
    Source: Metropolitan Police & City of London Police via police.uk |
    Food inflation: ONS CPI series D7G8 |
    Trend: Additive seasonal decomposition |
    Deprivation: MHCLG English Indices of Deprivation 2019
    """)


# ── Sub-renderers ─────────────────────────────────────────────────

def _render_crime_index_chart(indexed: pd.DataFrame):
    st.subheader("1. Which crimes went up and which went down?")
    st.markdown("""
    The chart shows how each crime type changed relative to January 2023.
    A value of +50 means 50% more incidents than at the start of the period.
    Zero means no change. Negative means it fell.
    """)

    economic_highlight = [
        "Shoplifting", "Theft from the person", "Drugs",
        "Vehicle crime", "Burglary",
    ]
    colors = {
        "Shoplifting":            "#e74c3c",
        "Theft from the person":  "#e67e22",
        "Drugs":                  "#9b59b6",
        "Vehicle crime":          "#95a5a6",
        "Burglary":               "#7f8c8d",
    }

    fig = go.Figure()
    for crime in economic_highlight:
        subset = indexed[indexed["crime_type"] == crime].copy()
        if subset.empty:
            continue
        subset["pct_change"] = subset["index_value"] - 100
        fig.add_trace(go.Scatter(
            x=subset["month"],
            y=subset["pct_change"],
            name=crime,
            line=dict(
                color=colors.get(crime, "#95a5a6"),
                width=3 if crime in ["Shoplifting", "Drugs"] else 1.5,
                dash="solid" if crime in [
                    "Shoplifting", "Drugs", "Theft from the person"
                ] else "dot",
            ),
            hovertemplate="%{x|%b %Y}: %{y:+.1f}%<extra>" + crime + "</extra>",
        ))

    fig.add_hline(
        y=0, line_dash="dash", line_color="white",
        opacity=0.3, annotation_text="No change from Jan 2023",
    )
    # Lines only — labels omitted to avoid crowding
    fig = add_economic_annotations(
        fig, indexed["index_value"].max() - 100, show_all=False
    )
    fig = apply_base_layout(
        fig, height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig = style_xaxis(fig)
    fig = style_yaxis(fig, title="% change since January 2023", ticksuffix="%")

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)


def _render_shoplifting_inflation_chart(
    monthly_shop: pd.DataFrame,
    food: pd.DataFrame,
    best_lag: pd.Series,
):
    st.subheader("2. Shoplifting kept rising after food prices stopped surging")
    st.markdown("""
    You might expect shoplifting to track food prices: rising when they spike,
    falling when they ease. The data shows something different.
    """)

    monthly_shop = monthly_shop.copy()
    merged     = monthly_shop.merge(food, on="month", how="inner")
    inflection = merged[merged["food_inflation"] < 5].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged["month"],
        y=merged["count"],
        name="Monthly shoplifting incidents",
        marker_color="rgba(231, 76, 60, 0.6)",
        yaxis="y1",
        hovertemplate="%{x|%b %Y}: %{y:,} incidents<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=merged["month"],
        y=merged["food_inflation"],
        name="Food price inflation (%)",
        line=dict(color="#f39c12", width=3),
        yaxis="y2",
        hovertemplate="%{x|%b %Y}: %{y:.1f}% food inflation<extra></extra>",
    ))
    fig.add_vline(
        x=inflection["month"], line_dash="dot",
        line_color="#f39c12", line_width=2, opacity=0.9,
    )
    fig.add_annotation(
        x=inflection["month"], y=1.08, yref="paper",
        text="Food inflation falls below 5%<br>Shoplifting continues rising",
        showarrow=False,
        font=dict(color="#f39c12", size=11),
        xanchor="left", xshift=8,
        bgcolor="rgba(0,0,0,0.5)",
    )
    fig.update_layout(
        yaxis=dict(
            title=dict(text="Monthly shoplifting incidents", font=dict(color="#e74c3c")),
            showspikes=False, gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis2=dict(
            title=dict(text="Food price inflation (%)", font=dict(color="#f39c12")),
            overlaying="y", side="right", showspikes=False,
        ),
    )
    fig = apply_base_layout(
        fig, height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig = style_xaxis(fig)

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Why didn't shoplifting fall when prices eased?**")
        st.markdown(f"""
        Food prices rose around 38% in total between 2020 and 2025. Even as
        the rate of increase slowed, prices stayed permanently higher and
        households remained in financial deficit from years of wages failing
        to keep pace.

        The data shows shoplifting responds to food price changes with roughly
        a **{int(best_lag['lag_months'])} month delay**. Financial damage takes
        several months to feed through into crime behaviour. By the time
        inflation eased in mid-2024, many households had already exhausted
        their savings and credit.
        """)
    with col2:
        st.markdown("**What does rising food inflation in 2025 mean?**")
        st.markdown("""
        Food inflation has been rising again since late 2024, sitting above
        4% through the end of 2025. If the same delay pattern holds, this
        points to continued upward pressure on shoplifting into 2026.

        The Joseph Rowntree Foundation's winter 2025 tracker found 61% of UK
        households reported their cost of living was still increasing, which
        suggests the structural damage from the crisis is not resolved.
        """)


def _render_decomposition_chart(decomp_clean: pd.DataFrame, trend_increase: float):
    st.subheader("3. The underlying trend")
    st.markdown("""
    Shoplifting naturally rises in summer and dips in winter as high streets
    get busier and opportunistic theft increases. To test whether the overall
    increase is real or just a seasonal pattern, we can remove those
    predictable fluctuations mathematically.

    The chart shows raw monthly figures in grey and the underlying trend in
    red, with seasonal variation removed.
    """)

    decomp_trimmed = decomp_clean[decomp_clean["month"] >= "2023-07-01"].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decomp_trimmed["month"],
        y=decomp_trimmed["observed"],
        name="Actual monthly count",
        line=dict(color="rgba(149,165,166,0.4)", width=1),
        fill="tozeroy",
        fillcolor="rgba(149,165,166,0.05)",
        hovertemplate="%{x|%b %Y}: %{y:,.0f} actual<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=decomp_trimmed["month"],
        y=decomp_trimmed["trend"],
        name="Underlying trend",
        line=dict(color="#e74c3c", width=3),
        hovertemplate="%{x|%b %Y}: %{y:,.0f} trend<extra></extra>",
    ))
    fig = apply_base_layout(
        fig, height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig = style_xaxis(fig)
    fig = style_yaxis(fig, title="Monthly shoplifting incidents")

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    st.info(f"""
    The underlying trend rose {trend_increase:.1f}% over this period. This is
    the genuine structural increase, separate from seasonal patterns. The
    trend shows no sign of reversing even as inflation eased in 2024,
    consistent with cumulative financial damage rather than a short-term
    price shock. Trend data begins in mid-2023 as seasonal decomposition
    requires several months of data to initialise.
    """)


def _render_drugs_changepoint_chart(
    monthly_drugs: pd.DataFrame,
    cp_date: pd.Timestamp,
    before_mean: float,
    after_mean: float,
    pct_increase: float,
):
    st.subheader("4. Drug offences: a sudden switch, not a gradual rise")
    st.markdown(f"""
    Unlike shoplifting, drug offences did not creep upward. They were broadly
    flat for 18 months, then jumped sharply in **{cp_date.strftime('%B %Y')}**
    and have remained at that higher level since.
    """)

    monthly_drugs = monthly_drugs.copy()

    fig = go.Figure()
    fig.add_vrect(
        x0=monthly_drugs["month"].min(), x1=cp_date,
        fillcolor="rgba(149,165,166,0.08)", layer="below", line_width=0,
        annotation_text=f"Average: {before_mean:.0f}/month",
        annotation_position="top left",
        annotation_font=dict(color="#95a5a6", size=11),
    )
    fig.add_vrect(
        x0=cp_date, x1=monthly_drugs["month"].max(),
        fillcolor="rgba(155,89,182,0.08)", layer="below", line_width=0,
        annotation_text=f"Average: {after_mean:.0f}/month",
        annotation_position="top right",
        annotation_font=dict(color="#9b59b6", size=11),
    )
    fig.add_trace(go.Scatter(
        x=monthly_drugs["month"],
        y=monthly_drugs["count"],
        line=dict(color="#9b59b6", width=2.5),
        showlegend=False,
        hovertemplate="%{x|%b %Y}<br>%{y:,} offences<extra></extra>",
    ))
    fig = add_vline_annotation(
        fig, cp_date,
        label=f"<b>{cp_date.strftime('%B %Y')}</b><br>Structural shift detected",
        color="#e74c3c",
        y_ref=monthly_drugs["count"].max() * 0.92,
    )
    fig = apply_base_layout(fig, height=360, showlegend=False)
    fig = style_xaxis(fig)
    fig = style_yaxis(fig, title="Monthly drug offences recorded")

    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{before_mean:.0f} to {after_mean:.0f} per month (+{pct_increase:.1f}%)**")
        st.markdown(f"""
        Labour took office in July 2024, one month before this shift. The
        Metropolitan Police's Drugs Action Plan and Operation Yamata
        specifically target organised drug supply networks across London.

        A sudden, sustained jump of this kind is more consistent with a
        deliberate operational decision than a change in underlying drug
        activity. The Policing section tests this interpretation directly
        against stop and search data.
        """)
    with col2:
        st.markdown("**Does more recorded drug crime mean more drugs?**")
        st.markdown("""
        Not necessarily. Recorded drug offences increase when police actively
        search for them through targeted operations and stop and search. The
        Policing section shows drug stop and search volumes were flat around
        this period, which supports the interpretation that the jump reflects
        changed recording practice rather than changed drug activity.
        """)


def _render_borough_chart(borough: pd.DataFrame):
    st.subheader("5. Which areas saw shoplifting rise most?")
    st.markdown("""
    The increase was not spread evenly. The five boroughs with the biggest
    increases and the five with the smallest are shown below, including one
    borough where shoplifting fell.
    """)
    st.caption("""
    Deprivation decile: areas ranked 1 to 10. Decile 1 is the most deprived
    10% of areas in England. Decile 10 is the least deprived.
    """)

    top5 = borough.nlargest(5,  "change_pct")
    bot5 = borough.nsmallest(5, "change_pct")
    display_boroughs = pd.concat([top5, bot5]).sort_values("change_pct").copy()

    fig = horizontal_bar_chart(
        df=display_boroughs,
        x_col="change_pct",
        y_col="borough",
        hover_template=(
            "<b>%{y}</b><br>"
            "Change: %{x:+.1f}%<br>"
            "Deprivation decile: %{customdata:.1f}"
            "<extra></extra>"
        ),
        customdata_col="avg_imd_decile",
        height=420,
        x_title="% change in shoplifting 2023 to 2025",
        x_suffix="%",
    )
    st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)

    display_cols = ["borough", "change_pct", "avg_imd_decile", "count_2023", "count_2025"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biggest increases**")
        st.dataframe(
            top5[display_cols].rename(columns=BOROUGH_RENAME).round(1),
            hide_index=True,
        )
    with col2:
        st.markdown("**Smallest increases / decreases**")
        st.dataframe(
            bot5[display_cols].rename(columns=BOROUGH_RENAME).round(1),
            hide_index=True,
        )

    barking_row = safe_get_borough(borough, "Barking and Dagenham")
    if barking_row is not None and barking_row["change_pct"] < 0:
        st.markdown(f"""
        **Note on {barking_row['borough']}:** Despite being one of London's most
        deprived boroughs (deprivation decile {barking_row['avg_imd_decile']:.1f}),
        shoplifting fell {abs(barking_row['change_pct']):.1f}%. This runs counter
        to the overall pattern and likely reflects local retail composition
        changes, store closures reducing opportunity, or targeted enforcement
        rather than improving economic conditions.
        """)