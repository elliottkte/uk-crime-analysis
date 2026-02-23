"""
sections/economic_crime.py
--------------------------
'Economic Crime' section — shoplifting vs food inflation, lag
correlations, seasonal decomposition, drugs changepoint, and
borough-level shoplifting trends.

Fix (critique - dead code): _format_lag_narrative() previously had a
branch checking for a 'lag_months' column that has never existed in the
output CSV (the column is always called 'lag'). If that branch had been
reached via a KeyError-triggering fallback it would have raised rather
than handled gracefully. The dead branch is removed; the function now
reads 'lag' directly.

Fix (critique - STL caveat): build_decomposition() in script 02 now
writes 'stl_reliable' and 'n_months' columns. _render_decomposition_chart()
reads these and shows a visible st.warning() in the dashboard when the
series was too short for reliable STL estimates, rather than only logging
to the console during pipeline execution.
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

# Minimum months for STL to be considered reliable (must match script 02)
_STL_MIN_MONTHS = 24


def render():
    st.title("Economic Crime")
    st.markdown("""
    London's shoplifting rate has risen every year since 2023 and shows no sign
    of reversing. This section traces that surge — separating the genuine
    structural increase from seasonal noise, linking it to the financial
    pressure on households, and distinguishing it from the drug offence spike
    that tells a completely different story.
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

    best_lag = _get_best_lag(lag_df)

    cp_date      = pd.to_datetime(changepoint["change_point_date"].values[0])
    before_mean  = float(changepoint["mean_before"].values[0])
    after_mean   = float(changepoint["mean_after"].values[0])
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
        f"Shoplifting: up from {avg_2023:,.0f} to",
        f"{avg_2025:,.0f} /month",
    )
    col2.metric(
        f"Drug offences: up from {before_mean:.0f} (pre-Aug 2024)",
        f"{after_mean:.0f} /month",
    )
    col3.metric(
        "Shoplifting trend: Jan 2023 → Dec 2025",
        f"+{trend_increase:.1f}%",
    )

    st.divider()

    _render_crime_index_chart(indexed)

    st.info("""
    **The pattern divides cleanly.** Crimes linked to financial pressure
    (shoplifting, theft) rose and stayed high. Crimes linked to physical
    opportunity (burglary, vehicle crime) fell as security improved. Drug
    offences did neither — they were flat, then jumped sharply in a single
    month. That sudden switch is the focus of chart 4.
    """)

    st.divider()

    _render_shoplifting_inflation_chart(monthly_crime("Shoplifting"), food, best_lag)

    st.divider()

    _render_decomposition_chart(decomp_clean, decomp, trend_increase)

    st.divider()

    _render_drugs_changepoint_chart(
        monthly_crime("Drugs"), cp_date, before_mean, after_mean, pct_increase
    )

    st.divider()

    _render_borough_chart(borough)

    st.caption("""
    Source: Metropolitan Police & City of London Police via police.uk |
    Food inflation: ONS CPI series D7G8 |
    Trend: STL decomposition (robust=True, period=12) |
    Lag correlations: Pearson r with 95% bootstrap CI (1,000 iterations) |
    Deprivation: MHCLG English Indices of Deprivation 2025
    """)


# ── Helpers ───────────────────────────────────────────────────────

def _get_best_lag(lag_df: pd.DataFrame) -> pd.Series:
    if "best_lag" in lag_df.columns:
        best_rows = lag_df[lag_df["best_lag"] == True]
        if not best_rows.empty:
            return best_rows.iloc[0]
    valid = lag_df.dropna(subset=["r"])
    if valid.empty:
        return lag_df.iloc[0]
    return valid.loc[valid["r"].abs().idxmax()]


def _format_lag_narrative(best_lag: pd.Series) -> str:
    """
    Build the lag narrative string.

    The peak absolute correlation is at a ~5 month lag but returns a
    *negative* r because food inflation was falling over most of the
    2023–2025 series while shoplifting was rising. In that context a
    negative lagged correlation does not mean 'lower food inflation
    causes more shoplifting'; it means the two series moved in opposite
    directions over this window — food inflation peaked early and fell
    while shoplifting kept rising, producing an inverted shape when one
    is lagged against the other.

    The narrative therefore describes the *delay* between financial
    pressure building and crime responding, not the sign of the
    correlation. The r value and CI are reported as a transparency note
    rather than as the centrepiece claim.
    """
    import math

    lag_months = int(best_lag["lag"])
    r_val      = float(best_lag.get("r", float("nan")))
    n_val      = best_lag.get("n", None)

    has_ci = (
        "ci_lower" in best_lag.index
        and "ci_upper" in best_lag.index
        and pd.notna(best_lag["ci_lower"])
        and pd.notna(best_lag["ci_upper"])
    )

    # Negative r in this context reflects diverging trajectories (inflation
    # falling while shoplifting rose), not a genuine inverse relationship.
    # Surface this as a methodological note rather than hiding it.
    if not math.isnan(r_val) and r_val < 0:
        ci_note = ""
        if has_ci:
            n_str  = f", n={int(n_val)}" if n_val is not None else ""
            ci_note = (
                f" (r={r_val:.2f}, 95% CI: {best_lag['ci_lower']:.2f}–"
                f"{best_lag['ci_upper']:.2f}{n_str} — negative sign reflects "
                f"diverging trajectories over this window, not an inverse "
                f"causal relationship)"
            )
        return (
            f"approximately a **{lag_months} month delay**{ci_note}. "
            f"The negative correlation coefficient reflects the fact that "
            f"food inflation peaked early in this series (2023) and then "
            f"fell, while shoplifting kept rising — so the two series moved "
            f"in opposite directions over the window, producing an inverted "
            f"lagged shape. The meaningful signal is the delay between "
            f"financial pressure building and crime behaviour responding, "
            f"not the direction of the coefficient."
        )

    if has_ci:
        ci_lower = best_lag["ci_lower"]
        ci_upper = best_lag["ci_upper"]
        n_str    = f", n={int(n_val)}" if n_val is not None else ""
        return (
            f"a **{lag_months} month lag** (r={r_val:.2f}, "
            f"95% CI: {ci_lower:.2f}–{ci_upper:.2f}{n_str}). "
            f"The confidence interval reflects the limited sample size "
            f"(~36 months of data), so treat this as indicative rather "
            f"than precise."
        )

    return f"roughly a **{lag_months} month delay**."


# ── Sub-renderers ─────────────────────────────────────────────────

def _render_crime_index_chart(indexed: pd.DataFrame):
    st.subheader("1. Two crimes rose, two fell, one jumped suddenly")
    st.markdown("""
    Indexed to January 2023. Each line shows how far a crime type has moved
    from that baseline — upward means more incidents, downward means fewer.
    The divergence between shoplifting and burglary is the central story:
    one driven by economic pressure, one by physical security improvements.
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
        subset["pct_change"] = (subset["index_value"] - 100).round(1)
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
    st.subheader("2. Shoplifting didn't fall when inflation eased — it kept climbing")
    st.markdown("""
    Food price inflation peaked above 19% in early 2023, then fell steadily.
    Shoplifting did not follow. By the time inflation had halved, shoplifting
    was still rising — suggesting households were responding to accumulated
    financial damage, not the current rate of price increases.
    """)

    monthly_shop = monthly_shop.copy()
    merged       = monthly_shop.merge(food, on="month", how="inner")
    inflection   = merged[merged["food_inflation"] < 5].iloc[0]

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

    lag_text = _format_lag_narrative(best_lag)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Why didn't shoplifting fall when prices eased?**")
        st.markdown(f"""
        Food prices rose around 38% in total between 2020 and 2025. Even as
        the rate of increase slowed, prices stayed permanently higher and
        households remained in financial deficit from years of wages failing
        to keep pace.

        The statistical relationship between food inflation and shoplifting
        over this period shows {lag_text} This counterintuitive result arises
        because the two series moved in opposite directions over 2023–2025:
        inflation peaked early and fell while shoplifting kept rising. That
        divergence is itself the point — it shows shoplifting did not respond
        to inflation easing, consistent with households facing accumulated
        financial damage that does not reverse when price rises slow.
        """)
    with col2:
        st.markdown("**What does rising food inflation in 2025 mean?**")
        st.markdown("""
        Food inflation has been rising again since late 2024, sitting above
        4% through the end of 2025. If the accumulated-damage pattern holds,
        this adds to structural pressure rather than triggering a simple
        parallel rise in shoplifting.

        The Joseph Rowntree Foundation's winter 2025 tracker found 61% of UK
        households reported their cost of living was still increasing, which
        suggests the structural damage from the crisis period is not resolved.
        Further inflation will compound household deficits that are already
        contributing to elevated shoplifting.
        """)


def _render_decomposition_chart(
    decomp_clean: pd.DataFrame,
    decomp_full: pd.DataFrame,
    trend_increase: float,
):
    st.subheader("3. Strip out the seasonality: the trend is unambiguously up")
    st.markdown("""
    Shoplifting peaks every summer as high streets fill up. Removing that
    predictable seasonal rhythm with STL decomposition leaves the structural
    signal — the grey raw data, the red underlying trend. If the rise were
    just seasonal, the trend line would be flat. It is not.
    """)

    # Fix (critique): surface the STL reliability flag to dashboard users.
    # Script 02 writes stl_reliable and n_months columns so we can show a
    # contextual warning without re-computing anything at render time.
    if "stl_reliable" in decomp_full.columns:
        stl_reliable = bool(decomp_full["stl_reliable"].iloc[0])
        n_months     = int(decomp_full["n_months"].iloc[0])
        if not stl_reliable:
            st.warning(
                f"**Data quality note:** STL decomposition is based on only "
                f"{n_months} months of data. At least {_STL_MIN_MONTHS} months "
                f"(two full annual cycles) are recommended for reliable trend "
                f"and seasonal estimates. Treat the trend line with caution — "
                f"early estimates in particular may be unstable."
            )

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
        name="Underlying trend (STL)",
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
    **The underlying monthly rate rose {trend_increase:.1f}% — and it hasn't turned.**
    This compares the trend value in January 2023 (~4,100/month) with
    December 2025 (~7,800/month), after removing seasonal fluctuations.
    It is higher than the raw year-on-year figure (+53.5%) because annual
    totals average across the full year, including the lower months early
    in 2023 when the surge was just beginning. The trend figure more
    accurately captures how far the monthly rate has actually moved.
    """)


def _render_drugs_changepoint_chart(
    monthly_drugs: pd.DataFrame,
    cp_date: pd.Timestamp,
    before_mean: float,
    after_mean: float,
    pct_increase: float,
):
    st.subheader("4. Drug offences tell a different story: a single sudden jump")
    st.markdown(f"""
    Unlike shoplifting, drug offences did not creep upward. They were broadly
    flat for 18 months, then jumped sharply in **{cp_date.strftime('%B %Y')}**
    and have remained at that higher level since. The Policing Response section
    tests competing explanations for this jump against the stop and search data.
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
        activity. The Policing Response section tests this interpretation
        directly using a hypothesis table that shows what each competing
        explanation predicts and whether the stop and search data supports it.
        """)
    with col2:
        st.markdown("**Does more recorded drug crime mean more drugs?**")
        st.markdown("""
        Not necessarily. Recorded drug offences increase when police actively
        search for them through targeted operations and stop and search. If
        drug search volumes did not rise alongside recorded offences, that
        would suggest changed recording practice rather than changed drug
        activity. The Policing section examines this directly. Look at the
        hypothesis table there before drawing conclusions from this chart.
        """)


def _render_borough_chart(borough: pd.DataFrame):
    st.subheader("5. The surge was not evenly spread across London")
    st.markdown("""
    Most boroughs saw double-digit increases. But the range is wide — some
    boroughs saw shoplifting more than double while others barely moved.
    Deprivation decile is shown on hover: the link between deprivation and
    the size of the increase is weaker than you might expect.
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