"""
sections/policing_response.py
-----------------------------
'Policing Response' section — stop and search ethnicity breakdown,
effectiveness by search type, drugs comparison, and borough map.

Fix (critique - drugs narrative): the hypothesis table previously
framed the 'Changed recording practice' explanation as near-settled.
In reality the table is ambiguous: if Operation Yamata targeted supply
networks (where arrest rates are structurally higher), arrest rates
might not fall even under the recording-change hypothesis. The section
now presents all three explanations as genuinely contested and makes
the ambiguity explicit in the interpretive text, rather than leading
readers to a predetermined conclusion.

Fix (critique - missing file message): changepoint_hypotheses now
carries a _missing_reason attribute set by data_loaders when the file
is absent, so the st.info() message can explain which specific upstream
scripts need to be run rather than giving a generic fallback.
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
from utils.data_loaders import load_policing_data, get_narrative_stat
from utils.helpers import get_ethnicity_val


def render():
    st.title("Policing Response")
    st.markdown("""
    The Metropolitan Police conducted hundreds of thousands of stop and
    searches between 2023 and 2025. Most found nothing. Black Londoners
    were stopped at more than four times their population share — and their
    arrest rate was almost identical to White Londoners. This section
    examines who is being stopped, whether it is working, and what the
    August 2024 drugs spike tells us about policing strategy.
    """)

    data = load_policing_data()

    outcomes_summary       = data["outcomes_summary"]
    ethnicity              = data["ethnicity"]
    outcomes_by_search     = data["outcomes_by_search"]
    ss_borough             = data["ss_borough"]
    drugs_comparison       = data["drugs_comparison"]
    narrative_stats        = data["narrative_stats"]
    changepoint_hypotheses = data["changepoint_hypotheses"]

    total_searches = int(outcomes_summary["total"].values[0])
    arrest_rate    = float(outcomes_summary["arrest_rate"].values[0])
    no_action_rate = float(outcomes_summary["no_action_rate"].values[0])

    black_ratio       = get_ethnicity_val(ethnicity, "Black", "stop_rate_ratio")
    black_stop_pct    = get_ethnicity_val(ethnicity, "Black", "stop_pct")
    black_pop_pct     = get_ethnicity_val(ethnicity, "Black", "population_pct")
    black_arrest_rate = get_ethnicity_val(ethnicity, "Black", "arrest_rate")
    white_arrest_rate = get_ethnicity_val(ethnicity, "White", "arrest_rate")

    r_dep_black    = get_narrative_stat(narrative_stats, "deprivation_black_stop_correlation")
    r_crime_search = get_narrative_stat(narrative_stats, "crime_rate_search_volume_correlation")

    cp_date         = pd.to_datetime("2024-08-01")
    before_searches = drugs_comparison[drugs_comparison["month"] <  cp_date]["drug_searches"].mean()
    after_searches  = drugs_comparison[drugs_comparison["month"] >= cp_date]["drug_searches"].mean()

    top_borough            = ss_borough.nlargest(1, "total_searches").iloc[0]
    highest_arrest_borough = ss_borough.nlargest(1, "arrest_rate").iloc[0]

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
        f"Black people: {black_stop_pct}% of stops vs {black_pop_pct}% of population",
        f"{black_ratio:.1f}× overrepresented",
    )

    st.divider()

    _render_ethnicity_chart(
        ethnicity,
        black_pop_pct, black_stop_pct,
        black_ratio, black_arrest_rate, white_arrest_rate,
        r_dep_black,
    )

    st.divider()

    _render_effectiveness_chart(
        outcomes_by_search, arrest_rate, total_searches, drug_search_pct,
        r_crime_search,
    )

    st.divider()

    _render_drugs_comparison_chart(
        drugs_comparison, cp_date,
        before_searches, after_searches,
        changepoint_hypotheses,
    )

    st.divider()

    _render_borough_map(ss_borough, top_borough, highest_arrest_borough, r_crime_search)

    st.caption("""
    Source: Metropolitan Police & City of London Police stop and search data
    via police.uk, 2023 to 2025 |
    Population figures: ONS Census 2021. Note: Metropolitan Police records
    four broad ethnicity categories (Asian, Black, White, Other). The ONS
    Mixed category (5.7% of London population) is combined with Other in
    all comparisons to match the recorded data. |
    Borough assignment based on nearest centroid from GPS coordinates |
    24,142 records (5.9%) had no GPS coordinates and could not be assigned
    to a borough.
    """)


# ── Sub-renderers ─────────────────────────────────────────────────

def _render_ethnicity_chart(
    ethnicity,
    black_pop_pct, black_stop_pct,
    black_ratio, black_arrest_rate, white_arrest_rate,
    r_dep_black: float,
):
    st.subheader("1. Black Londoners are stopped at four times their population share")
    st.markdown("""
    Red bars show each group's share of stops; grey bars show their share
    of London's population. The gap between the two is the disparity.
    For Black Londoners it is the largest by a wide margin — and it is
    consistent across all three years in the data.
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

    if not _is_nan(r_dep_black):
        dep_corr_text = f"r={r_dep_black:.2f}"
    else:
        dep_corr_text = "r not available — rerun scripts 03 and 06"

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
        deprivation and Black stop percentage is {dep_corr_text}, meaning the
        disparity is actually higher in wealthier boroughs.
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
    outcomes_by_search, arrest_rate, total_searches, drug_search_pct,
    r_crime_search: float,
):
    st.subheader("2. Most stops find nothing — but targeting makes the difference")
    st.markdown(f"""
    Across {total_searches:,} searches, only {arrest_rate}% resulted in an
    arrest. That average hides large variation. Intelligence-led searches
    for stolen goods produce arrests in roughly one in four stops.
    High-volume drug searches — the most common type — produce arrests
    in roughly one in eight.
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

    if not _is_nan(r_crime_search):
        search_corr_text = f"r={r_crime_search:.2f}"
    else:
        search_corr_text = "r not available — rerun scripts 03 and 06"

    st.markdown(f"""
    Searches for stolen goods (24.3%) and evidence of offences (21.6%) have
    the highest arrest rates. These are targeted, intelligence-led searches
    more likely to find what they are looking for. Drug searches account for
    approximately {drug_search_pct}% of all stops but result in arrest only
    13.2% of the time, below the overall average. Fireworks searches at 4.4%
    are the least productive.

    Search volume correlates with crime rate across boroughs at {search_corr_text},
    meaning police are broadly concentrating searches where crime is highest.
    """)


def _render_drugs_comparison_chart(
    drugs_comparison, cp_date,
    before_searches, after_searches,
    changepoint_hypotheses: pd.DataFrame,
):
    st.subheader("3. The August 2024 drugs spike: more crime, or more recording?")
    st.markdown("""
    Drug offences jumped in August 2024 and stayed high. Three explanations
    are plausible: more enforcement, changed recording practice, or genuinely
    more drug activity. The table below tests each against what the stop and
    search data actually shows.

    Note: none of these tests is conclusive on its own — Operation Yamata
    targeted supply networks where arrest rates are structurally higher,
    which complicates the recording-change test. Read each row as one piece
    of evidence, not a verdict.
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

    # ── Hypothesis table ──────────────────────────────────────────
    if not changepoint_hypotheses.empty:
        st.markdown("**What does the data say about each explanation?**")

        display_cols = ["hypothesis", "metric", "before", "after", "supports"]
        display_df   = changepoint_hypotheses[
            [c for c in display_cols if c in changepoint_hypotheses.columns]
        ].copy()
        display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        st.markdown("""
        **Reading the table:** each row tests one explanation against its
        own prediction. All three predictions could be partially true
        simultaneously — policing operations, recording changes, and
        genuine drug activity are not mutually exclusive. The table is a
        tool for interrogating the data, not for settling the question.

        The search volumes column ('More enforcement activity') is the
        most direct test: if drug searches did not rise alongside recorded
        offences, enforcement volume alone cannot explain the spike. The
        arrest rate column ('Changed recording practice') is more ambiguous
        because structural factors in targeted operations can affect rates
        independently of recording practice.
        """)
    else:
        # Fix (critique): show specific missing-file reason if available
        missing_reason = getattr(
            changepoint_hypotheses, "_missing_reason",
            "Hypothesis table not available. Run scripts 02 and 06 in sequence "
            "to generate ss_changepoint_hypotheses.csv."
        )
        st.info(missing_reason)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Search volumes in numbers**")
        st.markdown(f"""
        Drug stop and searches averaged {before_searches:,.0f} per month
        before August 2024 and {after_searches:,.0f} per month afterwards,
        a change of less than 0.3%. If the recorded crime spike had been
        driven solely by more searches, the purple line would rise alongside
        the red one. It does not — though this does not rule out that search
        targeting became more efficient rather than more frequent.
        """)
    with col2:
        st.markdown("**What the evidence points toward**")
        st.markdown("""
        The most parsimonious explanation consistent with the data is a
        change in how drug encounters were recorded, likely connected to
        Operation Yamata and the Metropolitan Police's Drugs Action Plan
        which explicitly targeted higher recording rates for drug supply
        networks.

        However, this remains a working interpretation rather than a firm
        conclusion. The absence of a rise in search volumes is necessary
        but not sufficient evidence for the recording-change explanation.
        The Policing Response section should be read with that caveat in mind.
        """)


def _render_borough_map(
    ss_borough, top_borough, highest_arrest_borough,
    r_crime_search: float,
):
    st.subheader("4. High volume, low effectiveness: the geographic picture")
    st.markdown("""
    Bubble size is total searches; colour is arrest rate. Large red bubbles
    are boroughs with high-volume searching that is not converting to
    arrests. Large green bubbles are boroughs where activity is more
    targeted and productive. The contrast tells you something about
    how intelligence-led the local approach is.
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

    if not _is_nan(r_crime_search):
        corr_text = f"r={r_crime_search:.2f}"
    else:
        corr_text = "r not available"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{top_borough['borough']}: most searched borough**")
        st.markdown(f"""
        {top_borough['borough']} has the highest stop and search volume with
        {int(top_borough['total_searches']):,} searches over the period. This
        is consistent with its position as London's highest crime rate borough.
        Search volume correlates with crime rate across all boroughs at
        {corr_text}, meaning police are broadly searching where crime is highest.
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


# ── Utility ───────────────────────────────────────────────────────

def _is_nan(value: float) -> bool:
    try:
        import math
        return math.isnan(value)
    except (TypeError, ValueError):
        return True