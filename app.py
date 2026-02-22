import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import requests

st.set_page_config(
    page_title="London Crime & the Cost of Living",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    .js-plotly-plot .plotly .cursor-crosshair {
        cursor: default !important;
    }
    .js-plotly-plot .plotly .cursor-pointer {
        cursor: default !important;
    }
    .js-plotly-plot .plotly .cursor-move {
        cursor: default !important;
    }
    .js-plotly-plot .plotly svg {
        cursor: default !important;
    }
    g.spikeline {
        display: none !important;
    }
    line.spikeline {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────
@st.cache_data
def load_full_street():
    df = pd.read_csv('data/processed/street_sample.csv')
    df['month'] = pd.to_datetime(df['month'])
    df['year'] = df['month'].dt.year
    return df


@st.cache_data
def load_modelling_data():
    return pd.read_csv('data/processed/modelling_data.csv')


@st.cache_data
def load_borough():
    london_boroughs = [
        'City of London', 'Barking and Dagenham', 'Barnet', 'Bexley', 'Brent',
        'Bromley', 'Camden', 'Croydon', 'Ealing', 'Enfield', 'Greenwich',
        'Hackney', 'Hammersmith and Fulham', 'Haringey', 'Harrow', 'Havering',
        'Hillingdon', 'Hounslow', 'Islington', 'Kensington and Chelsea',
        'Kingston upon Thames', 'Lambeth', 'Lewisham', 'Merton', 'Newham',
        'Redbridge', 'Richmond upon Thames', 'Southwark', 'Sutton',
        'Tower Hamlets', 'Waltham Forest', 'Wandsworth', 'Westminster'
    ]
    df = pd.read_csv('data/processed/borough_outliers.csv')
    return df[df['borough'].isin(london_boroughs)]


@st.cache_data
def load_stop_search():
    import glob
    files = glob.glob('data/raw/**/*stop*search*.csv', recursive=True)
    ss = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ss.columns = [c.lower().replace(' ', '_') for c in ss.columns]
    ss['date'] = pd.to_datetime(ss['date'], utc=True)
    ss['year'] = ss['date'].dt.year
    ss['hour'] = ss['date'].dt.hour
    return ss


@st.cache_data
def load_population():
    pop = pd.read_excel('data/raw/sapelsoasyoa20222024.xlsx',
                        sheet_name='Mid-2022 LSOA 2021',
                        skiprows=3, usecols=[2, 3, 4], header=0)
    pop.columns = ['lsoa_code', 'lsoa_name', 'population']
    pop = pop.dropna()
    pop['population'] = pd.to_numeric(pop['population'], errors='coerce')
    return pop.dropna(subset=['population'])


@st.cache_data
def get_live_data():
    url = "https://data.police.uk/api/crimes-street/all-crime"
    params = {'lat': 51.5074, 'lng': -0.1278, 'date': '2025-10'}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return pd.DataFrame(r.json())
    except:
        pass
    return None


def load_model():
    return joblib.load('models/crime_rate_model.pkl')


# ── Economic event annotations ────────────────────────────────────

ECONOMIC_EVENTS = [
    {'date': '2023-01-01', 'label': 'Energy bills at\ncrisis peak', 'color': '#e74c3c'},
    {'date': '2023-04-01', 'label': 'Core inflation\npeaks at 7.1%', 'color': '#e67e22'},
    {'date': '2023-07-01', 'label': 'Energy cap\nfalls to £2,074', 'color': '#2ecc71'},
    {'date': '2024-05-01', 'label': 'Inflation hits\n2% target', 'color': '#2ecc71'},
    {'date': '2024-07-01', 'label': 'Labour\nelected', 'color': '#3498db'},
    {'date': '2025-04-01', 'label': '"Awful April"\nbills rise again', 'color': '#e74c3c'},
]


def add_economic_annotations(fig, y_max, events=None, show_all=True):
    if events is None:
        events = ECONOMIC_EVENTS
    for i, event in enumerate(events):
        fig.add_vline(
            x=event['date'],
            line_dash='dash',
            line_color=event['color'],
            opacity=0.5
        )
        if show_all:
            fig.add_annotation(
                x=event['date'],
                y=y_max * (0.95 - (i % 2) * 0.15),
                text=event['label'],
                showarrow=False,
                font=dict(color=event['color'], size=9),
                xanchor='left',
                xshift=5
            )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────

st.sidebar.title("London Crime &\nthe Cost of Living")
st.sidebar.markdown("*2023–2025*")
section = st.sidebar.radio("Navigate", [
    "The Story",
    "Economic Crime",
    "Crime & Deprivation",
    "Policing Response",
    "Where is London Headed?"
])

st.sidebar.divider()
st.sidebar.caption("""
**Data sources**
- Metropolitan & City of London Police via police.uk
- ONS LSOA Population Estimates 2022
- Index of Multiple Deprivation 2019
- 2021 Census

**Methodology note**
Recorded crime reflects both actual crime levels and policing 
activity. An increase in recorded offences may indicate more 
enforcement rather than more crime. All crime rates are 
normalised per 1,000 residents.
""")

# ═══════════════════════════════════════════════════════════════════
# 1. THE STORY
# ═══════════════════════════════════════════════════════════════════

if section == "The Story":
    st.title("London Crime & the Cost of Living")
    st.markdown("""
    Between 2023 and 2025, London's recorded crime data told a story closely 
    linked to one of the most difficult economic periods in a generation. As 
    household finances were squeezed by inflation, rising energy bills, and 
    stagnant wages, certain crimes surged in ways that are hard to attribute 
    to anything other than economic pressure.

    This dashboard explores what three years of Metropolitan Police data reveals 
    about crime in the cost of living era — which crimes rose, which fell, which 
    areas were hit hardest, and what the data suggests about where London is headed.
    """)

    street = load_full_street()

    # ── Headline metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total crimes recorded", "3,416,295", "Jan 2023 – Dec 2025")
    col2.metric("Shoplifting", "+53%", "vs Jan 2023 baseline")
    col3.metric("Drug offences", "+50%", "vs Jan 2023 baseline")
    col4.metric("Vehicle crime", "-18%", "vs Jan 2023 baseline")

    st.divider()

    # ── Overview chart ──
    st.subheader("Which crimes rose and which fell?")
    st.markdown(
        "Not all crime increased. The picture is more nuanced — crimes driven by economic pressure surged, while others fell.")

    all_trends = street.groupby(['crime_type', 'year']).size().reset_index(name='count')
    pivot = all_trends.pivot(index='crime_type', columns='year', values='count')
    pivot['change'] = ((pivot[2025] - pivot[2023]) / pivot[2023] * 100).round(1)
    pivot = pivot.reset_index().sort_values('change', ascending=True)

    fig = px.bar(
        pivot, x='change', y='crime_type',
        orientation='h',
        color='change',
        color_continuous_scale='RdYlGn_r',
        labels={'change': '% change 2023–2025', 'crime_type': ''},
    )
    fig.add_vline(x=0, line_color='white', opacity=0.3)
    fig.update_layout(height=500, coloraxis_showscale=False,
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **The pattern:** Crimes of economic necessity — shoplifting, drug offences, 
    theft from the person — all rose sharply. Crimes less connected to financial 
    pressure — vehicle crime, burglary, bicycle theft — fell. This divergence is 
    consistent with the economic hardship hypothesis explored throughout this dashboard.
    """)

    st.divider()

    # ── Economic timeline ──
    st.subheader("The economic context")
    st.markdown("Key events in the UK cost of living crisis, mapped against London's overall monthly crime count.")

    monthly_all = street.groupby('month').size().reset_index(name='count')
    fig2 = px.line(monthly_all, x='month', y='count',
                   color_discrete_sequence=['#95a5a6'],
                   labels={'count': 'Monthly crimes recorded', 'month': ''})
    fig2 = add_economic_annotations(fig2, monthly_all['count'].max())
    fig2.update_layout(height=450, plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Source: Metropolitan Police & City of London Police via police.uk | 2023–2025")

# ═══════════════════════════════════════════════════════════════════
# 2. ECONOMIC CRIME
# ═══════════════════════════════════════════════════════════════════
elif section == "Economic Crime":
    st.title("Economic Crime")
    st.markdown("""
    As household finances were squeezed from 2023 onwards, certain crimes 
    responded in ways the data makes clear. This section examines which crimes 
    rose, why, and what the data suggests about where things are headed.
    """)


    # ── Load pre-computed data ──
    @st.cache_data
    def load_economic_crime_data():
        indexed = pd.read_csv('data/processed/crime_indexed.csv')
        indexed['month'] = pd.to_datetime(indexed['month'])
        corr_df = pd.read_csv('data/processed/food_inflation_correlations.csv')
        lag_df = pd.read_csv('data/processed/shoplifting_lag_correlations.csv')
        decomp = pd.read_csv('data/processed/shoplifting_decomposition.csv')
        decomp['month'] = pd.to_datetime(decomp['month'])
        changepoint = pd.read_csv('data/processed/drugs_changepoint.csv')
        borough = pd.read_csv('data/processed/borough_shoplifting_trend.csv')
        food = pd.read_csv('data/processed/food_inflation_ons.csv')
        food['month'] = pd.to_datetime(food['month'])
        scenarios = pd.read_csv('data/processed/shoplifting_scenarios.csv')
        scenarios['month'] = pd.to_datetime(scenarios['month'])
        return (indexed, corr_df, lag_df, decomp,
                changepoint, borough, food, scenarios)


    (indexed, corr_df, lag_df, decomp,
     changepoint, borough, food, scenarios) = load_economic_crime_data()

    street = load_full_street()

    # ── Derive narrative values ──
    corr_shop = corr_df[corr_df['crime_type'] == 'Shoplifting'].iloc[0]
    best_lag = lag_df.loc[lag_df['r'].abs().idxmax()]
    cp_date = pd.to_datetime(changepoint['change_point_date'].values[0])
    before_mean = float(changepoint['mean_before'].values[0])
    after_mean = float(changepoint['mean_after'].values[0])
    pct_increase = round((after_mean - before_mean) / before_mean * 100, 1)
    decomp_clean = decomp.dropna(subset=['trend'])
    trend_vals = decomp_clean['trend']
    trend_increase = round(
        (trend_vals.max() - trend_vals.min()) / trend_vals.min() * 100, 1
    )
    top_borough = borough.nlargest(1, 'change_pct').iloc[0]
    bot_borough = borough.nsmallest(1, 'change_pct').iloc[0]

    CHART_CONFIG = {'displayModeBar': False,
                    'scrollZoom': False,
                    'staticPlot': False,
                    'doubleClick': False}

    # ── Headline metrics ──
    st.caption("All figures based on 2025 full-year averages from Metropolitan "
               "and City of London Police recorded crime data.")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Shoplifting — 2025 monthly average",
        "7,344",
        "Up from 4,786/month in 2023"
    )
    col2.metric(
        "Drug offences — new monthly average",
        f"{after_mean:.0f}",
        f"Was {before_mean:.0f} before {cp_date.strftime('%b %Y')}"
    )
    col3.metric(
        "Shoplifting underlying trend increase",
        f"+{trend_increase}%",
        "Seasonal patterns removed"
    )

    st.divider()

    # ── Finding 1: % change chart ──
    st.subheader("1. Which crimes went up — and which went down?")
    st.markdown("""
    The chart below shows how each crime type changed relative to January 2023. 
    A value of +50 means 50% more incidents than at the start of the period. 
    Zero means no change. Negative means it fell.
    """)

    economic_highlight = ['Shoplifting', 'Theft From The Person', 'Drugs',
                          'Vehicle Crime', 'Burglary']
    colors = {
        'Shoplifting': '#e74c3c',
        'Theft From The Person': '#e67e22',
        'Drugs': '#9b59b6',
        'Vehicle Crime': '#95a5a6',
        'Burglary': '#7f8c8d'
    }

    fig1 = go.Figure()
    for crime in economic_highlight:
        subset = indexed[indexed['crime_type'] == crime].copy()
        if subset.empty:
            continue
        subset['pct_change'] = subset['index_value'] - 100
        fig1.add_trace(go.Scatter(
            x=subset['month'],
            y=subset['pct_change'],
            name=crime,
            line=dict(
                color=colors.get(crime, '#95a5a6'),
                width=3 if crime in ['Shoplifting', 'Drugs'] else 1.5,
                dash='solid' if crime in [
                    'Shoplifting', 'Drugs', 'Theft From The Person'
                ] else 'dot'
            ),
            hoverinfo='skip'
        ))

    fig1.add_hline(y=0, line_dash='dash', line_color='white',
                   opacity=0.3, annotation_text='No change from Jan 2023')
    fig1 = add_economic_annotations(
        fig1, indexed['index_value'].max() - 100, show_all=False
    )
    fig1.update_layout(
        height=430,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title='% change since January 2023',
        yaxis_ticksuffix='%',
        hovermode=False,
        dragmode=False,
        xaxis=dict(hoverformat=' ', showticklabels=False, showspikes=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig1, use_container_width=True, config=CHART_CONFIG)

    st.info("""
    **The pattern is striking:** Shoplifting and theft rose sharply and stayed 
    high — these are crimes most directly linked to financial pressure. Vehicle 
    crime and burglary both fell consistently. Drug offences were flat for 18 
    months then jumped suddenly — a completely different shape suggesting a 
    policing change rather than an economic one.
    """)

    st.divider()

    # ── Finding 2: Shoplifting vs food inflation ──
    st.subheader("2. Shoplifting kept rising even as food prices stopped surging")
    st.markdown("""
    You might expect shoplifting to track food price rises — go up when prices 
    spike, fall when they ease. The data tells a different story.
    """)

    monthly_shop = street[street['crime_type'] == 'Shoplifting'] \
        .groupby('month').size().reset_index(name='count')
    monthly_shop['month'] = pd.to_datetime(monthly_shop['month'])
    merged = monthly_shop.merge(food, on='month', how='inner')

    inflection = merged[merged['food_inflation'] < 5].iloc[0]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=merged['month'],
        y=merged['count'],
        name='Monthly shoplifting incidents',
        marker_color='rgba(231, 76, 60, 0.6)',
        yaxis='y1',
        hovertemplate='%{x|%b %Y}: %{y:,} incidents<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=merged['month'],
        y=merged['food_inflation'],
        name='Food price inflation (%)',
        line=dict(color='#f39c12', width=3),
        yaxis='y2',
        hovertemplate='%{x|%b %Y}: %{y:.1f}% food inflation<extra></extra>'
    ))
    fig2.add_vline(
        x=inflection['month'],
        line_dash='dot',
        line_color='#f39c12',
        line_width=2,
        opacity=0.9
    )
    fig2.add_annotation(
        x=inflection['month'],
        y=1.08,
        yref='paper',
        text="Food inflation falls below 5%<br>Shoplifting continues rising",
        showarrow=False,
        font=dict(color='#f39c12', size=11),
        xanchor='left',
        xshift=8,
        bgcolor='rgba(0,0,0,0.5)'
    )
    fig2.update_layout(
        yaxis=dict(
            title=dict(text='Monthly shoplifting incidents',
                       font=dict(color='#e74c3c')),
        ),
        yaxis2=dict(
            title=dict(text='Food price inflation (%)',
                       font=dict(color='#f39c12')),
            overlaying='y',
            side='right',
        ),
        xaxis=dict(
            showticklabels=False
        ),
        height=420,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified',
        dragmode=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Why didn't shoplifting fall when prices eased?**")
        st.markdown(f"""
        Food prices rose 38.6% in total between 2020 and 2025. Even when the 
        rate of increase slowed, prices themselves stayed permanently higher — 
        and households were still deep in deficit from years of wages failing 
        to keep up.

        Analysis of the data shows shoplifting responds to food price changes 
        with roughly a **{int(best_lag['lag_months'])} month delay** — the 
        financial damage takes several months to feed through into crime. By 
        the time inflation eased in mid-2024, millions of households had already 
        exhausted their savings and credit.
        """)
    with col2:
        st.markdown("**What does rising food inflation in 2025 mean?**")
        st.markdown("""
        Food inflation has been creeping back up since late 2024, sitting above 
        4% through the end of 2025. If the same delay pattern holds, this 
        suggests continued upward pressure on shoplifting into 2026.

        The Joseph Rowntree Foundation's winter 2025 tracker found 61% of UK 
        households reported their cost of living was still increasing — 
        suggesting the structural damage from the crisis is far from resolved.
        """)

    st.divider()

    # ── Finding 3: Trend line ──
    st.subheader("3. The underlying trend — stripping out seasonal noise")
    st.markdown("""
    Shoplifting naturally rises in summer and dips in winter — high streets are 
    busier, more people are out, and opportunistic theft increases. To see 
    whether the overall increase is real or just a seasonal pattern repeating, 
    we can mathematically remove those predictable swings.

    The chart below shows the raw monthly figures in grey, and the true 
    underlying trend in red — with seasonal fluctuations removed.
    """)

    decomp_trimmed = decomp_clean[
        decomp_clean['month'] >= '2023-07-01'
        ].copy()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=decomp_trimmed['month'],
        y=decomp_trimmed['observed'],
        name='Actual monthly count',
        line=dict(color='rgba(149,165,166,0.4)', width=1),
        fill='tozeroy',
        fillcolor='rgba(149,165,166,0.05)',
        hovertemplate='%{y:,.0f} actual<extra></extra>'
    ))
    fig3.add_trace(go.Scatter(
        x=decomp_trimmed['month'],
        y=decomp_trimmed['trend'],
        name='Underlying trend',
        line=dict(color='#e74c3c', width=3),
        hovertemplate='%{y:,.0f} trend<extra></extra>'
    ))
    fig3.update_layout(
        height=360,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title='Monthly shoplifting incidents',
        xaxis=dict(showticklabels=False),
        hovermode='x unified',
        dragmode=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig3, use_container_width=True, config=CHART_CONFIG)

    st.info(f"""
    **The underlying trend rose {trend_increase}% over this period** — this is 
    the genuine structural increase, entirely separate from seasonal patterns. 
    The trend shows no sign of reversing even as inflation eased in 2024, 
    reinforcing the idea that cumulative financial damage — not just the acute 
    price shock — is driving the increase.
    """)

    st.divider()

    # ── Finding 4: Drug change point ──
    st.subheader("4. Drug offences: a sudden switch, not a gradual rise")
    st.markdown(f"""
    Unlike shoplifting, drug offences didn't creep upward — they were broadly 
    flat for 18 months, then jumped sharply in **{cp_date.strftime('%B %Y')}** 
    and have stayed at that higher level ever since.
    """)

    monthly_drugs = street[street['crime_type'] == 'Drugs'] \
        .groupby('month').size().reset_index(name='count')
    monthly_drugs['month'] = pd.to_datetime(monthly_drugs['month'])

    fig4 = go.Figure()
    fig4.add_vrect(
        x0=monthly_drugs['month'].min(), x1=cp_date,
        fillcolor='rgba(149,165,166,0.08)', layer='below',
        line_width=0,
        annotation_text=f"Average: {before_mean:.0f}/month",
        annotation_position='top left',
        annotation_font=dict(color='#95a5a6', size=11)
    )
    fig4.add_vrect(
        x0=cp_date, x1=monthly_drugs['month'].max(),
        fillcolor='rgba(155,89,182,0.08)', layer='below',
        line_width=0,
        annotation_text=f"Average: {after_mean:.0f}/month",
        annotation_position='top right',
        annotation_font=dict(color='#9b59b6', size=11)
    )
    fig4.add_trace(go.Scatter(
        x=monthly_drugs['month'],
        y=monthly_drugs['count'],
        line=dict(color='#9b59b6', width=2.5),
        showlegend=False,
        hovertemplate='%{x|%b %Y}<br>%{y:,} offences<extra></extra>'
    ))
    fig4.add_vline(
        x=cp_date, line_dash='dash',
        line_color='#e74c3c', opacity=0.9, line_width=2
    )
    fig4.add_annotation(
        x=cp_date,
        y=monthly_drugs['count'].max() * 0.92,
        text=f"<b>{cp_date.strftime('%B %Y')}</b><br>Structural shift detected",
        showarrow=False,
        font=dict(color='#e74c3c', size=11),
        xanchor='left',
        xshift=10,
        bgcolor='rgba(0,0,0,0.5)'
    )
    fig4.update_layout(
    height=360,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    yaxis_title='Monthly drug offences recorded',
    xaxis=dict(showticklabels=False),
    hovermode='x unified',
    dragmode=False,
    showlegend=False
)
    st.plotly_chart(fig4, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{before_mean:.0f} → {after_mean:.0f} per month (+{pct_increase}%)**")
        st.markdown(f"""
        Labour took office in July 2024 — one month before this shift. The 
        Metropolitan Police's Drugs Action Plan and Operation Yamata 
        specifically target organised drug supply networks across London.

        A sudden, sustained jump like this — rather than a gradual rise — 
        strongly suggests a deliberate operational decision rather than a 
        change in the underlying level of drug activity.
        """)
    with col2:
        st.markdown("**Does more recorded drug crime mean more drugs?**")
        st.markdown("""
        Not necessarily. Recorded drug offences increase when police actively 
        look for them — more stop and searches, targeted operations, more 
        officers in known hotspots. This is an important caveat.

        What we can say confidently is that enforcement activity increased 
        sharply and has been sustained. Whether that reflects genuinely more 
        drug activity in London, or more policing of existing activity, 
        the data alone cannot tell us.
        """)

    st.divider()

    # ── Finding 5: Borough breakdown ──
    st.subheader("5. Which areas saw shoplifting rise most?")
    st.markdown("""
    The increase wasn't spread evenly. Below are the five boroughs with the 
    biggest increases and the five with the smallest.
    """)

    st.caption("""
    **Understanding deprivation decile:** Areas are ranked 1–10 by deprivation. 
    1 = most deprived 10% of areas in England. 10 = least deprived. 
    Lower number means higher unemployment, lower income, poorer housing.
    """)

    top5 = borough.nlargest(5, 'change_pct')
    bot5 = borough.nsmallest(5, 'change_pct')
    display_boroughs = pd.concat([top5, bot5]).sort_values('change_pct').copy()
    display_boroughs['deprivation_label'] = display_boroughs['avg_imd_decile'].apply(
        lambda x: f"{x:.1f} — {'More deprived' if x <= 3 else 'Average' if x <= 7 else 'Less deprived'}"
    )

    fig5 = px.bar(
        display_boroughs,
        x='change_pct',
        y='borough',
        orientation='h',
        color='change_pct',
        color_continuous_scale='RdYlGn_r',
        custom_data=['deprivation_label', 'count_2023', 'count_2025']
    )
    fig5.update_traces(
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Change: %{x:+.1f}%<br>'
            'Deprivation: %{customdata[0]}'
            '<extra></extra>'
        )
    )
    fig5.add_vline(x=0, line_color='white', opacity=0.4)
    fig5.update_layout(
        height=420,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title='% change in shoplifting 2023–2025',
        yaxis_title='',
        hovermode='y',
        dragmode=False,
        xaxis=dict(hoverformat=' ', showticklabels=False, showspikes=False)
    )
    st.plotly_chart(fig5, use_container_width=True, config=CHART_CONFIG)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biggest increases**")
        st.dataframe(
            top5[['borough', 'change_pct', 'avg_imd_decile',
                  'count_2023', 'count_2025']]
            .rename(columns={
                'borough': 'Borough',
                'change_pct': '% change',
                'avg_imd_decile': 'Deprivation decile',
                'count_2023': '2023 incidents',
                'count_2025': '2025 incidents'
            }).round(1),
            hide_index=True
        )
    with col2:
        st.markdown("**Smallest increases / decreases**")
        st.dataframe(
            bot5[['borough', 'change_pct', 'avg_imd_decile',
                  'count_2023', 'count_2025']]
            .rename(columns={
                'borough': 'Borough',
                'change_pct': '% change',
                'avg_imd_decile': 'Deprivation decile',
                'count_2023': '2023 incidents',
                'count_2025': '2025 incidents'
            }).round(1),
            hide_index=True
        )

    st.divider()

    # ── Finding 6: Scenario analysis ──
    st.subheader("6. What could 2026 look like? Three scenarios")
    st.markdown("""
    Rather than a single prediction, the data supports three plausible 
    trajectories depending on how economic and policy conditions develop.
    """)

    historical = street[street['crime_type'] == 'Shoplifting'] \
        .groupby('month').size().reset_index(name='count')
    historical['month'] = pd.to_datetime(historical['month'])

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=historical['month'],
        y=historical['count'],
        name='Recorded 2023–2025',
        line=dict(color='#e74c3c', width=2.5),
        hovertemplate='%{x|%b %Y}<br>%{y:,} incidents<extra></extra>'
    ))
    fig6.add_trace(go.Scatter(
        x=scenarios['month'],
        y=scenarios['pessimistic'],
        name='Pessimistic',
        line=dict(color='#e74c3c', width=1.5, dash='dot'),
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} projected<extra></extra>'
    ))
    fig6.add_trace(go.Scatter(
        x=scenarios['month'],
        y=scenarios['optimistic'],
        name='Optimistic',
        line=dict(color='#2ecc71', width=1.5, dash='dot'),
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} projected<extra></extra>'
    ))
    fig6.add_trace(go.Scatter(
        x=scenarios['month'],
        y=scenarios['central'],
        name='Central',
        line=dict(color='#f39c12', width=2, dash='dash'),
        hovertemplate='%{x|%b %Y}<br>%{y:,.0f} projected<extra></extra>'
    ))
    fig6.add_trace(go.Scatter(
        x=pd.concat([scenarios['month'], scenarios['month'][::-1]]),
        y=pd.concat([scenarios['pessimistic'],
                     scenarios['optimistic'][::-1]]),
        fill='toself',
        fillcolor='rgba(231,76,60,0.08)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig6.add_vline(
        x='2025-12-01', line_dash='dot',
        line_color='white', opacity=0.4
    )
    fig6.add_annotation(
        x='2025-12-01', y=1.05, yref='paper',
        text='2026 scenarios →',
        showarrow=False,
        font=dict(color='white', size=10),
        xanchor='left', xshift=5
    )
    fig6.update_layout(
        height=420,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_title='Monthly shoplifting incidents',
        xaxis=dict(showticklabels=False),
        hovermode='x unified',
        dragmode=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig6, use_container_width=True, config=CHART_CONFIG)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **Optimistic — ~6,000/month by Dec 2026**")
        st.markdown("""
        Food inflation falls back toward 2%. Labour's Crime and Policing Bill 
        passes and new retail crime powers come into effect. Wage growth 
        continues to outpace inflation, gradually rebuilding household finances. 
        Neighbourhood policing guarantee delivers visible presence in high streets.
        """)
    with col2:
        st.markdown("🟡 **Central — ~7,300/month through 2026**")
        st.markdown("""
        Current conditions persist. Food inflation stays in the 3–5% range, 
        household finances remain stretched but stable. Policing bill passes 
        but implementation takes time. Shoplifting plateaus at its current 
        elevated level — high, but no longer rising.
        """)
    with col3:
        st.markdown("🔴 **Pessimistic — ~8,500/month by Dec 2026**")
        st.markdown("""
        Food inflation continues rising — already back above 4% in late 2025. 
        "Awful April" 2026 brings further bill increases. Retail crime bill 
        delayed in Parliament. The five month delay pattern means pressure 
        building now feeds into crime by spring 2026.
        """)

    st.caption("""
    Source: Metropolitan Police & City of London Police via police.uk | 
    Food inflation: ONS CPI series D7G8 | 
    Trend: Additive seasonal decomposition | 
    Scenarios based on economic and policy assumptions — not statistical forecasts
    """)

# ═══════════════════════════════════════════════════════════════════
# 3. CRIME & DEPRIVATION
# ═══════════════════════════════════════════════════════════════════
elif section == "Crime & Deprivation":
    st.title("Crime & Deprivation")
    st.markdown("""
    Does deprivation predict crime in London? The answer is more nuanced than 
    a simple yes — it depends entirely on which crime you're looking at, and 
    which aspect of deprivation matters most.
    """)


    # ── Load data ──
    @st.cache_data
    def load_deprivation_data():
        lsoa = pd.read_csv('data/processed/lsoa_deprivation_crime.csv')
        domain_corr = pd.read_csv('data/processed/domain_crime_correlations.csv')
        return lsoa, domain_corr


    lsoa, domain_corr = load_deprivation_data()

    # ── Derive narrative values ──
    total_lsoas = len(lsoa)
    deprived_high = lsoa[lsoa['outlier_type'] == 'Deprived and high crime']
    affluent_high = lsoa[lsoa['outlier_type'] == 'Affluent but high crime']
    deprived_low = lsoa[lsoa['outlier_type'] == 'Deprived but low crime']

    top_affluent_borough = affluent_high['borough'].value_counts().index[0]
    top_affluent_count = affluent_high['borough'].value_counts().iloc[0]

    shop_corrs = domain_corr[domain_corr['crime_type'] == 'Shoplifting']
    shop_max_corr = shop_corrs.loc[shop_corrs['correlation'].abs().idxmax()]
    shop_income_corr = shop_corrs[
        shop_corrs['deprivation_domain'] == 'Income'
        ]['correlation'].values[0]

    strongest = domain_corr[domain_corr['significant']] \
        .sort_values('correlation', ascending=False).iloc[0]

    CHART_CONFIG = {'displayModeBar': False, 'scrollZoom': False}

    # ── Headline metrics ──
    st.caption("Analysis based on 4,653 Lower Super Output Areas (LSOAs) — "
               "small geographic units of ~1,500 residents each.")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "LSOAs analysed",
        f"{total_lsoas:,}",
        "Covering all 33 London boroughs"
    )
    col2.metric(
        "Deprived areas with surprisingly high crime",
        f"{len(deprived_high)}",
        f"{len(deprived_high) / total_lsoas * 100:.1f}% of all LSOAs"
    )
    col3.metric(
        "Affluent areas with surprisingly high crime",
        f"{len(affluent_high)}",
        f"Mostly {top_affluent_borough} & tourist zones"
    )

    st.divider()

    # ── Finding 1: Borough scatter plot ──
    st.subheader("1. Where deprivation and crime don't follow the rules")
    st.markdown("""
    Each bubble represents one London borough. The further right, the more 
    deprived. The higher up, the more crime per resident. Boroughs that sit 
    well above or below the expected line are the interesting ones.
    """)


    @st.cache_data
    def load_borough_deprivation():
        return pd.read_csv('data/processed/borough_outliers_deprivation.csv')


    borough_dep = load_borough_deprivation()

    color_map = {
        'As expected': '#4a4a4a',
        'Deprived and high crime': '#e74c3c',
        'Affluent but high crime': '#e67e22',
        'Deprived but low crime': '#2ecc71'
    }

    fig1 = go.Figure()
    for group in ['As expected', 'Deprived and high crime',
                  'Affluent but high crime', 'Deprived but low crime']:
        subset = borough_dep[borough_dep['dominant_outlier'] == group]
        if subset.empty:
            continue
        fig1.add_trace(go.Scatter(
            x=subset['avg_imd_score'],
            y=subset['avg_crime_rate'],
            mode='markers+text',
            name=group,
            text=subset['borough'],
            textposition='top center',
            textfont=dict(size=9, color='rgba(255,255,255,0.7)'),
            marker=dict(
                color=color_map[group],
                size=subset['total_lsoas'] / 8,
                opacity=0.8,
                line=dict(color='white', width=0.5)
            ),
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Avg deprivation score: %{x:.1f}<br>'
                'Avg crime rate: %{y:.0f} per 1,000<br>'
                'Outlier LSOAs: %{customdata[0]} (%{customdata[1]:.1f}%)'
                '<extra></extra>'
            ),
            customdata=subset[['pct_outlier', 'pct_outlier']].values
        ))

    fig1.update_layout(
        height=520,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='Average deprivation score (higher = more deprived)',
            showspikes=False,
            gridcolor='rgba(255,255,255,0.05)'
        ),
        yaxis=dict(
            title='Average crime rate per 1,000 residents',
            showspikes=False,
            gridcolor='rgba(255,255,255,0.05)'
        ),
        hovermode='closest',
        dragmode=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    st.plotly_chart(fig1, use_container_width=True, config=CHART_CONFIG)

    top_affluent_borough = borough_dep.sort_values(
        'affluent_high', ascending=False
    ).iloc[0]
    top_deprived_borough = borough_dep.sort_values(
        'avg_residual', ascending=False
    ).iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**🟠 {top_affluent_borough['borough']} — affluent but very high crime**")
        st.markdown(f"""
        {top_affluent_borough['borough']} has an average deprivation score of 
        {top_affluent_borough['avg_imd_score']:.1f} — relatively low — yet records 
        {top_affluent_borough['avg_crime_rate']:.0f} crimes per 1,000 residents, 
        far above what its deprivation level would predict. 

        {top_affluent_borough['affluent_high']} of its LSOAs are flagged as 
        affluent but high crime outliers. These are tourist and retail zones where 
        crime follows footfall — millions of visitors create high-volume theft and 
        anti-social behaviour regardless of local poverty levels.
        """)
    with col2:
        st.markdown(f"**🔴 {top_deprived_borough['borough']} — deprived and high crime**")
        st.markdown(f"""
        {top_deprived_borough['borough']} sits at the other extreme — high 
        deprivation score of {top_deprived_borough['avg_imd_score']:.1f} combined 
        with {top_deprived_borough['avg_crime_rate']:.0f} crimes per 1,000 
        residents. 

        {top_deprived_borough['deprived_high']} of its LSOAs are flagged as 
        deprived and high crime — areas where poverty and crime reinforce each 
        other, and where the cost of living crisis has hit hardest.
        """)

    st.divider()

        # ── Finding 2: Domain-crime heatmap ──
    st.subheader("2. Which type of deprivation drives which crime?")
    st.markdown("""
    Overall deprivation masks important differences between its components. 
    Income deprivation, poor housing conditions, barriers to services, and 
    health deprivation all affect crime — but in very different ways depending 
    on the crime type.
    """)

    heatmap_pivot = domain_corr.pivot(
        index='deprivation_domain',
        columns='crime_type',
        values='correlation'
    )

    # Mask non-significant values
    sig_pivot = domain_corr.pivot(
        index='deprivation_domain',
        columns='crime_type',
        values='significant'
    )

    fig2 = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale='RdYlGn',
        zmid=0,
        zmin=-0.2,
        zmax=0.5,
        text=heatmap_pivot.round(2).values,
        texttemplate='%{text}',
        hovertemplate=(
            '<b>%{y}</b> → <b>%{x}</b><br>'
            'Correlation: %{z:.3f}'
            '<extra></extra>'
        ),
        showscale=True,
        colorbar=dict(
            title='Correlation',
            tickvals=[-0.2, 0, 0.2, 0.4],
            ticktext=['−0.2', '0', '0.2', '0.4']
        )
    ))

    fig2.update_layout(
        height=380,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='',
            side='bottom',
            tickangle=-20,
            showspikes=False
        ),
        yaxis=dict(
            title='Deprivation domain',
            showspikes=False
        ),
        hovermode='closest',
        dragmode=False,
        margin=dict(l=140, b=100)
    )
    st.plotly_chart(fig2, use_container_width=True, config=CHART_CONFIG)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🏚️ Living environment drives burglary and robbery**")
        living_burg = domain_corr[
            (domain_corr['crime_type'] == 'Burglary') &
            (domain_corr['deprivation_domain'] == 'Living Env')
            ]['correlation'].values[0]
        st.markdown(f"""
        The strongest relationship in the entire dataset is between living 
        environment deprivation and burglary (r={living_burg:.2f}). Poor street 
        lighting, run-down housing, lack of natural surveillance — these 
        physical conditions create opportunity regardless of income levels. 
        Robbery shows an almost identical pattern.
        """)
    with col2:
        st.markdown("**💷 Income and employment drive violence and drugs**")
        income_violence = domain_corr[
            (domain_corr['crime_type'] == 'Violence And Sexual Offences') &
            (domain_corr['deprivation_domain'] == 'Income')
            ]['correlation'].values[0]
        st.markdown(f"""
        Violence and drug offences are most strongly predicted by income and 
        employment deprivation (r={income_violence:.2f}). These are the crimes 
        most directly linked to financial desperation and the street economy — 
        areas with higher unemployment and lower incomes see significantly more 
        of both.
        """)
    with col3:
        st.markdown("**🛒 Shoplifting is predicted by almost nothing**")
        st.markdown(f"""
        Shoplifting shows near-zero correlation with income (r={shop_income_corr:.3f}), 
        employment, health, and barriers to services. The only weak signal comes 
        from living environment (r={shop_max_corr['correlation']:.2f}).

        This reinforces the economic crime analysis: shoplifting is no longer 
        concentrated in deprived areas. It has become a cross-demographic 
        response to the cost of living crisis, happening in wealthy high streets 
        as much as deprived ones.
        """)

    st.divider()

    # ── Finding 3: Borough map ──
    st.subheader("3. Where are the outlier boroughs?")
    st.markdown("""
        The map shows each borough coloured by whether it has more deprived-and-high-crime 
        or affluent-and-high-crime outlier areas. Size reflects the proportion of 
        outlier LSOAs within the borough.
        """)

    map_data = borough_dep[borough_dep['dominant_outlier'] != 'As expected'].copy()
    map_data['size'] = (map_data['pct_outlier'] * 3).clip(lower=8)

    fig3 = px.scatter_mapbox(
        map_data,
        lat='latitude',
        lon='longitude',
        color='dominant_outlier',
        color_discrete_map={
            'Deprived and high crime': '#e74c3c',
            'Affluent but high crime': '#e67e22',
            'Deprived but low crime': '#2ecc71'
        },
        size='size',
        size_max=30,
        hover_name='borough',
        hover_data={
            'avg_imd_score': ':.1f',
            'avg_crime_rate': ':.0f',
            'pct_outlier': ':.1f',
            'latitude': False,
            'longitude': False,
            'dominant_outlier': False,
            'size': False
        },
        labels={
            'avg_imd_score': 'Avg deprivation score',
            'avg_crime_rate': 'Avg crime rate per 1,000',
            'pct_outlier': '% outlier LSOAs'
        },
        zoom=9,
        center=dict(lat=51.509, lon=-0.118),
        mapbox_style='carto-darkmatter',
        height=500
    )
    fig3.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode=False,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            title=''
        )
    )
    st.plotly_chart(fig3, use_container_width=True, config=CHART_CONFIG)



    st.caption("""
    Source: Metropolitan Police & City of London Police via police.uk | 
    Deprivation: Ministry of Housing, Communities & Local Government, 
    English Indices of Deprivation 2019 | 
    Outliers defined as areas with residuals >1.5 standard deviations 
    from the deprivation-crime regression line
    """)

# ═══════════════════════════════════════════════════════════════════
# 4. POLICING RESPONSE
# ═══════════════════════════════════════════════════════════════════

elif section == "Policing Response":
    st.title("The Policing Response")
    st.markdown("""
    How did policing adapt to the cost of living era? Stop and search data 
    reveals patterns in enforcement activity — who is being stopped, for what, 
    and whether those stops are equally justified across different groups.
    """)

    ss = load_stop_search()

    # ── Overall stats ──
    total = len(ss)
    arrests = ss['outcome'].str.contains('Arrest', na=False).sum()
    hit_rate = arrests / total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total stop and searches", f"{total:,}")
    col2.metric("Resulted in arrest", f"{arrests:,}")
    col3.metric("Overall arrest rate", f"{hit_rate:.1f}%")

    st.divider()

    # ── Finding 1: Drug stops ──
    st.subheader("Finding 1 — Drug stops of Asian people are the least productive")
    st.markdown("""
    68% of stops of Asian people are for drugs — the highest of any group. 
    Yet these stops result in arrest only 11.3% of the time, the lowest rate. 
    By contrast, weapon stops show broadly similar arrest rates across all ethnicities 
    (18–21%), suggesting those stops are more equally founded.
    """)

    drug_stops = ss[ss['object_of_search'] == 'Controlled drugs']
    total_by_eth = ss.groupby('officer-defined_ethnicity').size()
    drug_by_eth = drug_stops.groupby('officer-defined_ethnicity').size()

    drug_stats = drug_stops.groupby('officer-defined_ethnicity').apply(
        lambda x: pd.Series({
            'arrest_rate': round(x['outcome'].str.contains('Arrest', na=False).sum() / len(x) * 100, 1),
        }), include_groups=False
    ).reset_index()
    drug_stats['pct_stops_for_drugs'] = (drug_by_eth / total_by_eth * 100).round(1).values
    drug_stats = drug_stats[drug_stats['officer-defined_ethnicity'].isin(['Asian', 'Black', 'White', 'Other'])]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(drug_stats, x='officer-defined_ethnicity', y='pct_stops_for_drugs',
                     color='officer-defined_ethnicity',
                     labels={'pct_stops_for_drugs': '% of stops that are drug stops',
                             'officer-defined_ethnicity': ''},
                     title='How often is each group stopped for drugs?',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, height=320,
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(drug_stats, x='officer-defined_ethnicity', y='arrest_rate',
                      color='officer-defined_ethnicity',
                      labels={'arrest_rate': 'Arrest rate (%)',
                              'officer-defined_ethnicity': ''},
                      title='How often do those drug stops result in arrest?',
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(showlegend=False, height=320,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Finding 2: Trend ──
    st.subheader("Finding 2 — The arrest rate gap by ethnicity is widening")

    trend = ss.groupby(['year', 'officer-defined_ethnicity']).apply(
        lambda x: round(x['outcome'].str.contains('Arrest', na=False).sum() / len(x) * 100, 1),
        include_groups=False
    ).reset_index()
    trend.columns = ['year', 'ethnicity', 'arrest_rate']
    trend = trend[trend['ethnicity'].isin(['Asian', 'Black', 'White'])]

    fig3 = px.line(trend, x='year', y='arrest_rate', color='ethnicity',
                   markers=True,
                   labels={'arrest_rate': 'Arrest rate (%)', 'year': '', 'ethnicity': ''},
                   title='Arrest rate by ethnicity 2023–2025',
                   color_discrete_map={'Asian': '#e74c3c', 'Black': '#3498db', 'White': '#2ecc71'})
    fig3.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    White arrest rates have risen consistently (15.9% → 19.1%) while Asian rates 
    plateaued after 2023 (12.4% → 15.1% → 15.1%). The gap between White and Asian 
    arrest rates grew from 3.5 to 4 percentage points between 2023 and 2025.
    """)

    st.divider()

    # ── Finding 3: Outcomes ──
    st.subheader("What happens when someone is stopped?")
    outcome_counts = ss['outcome'].value_counts().head(8).reset_index()
    outcome_counts.columns = ['outcome', 'count']
    fig4 = px.bar(outcome_counts, x='count', y='outcome', orientation='h',
                  color_discrete_sequence=['#3498db'],
                  labels={'count': 'Number of stops', 'outcome': ''})
    fig4.update_layout(height=380, plot_bgcolor='rgba(0,0,0,0)',
                       paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4, use_container_width=True)

    st.caption("Data: Metropolitan & City of London Police stop and search records, 2023–2025.")

# ═══════════════════════════════════════════════════════════════════
# 5. WHERE IS LONDON HEADED?
# ═══════════════════════════════════════════════════════════════════

elif section == "Where is London Headed?":
    st.title("Where is London Headed?")
    st.markdown("""
    A Random Forest model trained on deprivation indicators provides a data-driven 
    basis for thinking about where crime is likely to remain elevated — and which 
    areas may be most at risk if economic conditions deteriorate further.
    """)

    df = load_modelling_data()

    # ── Key finding ──
    st.subheader("Deprivation explains two thirds of crime variation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model R²", "0.659", "66% of variation explained")
    col2.metric("Key driver", "IMD Score", "Strongest predictor")
    col3.metric("LSOAs modelled", f"{len(df):,}", "London small areas")

    st.info("""
    **What this means:** Two thirds of the variation in local crime rates across 
    London can be predicted purely from deprivation indicators — income, employment, 
    health, education, and living environment scores. The remaining third reflects 
    factors not captured here, including tourist footfall, policing intensity, 
    and local geography.

    **The implication:** Areas with persistently high deprivation are structurally 
    predisposed to higher crime regardless of policing activity. Without improvement 
    in underlying economic conditions, crime in these areas is unlikely to fall 
    sustainably.
    """)

    st.divider()

    # ── Feature importance ──
    st.subheader("What drives crime rates most?")
    model = load_model()
    features = ['imd_score', 'income_score', 'employment_score',
                'education_score', 'health_score', 'barriers_score', 'living_env_score']
    feature_labels = {
        'imd_score': 'Overall deprivation (IMD)',
        'income_score': 'Income deprivation',
        'employment_score': 'Employment deprivation',
        'education_score': 'Education & skills',
        'health_score': 'Health deprivation',
        'barriers_score': 'Barriers to housing & services',
        'living_env_score': 'Living environment'
    }

    importance = pd.DataFrame({
        'feature': [feature_labels[f] for f in features],
        'importance': model.feature_importances_
    }).sort_values('importance')

    fig = px.bar(importance, x='importance', y='feature', orientation='h',
                 color='importance', color_continuous_scale='Blues',
                 labels={'importance': 'Relative importance', 'feature': ''})
    fig.update_layout(height=400, coloraxis_showscale=False,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Deprivation vs crime scatter ──
    st.subheader("Deprivation vs crime rate across London LSOAs")
    fig2 = px.scatter(
        df[df['crime_rate'] <= df['crime_rate'].quantile(0.99)],
        x='imd_score', y='crime_rate',
        color='imd_score', color_continuous_scale='Reds',
        opacity=0.4,
        labels={'crime_rate': 'Crime rate per 1,000 residents',
                'imd_score': 'Deprivation score (higher = more deprived)'},
        title='Each dot is one London LSOA'
    )
    fig2.update_layout(height=500, coloraxis_showscale=False,
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Forward outlook ──
    st.subheader("The outlook")
    st.markdown("""
    Based on what the data shows, here is a data-informed assessment of where 
    London crime is likely to go:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Reasons crime may continue at elevated levels**")
        st.markdown("""
        - Real household disposable incomes are not forecast to return to 
          2021 levels until 2027 at the earliest
        - "Awful April" 2025 saw bills rise again — water bills up 26%, 
          energy bills rising — suggesting financial pressure continues
        - Shoplifting has shown no sign of falling despite inflation easing, 
          suggesting structural rather than temporary change
        - The IMD data used in the model is from 2019 — deprivation in many 
          London areas has likely worsened since then
        """)
    with col2:
        st.markdown("**Reasons for cautious optimism**")
        st.markdown("""
        - Wage growth has outpaced inflation since mid-2023, gradually 
          rebuilding household finances
        - Vehicle crime, burglary, and bicycle theft all fell over the period, 
          suggesting policing improvements in some areas
        - Labour's neighbourhood policing guarantee — 13,000 additional 
          officers nationally — may increase deterrence
        - The partial recovery in theft from the person in 2025 suggests 
          some crimes are responsive to increased policing presence
        """)

    st.divider()
    st.markdown("""
    **Caveat:** This analysis uses recorded crime data and 2019 deprivation scores. 
    Recorded crime reflects enforcement activity as well as actual crime levels. 
    Predictions about future crime trends are inherently uncertain and this analysis 
    should be treated as indicative rather than definitive.
    """)

    st.divider()

    # ── Technical detail ──
    with st.expander("Technical detail — model methodology"):
        st.markdown("""
        **Model:** Random Forest Regressor (100 estimators, random_state=42)

        **Target variable:** Crime rate per 1,000 residents per LSOA, 
        log-transformed to address right skew. Extreme outliers (central London 
        areas with very small residential populations) capped at the 99th percentile.

        **Features:** Seven deprivation domain scores from the 2019 Index of 
        Multiple Deprivation — overall IMD score, income, employment, education 
        & skills, health & disability, barriers to housing & services, and living 
        environment.

        **Train/test split:** 80/20, stratified by random seed.

        **Performance:** R² = 0.659 on held-out test set. MAE = 0.952 on 
        log-transformed scale.

        **Limitations:** IMD data is from 2019. LSOA boundary changes between 
        2011 (IMD) and 2021 (population estimates) result in some LSOAs being 
        excluded from the joined dataset. The model captures structural deprivation 
        but not dynamic factors such as policing operations, local events, or 
        short-term economic shocks.
        """)