import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import requests

st.set_page_config(
    page_title="London Crime Analysis",
    page_icon="🔍",
    layout="wide"
)

# ── Data loading ──────────────────────────────────────────────────

@st.cache_data
def load_street():
    df = pd.read_csv('data/processed/street_clean.csv')
    df['month'] = pd.to_datetime(df['month'])
    return df

@st.cache_data
def load_modelling_data():
    return pd.read_csv('data/processed/modelling_data.csv')

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

# ── Sidebar ───────────────────────────────────────────────────────

st.sidebar.title("London Crime Analysis")
section = st.sidebar.radio("Navigate", [
    "Overview",
    "Crime Map",
    "Stop & Search",
    "Predictive Model",
    "Live Data"
])

# ── Overview ──────────────────────────────────────────────────────

if section == "Overview":
    st.title("London Crime Analysis")
    st.markdown("""
    This dashboard explores crime patterns across London using three years of 
    police data (2023–2025), ONS population estimates, and the Index of Multiple 
    Deprivation. All crime rates are normalised per 1,000 residents to allow 
    fair comparison across areas of different sizes.
    """)

    street = load_street()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total crimes recorded", f"{len(street):,}")
    col2.metric("Date range", "Jan 2023 – Dec 2025")
    col3.metric("London LSOAs", f"{street['lsoa_code'].nunique():,}")

    st.subheader("Crime types")
    crime_counts = street['crime_type'].value_counts().reset_index()
    crime_counts.columns = ['crime_type', 'count']
    fig = px.bar(crime_counts, x='count', y='crime_type', orientation='h',
                 color_discrete_sequence=['steelblue'])
    fig.update_layout(yaxis_title='', xaxis_title='Number of incidents',
                      height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly crime trend")
    monthly = street.groupby('month').size().reset_index(name='count')
    fig2 = px.line(monthly, x='month', y='count',
                   color_discrete_sequence=['steelblue'])
    fig2.update_layout(xaxis_title='', yaxis_title='Crimes recorded')
    st.plotly_chart(fig2, use_container_width=True)

# ── Crime Map ─────────────────────────────────────────────────────

elif section == "Crime Map":
    st.title("Crime Clusters Map")
    st.markdown("""
    LSOAs clustered by crime profile, normalised by population. Each dot 
    represents one LSOA coloured by its dominant crime type cluster.
    """)

    street = load_street()
    pop = load_population()

    crime_profile = street.groupby(['lsoa_code', 'crime_type']).size().unstack(fill_value=0)
    crime_profile = crime_profile.merge(pop[['lsoa_code', 'population']], on='lsoa_code', how='inner')
    crime_cols = [c for c in crime_profile.columns if c not in ['lsoa_code', 'population']]
    for col in crime_cols:
        crime_profile[col] = (crime_profile[col] / crime_profile['population']) * 1000

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings('ignore')

    scaler = StandardScaler()
    X = scaler.fit_transform(crime_profile[crime_cols])
    kmeans = KMeans(n_clusters=4, random_state=42)
    crime_profile['cluster'] = kmeans.fit_predict(X)
    cluster_means = crime_profile.groupby('cluster')[crime_cols].mean()
    crime_profile['dominant_crime'] = crime_profile['cluster'].map(
        cluster_means.idxmax(axis=1).to_dict()
    )

    lsoa_coords = street.groupby('lsoa_code')[['latitude', 'longitude']].median().reset_index()
    map_data = crime_profile.merge(lsoa_coords, on='lsoa_code', how='inner')

    fig = px.scatter_mapbox(
        map_data, lat='latitude', lon='longitude',
        color='dominant_crime',
        hover_name='lsoa_code',
        hover_data={'latitude': False, 'longitude': False},
        title='Crime profile clusters across London LSOAs',
        mapbox_style='carto-positron',
        zoom=9, height=650
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Stop & Search ─────────────────────────────────────────────────

elif section == "Stop & Search":
    st.title("Stop & Search Analysis")

    @st.cache_data
    def load_stop_search():
        import glob, os
        files = glob.glob('data/raw/**/*stop*search*.csv', recursive=True)
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            df['force'] = os.path.basename(f).split('-')[2]
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    ss = load_stop_search()
    ss.columns = [c.lower().replace(' ', '_') for c in ss.columns]

    st.subheader("Outcomes")
    outcome_counts = ss['outcome'].value_counts().reset_index()
    outcome_counts.columns = ['outcome', 'count']
    fig = px.bar(outcome_counts, x='count', y='outcome', orientation='h',
                 color_discrete_sequence=['coral'])
    fig.update_layout(yaxis_title='', xaxis_title='Number of stops', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Officer-defined ethnicity vs self-defined ethnicity")
    eth = ss.groupby(['officer-defined_ethnicity', 'self-defined_ethnicity']).size().reset_index(name='count')
    fig2 = px.bar(eth, x='officer-defined_ethnicity', y='count',
                  color='self-defined_ethnicity', barmode='group',
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(xaxis_title='Officer-defined ethnicity',
                       yaxis_title='Number of stops', height=500)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Object of search")
    obj_counts = ss['object_of_search'].value_counts().head(10).reset_index()
    obj_counts.columns = ['object', 'count']
    fig3 = px.bar(obj_counts, x='count', y='object', orientation='h',
                  color_discrete_sequence=['steelblue'])
    fig3.update_layout(yaxis_title='', xaxis_title='Number of stops', height=400)
    st.plotly_chart(fig3, use_container_width=True)

# ── Predictive Model ──────────────────────────────────────────────

elif section == "Predictive Model":
    st.title("Predicting Crime Rates from Deprivation")
    st.markdown("""
    A Random Forest model trained to predict LSOA-level crime rates per 1,000 
    residents using deprivation indicators. Extreme outliers (central London 
    tourist areas) were capped at the 99th percentile and the target was 
    log-transformed to account for skew. The model achieves an R² of 0.659, 
    meaning deprivation scores explain around two thirds of variation in 
    local crime rates.
    """)

    df = load_modelling_data()
    model = load_model()

    features = ['imd_score', 'income_score', 'employment_score',
                'education_score', 'health_score', 'barriers_score', 'living_env_score']

    st.subheader("Feature importance")
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance')

    fig = px.bar(importance, x='importance', y='feature', orientation='h',
                 color_discrete_sequence=['steelblue'])
    fig.update_layout(yaxis_title='', xaxis_title='Importance', height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predicted vs actual crime rate")
    fig2 = px.scatter(df, x='crime_rate', y='imd_score',
                      color='imd_score',
                      color_continuous_scale='Reds',
                      labels={'crime_rate': 'Crime rate per 1,000 residents',
                              'imd_score': 'IMD Score'},
                      title='Crime rate vs deprivation score by LSOA')
    st.plotly_chart(fig2, use_container_width=True)

# ── Live Data ─────────────────────────────────────────────────────

elif section == "Live Data":
    st.title("Live Data — Central London")
    st.markdown("Most recent crimes within 1 mile of Central London via the police.uk API.")

    with st.spinner("Fetching live data..."):
        live = get_live_data()

    if live is not None and len(live) > 0:
        st.success(f"{len(live):,} crimes loaded")
        cat_counts = live['category'].value_counts().reset_index()
        cat_counts.columns = ['category', 'count']
        fig = px.bar(cat_counts, x='count', y='category', orientation='h',
                     color_discrete_sequence=['steelblue'])
        fig.update_layout(yaxis_title='', xaxis_title='Number of incidents',
                          title='Crime categories — Central London, October 2025')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch live data from the API. Please try again later.")