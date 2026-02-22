"""
app.py
------
Entry point for the London Crime & the Cost of Living dashboard.
Handles page config, sidebar navigation, and routing to section
modules. No chart or data logic lives here.

Run with:
    streamlit run app.py
"""

import streamlit as st

from sections import (
    crime_deprivation,
    economic_crime,
    policing_response,
    the_story,
    where_headed,
)

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="London Crime & the Cost of Living",
    page_icon="🔍",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────
st.markdown("""
    <style>
    .js-plotly-plot .plotly .cursor-crosshair { cursor: default !important; }
    .js-plotly-plot .plotly .cursor-pointer   { cursor: default !important; }
    .js-plotly-plot .plotly .cursor-move      { cursor: default !important; }
    .js-plotly-plot .plotly svg               { cursor: default !important; }
    g.spikeline    { display: none !important; }
    line.spikeline { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.title("London Crime Analysis\nLondon Crime & the Cost of Living")
st.sidebar.markdown("*2023 – 2025*")

SECTIONS = [
    "The Story",
    "Economic Crime",
    "Crime & Deprivation",
    "Policing Response",
    "Where is London Headed?",
]

section = st.sidebar.radio("Navigate", SECTIONS)

st.sidebar.divider()
st.sidebar.caption("""
**Data sources**
- Metropolitan & City of London Police via police.uk
- ONS LSOA Population Estimates 2022
- Index of Multiple Deprivation 2019
- ONS Census 2021
- ONS CPI food inflation series D7G8

**Methodology note:**
Recorded crime reflects both actual crime levels and policing
activity. An increase in recorded offences may indicate more
enforcement rather than more crime. All crime rates are
normalised per 1,000 residents. Stop and search records are
assigned to boroughs via nearest centroid from GPS coordinates.
""")

# ── Routing ───────────────────────────────────────────────────────
ROUTES = {
    "The Story":               the_story.render,
    "Economic Crime":          economic_crime.render,
    "Crime & Deprivation":     crime_deprivation.render,
    "Policing Response":       policing_response.render,
    "Where is London Headed?": where_headed.render,
}

ROUTES[section]()