"""
utils/constants.py
------------------
Shared constants used across the dashboard sections.
Import from here rather than defining locally in section files.

Fix (critique): IMD_DOMAIN_LABELS previously used short keys like
'imd_score', 'income_score' that did not match the actual long column
names produced by the IMD CSV after keyword matching in script 04.
This caused _render_model_importance in where_headed.py to silently
produce empty bars when model.feature_names_in_ was used.

The mapping is now keyed on the actual IMD column name substrings
that the DOMAIN_KEYWORDS matching in 04_train_model.py would produce,
with a runtime fallback in data_loaders.build_imd_label_map() that
constructs the label map from the model's stored feature_names_in_
so there is no dependency on these keys being exact.
"""

# ── Plotly chart config ───────────────────────────────────────────
CHART_CONFIG = {'displayModeBar': False, 'scrollZoom': False}

# ── Economic event annotations ────────────────────────────────────
ECONOMIC_EVENTS = [
    {'date': '2023-01-01', 'label': 'Energy bills at\ncrisis peak',   'color': '#e74c3c'},
    {'date': '2023-04-01', 'label': 'Core inflation\npeaks at 7.1%', 'color': '#e67e22'},
    {'date': '2023-07-01', 'label': 'Energy cap\nfalls to £2,074',   'color': '#2ecc71'},
    {'date': '2024-05-01', 'label': 'Inflation hits\n2% target',      'color': '#2ecc71'},
    {'date': '2024-07-01', 'label': 'Labour\nelected',                 'color': '#3498db'},
    {'date': '2025-04-01', 'label': '"Awful April"\nbills rise again', 'color': '#e74c3c'},
]

# ── Colour palette ────────────────────────────────────────────────
CRIME_COLOURS = {
    'Shoplifting':            '#e74c3c',
    'Theft from the person':  '#e67e22',
    'Drugs':                  '#9b59b6',
    'Vehicle crime':          '#95a5a6',
    'Burglary':               '#7f8c8d',
    'Violence and sexual offences': '#3498db',
    'Robbery':                '#1abc9c',
}

RISK_COLOURS = {
    'Higher risk': '#e74c3c',
    'Medium risk': '#f39c12',
    'Lower risk':  '#2ecc71',
}

DEPRIVATION_OUTLIER_COLOURS = {
    'Deprived and high crime': '#e74c3c',
    'Affluent but high crime': '#e67e22',
}

# ── DataFrame column rename mappings ─────────────────────────────
BOROUGH_RENAME = {
    'borough':         'Borough',
    'change_pct':      '% change',
    'avg_imd_decile':  'Deprivation decile',
    'count_2023':      '2023 incidents',
    'count_2025':      '2025 incidents',
}

VULNERABILITY_RENAME = {
    'borough':             'Borough',
    'vulnerability_score': 'Score',
    'risk_tier':           'Risk tier',
}

# ── IMD domain display labels ─────────────────────────────────────
# Fix (critique): previously used short keys ('imd_score', 'income_score')
# that did not match actual IMD column names from the raw CSV, causing
# the feature importance chart to silently display wrong/empty labels.
#
# This mapping now uses lowercase substrings of the actual column names
# as keys. build_imd_label_map() in data_loaders.py uses this to
# construct a {actual_feature_name: display_label} dict at runtime by
# matching each feature name against these substrings, so the chart
# labels work regardless of the exact column name in the raw IMD file.
#
# Do not change these keys without also updating build_imd_label_map().
IMD_DOMAIN_LABELS = {
    'index of multiple deprivation (imd) score': 'Overall deprivation',
    'income score':                               'Income deprivation',
    'employment score':                           'Employment deprivation',
    'education, skills and training score':       'Education & skills',
    'health deprivation and disability score':    'Health deprivation',
    'barriers to housing and services score':     'Barriers to housing',
    'living environment score':                   'Living environment',
}

# Fallback short display labels used when a feature name does not
# match any key in IMD_DOMAIN_LABELS. build_imd_label_map() will
# return the raw feature name if no match is found.
IMD_DOMAIN_SHORT_LABELS = {
    'imd':         'Overall deprivation',
    'income':      'Income deprivation',
    'employment':  'Employment deprivation',
    'education':   'Education & skills',
    'health':      'Health deprivation',
    'barriers':    'Barriers to housing',
    'living':      'Living environment',
}

# ── Date range covered by the dashboard ──────────────────────────
DATE_MIN = '2023-01'
DATE_MAX = '2025-12'

# ── Map defaults ──────────────────────────────────────────────────
LONDON_MAP_CENTRE = {'lat': 51.509, 'lon': -0.118}
LONDON_MAP_ZOOM   = 9
MAPBOX_STYLE      = 'carto-darkmatter'

# ── Shared layout defaults applied to all Plotly figures ─────────
BASE_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    dragmode=False,
    hovermode='x unified',
)

AXIS_DEFAULTS = dict(
    showspikes=False,
    gridcolor='rgba(255,255,255,0.05)',
)

LEGEND_TOP = dict(
    orientation='h',
    yanchor='bottom',
    y=1.02,
)