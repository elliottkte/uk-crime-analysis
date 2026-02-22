"""
utils/constants.py
------------------
Shared constants used across the dashboard sections.
Import from here rather than defining locally in section files.
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
IMD_DOMAIN_LABELS = {
    'imd_score':         'Overall deprivation',
    'income_score':      'Income deprivation',
    'employment_score':  'Employment deprivation',
    'education_score':   'Education & skills',
    'health_score':      'Health deprivation',
    'barriers_score':    'Barriers to housing',
    'living_env_score':  'Living environment',
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