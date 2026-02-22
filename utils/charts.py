"""
utils/charts.py
---------------
Shared chart helpers used across dashboard sections.
All functions return a Plotly figure object.

Import example:
    from utils.charts import add_economic_annotations, apply_base_layout
"""

import pandas as pd
import plotly.graph_objects as go
from utils.constants import (
    ECONOMIC_EVENTS,
    BASE_LAYOUT,
    AXIS_DEFAULTS,
)


# ── Layout helpers ────────────────────────────────────────────────

def apply_base_layout(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    """
    Apply the standard dark transparent background and drag/spike
    settings to a figure. Additional layout kwargs are passed through
    so callers can override individual properties.

    Usage:
        fig = apply_base_layout(fig, height=360, hovermode='y')
    """
    layout = {**BASE_LAYOUT, "height": height, **kwargs}
    fig.update_layout(**layout)
    return fig


def style_xaxis(fig: go.Figure, show_labels: bool = False, **kwargs) -> go.Figure:
    """Apply standard x-axis defaults. Labels hidden by default."""
    props = {**AXIS_DEFAULTS, "showticklabels": show_labels, **kwargs}
    fig.update_xaxes(**props)
    return fig


def style_yaxis(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    """Apply standard y-axis defaults."""
    props = {**AXIS_DEFAULTS, "title": title, **kwargs}
    fig.update_yaxes(**props)
    return fig


# ── Annotation helpers ────────────────────────────────────────────

def add_economic_annotations(
    fig: go.Figure,
    y_max: float,
    events: list | None = None,
    show_all: bool = True,
) -> go.Figure:
    """
    Overlay dashed vertical lines marking key economic events.

    Args:
        fig:      Plotly figure to annotate.
        y_max:    Upper y-value used to stagger annotation label positions.
        events:   List of event dicts with 'date', 'label', 'color' keys.
                  Defaults to the module-level ECONOMIC_EVENTS list.
        show_all: When True (default), text labels are drawn beside each
                  line. Pass False to draw lines only — use this on charts
                  where labels would overlap other content.
    """
    if events is None:
        events = ECONOMIC_EVENTS

    for i, event in enumerate(events):
        fig.add_vline(
            x=event["date"],
            line_dash="dash",
            line_color=event["color"],
            opacity=0.5,
        )
        if show_all:
            fig.add_annotation(
                x=event["date"],
                y=y_max * (0.95 - (i % 2) * 0.15),
                text=event["label"],
                showarrow=False,
                font=dict(color=event["color"], size=9),
                xanchor="left",
                xshift=5,
            )

    return fig


def add_vline_annotation(
    fig: go.Figure,
    x: str | pd.Timestamp,
    label: str,
    color: str = "white",
    y_ref: float = 0.92,
    y_paper: bool = False,
) -> go.Figure:
    """
    Add a single annotated vertical dashed line.

    Args:
        x:        Date string or Timestamp for the line position.
        label:    Text to display beside the line.
        color:    Line and text colour.
        y_ref:    Vertical position of the annotation label.
        y_paper:  If True, y_ref is in paper (0–1) coordinates.
                  If False, y_ref is in data coordinates.
    """
    fig.add_vline(
        x=x,
        line_dash="dash",
        line_color=color,
        opacity=0.9,
        line_width=2,
    )
    fig.add_annotation(
        x=x,
        y=y_ref,
        yref="paper" if y_paper else "y",
        text=label,
        showarrow=False,
        font=dict(color=color, size=11),
        xanchor="left",
        xshift=10,
        bgcolor="rgba(0,0,0,0.5)",
    )
    return fig


# ── Reusable chart builders ───────────────────────────────────────

def horizontal_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    colorscale: str = "RdYlGn_r",
    hover_template: str | None = None,
    customdata_col: str | None = None,
    height: int = 420,
    x_title: str = "",
    x_suffix: str = "",
) -> go.Figure:
    """
    Standard horizontal bar chart with a diverging colour scale.
    Used for % change charts across crime types and boroughs.

    Args:
        df:              Source DataFrame.
        x_col:           Column for bar length (numeric).
        y_col:           Column for bar labels (categorical).
        colorscale:      Plotly colour scale name.
        hover_template:  Custom hovertemplate string.
        customdata_col:  Column to attach as customdata for hover.
        height:          Chart height in pixels.
        x_title:         X-axis title.
        x_suffix:        Suffix appended to x-axis tick labels (e.g. '%').
    """
    bar_kwargs: dict = dict(
        x=df[x_col],
        y=df[y_col],
        orientation="h",
        marker=dict(
            color=df[x_col],
            colorscale=colorscale,
            showscale=False,
        ),
    )

    if hover_template:
        bar_kwargs["hovertemplate"] = hover_template
    if customdata_col and customdata_col in df.columns:
        bar_kwargs["customdata"] = df[customdata_col]

    fig = go.Figure()
    fig.add_trace(go.Bar(**bar_kwargs))
    fig.add_vline(x=0, line_color="white", opacity=0.3)

    fig = apply_base_layout(fig, height=height, hovermode="y")
    fig = style_xaxis(fig, show_labels=True, title=x_title, ticksuffix=x_suffix)
    fig = style_yaxis(fig)

    return fig


def time_series_chart(
    traces: list[dict],
    height: int = 420,
    y_title: str = "",
    show_x_labels: bool = False,
) -> go.Figure:
    """
    Build a multi-trace time series figure.

    Each item in `traces` is a dict with keys:
        x, y       – data arrays
        name       – legend label
        color      – line colour
        width      – line width (default 2)
        dash       – line dash style (default 'solid')
        fill       – fill mode (optional, e.g. 'tozeroy')
        fillcolor  – fill colour (optional)
        hover      – hovertemplate string (optional)
        yaxis      – secondary axis key e.g. 'y2' (optional)

    Usage:
        fig = time_series_chart([
            {'x': df['month'], 'y': df['count'], 'name': 'Shoplifting',
             'color': '#e74c3c', 'hover': '%{x|%b %Y}: %{y:,}<extra></extra>'},
        ], y_title='Monthly incidents')
    """
    fig = go.Figure()

    for t in traces:
        scatter_kwargs = dict(
            x=t["x"],
            y=t["y"],
            name=t.get("name", ""),
            line=dict(
                color=t.get("color", "#95a5a6"),
                width=t.get("width", 2),
                dash=t.get("dash", "solid"),
            ),
            showlegend=t.get("name") is not None,
        )
        if "fill" in t:
            scatter_kwargs["fill"]      = t["fill"]
        if "fillcolor" in t:
            scatter_kwargs["fillcolor"] = t["fillcolor"]
        if "hover" in t:
            scatter_kwargs["hovertemplate"] = t["hover"]
        if "yaxis" in t:
            scatter_kwargs["yaxis"] = t["yaxis"]

        fig.add_trace(go.Scatter(**scatter_kwargs))

    fig = apply_base_layout(
        fig, height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig = style_xaxis(fig, show_labels=show_x_labels)
    fig = style_yaxis(fig, title=y_title)

    return fig


# ── Specific reusable figures ─────────────────────────────────────

def crime_change_overview_chart(crime_by_year: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart showing % change for every crime type,
    sorted ascending. Used on The Story page.
    """
    pivot = crime_by_year.reset_index().sort_values("change", ascending=True)

    return horizontal_bar_chart(
        df=pivot,
        x_col="change",
        y_col="crime_type",
        colorscale="RdYlGn_r",
        hover_template="<b>%{y}</b><br>%{x:+.1f}% change 2023 to 2025<extra></extra>",
        height=500,
        x_title="% change 2023 to 2025",
        x_suffix="%",
    )


def monthly_total_chart(
    monthly_all: pd.DataFrame,
    annotate_events: bool = True,
) -> go.Figure:
    """
    Area chart of total monthly recorded crime with optional
    economic event annotations. Used on The Story page.
    """
    fig = time_series_chart(
        traces=[{
            "x":         monthly_all["month"],
            "y":         monthly_all["count"],
            "name":      None,
            "color":     "#95a5a6",
            "width":     2,
            "fill":      "tozeroy",
            "fillcolor": "rgba(149,165,166,0.08)",
            "hover":     "%{x|%b %Y}<br>%{y:,} crimes recorded<extra></extra>",
        }],
        height=420,
        y_title="Monthly crimes recorded",
    )

    if annotate_events:
        fig = add_economic_annotations(fig, monthly_all["count"].max())

    return fig