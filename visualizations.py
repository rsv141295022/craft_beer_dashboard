"""
Reusable visualization functions for Craft Beer Recipe Dashboard.
All functions return Plotly figure objects for use with Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import gaussian_kde, pearsonr
from statsmodels.nonparametric.smoothers_lowess import lowess


# =============================================================================
# BELL CURVE / DISTRIBUTION PLOTS
# =============================================================================

def plot_bell_curve(
    data,
    title,
    bjcp_range_x0=None,
    bjcp_range_x1=None,
    height=400
):
    """
    Plot a skewed bell curve (skew-normal distribution) fitted to data.

    Parameters:
    -----------
    data : array-like
        Data values to fit the distribution to
    title : str
        Chart title
    bjcp_range_x0 : float, optional
        Left bound of BJCP guideline range
    bjcp_range_x1 : float, optional
        Right bound of BJCP guideline range
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
    """
    # Remove NaN values
    data = np.array(data)
    data = data[~np.isnan(data)]

    if len(data) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data points",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Calculate stats
    mean = np.mean(data)
    std_dev = np.std(data)
    median = np.median(data)

    # Fit skew normal distribution
    try:
        params = stats.skewnorm.fit(data)
        a, loc, scale = params
    except Exception:
        # Fallback to normal distribution
        a, loc, scale = 0, mean, std_dev

    # Generate curve
    x = np.linspace(min(data) - 0.5 * std_dev, max(data) + 0.5 * std_dev, 1000)
    y = stats.skewnorm.pdf(x, a, loc, scale)
    y_max = max(y) if max(y) > 0 else 1

    fig = go.Figure()

    # Add BJCP range shading first (background)
    if bjcp_range_x0 is not None and bjcp_range_x1 is not None:
        fig.add_vrect(
            x0=bjcp_range_x0, x1=bjcp_range_x1,
            fillcolor='rgba(200, 200, 200, 0.3)',
            line_width=0,
            layer="below"
        )
        fig.add_annotation(
            x=(bjcp_range_x0 + bjcp_range_x1) / 2,
            y=y_max * 1.35,
            text="BJCP<br>RANGE",
            showarrow=False,
            font=dict(size=10)
        )

    # Add reference lines
    # Mean line
    fig.add_shape(
        type="line",
        x0=mean, x1=mean,
        y0=0, y1=y_max * 1.08,
        line=dict(color="red", width=2, dash="solid"),
        layer="below"
    )

    # Median line
    fig.add_shape(
        type="line",
        x0=median, x1=median,
        y0=0, y1=y_max * 1.08,
        line=dict(color="green", width=2, dash="dash"),
        layer="below"
    )

    # Std dev lines
    fig.add_shape(
        type="line",
        x0=mean - std_dev, x1=mean - std_dev,
        y0=0, y1=y_max * 1.08,
        line=dict(color="orange", width=1.5, dash="dot"),
        layer="below"
    )
    fig.add_shape(
        type="line",
        x0=mean + std_dev, x1=mean + std_dev,
        y0=0, y1=y_max * 1.08,
        line=dict(color="orange", width=1.5, dash="dot"),
        layer="below"
    )

    # Add bell curve
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color='#4a7cb5', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(74, 124, 181, 0.1)',
        name='Distribution'
    ))

    # Add data points on x-axis
    fig.add_trace(go.Scatter(
        x=data,
        y=[0] * len(data),
        mode='markers',
        marker=dict(color='darkblue', size=8, symbol='circle', opacity=0.6),
        name='Data Points'
    ))

    # Annotations
    fig.add_annotation(
        x=mean, y=y_max * 1.15,
        text=f"Mean<br>{mean:.3f}",
        showarrow=False,
        font=dict(color="red", size=9)
    )
    fig.add_annotation(
        x=median, y=y_max * 1.15,
        text=f"Median<br>{median:.3f}",
        showarrow=False,
        font=dict(color="green", size=9)
    )
    fig.add_annotation(
        x=mean - std_dev, y=y_max * 1.15,
        text=f"-1σ<br>{mean - std_dev:.3f}",
        showarrow=False,
        font=dict(color="orange", size=9)
    )
    fig.add_annotation(
        x=mean + std_dev, y=y_max * 1.15,
        text=f"+1σ<br>{mean + std_dev:.3f}",
        showarrow=False,
        font=dict(color="orange", size=9)
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickangle=-45,
            tickformat='.3f',
            showgrid=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        template='plotly_white',
        showlegend=False,
        height=height
    )

    return fig


# =============================================================================
# SCATTER PLOTS WITH TREND LINE
# =============================================================================

def plot_scatter_trend(
    data,
    x_col,
    y_col,
    title,
    x_label=None,
    y_label=None,
    height=400
):
    """
    Plot scatter with OLS trend line and Pearson correlation.

    Parameters:
    -----------
    data : pd.DataFrame
        Data containing x and y columns
    x_col : str
        Column name for x-axis
    y_col : str
        Column name for y-axis
    title : str
        Chart title
    x_label : str, optional
        Label for x-axis
    y_label : str, optional
        Label for y-axis
    height : int
        Chart height in pixels

    Returns:
    --------
    tuple: (go.Figure, dict with stats)
    """
    data = data.dropna(subset=[x_col, y_col])

    if len(data) < 3:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data points",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig, {}

    x = data[x_col].values
    y = data[y_col].values

    pearson_r, pearson_p = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(color="#4a7cb5", size=8, opacity=0.8),
        name='Recipes',
        hovertemplate=f'{x_label or x_col}: %{{x}}<br>{y_label or y_col}: %{{y:.3f}}<extra></extra>'
    ))

    # Trend line
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color="darkgray", dash="dash", width=3),
        name='Trend'
    ))

    # Annotation with stats
    fig.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"Pearson r = {pearson_r:.3f}<br>p-value = {pearson_p:.4f}",
        showarrow=False,
        font=dict(color="black", size=11),
        align="left",
        bgcolor="rgba(255,255,255,0.8)"
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        template="plotly_white",
        height=height,
        showlegend=False
    )

    stats_dict = {
        'pearson_r': pearson_r,
        'p_value': pearson_p,
        'r_squared': pearson_r ** 2,
        'slope': slope,
        'intercept': intercept,
        'trend': 'Increasing' if pearson_r > 0 else 'Decreasing'
    }

    return fig, stats_dict


# =============================================================================
# DONUT / PIE CHARTS
# =============================================================================

def plot_donut(
    data,
    names_col,
    values_col,
    title=None,
    hole=0.5,
    height=400,
    color_sequence=None
):
    """
    Plot a donut (pie with hole) chart.

    Parameters:
    -----------
    data : pd.DataFrame
        Data containing names and values columns
    names_col : str
        Column name for labels
    values_col : str
        Column name for values
    title : str, optional
        Chart title
    hole : float
        Size of center hole (0-1)
    height : int
        Chart height in pixels
    color_sequence : list, optional
        Custom color sequence

    Returns:
    --------
    go.Figure
    """
    fig = px.pie(
        data,
        names=names_col,
        values=values_col,
        hole=hole,
        color_discrete_sequence=color_sequence
    )

    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        hovertemplate="%{label}: %{value:.2f}<extra></extra>",
        marker=dict(line=dict(color="white", width=2))
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        legend_title_text=names_col,
        height=height
    )

    return fig


# =============================================================================
# KDE (KERNEL DENSITY ESTIMATION) PLOTS
# =============================================================================

def plot_malt_kde(
    df,
    value_col="pct_of_grist",
    type_col="malt_type",
    title="Distribution of Malt Types if Used",
    min_points=10,
    height=400
):
    """
    Plot KDE distribution of malt percentages by type.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with malt percentage and type columns
    value_col : str
        Column containing percentage values
    type_col : str
        Column containing malt type categories
    title : str
        Chart title
    min_points : int
        Minimum data points required for KDE
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
    """
    fig = go.Figure()
    x_grid = np.linspace(0, 100, 1000)

    for malt, grp in df.groupby(type_col):
        values = grp[value_col].values

        if len(values) < min_points:
            continue

        kde = gaussian_kde(values)
        y = kde(x_grid)
        y = y / y.max() * 100  # normalize to percentage

        fig.add_trace(go.Scatter(
            x=x_grid,
            y=y,
            mode="lines",
            name=str(malt).title(),
            hovertemplate=(
                f"{malt}<br>"
                "% of grist: %{x:.1f}<br>"
                "% of recipes: %{y:.1f}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title="% of Grist",
        yaxis_title="% of Recipes that Used This Malt",
        template="plotly_white",
        legend_title_text="Malt Type",
        hovermode="x unified",
        height=height
    )

    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])

    return fig


def plot_hop_rate_kde(
    df,
    rate_col="oz_per_gal",
    stage_col="hop_type",
    stages=("flavour", "aroma"),
    title="Hop Rate Distribution by Addition Type",
    bandwidth=None,
    height=400
):
    """
    Plot KDE distribution of hop rates by usage stage.

    Parameters:
    -----------
    df : pd.DataFrame
        Data with hop rate and stage columns
    rate_col : str
        Column containing hop rate values
    stage_col : str
        Column containing hop usage stage
    stages : tuple
        Stages to include in plot
    title : str
        Chart title
    bandwidth : float, optional
        KDE bandwidth parameter
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
    """
    fig = go.Figure()

    x_grid = np.linspace(
        df[rate_col].min(),
        df[rate_col].max(),
        500
    )

    for stage in stages:
        values = df.loc[df[stage_col] == stage, rate_col].dropna().values

        if len(values) < 10:
            continue

        kde = gaussian_kde(values, bw_method=bandwidth)
        y = kde(x_grid)
        y = y / y.max() * 100

        fig.add_trace(go.Scatter(
            x=x_grid,
            y=y,
            mode="lines",
            name=stage.capitalize(),
            hovertemplate=(
                f"{stage.capitalize()}<br>"
                "Hop rate: %{x:.3f} oz/gal<br>"
                "Recipes: %{y:.1f}%"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Hop Rate (oz / gallon)",
        yaxis_title="Percent of Recipes Using This Rate",
        template="plotly_white",
        hovermode="x unified",
        height=height
    )

    fig.update_yaxes(range=[0, 100])
    return fig


# =============================================================================
# TREND ANALYSIS WITH LOESS
# =============================================================================

def _loess_ci(x, y, frac=0.4, n_boot=200, grid_size=200):
    """Calculate LOESS fit with bootstrap confidence intervals."""
    x = np.asarray(x)
    y = np.asarray(y)

    x_grid = np.linspace(x.min(), x.max(), grid_size)

    # LOESS on original data
    loess_fit = lowess(y, x, frac=frac, return_sorted=True)
    y_loess = np.interp(x_grid, loess_fit[:, 0], loess_fit[:, 1])

    # Bootstrap for CI
    boot_curves = []
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        xb, yb = x[idx], y[idx]
        fit = lowess(yb, xb, frac=frac, return_sorted=True)
        yb_interp = np.interp(x_grid, fit[:, 0], fit[:, 1])
        boot_curves.append(yb_interp)

    boot_curves = np.vstack(boot_curves)
    lower = np.percentile(boot_curves, 2.5, axis=0)
    upper = np.percentile(boot_curves, 97.5, axis=0)

    return x_grid, y_loess, lower, upper


def plot_late_hop_usage(
    yearly_df,
    stages=("flavour", "aroma"),
    show_loess=True,
    title="Usage of Late Hops Over Time",
    height=450
):
    """
    Plot late hop usage trends with linear trend and optional LOESS.

    Parameters:
    -----------
    yearly_df : pd.DataFrame
        Data with columns: year, hop_type, pct
    stages : tuple
        Hop stages to plot
    show_loess : bool
        Whether to show LOESS smoothing with CI
    title : str
        Chart title
    height : int
        Chart height in pixels

    Returns:
    --------
    go.Figure
    """
    fig = go.Figure()

    colors = {
        "bittering": "#c44e52",
        "flavour": "#4a7cb5",
        "aroma": "#d07c2c",
        "dry_hop": "#55a868"
    }

    for stage in stages:
        d = yearly_df[yearly_df["hop_type"] == stage].sort_values("year")

        if len(d) < 3:
            continue

        x = d["year"].values
        y = d["pct"].values
        color = colors.get(stage, "#888888")

        # Scatter points
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="markers",
            name=stage.capitalize(),
            marker=dict(size=9, color=color)
        ))

        # Linear trend
        m, b = np.polyfit(x, y, 1)
        x_fit = np.array([x.min(), x.max()])
        y_fit = m * x_fit + b

        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit,
            mode="lines",
            name=f"Linear ({stage.capitalize()})",
            line=dict(color=color, dash="dot", width=2),
            showlegend=False
        ))

        # Pearson annotation
        r, _ = pearsonr(x, y)
        fig.add_annotation(
            x=x_fit.mean(),
            y=y_fit.mean() + 5,
            text=f"r = {r:.2f}",
            showarrow=False,
            font=dict(size=10, color=color)
        )

        # LOESS with CI
        if show_loess and len(x) >= 10:
            try:
                xg, y_loess, lo, hi = _loess_ci(x, y, frac=0.4)

                # CI band
                fig.add_trace(go.Scatter(
                    x=np.concatenate([xg, xg[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor=f"rgba(128, 128, 128, 0.15)",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False
                ))

                # LOESS line
                fig.add_trace(go.Scatter(
                    x=xg, y=y_loess,
                    mode="lines",
                    name=f"LOESS ({stage.capitalize()})",
                    line=dict(color=color, width=3),
                    showlegend=False
                ))
            except Exception:
                pass

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="% of Recipes",
        template="plotly_white",
        hovermode="x unified",
        height=height
    )

    fig.update_yaxes(range=[0, 100])
    return fig


# =============================================================================
# VIOLIN / BOX PLOTS
# =============================================================================

def plot_violin(
    df,
    x_col,
    y_col,
    title,
    x_label=None,
    y_label=None,
    show_box=True,
    points="outliers",
    height=400
):
    """
    Plot violin chart with optional box plot overlay.

    Parameters:
    -----------
    df : pd.DataFrame
        Data containing x and y columns
    x_col : str
        Column for x-axis (categories)
    y_col : str
        Column for y-axis (values)
    title : str
        Chart title
    x_label : str, optional
        X-axis label
    y_label : str, optional
        Y-axis label
    show_box : bool
        Whether to show box plot inside violin
    points : str
        How to show individual points ("outliers", "all", False)
    height : int
        Chart height

    Returns:
    --------
    go.Figure
    """
    fig = px.violin(
        df,
        x=x_col,
        y=y_col,
        box=show_box,
        points=points
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label or x_col,
        yaxis_title=y_label or y_col,
        template="plotly_white",
        showlegend=False,
        height=height
    )

    return fig


def plot_box_ions(
    df,
    ions,
    title="Water Ion Concentrations",
    height=400
):
    """
    Plot box plots for water ion concentrations.

    Parameters:
    -----------
    df : pd.DataFrame
        Data containing ion columns
    ions : list
        List of ion column names
    title : str
        Chart title
    height : int
        Chart height

    Returns:
    --------
    go.Figure
    """
    # Filter to available columns
    available_ions = [col for col in ions if col in df.columns]

    if not available_ions:
        fig = go.Figure()
        fig.add_annotation(
            text="No ion data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    water_long = (
        df[available_ions]
        .dropna(how="all")
        .melt(var_name="ion", value_name="ppm")
        .dropna(subset=["ppm"])
    )

    if len(water_long) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No ion data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = px.box(
        water_long,
        y="ppm",
        x="ion",
        color="ion",
        points="all"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Ion",
        yaxis_title="Concentration (ppm)",
        template="plotly_white",
        showlegend=False,
        height=height
    )

    return fig


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_stats_summary(data, column):
    """
    Get summary statistics for a data column.

    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Data to summarize
    column : str, optional
        Column name if data is DataFrame

    Returns:
    --------
    dict
    """
    if isinstance(data, pd.DataFrame):
        values = data[column].dropna()
    else:
        values = data.dropna()

    return {
        'count': len(values),
        'mean': values.mean(),
        'median': values.median(),
        'std': values.std(),
        'min': values.min(),
        'max': values.max(),
        'q25': values.quantile(0.25),
        'q75': values.quantile(0.75)
    }
