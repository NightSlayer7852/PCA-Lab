import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pca_utils import SECTOR_COLORS


# ── Shared layout config ─────────────────────────────────────────────────────
_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(family="DM Sans, sans-serif", size=12, color="#222220"),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#D0D0CC",
        font=dict(family="DM Mono, monospace", size=11, color="#111110"),
    ),
    margin=dict(l=60, r=30, t=50, b=60),
)


def _sector_color_map():
    """Return the canonical sector→color dict for plotly color_discrete_map."""
    return SECTOR_COLORS.copy()


# ── 1. Bar Graph: Mean PC1 by Sector ─────────────────────────────────────────
def create_bar_graph(stock_pca):
    """Horizontal bar showing each sector's mean PC1 score (market beta).

    PC1 captures the "market factor". Sectors with higher absolute mean PC1
    load more strongly on the overall market direction.
    """
    sector_avg = (
        stock_pca.groupby("Sector")["PC1"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean")
    )
    colors = [SECTOR_COLORS.get(s, "#9CA3AF") for s in sector_avg["Sector"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sector_avg["Sector"],
        x=sector_avg["mean"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=1, color="#FFFFFF"),
        ),
        error_x=dict(
            type="data",
            array=sector_avg["std"].values,
            visible=True,
            color="#888888",
            thickness=1.5,
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean PC1: %{x:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text="Average Market Exposure (PC1) by Sector",
            font=dict(family="DM Serif Display, serif", size=18, color="#111110"),
        ),
        xaxis=dict(
            title="Mean PC1 Score",
            title_font=dict(color="#222220"),
            showgrid=True, gridcolor="#EEEEE8",
            zeroline=True, zerolinecolor="#CCCCCC", zerolinewidth=1.5,
            tickfont=dict(family="DM Mono, monospace", size=10, color="#222220"),
        ),
        yaxis=dict(
            tickfont=dict(family="DM Sans, sans-serif", size=11, color="#222220"),
            showgrid=False,
        ),
        showlegend=False,
        height=420,
    )
    return fig


# ── 2. Histogram: Distribution of PC1 scores ─────────────────────────────────
def create_histogram(stock_pca):
    """Stacked histogram of PC1 scores across all stocks, coloured by sector.

    Shows how stocks are distributed along the primary market factor axis.
    Box-plot marginal gives quick insight into spread and outliers.
    """
    fig = px.histogram(
        stock_pca,
        x="PC1",
        color="Sector",
        color_discrete_map=_sector_color_map(),
        nbins=20,
        barmode="overlay",
        marginal="box",
        title="Distribution of Market Factor (PC1) Scores",
        hover_data=["Ticker"],
        labels={"PC1": "PC1 Score"},
    )
    fig.update_traces(opacity=0.75)
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text="Distribution of Market Factor (PC1) Scores",
            font=dict(family="DM Serif Display, serif", size=18, color="#111110"),
        ),
        xaxis=dict(
            title="PC1 Score",
            title_font=dict(color="#222220"),
            showgrid=True, gridcolor="#EEEEE8",
            tickfont=dict(family="DM Mono, monospace", size=10, color="#222220"),
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(color="#222220"),
            showgrid=True, gridcolor="#EEEEE8",
            tickfont=dict(family="DM Mono, monospace", size=10, color="#222220"),
        ),
        legend=dict(
            title=dict(text="Sector", font=dict(family="DM Mono, monospace", size=10, color="#222220")),
            font=dict(size=11, color="#222220"),
        ),
        height=480,
    )
    return fig


# ── 3. Pairplot: Scatter matrix of PC dimensions ─────────────────────────────
def create_pairplot(stock_pca, show_3d=False):
    """Scatter matrix (pairplot) of principal components.

    Reveals how stocks cluster across different PC combinations. When 3D mode
    is active, PC3 is included automatically.
    """
    dims = ["PC1", "PC2"]
    if show_3d and "PC3" in stock_pca.columns:
        dims.append("PC3")

    fig = px.scatter_matrix(
        stock_pca,
        dimensions=dims,
        color="Sector",
        color_discrete_map=_sector_color_map(),
        hover_name="Ticker",
        title="Pairwise Relationships of Principal Components",
        labels={d: d for d in dims},
    )
    fig.update_traces(
        diagonal_visible=True,
        marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color="white")),
    )
    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text="Pairwise Relationships of Principal Components",
            font=dict(family="DM Serif Display, serif", size=18, color="#111110"),
        ),
        legend=dict(
            title=dict(text="Sector", font=dict(family="DM Mono, monospace", size=10, color="#222220")),
            font=dict(size=11, color="#222220"),
        ),
        height=700 if len(dims) == 3 else 500,
    )
    return fig


# ── 4. Heatmap: Stock-to-stock return correlations ───────────────────────────
def create_heatmap(returns, max_stocks=30):
    """Correlation heatmap of daily returns.

    Uses the full return matrix (up to *max_stocks*) so the colours
    faithfully represent how similarly two stocks move day-to-day.
    Diverging RdBu palette: blue = positive correlation, red = negative.
    """
    subset = returns.iloc[:, :max_stocks]
    corr = subset.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title=dict(text="Correlation", font=dict(size=12, color="#222220")),
            tickfont=dict(family="DM Mono, monospace", size=10, color="#222220"),
            thickness=15,
        ),
        hovertemplate=(
            "<b>%{x} × %{y}</b><br>"
            "Correlation: %{z:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    n = len(corr.columns)
    fig.update_layout(**_LAYOUT_BASE)
    fig.update_layout(
        title=dict(
            text=f"Return Correlation Heatmap ({n} Stocks)",
            font=dict(family="DM Serif Display, serif", size=18, color="#111110"),
        ),
        xaxis=dict(
            tickfont=dict(family="DM Mono, monospace", size=9, color="#222220"),
            tickangle=-45,
            showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(family="DM Mono, monospace", size=9, color="#222220"),
            showgrid=False,
            autorange="reversed",
        ),
        height=650,
        margin=dict(l=80, r=30, t=50, b=100),
    )
    return fig
