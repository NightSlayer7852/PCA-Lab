import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

from pca_utils import (
    load_and_prepare,
    get_stock_pca_positions,
    get_loadings,
    SECTOR_COLORS,
)

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCA · Stock Market Lens",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Reset & Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background-color: #FAFAF8;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8E8E4;
}
[data-testid="stSidebar"] .css-1d391kg { padding-top: 2rem; }

/* Main title */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: #111110;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 0.15rem;
}
.main-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #888882;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E4;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    flex: 1;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #999994;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #111110;
    line-height: 1;
}
.metric-unit {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #888882;
    margin-top: 0.2rem;
}

/* Section headings */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #AAAAAA;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    margin-top: 0.2rem;
}

/* Info box */
.info-box {
    background: #F3F3EF;
    border-left: 3px solid #111110;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.86rem;
    color: #444440;
    line-height: 1.6;
}

/* Sidebar section headers */
.sidebar-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #BBBBBB;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 1.4rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #F0F0EC;
}

/* Divider */
.thin-divider {
    height: 1px;
    background: #E8E8E4;
    margin: 1.8rem 0;
}

/* Tooltip / legend label */
.legend-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}

/* Stock table */
.dataframe { font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; }

/* Plotly chart border */
.chart-container {
    border: 1px solid #E8E8E4;
    border-radius: 12px;
    overflow: hidden;
    background: #FFFFFF;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background: #FAFAF8;
    border: 1.5px dashed #D8D8D4;
    border-radius: 10px;
    padding: 0.5rem;
}

/* Slider */
[data-testid="stSlider"] > div { color: #111110; }

/* Selectbox */
[data-testid="stSelectbox"] label { font-size: 0.82rem; color: #555550; }

/* Checkbox */
[data-testid="stCheckbox"] label { font-size: 0.82rem; color: #555550; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title" style="font-size:1.5rem;">📈 PCA Lens</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle" style="font-size:0.65rem;">Stock Market · Dimensionality Reduction</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio(
        "Choose data source",
        ["📂 Upload Kaggle CSV", "🎲 Use Sample Data"],
        label_visibility="collapsed"
    )

    uploaded_file = None
    if data_source == "📂 Upload Kaggle CSV":
        uploaded_file = st.file_uploader(
            "Upload all_stocks_5yr.csv",
            type=["csv"],
            help="Download from: kaggle.com/datasets/camnugent/sandp500"
        )
        st.markdown(
            '<div style="font-size:0.72rem; color:#AAAAAA; margin-top:0.5rem;">'
            '🔗 <a href="https://www.kaggle.com/datasets/camnugent/sandp500" target="_blank" '
            'style="color:#555550;">Download dataset from Kaggle</a></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-section">Date Range</div>', unsafe_allow_html=True)
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        start_date = st.date_input("From", value=pd.to_datetime("2016-01-01"), label_visibility="visible")
    with col_s2:
        end_date = st.date_input("To", value=pd.to_datetime("2018-01-01"), label_visibility="visible")

    st.markdown('<div class="sidebar-section">PCA Settings</div>', unsafe_allow_html=True)
    max_stocks = st.slider("Max stocks to include", min_value=10, max_value=100, value=50, step=5)
    num_pcs = st.slider("Number of Principal Components", min_value=2, max_value=10, value=2, step=1)

    st.markdown('<div class="sidebar-section">Display</div>', unsafe_allow_html=True)
    show_labels = st.checkbox("Show ticker labels on chart", value=True)
    selected_sectors = st.multiselect(
        "Filter by sector",
        options=list(SECTOR_COLORS.keys()),
        default=[],
        placeholder="All sectors"
    )



# ── Sample data generator ────────────────────────────────────────────────────
@st.cache_data
def generate_sample_data(seed=42):
    """Generate realistic-looking synthetic S&P 500 style data."""
    np.random.seed(seed)
    tickers = [
        "AAPL","MSFT","GOOGL","META","NVDA","INTC","CSCO","IBM","ORCL","AMD",
        "JPM","BAC","WFC","GS","MS","C","AXP","BLK","USB","COF",
        "JNJ","PFE","UNH","MRK","ABT","TMO","MDT","AMGN","GILD","BMY",
        "XOM","CVX","COP","SLB","EOG","PSX","VLO","OXY",
        "AMZN","TSLA","HD","NKE","MCD","SBUX","TGT","LOW",
        "PG","KO","PEP","WMT",
    ]
    dates = pd.date_range("2016-01-01", "2018-01-01", freq="B")
    n_days = len(dates)
    rows = []
    # Sector factor returns (to introduce correlation structure)
    sector_factors = {
        "Tech": np.random.randn(n_days) * 0.008,
        "Finance": np.random.randn(n_days) * 0.007,
        "Health": np.random.randn(n_days) * 0.006,
        "Energy": np.random.randn(n_days) * 0.012,
        "Consumer": np.random.randn(n_days) * 0.007,
        "Staples": np.random.randn(n_days) * 0.005,
    }
    market_factor = np.random.randn(n_days) * 0.005

    sector_assignment = {
        "AAPL":"Tech","MSFT":"Tech","GOOGL":"Tech","META":"Tech","NVDA":"Tech",
        "INTC":"Tech","CSCO":"Tech","IBM":"Tech","ORCL":"Tech","AMD":"Tech",
        "JPM":"Finance","BAC":"Finance","WFC":"Finance","GS":"Finance","MS":"Finance",
        "C":"Finance","AXP":"Finance","BLK":"Finance","USB":"Finance","COF":"Finance",
        "JNJ":"Health","PFE":"Health","UNH":"Health","MRK":"Health","ABT":"Health",
        "TMO":"Health","MDT":"Health","AMGN":"Health","GILD":"Health","BMY":"Health",
        "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy","EOG":"Energy",
        "PSX":"Energy","VLO":"Energy","OXY":"Energy",
        "AMZN":"Consumer","TSLA":"Consumer","HD":"Consumer","NKE":"Consumer",
        "MCD":"Consumer","SBUX":"Consumer","TGT":"Consumer","LOW":"Consumer",
        "PG":"Staples","KO":"Staples","PEP":"Staples","WMT":"Staples",
    }

    for ticker in tickers:
        price = 100.0
        sec = sector_assignment.get(ticker, "Tech")
        idio = np.random.randn(n_days) * 0.009
        returns = market_factor + sector_factors[sec] * 0.6 + idio
        prices = price * np.cumprod(1 + returns)
        for i, d in enumerate(dates):
            rows.append({"date": d, "Name": ticker, "close": prices[i]})

    return pd.DataFrame(rows)


# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_kaggle_csv(file_bytes, start_date, end_date, max_stocks):
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Auto-detect columns
    date_col = next((c for c in df.columns if c.lower() in ["date","Date"]), df.columns[0])
    name_col = next((c for c in df.columns if c.lower() in ["name","ticker","symbol"]), df.columns[1])
    price_col = next((c for c in df.columns if c.lower() in ["close","adj close","price"]), df.columns[-1])

    returns = load_and_prepare(df, date_col, name_col, price_col, start_date, end_date)
    # Limit stocks
    if returns.shape[1] > max_stocks:
        returns = returns.iloc[:, :max_stocks]
    return returns


@st.cache_data(show_spinner=False)
def load_sample(start_date, end_date, max_stocks):
    df = generate_sample_data()
    from pca_utils import load_and_prepare
    returns = load_and_prepare(df, "date", "Name", "close", start_date, end_date)
    if returns.shape[1] > max_stocks:
        returns = returns.iloc[:, :max_stocks]
    return returns


# ── Main Layout ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Stock Market · PCA Lens</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Principal Component Analysis · Dimensionality Reduction · S&P 500</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="info-box">'
    '<b>How it works:</b> PCA identifies hidden structure across hundreds of correlated stocks by compressing '
    'daily return patterns into two principal components. Each dot represents one stock — stocks that move '
    'similarly cluster together, revealing sector groupings and market dynamics invisible in raw data.'
    '</div>',
    unsafe_allow_html=True
)

# ── Load the data ─────────────────────────────────────────────────────────────
with st.spinner("Loading returns data…"):
    try:
        if data_source == "📂 Upload Kaggle CSV" and uploaded_file is not None:
            raw_returns = load_kaggle_csv(uploaded_file.read(), start_date, end_date, max_stocks)
            data_label = "Kaggle S&P 500"
        else:
            raw_returns = load_sample(start_date, end_date, max_stocks)
            data_label = "Sample Data (synthetic S&P 500)"
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        st.stop()

# ── What-If Analysis (Sidebar) ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">What-If Analysis</div>', unsafe_allow_html=True)
    
    # Calculate baseline top drivers to show in the UI
    from sklearn.decomposition import PCA as _PCA_Base
    from sklearn.preprocessing import StandardScaler as _SS_Base
    _Xr_base = _SS_Base().fit_transform(raw_returns.values)
    _pca_base = _PCA_Base(n_components=1).fit(_Xr_base)
    _load_base = pd.Series(_pca_base.components_[0], index=raw_returns.columns)
    
    # Identify top 10 absolute drivers of PC1
    top_10_drivers = _load_base.abs().nlargest(10).index.tolist()
    
    exclude_top_n = st.slider(
        "Exclude Top Market Drivers", 
        min_value=0, max_value=10, value=0,
        help="Removes the stocks that drive the most variance in PC1, revealing how the rest of the market clusters."
    )
    
    auto_excluded = top_10_drivers[:exclude_top_n]
    if auto_excluded:
        st.markdown(f"<div style='font-size:0.75rem; color:#888; margin-top:-10px; margin-bottom:10px;'><b>Excluding:</b> {', '.join(auto_excluded)}</div>", unsafe_allow_html=True)
        
    manual_exclude = st.multiselect(
        "Manually exclude stocks", 
        options=sorted(raw_returns.columns.tolist()),
        default=[],
        placeholder="Select stocks to drop"
    )
    
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.68rem; color:#CCCCCC; line-height:1.6;">'
        'Paper: Ghorbani & Chong (2020)<br>'
        'PLOS ONE · PCA for Stock Prediction<br><br>'
        'Dataset: S&P 500 · Cam Nugent · Kaggle'
        '</div>',
        unsafe_allow_html=True
    )

# Filter returns based on exclusions
exclude_list = list(set(auto_excluded + manual_exclude))
returns = raw_returns.drop(columns=exclude_list, errors="ignore")

if returns.shape[1] < 3:
    st.error("⚠️ Not enough stocks remaining to run PCA. Please reduce exclusions.")
    st.stop()

with st.spinner("Running PCA…"):
    try:
        stock_pca, ev_ratio, pca_obj = get_stock_pca_positions(returns, n_components=2)
    except Exception as e:
        st.error(f"⚠️ Error running PCA: {e}")
        st.stop()

# ── Filter by sector ──────────────────────────────────────────────────────────
plot_df = stock_pca.copy()
if selected_sectors:
    plot_df = plot_df[plot_df["Sector"].isin(selected_sectors)]

# ── Metric Cards ──────────────────────────────────────────────────────────────
n_stocks = len(plot_df)
n_days = len(returns)
pc1_var = round(ev_ratio[0] * 100, 1)
pc2_var = round(ev_ratio[1] * 100, 1)
total_var = round((ev_ratio[0] + ev_ratio[1]) * 100, 1)

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-label">Stocks Analysed</div>
    <div class="metric-value">{n_stocks}</div>
    <div class="metric-unit">{data_label}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Trading Days</div>
    <div class="metric-value">{n_days}</div>
    <div class="metric-unit">{str(start_date)} → {str(end_date)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">PC1 Variance Explained</div>
    <div class="metric-value">{pc1_var}%</div>
    <div class="metric-unit">First principal component</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">PC2 Variance Explained</div>
    <div class="metric-value">{pc2_var}%</div>
    <div class="metric-unit">Second principal component</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Total Captured</div>
    <div class="metric-value">{total_var}%</div>
    <div class="metric-unit">PC1 + PC2 combined</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── 2D PCA Scatter Plot ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">2D PCA Scatter — Each dot is one stock</div>', unsafe_allow_html=True)

fig = go.Figure()

sectors_present = plot_df["Sector"].unique()
from pca_utils import SECTOR_COLORS

for sector in sorted(sectors_present):
    sdf = plot_df[plot_df["Sector"] == sector]
    color = SECTOR_COLORS.get(sector, "#9CA3AF")
    fig.add_trace(go.Scatter(
        x=sdf["PC1"],
        y=sdf["PC2"],
        mode="markers+text" if show_labels else "markers",
        name=sector,
        text=sdf["Ticker"] if show_labels else None,
        textposition="top center",
        textfont=dict(family="DM Mono, monospace", size=9, color="#444440"),
        marker=dict(
            color=color,
            size=11,
            opacity=0.82,
            line=dict(width=1.2, color="white"),
            symbol="circle",
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Sector: " + sector + "<br>"
            "PC1: %{x:.3f}<br>"
            "PC2: %{y:.3f}<br>"
            "<extra></extra>"
        ),
        customdata=sdf["Ticker"],
    ))

# Zero lines
fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#DDDDDD")
fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#DDDDDD")

fig.update_layout(
    xaxis_title=f"PC1  ({pc1_var}% variance explained)",
    yaxis_title=f"PC2  ({pc2_var}% variance explained)",
    font=dict(family="DM Sans, sans-serif", size=12, color="#333330"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    legend=dict(
        title=dict(text="Sector", font=dict(family="DM Mono, monospace", size=10)),
        font=dict(family="DM Sans, sans-serif", size=11),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E8E8E4",
        borderwidth=1,
        x=1.01, y=1,
    ),
    xaxis=dict(
        showgrid=True, gridcolor="#F0F0EC", gridwidth=1,
        zeroline=False,
        tickfont=dict(family="DM Mono, monospace", size=10),
        title_font=dict(family="DM Sans, sans-serif", size=12, color="#666660"),
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#F0F0EC", gridwidth=1,
        zeroline=False,
        tickfont=dict(family="DM Mono, monospace", size=10),
        title_font=dict(family="DM Sans, sans-serif", size=12, color="#666660"),
    ),
    margin=dict(l=60, r=180, t=30, b=60),
    height=560,
    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#E8E8E4",
        font=dict(family="DM Mono, monospace", size=11),
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Divider ────────────────────────────────────────────────────────────────────
st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

# ── Bottom: Explained Variance Bar + Top Loadings ─────────────────────────────
col_a, col_b = st.columns([1, 1], gap="large")

with col_a:
    st.markdown('<div class="section-label">Variance Explained per Component</div>', unsafe_allow_html=True)

    # Compute full PCA for scree-style bar
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _SS
    _X = _SS().fit_transform(returns.T.values)
    _n = min(num_pcs, _X.shape[1])
    _pca_full = _PCA(n_components=_n).fit(_X)
    _ev = _pca_full.explained_variance_ratio_ * 100

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(len(_ev))],
        y=_ev,
        marker=dict(
            color=["#111110" if i < 2 else "#DDDDDA" for i in range(len(_ev))],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x}</b><br>%{y:.2f}% variance<extra></extra>",
    ))
    bar_fig.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(len(_ev))],
        y=np.cumsum(_ev),
        mode="lines+markers",
        name="Cumulative",
        line=dict(color="#2563EB", width=2, dash="dot"),
        marker=dict(size=5, color="#2563EB"),
        yaxis="y2",
        hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
    ))
    bar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        showlegend=False,
        xaxis=dict(tickfont=dict(family="DM Mono, monospace", size=10), showgrid=False),
        yaxis=dict(title="% Variance", tickfont=dict(family="DM Mono, monospace", size=10),
                   showgrid=True, gridcolor="#F0F0EC"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    tickfont=dict(family="DM Mono, monospace", size=10),
                    showgrid=False, range=[0, 105]),
        margin=dict(l=50, r=60, t=20, b=40),
        height=300,
        font=dict(family="DM Sans, sans-serif", size=11),
        hoverlabel=dict(bgcolor="white", bordercolor="#E8E8E4",
                        font=dict(family="DM Mono, monospace", size=11)),
    )
    st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

with col_b:
    st.markdown(f'<div class="section-label">Top Stock Contributions (PC1 to PC{num_pcs})</div>', unsafe_allow_html=True)

    # Compute contributions: run PCA on original returns (rows=days, cols=stocks)
    # loadings = pca.components_.T  shape (n_stocks, n_components)
    from sklearn.decomposition import PCA as _PCA2
    from sklearn.preprocessing import StandardScaler as _SS2
    _Xr = _SS2().fit_transform(returns.values)   # (n_days, n_stocks)
    
    _n_comp = min(num_pcs, _Xr.shape[1])
    _pca2 = _PCA2(n_components=_n_comp).fit(_Xr)
    _load_matrix = _pca2.components_.T           # (n_stocks, _n_comp)
    cols = [f"PC{i+1}" for i in range(_n_comp)]
    load_df = pd.DataFrame(_load_matrix, index=returns.columns, columns=cols)

    # Gather top drivers from all PCs
    top_stocks = []
    # If N is large, take fewer top stocks per PC so it fits on screen nicely
    top_n_per_pc = max(2, int(10 / _n_comp)) 
    for c in cols:
        top_stocks.extend(load_df[c].abs().nlargest(top_n_per_pc).index.tolist())
    combined = list(dict.fromkeys(top_stocks))[:12]
    display_load = load_df.loc[combined, cols].round(4)

    load_fig = go.Figure()
    colors = ["#111110", "#BBBBBB", "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2", "#BE185D", "#EA580C"]
    
    for i, c in enumerate(cols):
        load_fig.add_trace(go.Bar(
            name=c,
            x=display_load.index,
            y=display_load[c],
            marker_color=colors[i % len(colors)],
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>" + c + " loading: %{y:.4f}<extra></extra>",
        ))
    load_fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        legend=dict(font=dict(family="DM Mono, monospace", size=10)),
        xaxis=dict(tickfont=dict(family="DM Mono, monospace", size=10), showgrid=False),
        yaxis=dict(title="Loading", tickfont=dict(family="DM Mono, monospace", size=10),
                   showgrid=True, gridcolor="#F0F0EC"),
        margin=dict(l=50, r=20, t=20, b=40),
        height=300,
        font=dict(family="DM Sans, sans-serif", size=11),
        hoverlabel=dict(bgcolor="white", bordercolor="#E8E8E4",
                        font=dict(family="DM Mono, monospace", size=11)),
    )
    st.plotly_chart(load_fig, use_container_width=True, config={"displayModeBar": False})


# ── Divider ────────────────────────────────────────────────────────────────────
st.markdown('<div class="thin-divider"></div>', unsafe_allow_html=True)

# ── Stock Data Table ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Stock PCA Coordinates</div>', unsafe_allow_html=True)

table_df = stock_pca[["Ticker", "Sector", "PC1", "PC2"]].copy()
table_df["PC1"] = table_df["PC1"].round(4)
table_df["PC2"] = table_df["PC2"].round(4)
if selected_sectors:
    table_df = table_df[table_df["Sector"].isin(selected_sectors)]
table_df = table_df.sort_values("PC1").reset_index(drop=True)

st.dataframe(
    table_df,
    use_container_width=True,
    height=260,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Sector": st.column_config.TextColumn("Sector", width="medium"),
        "PC1": st.column_config.NumberColumn("PC1 Score", format="%.4f"),
        "PC2": st.column_config.NumberColumn("PC2 Score", format="%.4f"),
    },
    hide_index=True,
)

# ── Download ───────────────────────────────────────────────────────────────────
csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇  Download PCA Results as CSV",
    data=csv_bytes,
    file_name="pca_stock_results.csv",
    mime="text/csv",
)