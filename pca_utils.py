import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# S&P 500 sector mapping (ticker -> sector)
SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "GOOG": "Technology",
    "META": "Technology", "NVDA": "Technology", "INTC": "Technology", "CSCO": "Technology",
    "IBM": "Technology", "ORCL": "Technology", "AMD": "Technology", "QCOM": "Technology",
    "TXN": "Technology", "ADBE": "Technology", "CRM": "Technology", "AVGO": "Technology",
    "HPQ": "Technology", "MU": "Technology",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
    "MS": "Financials", "C": "Financials", "AXP": "Financials", "BLK": "Financials",
    "USB": "Financials", "PNC": "Financials", "SCHW": "Financials", "COF": "Financials",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", "MRK": "Healthcare",
    "ABT": "Healthcare", "TMO": "Healthcare", "MDT": "Healthcare", "AMGN": "Healthcare",
    "GILD": "Healthcare", "BMY": "Healthcare", "BIIB": "Healthcare", "ISRG": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "PSX": "Energy", "VLO": "Energy", "OXY": "Energy",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "LOW": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "CL": "Consumer Staples",
    "GE": "Industrials", "BA": "Industrials", "CAT": "Industrials", "MMM": "Industrials",
    "UPS": "Industrials", "HON": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate", "SPG": "Real Estate",
    "LIN": "Materials", "APD": "Materials", "ECL": "Materials", "NEM": "Materials",
    "T": "Communication", "VZ": "Communication", "NFLX": "Communication", "DIS": "Communication",
}

SECTOR_COLORS = {
    "Technology": "#2563EB",
    "Financials": "#16A34A",
    "Healthcare": "#DC2626",
    "Energy": "#D97706",
    "Consumer Discretionary": "#7C3AED",
    "Consumer Staples": "#0891B2",
    "Industrials": "#BE185D",
    "Utilities": "#65A30D",
    "Real Estate": "#EA580C",
    "Materials": "#0D9488",
    "Communication": "#6366F1",
    "Unknown": "#9CA3AF",
}


def load_and_prepare(df: pd.DataFrame, date_col: str, ticker_col: str,
                     price_col: str, start_date, end_date,
                     min_stocks: int = 10) -> pd.DataFrame:
    """
    Pivot raw stock CSV into a wide returns matrix.
    Returns: DataFrame where rows=dates, cols=tickers, values=daily returns.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[(df[date_col] >= pd.to_datetime(start_date)) &
            (df[date_col] <= pd.to_datetime(end_date))]

    pivot = df.pivot_table(index=date_col, columns=ticker_col, values=price_col)
    pivot = pivot.sort_index()

    # Drop tickers with too many NaNs (keep those with >80% data)
    threshold = 0.8 * len(pivot)
    pivot = pivot.dropna(axis=1, thresh=int(threshold))

    # Compute daily returns
    returns = pivot.pct_change().dropna()

    # Drop columns still having any NaN
    returns = returns.dropna(axis=1)

    if returns.shape[1] < min_stocks:
        raise ValueError(f"Not enough stocks after cleaning. Found {returns.shape[1]}, need at least {min_stocks}.")

    return returns


def run_pca(returns: pd.DataFrame, n_components: int = 2):
    """
    Scale returns and apply PCA.
    Returns: (scores_df, pca_object, scaler_object, explained_variance_ratio)
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    scores_df = pd.DataFrame(
        scores,
        index=returns.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return scores_df, pca, scaler


def get_stock_pca_positions(returns: pd.DataFrame, n_components: int = 2):
    """
    Run PCA on the TRANSPOSED returns (stocks as observations, dates as features).
    Each stock becomes a point in PC space — for stock-level scatter plots.
    Returns: DataFrame with PC1, PC2, ticker, sector, color.
    """
    # Transpose: rows=stocks, cols=dates
    X = returns.T.values           # shape (n_stocks, n_days)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)   # still (n_stocks, n_days)

    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X_scaled)  # (n_stocks, 2)

    tickers = returns.columns.tolist()
    sectors = [SECTOR_MAP.get(t, "Unknown") for t in tickers]
    colors = [SECTOR_COLORS.get(s, "#9CA3AF") for s in sectors]

    result_dict = {
        "Ticker": tickers,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "Sector": sectors,
        "Color": colors,
    }
    if n_components >= 3:
        result_dict["PC3"] = coords[:, 2]

    result = pd.DataFrame(result_dict)

    ev = pca.explained_variance_ratio_
    return result, ev, pca


def get_loadings(pca, feature_names, n_top: int = 10):
    """
    Return loadings DataFrame.
    When PCA is run on transposed matrix (stocks=rows, dates=cols):
      pca.components_ shape = (n_components, n_days)  → not useful for stock labels.
    So we re-derive loadings via the scores: correlation of each stock's
    returns with the PC scores.  feature_names = stock tickers.
    This gives intuitive 'which stocks load heavily on PC1/PC2' info.
    """
    n_components, n_features = pca.components_.shape
    # components_.T => (n_features, n_components) = (n_days, n_components)
    # We just return what we have, labelled by index
    loadings = pd.DataFrame(
        pca.components_.T[:len(feature_names)],
        index=feature_names[:n_components],  # use component index labels
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    return loadings