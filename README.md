# 📈 PCA Stock Market Lens

A Streamlit application that applies **Principal Component Analysis (PCA)** to S&P 500 stock returns, compressing hundreds of correlated stocks into 2 dimensions to reveal sector clustering and hidden market structure.

---

## Research Foundation

**Paper:** Ghorbani, M. & Chong, E.K.P. (2020). *Stock price prediction using principal components.* PLOS ONE.
🔗 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7083277/

**Dataset:** S&P 500 Stock Data — Cam Nugent, Kaggle
🔗 https://www.kaggle.com/datasets/camnugent/sandp500

---

## Features

- **2D PCA Scatter Plot** — Each stock as a point in principal component space, colored by sector
- **Variance Explained Bar Chart** — Scree-style chart with cumulative variance overlay
- **Top Loadings Chart** — Which stocks drive PC1 and PC2 the most
- **Interactive Filters** — Filter by date range, sector, number of stocks
- **Download Results** — Export PCA coordinates as CSV
- **Sample Data Mode** — Works out of the box without uploading any file

---

## Setup

### 1. Clone / copy the project
```bash
cd pca_stock_app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. (Optional) Upload Kaggle Data
- Download `all_stocks_5yr.csv` from the Kaggle link above
- In the sidebar, select **"Upload Kaggle CSV"** and upload the file

---

## Project Structure

```
pca_stock_app/
│
├── app.py              # Main Streamlit application
├── pca_utils.py        # PCA logic: preprocessing, fitting, loadings
├── requirements.txt    # Python dependencies
└── README.md
```

---

## How PCA Works Here

1. **Daily Returns Matrix** — For each stock, compute daily % change in closing price
2. **Standardize** — Zero mean, unit variance using `StandardScaler`
3. **Transpose** — Rows = stocks, Columns = days (stocks become observations)
4. **PCA** — Extract PC1 and PC2 (the two directions of maximum variance)
5. **Plot** — Each stock is a point at `(PC1_score, PC2_score)`

Stocks that move together in similar patterns cluster near each other on the plot — typically revealing **sector groupings** (Tech, Energy, Financials, etc.) without any labels.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Streamlit |
| ML | scikit-learn (PCA, StandardScaler) |
| Data | pandas, numpy |
| Charts | Plotly |
| Fonts | DM Serif Display, DM Mono, DM Sans |# PCA-Lab
# PCA-Lab
