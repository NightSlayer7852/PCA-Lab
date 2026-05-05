import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
import warnings

# Suppress sklearn warnings for cleaner Streamlit output
warnings.filterwarnings('ignore')

def train_sector_model(stock_pca_df):
    """
    Predict stock sector based on PCA coordinates.
    Input: DataFrame with PC1, PC2, ..., and Sector labels.
    """
    # Filter out Unknown sectors
    df = stock_pca_df[stock_pca_df["Sector"] != "Unknown"].dropna()
    if len(df) < 10:
        return None, 0.0
        
    # Use available PCs
    pc_cols = [c for c in df.columns if c.startswith("PC")]
    X = df[pc_cols]
    y = df["Sector"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return clf, acc, pc_cols, clf.feature_importances_


def create_features_for_stock(returns_df, ticker):
    """
    Create rolling and lag features for a specific stock.
    Returns: DataFrame with features and target columns.
    """
    df = pd.DataFrame({"Return": returns_df[ticker]})
    
    # Features
    df["Lag_1"] = df["Return"].shift(1)
    df["Lag_2"] = df["Return"].shift(2)
    df["Lag_3"] = df["Return"].shift(3)
    df["Rolling_Mean_5"] = df["Return"].rolling(5).mean()
    df["Rolling_Std_5"] = df["Return"].rolling(5).std()
    
    # Market features (Average of all stocks)
    market_avg = returns_df.mean(axis=1)
    df["Market_Lag_1"] = market_avg.shift(1)
    
    # Targets
    df["Target_Movement"] = (df["Return"].shift(-1) > 0).astype(int) # 1 if up tomorrow, 0 if down
    df["Target_Vol_5d"] = df["Return"].shift(-5).rolling(5).std() # Volatility over next 5 days
    
    df = df.dropna()
    return df


def train_stock_models(returns_df, ticker):
    """
    Train Movement (Classification) and Volatility (Regression) models for a stock.
    Returns dictionaries with models and metrics.
    """
    df = create_features_for_stock(returns_df, ticker)
    
    if len(df) < 50:
        return None, None
        
    feature_cols = ["Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_5", "Rolling_Std_5", "Market_Lag_1"]
    
    # --- Movement Prediction (Classification) ---
    X_move = df[feature_cols]
    y_move = df["Target_Movement"]
    Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_move, y_move, test_size=0.2, shuffle=False)
    
    clf_move = LogisticRegression(class_weight="balanced")
    clf_move.fit(Xm_train, ym_train)
    
    acc_move = accuracy_score(ym_test, clf_move.predict(Xm_test))
    
    # Extract the last row to make a future prediction
    latest_features = df[feature_cols].iloc[-1:]
    pred_move = clf_move.predict(latest_features)[0]
    pred_move_prob = clf_move.predict_proba(latest_features)[0][1]
    
    movement_res = {
        "model": clf_move,
        "accuracy": acc_move,
        "prediction": "Up 📈" if pred_move == 1 else "Down 📉",
        "probability": pred_move_prob,
        "feature_names": feature_cols,
        "coefficients": clf_move.coef_[0]
    }
    
    # --- Volatility Prediction (Regression) ---
    # Need to drop na for Vol target specifically
    df_vol = df.dropna(subset=["Target_Vol_5d"])
    if len(df_vol) > 50:
        X_vol = df_vol[feature_cols]
        y_vol = df_vol["Target_Vol_5d"]
        Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_vol, y_vol, test_size=0.2, shuffle=False)
        
        reg_vol = LinearRegression()
        reg_vol.fit(Xv_train, yv_train)
        
        rmse_vol = np.sqrt(mean_squared_error(yv_test, reg_vol.predict(Xv_test)))
        
        # Predict future vol (using latest feature row, even if target was NaN for it)
        pred_vol = reg_vol.predict(latest_features)[0]
        
        # Prevent negative volatility prediction (Linear regression artifact)
        pred_vol = max(0.001, pred_vol)
        
        volatility_res = {
            "model": reg_vol,
            "rmse": rmse_vol,
            "prediction": pred_vol,
            "baseline": df["Rolling_Std_5"].iloc[-1], # Current 5d vol for comparison
            "test_dates": yv_test.index,
            "test_actual": yv_test.values,
            "test_pred": reg_vol.predict(Xv_test)
        }
    else:
        volatility_res = None
        
    return movement_res, volatility_res
