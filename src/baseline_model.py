import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def create_lag_features(df, lags=[1,7,30]):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    return df

def train_xgboost(df):
    df_lag = create_lag_features(df)
    
    features = [col for col in df_lag.columns if col not in ["ds", "y"]]
    
    X = df_lag[features]
    y = df_lag["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )
    
    model = XGBRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test
