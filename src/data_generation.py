import pandas as pd
import numpy as np

def generate_time_series():
    np.random.seed(42)
    
    dates = pd.date_range(start="2018-01-01", end="2021-12-31", freq="D")
    n = len(dates)
    
    # Trend
    trend = 0.05 * np.arange(n)
    
    # Yearly seasonality
    yearly = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
    
    # Weekly seasonality
    weekly = 3 * np.sin(2 * np.pi * np.arange(n) / 7)
    
    # External Regressor 1 (Price Index)
    price_index = 50 + 0.01*np.arange(n) + np.random.normal(0, 2, n)
    
    # External Regressor 2 (Marketing Spend)
    marketing_spend = 200 + 20*np.sin(2 * np.pi * np.arange(n) / 30) + np.random.normal(0, 5, n)
    
    noise = np.random.normal(0, 2, n)
    
    y = 100 + trend + yearly + weekly - 0.5*price_index + 0.3*marketing_spend + noise
    
    df = pd.DataFrame({
        "ds": dates,
        "y": y,
        "price_index": price_index,
        "marketing_spend": marketing_spend
    })
    
    return df
