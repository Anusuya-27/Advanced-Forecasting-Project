import pandas as pd

def create_lag_features(df, lags=[1, 7]):
    df = df.copy()

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    return df


def prepare_features(df):
    df = create_lag_features(df)

    # Drop rows with NaN values created by lagging
    df = df.dropna().reset_index(drop=True)

    return df
