from xgboost import XGBRegressor
from src.feature_engineering import prepare_features


def train_xgboost(df):
    """
    Trains an XGBoost regression model using time-aware train-test split.

    Steps:
    - Apply feature engineering
    - Create supervised dataset
    - Perform chronological split (no shuffling)
    - Train XGBRegressor
    """

    # Apply feature engineering (lag creation, etc.)
    df = prepare_features(df)

    # Define features and target
    X = df.drop(columns=["ds", "y"])
    y = df["y"]

    # Time-series split (80% train, 20% test)
    train_size = int(len(df) * 0.8)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    # Initialize model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        objective="reg:squarederror"
    )

    # Train model
    model.fit(X_train, y_train)

    return model, X_test, y_test
