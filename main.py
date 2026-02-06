from src.data_generation import generate_time_series
from src.prophet_model import train_prophet
from src.baseline_model import train_xgboost
from src.evaluation import evaluate
from src.shap_analysis import run_shap
import pandas as pd

# =========================
# 1. Generate Data
# =========================
df = generate_time_series()

# =========================
# 2. Train/Test Split (Time-aware)
# =========================
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# =========================
# 3. Prophet Model
# =========================
prophet_model, best_params, best_rmse = train_prophet(train_df)

print("Best Prophet Params:", best_params)
print("Best Prophet CV RMSE:", best_rmse)

# Create future dataframe for test period only
future = prophet_model.make_future_dataframe(periods=len(test_df))

# Merge regressors
future = future.merge(
    df[["ds", "price_index", "marketing_spend"]],
    on="ds",
    how="left"
)

# Forward-fill assumption for synthetic experiment
future[["price_index", "marketing_spend"]] = \
    future[["price_index", "marketing_spend"]].ffill()

forecast = prophet_model.predict(future)

# Extract Prophet test predictions
y_pred_prophet = forecast["yhat"].iloc[-len(test_df):].values
y_test_prophet = test_df["y"].values

# Evaluate Prophet properly
prophet_metrics = evaluate(y_test_prophet, y_pred_prophet)
print("Prophet Metrics:", prophet_metrics)

# Optional: Plot
prophet_model.plot(forecast)
prophet_model.plot_components(forecast)


# =========================
# 4. XGBoost Baseline
# =========================
xgb_model, X_test, y_test = train_xgboost(df)

y_pred_xgb = xgb_model.predict(X_test)

xgb_metrics = evaluate(y_test, y_pred_xgb)
print("XGBoost Metrics:", xgb_metrics)


# =========================
# 5. SHAP Analysis
# =========================
run_shap(xgb_model, X_test)
