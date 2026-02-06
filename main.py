from src.data_generation import generate_time_series
from src.prophet_model import train_prophet
from src.baseline_model import train_xgboost
from src.evaluation import evaluate
from src.shap_analysis import run_shap
import pandas as pd

# Generate Data
df = generate_time_series()

# Prophet
prophet_model, best_params, best_rmse = train_prophet(df)

print("Best Prophet Params:", best_params)
print("Best Prophet CV RMSE:", best_rmse)

future = prophet_model.make_future_dataframe(periods=365)

# Merge historical regressors
future = future.merge(
    df[["ds", "price_index", "marketing_spend"]],
    on="ds",
    how="left"
)

# Forward fill for future periods
future["price_index"] = future["price_index"].ffill()
future["marketing_spend"] = future["marketing_spend"].ffill()



forecast = prophet_model.predict(future)
prophet_model.plot(forecast)
prophet_model.plot_components(forecast)


# Baseline
xgb_model, X_test, y_test = train_xgboost(df)

y_pred = xgb_model.predict(X_test)

metrics = evaluate(y_test, y_pred)
print("XGBoost Metrics:", metrics)

# SHAP
run_shap(xgb_model, X_test)
