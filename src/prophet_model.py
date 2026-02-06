from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
import itertools

def train_prophet(df):
    
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 5.0, 10.0]
    }
    
    all_params = [dict(zip(param_grid.keys(), v)) 
                  for v in itertools.product(*param_grid.values())]
    
    best_rmse = float("inf")
    best_params = None
    best_model = None
    
    for params in all_params:
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            yearly_seasonality=True,
            weekly_seasonality=True
        )
        
        model.add_regressor("price_index")
        model.add_regressor("marketing_spend")
        
        model.fit(df)
        
        df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
        df_p = performance_metrics(df_cv)
        
        rmse = df_p['rmse'].mean()
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model
    
    return best_model, best_params, best_rmse
