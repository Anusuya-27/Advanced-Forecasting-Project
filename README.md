Advanced Time Series Forecasting Project

1. Project Overview
This project implements a complete time series forecasting pipeline using:

    - Prophet (with external regressors and hyperparameter tuning)

    - XGBoost baseline model

    - SHAP for model interpretability

    - Cross-validation for robust model evaluation

The objective is to compare statistical forecasting (Prophet) with machine learning approaches (XGBoost) and evaluate performance using standard regression metrics.

2. Data Generation

A synthetic time series dataset was generated to simulate realistic forecasting scenarios.

The dataset includes:

    - A clear upward trend

    - Seasonality

    - Two external regressors

    - Random noise component

This setup allows testing:

    -  Whether Prophet effectively captures seasonality

    -  Whether machine learning models benefit from engineered lag features

    - How external regressors influence predictions

3. Feature Engineering

For the XGBoost model, the following time-series features were created:

    - Lag features (previous time steps)

    - Trend-related transformations

    - External regressors

These features allow XGBoost to learn temporal dependencies without explicitly modeling seasonality like Prophet.    

4. Models Implemented

4.1 Prophet Model

    - External regressors added

    - Hyperparameter tuning performed using cross-validation

    - Parameters tuned:

         - changepoint_prior_scale

        - seasonality_prior_scale

    - Best model selected based on lowest cross-validated RMSE

    - Final model re-trained using optimal hyperparameters

Best Hyperparameters

Best changepoint_prior_scale: 0.01  
Best seasonality_prior_scale: 1.0  
Best Cross-Validated RMSE: 2.076

- The low changepoint_prior_scale (0.01) indicates the model prefers a smoother, less flexible trend, reducing overfitting.
- The seasonality_prior_scale (1.0) suggests moderate seasonal influence.
- The cross-validated RMSE of 2.076 confirms strong generalization performance after tuning.


4.2 XGBoost Baseline

  - Trained using supervised learning approach

  - Time-series aware train-test split (shuffle=False)

  - Performance compared against Prophet

  - Feature importance analyzed using SHAP    

5. Cross-Validation Strategy

For Prophet:

    - Cross-validation performed using rolling forecast origin

    - Multiple parameter combinations evaluated

    - RMSE calculated for each fold

    - Mean RMSE used to determine best parameter combination

This ensures model selection is robust and avoids overfitting.

6. Model Evaluation Metrics

Models were evaluated using:

    - MAE (Mean Absolute Error)

    - RMSE (Root Mean Squared Error)

    - MAPE (Mean Absolute Percentage Error)

Model Comparison

| Model   | MAE  | RMSE | MAPE |
| ------- | ---- | ---- | ---- |
| Prophet | 1.66 | 2.08 | 0.86 |
| XGBoost | 4.28 | 5.27 | 2.23 |

Performance Analysis:

Prophet significantly outperforms XGBoost across all evaluation metrics.

RMSE reduced from 5.27 (XGBoost) to 2.08 (Prophet)

MAPE reduced from 2.23% to 0.86%

Cross-validated RMSE (2.06) closely matches test RMSE (2.08), indicating stable generalization

This suggests that the synthetic dataset’s structured trend and seasonality are better captured by Prophet’s additive modeling framework.
XGBoost performs reasonably well but depends heavily on engineered lag features and lacks explicit seasonal decomposition.


7. SHAP Analysis (Model Interpretability)

SHAP analysis shows that:

- Lag features have the highest contribution magnitude, confirming temporal dependency.
- Marketing_spend has a positive SHAP contribution, indicating higher marketing increases forecasted values.
- Price_index shows moderate impact with asymmetric effects across observations.
- Trend-related features dominate long-term forecast behavior.

This confirms that the model is leveraging both temporal memory and external regressors.


    

8. Key Findings

    - Prophet effectively captures seasonality and long-term trend.

    - XGBoost performs competitively when sufficient lag features are provided.

    - External regressors improve forecasting accuracy in both models.

    - Hyperparameter tuning significantly improves Prophet performance.

    -  SHAP provides transparency into model behavior.
    
    - Prophet significantly outperformed XGBoost on the synthetic dataset, achieving lower RMSE after hyperparameter tuning.

9. Project Structure

advanced-forecasting-project/
│
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── baseline_model.py
│   ├── prophet_model.py
│   ├── evaluation.py
│   └── shap_analysis.py


10. How to Run

    1.Install dependencies:

       pip install -r requirements.txt

    2.Run main script:
       python main.py

11. Conclusion

This project demonstrates a full forecasting workflow including:

        - Data simulation

        - Feature engineering

        - Statistical vs ML comparison

        - Cross-validation

        - Hyperparameter tuning

        - Model interpretability

It highlights the importance of both predictive performance and explainability in real-world forecasting systems.       



