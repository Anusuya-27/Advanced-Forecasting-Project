import shap
import matplotlib.pyplot as plt

def run_shap(model, X_sample):
    
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    
    shap.summary_plot(shap_values, X_sample)
    
    return shap_values
