from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def train_sarima_all_features(train_data, test_data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 24)):

    metrics = []
    for feature in train_data.columns:
        train_series = train_data[feature]
        test_series = test_data[feature]
        
        model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        predictions = model_fit.forecast(steps=len(test_series))
        
        mae = mean_absolute_error(test_series, predictions)
        rmse = np.sqrt(mean_squared_error(test_series, predictions))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse})
    
    metrics_df = pd.DataFrame(metrics)
    overall_mae = metrics_df['MAE'].mean()
    return metrics_df, overall_mae

