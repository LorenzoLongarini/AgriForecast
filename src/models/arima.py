from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def train_arima_all_features(train_data, test_data, order=(1, 1, 1)):

    metrics = []
    for feature in train_data.columns:
        train_series = train_data[feature]
        test_series = test_data[feature]
        
        model = ARIMA(train_series, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test_series))
        
        mae = mean_absolute_error(test_series, predictions)
        rmse = np.sqrt(mean_squared_error(test_series, predictions))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse})
    
    metrics_df = pd.DataFrame(metrics)
    overall_mae = metrics_df['MAE'].mean()
    return metrics_df, overall_mae

