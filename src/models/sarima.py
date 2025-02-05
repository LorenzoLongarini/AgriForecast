from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from callbacks import CallbackHandler


def train_sarima_all_features(train_data, test_data, order=(2, 1, 2), seasonal_order=(0, 1, 1, 24)):

    metrics = []
    metrics_callback = CallbackHandler()

    for feature in train_data.columns:
        train_series = train_data[feature]
        test_series = test_data[feature]
        
        model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order)

        metrics_callback.start()
        model_fit = model.fit(disp=False)
        metrics_callback.stop()
        train_efficency_metric = metrics_callback.collect(key = 'train')

        metrics_callback.start()
        predictions = model_fit.forecast(steps=len(test_series))
        metrics_callback.stop()
        test_efficency_metric = metrics_callback.collect(key = 'test')
        
        mae = mean_absolute_error(test_series, predictions)
        rmse = np.sqrt(mean_squared_error(test_series, predictions))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse, **train_efficency_metric, **test_efficency_metric})

    
    white_list = [ 'MAE', 'RMSE'] + list(train_efficency_metric.keys()) + list(test_efficency_metric.keys())
    metrics_df = pd.DataFrame(metrics)
    overall = {c: metrics_df[c].mean() for c in white_list}

    return metrics_df, overall

