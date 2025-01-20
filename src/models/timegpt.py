import pandas as pd
from nixtla import NixtlaClient
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from callbacks import CallbackHandler


def preprocess_time_column(data, time_col):
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(by=time_col) #.drop_duplicates(subset=[time_col])
    data = data.set_index(time_col)

    # Generate uniform time index
    full_time_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='H')
    data = data.reindex(full_time_index)

    # Interpolates missing values
    if data.isnull().any().any():
        print("Missing values found. Interpolating...")
        data = data.interpolate(method='time')

    # Reset index to have time as a column
    data = data.reset_index().rename(columns={'index': time_col})
    return data


def train_timegpt_all_features(train_data, test_data, api_key, fine_tune=False, preprocess=True, h_len = 24):

    client = NixtlaClient(api_key=api_key)
    
    metrics = []
    metrics_callback = CallbackHandler()

    timestamp_col = None
    for col in train_data.columns:
        if col.lower() in "dateandtime":
            timestamp_col = col
            break
    
    if not timestamp_col:
        raise ValueError("Colonna dei timestamp non trovata nel dataset.")

    if preprocess:
        train_data = preprocess_time_column(train_data, time_col=timestamp_col)
        test_data = preprocess_time_column(test_data, time_col=timestamp_col)

    for feature in train_data.columns:
        if feature == timestamp_col:
            continue 

        train_series = train_data[[timestamp_col, feature]].rename(columns={timestamp_col: "ds", feature: "y"})
        test_series = test_data[[timestamp_col, feature]].rename(columns={timestamp_col: "ds", feature: "y"})

        # print(len(test_series), len(train_series))

        forecast_args = {
            "df": train_series,
            "h": max(h_len, len(test_series)),
        }

        if fine_tune:
            forecast_args["finetune_steps"] = 5 
            forecast_args["finetune_depth"] = 2 

        metrics_callback.start()
        forecast = client.forecast(**forecast_args)
        metrics_callback.stop()
        test_efficency_metric = metrics_callback.collect(key = 'test')

        
        # print(f"Forecast for feature {feature}:")
        # print(forecast.head())

        predicted_col = "y" if "y" in forecast.columns else forecast.columns[-1]

        mae = mean_absolute_error(test_series["y"], forecast[predicted_col])
        rmse = np.sqrt(mean_squared_error(test_series["y"], forecast[predicted_col]))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse,  **test_efficency_metric})

    white_list = [ 'MAE', 'RMSE'] + list(test_efficency_metric.keys())
    metrics_df = pd.DataFrame(metrics)
    overall = {c: metrics_df[c].mean() for c in white_list}

    return metrics_df, overall

