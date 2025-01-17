import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tqdm import tqdm 
from pyESN import ESN  # Assicurati di installare questa libreria

def train_esn_all_features(train_series, test_series, n_reservoir=100, sparsity=0.2, spectral_radius=0.95):

    metrics = []
    for feature in tqdm(train_series.columns, desc="Training ESN per le feature"):
        
        train_input = train_series[:-1].values.reshape(-1, 1)
        train_output = train_series[1:].values.reshape(-1, 1)
        test_input = test_series[:-1].values.reshape(-1, 1)
        test_output = test_series[1:].values.reshape(-1, 1)
        
        esn = ESN(
            n_inputs=1, 
            n_outputs=1, 
            n_reservoir=n_reservoir, 
            sparsity=sparsity, 
            spectral_radius=spectral_radius)
        
        esn.fit(train_input, train_output)
        predictions = esn.predict(test_input)
        
        mae = mean_absolute_error(test_output, predictions)
        rmse = np.sqrt(mean_squared_error(test_output, predictions))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse})
    
    metrics_df = pd.DataFrame(metrics)
    overall_mae = metrics_df['MAE'].mean()


    return metrics_df, overall_mae
