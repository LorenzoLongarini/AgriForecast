import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tqdm import tqdm 
import torch
# from pyESN import ESN  # Assicurati di installare questa libreriaù
import sys
sys.path.append("./src/models")
from callbacks import CallbackHandler
from torchdyno.models.esn import EchoStateNetwork as ESN


def train_esn_all_features(train_series, test_series, n_reservoir=100, sparsity=0.2, spectral_radius=0.95):

    metrics = []
    metrics_callback = CallbackHandler()
    for feature in tqdm(train_series.columns[:2], desc="Training ESN per le feature"):
        

        train_input = train_series[feature].values.astype(np.float32)[: -1]
        train_output = train_series[feature].values.astype(np.float32)[1:]
        test_input = test_series[feature].values.astype(np.float32)[: -1]
        test_output = test_series[feature].values.astype(np.float32)[1:]
        
        train_input = torch.from_numpy(train_input).to(torch.float32).reshape(-1, 1, 1)
        train_output = torch.from_numpy(train_output).to(torch.float32).reshape(-1, 1, 1)
        test_input_torch = torch.from_numpy(test_input).to(torch.float32).reshape(-1, 1, 1)
        # test_output_torch = torch.from_numpy(test_output).to(torch.float32).reshape(-1, 1, 1)

        train_set = torch.utils.data.TensorDataset(train_input, train_output)
        print(train_input.shape, train_output.shape, test_input.shape, test_output.shape)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        esn = ESN(
            input_size=train_input.shape[-1], 
            output_size=train_input.shape[-1], 
            layer_sizes=[n_reservoir]
        )
            # sparsity=sparsity, 
            # spectral_radius=spectral_radius)
        
        metrics_callback.start()
        esn.fit_readout(train_loader, l2_value=0.1)
        metrics_callback.stop()
        train_efficency_metric = metrics_callback.collect(key = 'train')


        metrics_callback.start()
        predictions = esn(test_input_torch).detach().numpy().flatten() #.evaluate(test_loader)
        metrics_callback.stop()
        test_efficency_metric = metrics_callback.collect(key = 'test')


        mae = mean_absolute_error(test_output, predictions)
        rmse = np.sqrt(mean_squared_error(test_output, predictions))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse, **train_efficency_metric, **test_efficency_metric})


    white_list = [ 'MAE', 'RMSE'] + list(train_efficency_metric.keys()) + list(test_efficency_metric.keys())

    metrics_df = pd.DataFrame(metrics)
    overall = {c: metrics_df[c].mean() for c in white_list}
    # overall_mae = metrics_df['MAE'].mean()
    # overall_rmse = metrics_df['RMSE'].mean()


    return metrics_df, overall
