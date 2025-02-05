import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tqdm import tqdm 
import torch
import sys
sys.path.append("./src/models")
from callbacks import CallbackHandler
from torchdyno.models.esn import EchoStateNetwork as ESN

def train_esn_all_features(train_series, test_series, n_reservoir=20, sparsity=0.2, spectral_radius=0.95, horizon=24):
    metrics = []
    metrics_callback = CallbackHandler()
    
    for feature in tqdm(train_series.columns, desc="Training ESN per le feature"):
        train_input = train_series[feature].values.astype(np.float32)[: -horizon].reshape(-1, 1, 1)
        test_input = test_series[feature].values.astype(np.float32)[: -horizon].reshape(-1, 1, 1)
        
        train_output = []
        test_output = []
        
        for i in range(horizon):
            start = i + 1
            end = - horizon + i + 1
            if end == 0:
                end = None
            train_output.append(train_series[feature].values.astype(np.float32)[start:end])
            test_output.append(test_series[feature].values.astype(np.float32)[start:end])
        
        train_output = np.stack(train_output).T.reshape(-1, 1, horizon)
        test_output =  np.stack(test_output).T.reshape(-1, 1, horizon)
        
        train_input = torch.from_numpy(train_input).to(torch.float32)
        train_output = torch.from_numpy(train_output).to(torch.float32)
        test_input_torch = torch.from_numpy(test_input).to(torch.float32)
        test_output_torch = torch.from_numpy(test_output).to(torch.float32)
        
        train_set = torch.utils.data.TensorDataset(train_input, train_output)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)
        
        esn = ESN(
            input_size=train_input.shape[-1], 
            output_size=train_output.shape[-1], 
            layer_sizes=[n_reservoir],
            bias=True,
        )
        
        metrics_callback.start()
        esn.fit_readout(train_loader, l2_value=0.1)
        metrics_callback.stop()
        train_efficency_metric = metrics_callback.collect(key='train')
        
        model_path = f"C:\\Users\\lollo\\Universita\\Tesi\\progetti\\AgriForecast\\models\\{feature}.pth"
        torch.save(esn.state_dict(), model_path)
        print(f"Model saved for feature: {feature}")
        
        # dummy_input = torch.randn(1, horizon, 1)
        # onnx_model_path = f"C:\\Users\\lollo\\Universita\\Tesi\\progetti\\AgriForecast\\models\\{feature}.onnx"
        # torch.onnx.export(
        #     esn, dummy_input, onnx_model_path, opset_version=11,
        #     input_names=["input"], output_names=["output"]
        # )
        
        metrics_callback.start()
        predictions = esn(test_input_torch).detach().numpy()
        metrics_callback.stop()
        test_efficency_metric = metrics_callback.collect(key='test')
        
        test_output = test_output[:, 0, :].reshape(-1, horizon)
        predictions = predictions[:, 0, :].reshape(-1, horizon)
        mae = mean_absolute_error(test_output, predictions)
        rmse = np.sqrt(mean_squared_error(test_output, predictions))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse, **train_efficency_metric, **test_efficency_metric})
    
    white_list = ['MAE', 'RMSE'] + list(train_efficency_metric.keys()) + list(test_efficency_metric.keys())
    metrics_df = pd.DataFrame(metrics)
    overall = {c: metrics_df[c].mean() for c in white_list}
    
    return metrics_df, overall
