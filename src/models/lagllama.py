import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.pandas import PandasDataset

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset

from callbacks import CallbackHandler


# Se il tuo repo è locale o specifico, fai la pip install da git, per esempio:
# pip install lag-llama "git+https://github.com/time-series-foundation-models/lag-llama.git@update-gluonts"
# huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./path/to/save/directory


class PartialFineTuneLagLlamaEstimator(LagLlamaEstimator):
    """
    Sottoclasse di LagLlamaEstimator che congela tutti i layer
    tranne gli ultimi 2.
    """
    def create_lightning_module(self):
        lightning_module = super().create_lightning_module()
        model = lightning_module.model 

        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        for layer in model.transformer.h[-1:]:
            for param in layer.parameters():
                param.requires_grad = True

        return lightning_module


def df_to_pandas_dataset(df, date_col=None, target_col=None, freq="1H"):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.asfreq(freq)

    # Interpolation 
    if df.isnull().values.any():
        print(f"[WARNING] Missing data in '{target_col}'. Filling with interpolation.")
        df = df.interpolate(method="time")

    # Create GluonTS ListDataset
    dataset = ListDataset(
        [{"start": df.index[0], "target": df[target_col].values}],
        freq=freq
    )
    return dataset


def train_lag_llama_all_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    ckpt_path="lag-llama.ckpt",
    date_col="dateandtime",
    device="cuda",
    prediction_length=24,
    context_length=32,
    freq="1H",
    fine_tune=False,
    max_epochs=10
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Carica il checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    # Selezioniamo le feature numeriche
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns

    all_metrics = []
    metrics_callback = CallbackHandler()

    for feature in tqdm(numeric_cols, desc="Iterating over numeric features"):

        # Crea i dataset per la singola feature
        train_dataset = df_to_pandas_dataset(
            train_data,
            date_col=date_col,
            target_col=feature,
            freq=freq
        )
        test_dataset = df_to_pandas_dataset(
            test_data,
            date_col=date_col,
            target_col=feature,
            freq=freq
        )

        # Se fine_tune = True -> uso la classe che congela tutti i layer
        # tranne gli ultimi 2. Altrimenti, uso la classe standard.
        EstimatorClass = PartialFineTuneLagLlamaEstimator if fine_tune else LagLlamaEstimator

        # Inizializza l'estimator
        estimator = EstimatorClass(
            ckpt_path=ckpt_path,
            prediction_length=prediction_length,
            context_length=context_length,

            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],

            nonnegative_pred_samples=True,
            batch_size=64,
            num_parallel_samples=20,

            # Decommenta se vuoi usare rope scaling
            # rope_scaling={
            #     "type": "linear",
            #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            # },

            trainer_kwargs={"max_epochs": max_epochs} if fine_tune else None
        )

        # Se fine_tune=True, scatta l'allenamento con i layer parzialmente scongelati
        if fine_tune:
            metrics_callback.start()
            predictor = estimator.train(train_dataset, cache_data=True, shuffle_buffer_length=1000)
            metrics_callback.stop()
            train_efficency_metric = metrics_callback.collect(key = 'train')

        else:
            # Se non facciamo fine-tuning, creiamo direttamente il predictor dal modello base
            lightning_module = estimator.create_lightning_module()
            transformation = estimator.create_transformation()
            predictor = estimator.create_predictor(transformation, lightning_module)

        # Valutazione
        metrics_callback.start()
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset,
            predictor=predictor,
            num_samples=20
        )
        metrics_callback.stop()
        test_efficency_metric = metrics_callback.collect(key = 'test')

        forecasts = list(forecast_it)
        tss = list(ts_it)

        if len(forecasts) == 0:
            print(f"[WARNING] No forecast for feature '{feature}'")
            continue

        forecast = forecasts[0]
        ts = tss[0]

        predicted = forecast.samples.mean(axis=0)
        original = ts.values[-len(predicted):]

        if len(predicted) != len(original):
            print(f"[WARNING] Length mismatch for feature '{feature}'")
            continue

        mae = np.mean(np.abs(predicted - original))
        rmse = np.sqrt(np.mean((predicted - original) ** 2))
        all_metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse, **train_efficency_metric, **test_efficency_metric})


    # Raccolta delle metriche generali
    white_list = [ 'MAE', 'RMSE'] + list(train_efficency_metric.keys()) + list(test_efficency_metric.keys())
    metrics_df = pd.DataFrame(all_metrics)
    overall = {c: metrics_df[c].mean() for c in white_list}


    return metrics_df, overall