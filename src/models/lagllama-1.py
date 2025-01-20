import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.pandas import PandasDataset

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.common import ListDataset
# pip install lag-llama "git+https://github.com/time-series-foundation-models/lag-llama.git@update-gluonts"
# huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir ./path/to/save/directory

def df_to_pandas_dataset(df, date_col=None, target_col=None, freq="1H"):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.asfreq(freq)

    # Interpolation
    if df.isnull().values.any():
        print(f"[WARNING] Missing data detected in '{target_col}'. Filling with interpolation.")
        df = df.interpolate(method="time")
    # GluonTS ListDataset
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

    # Carichiamo il checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    # Selezioniamo le feature numeriche (escludendo eventuali date)
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns

    all_metrics = []

    for feature in tqdm(numeric_cols, desc="Iterating over numeric features"):

        train_dataset = df_to_pandas_dataset(
            train_data,
            date_col=date_col,
            target_col=feature,
            freq=freq,
            # item_id=f"{feature}"
        )
        test_dataset = df_to_pandas_dataset(
            test_data,
            date_col=date_col,
            target_col=feature,
            freq=freq,
            # item_id=f"{feature}"
        )

        # Initialize the estimator
        estimator = LagLlamaEstimator(
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

            # Decomment to use rope scaling
            # rope_scaling={
            #     "type": "linear",
            #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
            # },

            trainer_kwargs={"max_epochs": max_epochs} if fine_tune else None
        )

        if fine_tune:
            predictor = estimator.train(train_dataset, cache_data=True, shuffle_buffer_length=1000)
        else:
            lightning_module = estimator.create_lightning_module()
            transformation = estimator.create_transformation()
            predictor = estimator.create_predictor(transformation, lightning_module)

        # Prediction
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset,
            predictor=predictor,
            num_samples=20
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)

        if len(forecasts) == 0:
            print(f"[WARNING] Nessun forecast per feature '{feature}'")
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

        print(f"MAE for feature '{feature}': {mae}")
        print(f"RMSE for feature '{feature}': {rmse}")

        all_metrics.append({
            "Feature": feature,
            "MAE": mae,
            "RMSE": rmse
        })

    metrics_df = pd.DataFrame(all_metrics)
    overall_mae = metrics_df["MAE"].mean() if not metrics_df.empty else None
    overall_rmse = metrics_df["RMSE"].mean() if not metrics_df.empty else None

    return metrics_df, overall_mae, overall_rmse
