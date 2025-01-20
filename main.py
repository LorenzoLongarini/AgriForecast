import sys
import os
import json
from dotenv import load_dotenv
from glob import glob
sys.path.append("./src/dataloader")
sys.path.append("./src/models")
from data_preparation import prepare_data
from arima import train_arima_all_features
from timegpt import train_timegpt_all_features
from sarima import train_sarima_all_features
from lstm import train_lstm_all_features
from esn import train_esn_all_features
from lagllama import train_lag_llama_all_features
import numpy as np


def save_results(file_name, metrics, overall):
    results_dir = f"./assets/results/{file_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Convertiamo eventuali np.float32 in float per JSON
    results = {
        "overall": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in overall.items()},
        # "rmse_overall": float(overall_rmse) if overall_rmse is not None else None,
        "features": [
            {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in feature.items()}
            for feature in metrics
        ]
    }
    
    with open(f"{results_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)


def analyze_results(results_folder):
    mse_values = []
    rmse_values = []

    for root, _, files in os.walk(results_folder):
        for file in files:
            if file.endswith("results.json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    mse_values.append(data.get("mae_overall", float("inf")))
                    rmse_values.append(data.get("rmse_overall", float("inf")))

    best_mse = min(mse_values) if mse_values else None
    worst_mse = max(mse_values) if mse_values else None
    avg_mse = sum(mse_values) / len(mse_values) if mse_values else None

    best_rmse = min(rmse_values) if rmse_values else None
    worst_rmse = max(rmse_values) if rmse_values else None
    avg_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else None
    results = {
        "best_mae": best_mse,
        "worst_mae": worst_mse,
        "avg_mae": avg_mse,
        "best_rmse": best_rmse,
        "worst_rmse": worst_rmse,
        "avg_rmse": avg_rmse
    }
    with open(f"{results_folder}/tot_results.json", "w") as f:
        json.dump(results,f, indent=4)
    return results

if __name__ == "__main__":
    # file_path = ".//assets//datimerged//A84041A9018859AC_merged.csv"
    data_files = "./assets/datimerged/"
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    

    # for file_name in os.listdir(data_files):
    #     file_path = os.path.join(data_files, file_name)
    #     base_file_name = os.path.splitext(file_name)[0]
    #     train_data, test_data = prepare_data(file_path, keep_dateandtime=False)

    #     print("Testing ARIMA...")
    #     model_results_dir = f"ARIMA/{base_file_name}"
    #     arima_metrics, arima_overall_mae, arima_overall_rmse = train_arima_all_features(train_data, test_data)
    #     save_results(model_results_dir, arima_metrics.to_dict(orient='records'), arima_overall_mae, arima_overall_rmse)
    
    
    # for file_name in os.listdir(data_files):
    #     file_path = os.path.join(data_files, file_name)
    #     base_file_name = os.path.splitext(file_name)[0]
    #     train_data, test_data = prepare_data(file_path, keep_dateandtime=False)

    #     print("Testing SARIMA...")
    #     model_results_dir = f"SARIMA/{base_file_name}"
    #     sarima_metrics, sarima_overall_mae, sarima_overall_rmse = train_sarima_all_features(train_data, test_data)
    #     save_results(model_results_dir, sarima_metrics.to_dict(orient='records'), sarima_overall_mae, sarima_overall_rmse)
    

    # for file_name in os.listdir(data_files):
    #     file_path = os.path.join(data_files, file_name)
    #     base_file_name = os.path.splitext(file_name)[0]
    #     train_data, test_data = prepare_data(file_path, keep_dateandtime=False)

    #     print("Testing LSTM...")
    #     model_results_dir = f"LSTM/{base_file_name}"
    #     lstm_metrics, lstm_overall_mae, lstm_overall_rmse = train_lstm_all_features(train_data, test_data)
    #     save_results(model_results_dir, lstm_metrics.to_dict(orient='records'), lstm_overall_mae, lstm_overall_rmse)
    

    for file_name in os.listdir(data_files):
        file_path = os.path.join(data_files, file_name)
        base_file_name = os.path.splitext(file_name)[0]
        train_data, test_data = prepare_data(file_path, keep_dateandtime=False)

        print("Testing ESN...")
        model_results_dir = f"ESN/{base_file_name}"
        esn_metrics, esn_overall = train_esn_all_features(train_data, test_data)
        save_results(model_results_dir, esn_metrics.to_dict(orient='records'), esn_overall)
    

    # for file_name in os.listdir(data_files):
    #     file_path = os.path.join(data_files, file_name)
    #     base_file_name = os.path.splitext(file_name)[0]
    #     train_data_llm, test_data_llm = prepare_data(file_path, keep_dateandtime=True)

    #     print("Testing TimeGPT...")
    #     model_results_dir = f"TIMEGPT/{base_file_name}"
    #     timegpt_metrics, timegpt_overall_mae, timegpt_overall_rmse = train_timegpt_all_features(train_data_llm, test_data_llm, API_KEY, fine_tune=True, preprocess=True)
    #     save_results(model_results_dir, timegpt_metrics.to_dict(orient='records'), timegpt_overall_mae, timegpt_overall_rmse)
    

    # for file_name in os.listdir(data_files):
    #     file_path = os.path.join(data_files, file_name)
    #     base_file_name = os.path.splitext(file_name)[0]
    #     train_data_llm, test_data_llm = prepare_data(file_path, keep_dateandtime=True)

    #     print("Testing Lag-Llama...")
    #     model_name = "lag-llama.ckpt"  # Sostituisci con un modello valido di Hugging Face
    #     model_results_dir = f"LAGLLAMA/{base_file_name}"
    #     lag_llama_metrics, lag_llama_overall_mae, lag_llama_overall_rmse = train_lag_llama_all_features(train_data_llm, test_data_llm, model_name, fine_tune=True)
    #     save_results(model_results_dir, lag_llama_metrics.to_dict(orient='records'), lag_llama_overall_mae, lag_llama_overall_rmse)



    results_path = "./assets/results"
    for file_name in os.listdir(results_path):
        if file_name not in '.gitkeep':
            file_path = os.path.join(results_path, file_name)
            results_analysis = analyze_results(file_path)
            print("Results Analysis:")
            print(json.dumps(results_analysis, indent=4))