import sys
import os
from dotenv import load_dotenv
sys.path.append("./src/dataloader")
sys.path.append("./src/models")
from data_preparation import prepare_data
from arima import train_arima_all_features
from timegpt import train_timegpt_all_features
from sarima import train_sarima_all_features
from lstm import train_lstm_all_features
from esn import train_esn_all_features
from lagllama import train_lag_llama_all_features

if __name__ == "__main__":
    file_path = ".//assets//datimerged//A84041A9018859AC_merged.csv"

    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    

    train_data, test_data = prepare_data(file_path, keep_dateandtime=False)

    # print(train_data.head())
    
    # print("Testing ARIMA...")
    # arima_metrics, arima_overall_mae = train_arima_all_features(train_data, test_data)
    # arima_metrics.to_csv("arima_metrics.csv", index=False)
    # print(f"ARIMA Overall MAE: {arima_overall_mae}")
    
    # print("Testing SARIMA...")
    # sarima_metrics, sarima_overall_mae = train_sarima_all_features(train_data, test_data)
    # sarima_metrics.to_csv("sarima_metrics.csv", index=False)
    # print(f"SARIMA Overall MAE: {sarima_overall_mae}")
    
    # print("Testing LSTM...")
    # lstm_metrics, lstm_overall_mae = train_lstm_all_features(train_data, test_data)
    # lstm_metrics.to_csv("lstm_metrics.csv", index=False)
    # print(f"LSTM Overall MAE: {lstm_overall_mae}")
    
    # print("Testing ESN...")
    # esn_metrics, esn_overall_mae = train_esn_all_features(train_data, test_data)
    # esn_metrics.to_csv("esn_metrics.csv", index=False)
    # print(f"ESN Overall MAE: {esn_overall_mae}")

    # print("Testing TimeGPT...")
    # timegpt_metrics, timegpt_overall_mae = train_timegpt_all_features(train_data, test_data, API_KEY, fine_tune=False, preprocess=True)
    # timegpt_metrics.to_csv("timegpt_metrics.csv", index=False)
    # print(f"TimeGPT Overall MAE: {timegpt_overall_mae}")
    
    print("Testing Lag-Llama...")
    model_name = "facebook/opt-125m"  # Sostituisci con un modello valido di Hugging Face
    lag_llama_metrics, lag_llama_overall_mae = train_lag_llama_all_features(train_data, test_data, model_name, fine_tune=False)
    lag_llama_metrics.to_csv("lag_llama_metrics.csv", index=False)
    print(f"Lag-Llama Overall MAE: {lag_llama_overall_mae}")