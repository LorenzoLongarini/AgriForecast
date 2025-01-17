import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.pandas import PandasDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

def train_lag_llama_all_features(
    train_data,
    test_data,
    model_checkpoint="lag-llama.ckpt",
    device="cuda",
    prediction_length=24,
    fine_tune=False,
    num_epochs=3,
    batch_size=8
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Caricamento del modello e del tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model.to(device)

    metrics = []
    for feature in tqdm(train_data.columns, desc="Processing features"):
        train_series = train_data[feature].tolist()
        test_series = test_data[feature].tolist()

        if fine_tune:
            # Prepara i dati per il fine-tuning
            training_texts = " ".join(map(str, train_series))
            inputs = tokenizer(training_texts, return_tensors="pt", max_length=512, truncation=True).to(device)
            labels = inputs["input_ids"].clone()

            # Configura il Trainer per il fine-tuning
            training_args = TrainingArguments(
                output_dir="./results",
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                save_steps=10_000,
                save_total_limit=2,
                logging_dir="./logs",
                logging_steps=500,
                do_train=True,
                do_eval=False,
                disable_tqdm=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=inputs["input_ids"],
            )
            trainer.train()

        # Previsione zero-shot o post-fine-tuning
        inputs = tokenizer(" ".join(map(str, train_series)), return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(inputs["input_ids"], max_new_tokens=len(test_series))
        
        # Estrai le previsioni
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split()
        predictions = list(map(float, predictions[-len(test_series):]))

        # Calcolo delle metriche
        mae = mean_absolute_error(test_series, predictions)
        rmse = np.sqrt(mean_squared_error(test_series, predictions))
        metrics.append({'Feature': feature, 'MAE': mae, 'RMSE': rmse})

    # Creazione del DataFrame delle metriche
    metrics_df = pd.DataFrame(metrics)
    overall_mae = metrics_df['MAE'].mean()

    return metrics_df, overall_mae