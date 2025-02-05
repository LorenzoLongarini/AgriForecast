import pandas as pd
import json

# Carica il file CSV
csv_file_path = "C:\\Users\\lollo\\Universita\\Tesi\\progetti\\AgriForecast\\assets\\datimerged\\A84041C3C18859AE_merged.csv"
data = pd.read_csv(csv_file_path)

# Rimuovi le righe contenenti valori NaN
data = data.dropna()

# Ottieni le feature escludendo la colonna temporale
features = data.columns[1:].tolist()

# Ristruttura i dati per avere ogni chiave associata a una lista di valori
structured_features = {feature: data[feature].tolist() for feature in features}

# Crea il JSON per l'inserimento del sensore
sensor_json_with_lists = {
    "IdUtente": 1,  # Identificativo utente
    "Nome": "A84041C3C18859AE",  # Nome del sensore
    "Features": structured_features,  # Dati delle feature come liste
    "DataInstallazione": data.iloc[0]["dateandtime"]  # Timestamp iniziale
}

# Esporta il JSON in un file o stampalo
output_file_path = "sensor_data3.json"
with open(output_file_path, "w") as json_file:
    json.dump(sensor_json_with_lists, json_file, indent=4)

print(f"JSON generato e salvato in {output_file_path}")
