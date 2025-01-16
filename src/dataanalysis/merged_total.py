import pandas as pd
import os
from glob import glob

# Caricamento del file del report meteo
report_path = '.\\assets\\Report.xlsx'
report_df = pd.read_excel(report_path)

# Controllo e rinomina delle colonne per uniformarle al report
report_df.columns = [col.strip().lower() for col in report_df.columns]

# Uniformare la colonna della data e ora nel report
report_df['dateandtime'] = pd.to_datetime(report_df['dateandtime'], format='%d/%m/%Y %H:%M', errors='coerce')

# Percorso della cartella dei sensori
sensor_folder_path = 'C:\\Users\\lollo\\Universita\\Tesi\\dati-riassunti'
# Trova tutti i file CSV nella cartella
sensor_file_paths = glob(os.path.join(sensor_folder_path, '*.csv'))

# Inizializza un DataFrame vuoto per memorizzare tutti i dati uniti
merged_total_df = pd.DataFrame()

# Processa ogni file CSV nella cartella
for sensor_file_path in sensor_file_paths:
    # Estrarre la data dal nome del file
    file_name = os.path.splitext(os.path.basename(sensor_file_path))[0]
    file_date = pd.to_datetime(file_name, format='%Y%m%d', errors='coerce')

    # Caricare i dati del sensore
    sensor_df = pd.read_csv(sensor_file_path, sep=';')

    # Controllo e rinomina delle colonne per uniformarle al report
    sensor_df.columns = [col.strip().lower() for col in sensor_df.columns]

    # Uniformare la colonna dell'ora e creare la colonna dateandtime
    sensor_df['hour'] = pd.to_numeric(sensor_df['hour'], errors='coerce')
    sensor_df['dateandtime'] = file_date + pd.to_timedelta(sensor_df['hour'], unit='h')

    # Eseguire il join: replicare le righe del report in base alle corrispondenze con i sensori
    merged_df = pd.merge(report_df, sensor_df, on='dateandtime', how='inner')

    # Aggiungi i dati uniti al DataFrame totale
    merged_total_df = pd.concat([merged_total_df, merged_df], ignore_index=True)

# Percorso della cartella di output
output_folder_path = '.\\assets\\datimerged'
os.makedirs(output_folder_path, exist_ok=True)

# Raggruppa i dati per il tipo di sensore e salva un file per ciascun sensore
for sensor_id, group in merged_total_df.groupby('id'):
    output_file_path = os.path.join(output_folder_path, f'{sensor_id}_merged.csv')
    group.to_csv(output_file_path, index=False)
    print(f"Merged data for sensor {sensor_id} saved to {output_file_path}")