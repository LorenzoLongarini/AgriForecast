import matplotlib
matplotlib.use('Agg')  # Utilizza il backend Agg per evitare problemi con Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import os
from glob import glob

# Funzione per generare boxplot multipli su feature selezionate
def generate_boxplots(data, output_dir):
    categories = {
        'vwc': ['vwc10', 'vwc20', 'vwc30', 'vwc40', 'vwc50', 'vwc60'],
        'salinity': ['salinity10', 'salinity20', 'salinity30', 'salinity40', 'salinity50', 'salinity60'],
        'temp': ['temp10', 'temp20', 'temp30', 'temp40', 'temp50', 'temp60']
    }
    
    titles = {
        'vwc': 'Volumetric Water Content',
        'salinity': 'Salinity',
        'temp': 'Temperature'
    }
    
    units = {
        'vwc': 'm³/m³',
        'salinity': 'dS/m',
        'temp': '°C'
    }
    
    for category, features in categories.items():
        # Determinare il numero di righe e colonne per il layout
        num_features = len(features)
        cols = 3  # Numero di colonne desiderato
        rows = math.ceil(num_features / cols)

        # Creare i subplot
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 5))
        axes = axes.flatten()  # Riorganizzare gli assi in una lista

        # Generare un boxplot per ogni feature selezionata
        palette = sns.color_palette("husl", num_features)
        for i, column in enumerate(features):
            if column in data.columns:
                sns.boxplot(data=data, y=column, ax=axes[i], palette=[palette[0]])
                axes[i].set_title(f'{titles[category]} at {column[-2:]} cm')
                axes[i].set_ylabel(units[category])
                axes[i].set_xlabel('')
            else:
                print(f"Column {column} not found in data")

        # Rimuovere i subplot vuoti (se esistenti)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Migliorare il layout
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{category}_boxplots.png'))  # Salva il grafico come immagine
        plt.close()

# Percorso della cartella locale
folder_path = 'C:\\Users\\lollo\\Universita\\Tesi\\datipersensore\\'
# Trova tutti i file CSV nella cartella
file_paths = glob(os.path.join(folder_path, '*.csv'))

# Processa ogni file CSV nella cartella
for file_path in file_paths:
    # Caricare i dati
    data = pd.read_csv(file_path, sep=';')
    print("Columns in data:", data.columns)  # Stampa i nomi delle colonne nel DataFrame
    print(data.head())  # Stampa le prime righe del DataFrame per verificare che sia stato letto correttamente

    # Creare la cartella di output con il nome del file originale
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join('.\\assets\\plot\\plotsensori', file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Generare i boxplot
    generate_boxplots(data, output_dir)

    # Determinare il periodo di copertura
    first_date = data['datetimeV'].min()
    last_date = data['datetimeV'].max()
    sensor_id = data['id'].iloc[0] if 'id' in data.columns else "Unknown Sensor"

    # Salva i dati del periodo di copertura in un file di testo
    txt_output_file = os.path.join(output_dir, f'{file_name}.txt')
    with open(txt_output_file, 'w') as f:
        f.write(f"Sensor ID: {sensor_id}\n")
        f.write(f"First Date: {first_date}\n")
        f.write(f"Last Date: {last_date}\n")
    print(f"Coverage period for sensor {sensor_id} saved to {txt_output_file}")