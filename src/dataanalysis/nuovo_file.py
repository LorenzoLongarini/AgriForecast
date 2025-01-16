import matplotlib
matplotlib.use('Agg')  # Utilizza il backend Agg per evitare problemi con Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

# Funzione per generare boxplot multipli su feature selezionate
def generate_boxplots(selected_features, data):
    # Determinare il numero di righe e colonne per il layout
    num_features = len(selected_features)
    cols = 4  # Numero di colonne desiderato
    rows = math.ceil(num_features / cols)

    # Creare i subplot
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 5))
    axes = axes.flatten()  # Riorganizzare gli assi in una lista

    list_values = ['Vento massimo', 'Vento minimo', 'Vento medio', 'Direzione del vento']  # Esempio: specifica i valori desiderati
    list_units = ['m/s', 'm/s', 'm/s', '°']  # Esempio: specifica le unità di misura
    # Generare un boxplot per ogni feature selezionata
    for i, column in enumerate(selected_features):
        sns.boxplot(data=data, y=column, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{list_values[i]}')
        axes[i].set_ylabel(list_units[i])
        axes[i].set_xlabel('')

    # Rimuovere i subplot vuoti (se esistenti)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Migliorare il layout
    plt.tight_layout()
    plt.savefig('boxplots.png')  # Salva il grafico come immagine
    plt.close()

# Percorso del file locale
file_path = '.\\assets\\Report.xlsx'
# Caricare i dati
data = pd.read_excel(file_path)
# Chiedi all'utente di selezionare le feature
selected_features = ['wind_max', 'wind_min', 'wind_mean', 'wind_dir']  # Esempio: specifica le colonne desiderate
generate_boxplots(selected_features, data)
