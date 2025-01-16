import os
import pandas as pd
import numpy as np

def correggi_formattazione(valore):
    """
    Corregge la formattazione dei valori numerici.
    Rimuove i punti che separano le migliaia e mantiene il punto decimale.
    Arrotonda i valori a sei cifre decimali.
    """
    try:
        # Se il valore è già un numero, restituiscilo arrotondato
        if isinstance(valore, (int, float)):
            return round(valore, 6)
        # Rimuove i punti che separano le migliaia
        if valore.count('.') > 1:
            valore = valore.replace('.', '', valore.count('.') - 1)
        # Converte il valore in float e arrotonda a sei cifre decimali
        valore = round(float(valore), 6)
    except (ValueError, TypeError):
        pass
    return valore

def verifica_e_riassumi_file_csv(cartella, cartella_output):
    """
    Verifica se i file CSV nella cartella sono vuoti. Se non sono vuoti, crea un nuovo file
    nella cartella di output con lo stesso nome, contenente il valore minimo, massimo e medio
    per ogni colonna specificata, raggruppato per 'id' e per ogni ora. Assegna l'ID corrente
    e l'ultimo valore del gruppo per 'datetimeV' e 'timestampV', ordinando per 'timestampV'.

    :param cartella: Percorso della cartella contenente i file CSV.
    :param cartella_output: Percorso della cartella di output per i file riassunti.
    """
    # Crea la cartella di output se non esiste
    os.makedirs(cartella_output, exist_ok=True)

    # Colonne per cui calcolare i valori min, max e mean
    colonne_da_calcolare = [
        'vwc10', 'vwc20', 'vwc30', 'vwc40', 'vwc50', 'vwc60',
        'salinity10', 'salinity20', 'salinity30', 'salinity40', 'salinity50', 'salinity60',
        'temp10', 'temp20', 'temp30', 'temp40', 'temp50', 'temp60'
    ]

    # Itera attraverso tutti i file nella cartella
    for nome_file in os.listdir(cartella):
        if nome_file.endswith(".csv") and nome_file.startswith("2024"):
            percorso_file = os.path.join(cartella, nome_file)
            try:
                # Legge il file CSV con separatore ';'
                df = pd.read_csv(percorso_file, sep=';')
                
                # Verifica se il file è vuoto
                if df.empty:
                    print(f"Il file {nome_file} è vuoto.")
                elif 'id' not in df.columns:
                    print(f"Il file {nome_file} non contiene la colonna 'id'.")
                else:
                    # Corregge la formattazione dei valori nelle colonne specificate
                    for col in colonne_da_calcolare:
                        if col in df.columns:
                            df[col] = df[col].apply(correggi_formattazione)
                    
                    # Sostituisce i valori errati con NaN
                    df.replace(-999.999, np.nan, inplace=True)
                    
                    # Ordina il DataFrame per 'timestampV'
                    df = df.sort_values(by='timestampV')
                    
                    # Convert 'datetimeV' to datetime type
                    df['datetimeV'] = pd.to_datetime(df['datetimeV'])
                    
                    # Extract hour from 'datetimeV'
                    df['hour'] = df['datetimeV'].dt.hour
                    
                    # Raggruppa per 'id' e poi per 'hour'
                    df_grouped_double = df.groupby(['id', 'hour']).agg(
                        {col: ['min', 'max', 'mean'] for col in colonne_da_calcolare}
                    ).reset_index()
                    
                    # Rimuove il multi-index delle colonne
                    df_grouped_double.columns = [
                        col[0] if col[0] in ['id', 'hour'] else '_'.join(col).strip()
                        for col in df_grouped_double.columns
                    ]
                    
                    # Sostituisce i valori NaN con la media della colonna
                    for col in df_grouped_double.columns:
                        if col not in ['id', 'hour']:
                            df_grouped_double[col].fillna(df_grouped_double[col].mean(), inplace=True)
                    
                    # Crea un DataFrame con tutte le ore per ogni 'id'
                    ids = df['id'].unique()
                    hours = pd.DataFrame({'hour': range(24)})
                    df_all_hours = pd.concat([pd.DataFrame({'id': id_}, index=hours.index).join(hours) for id_ in ids])
                    
                    # Unisce il DataFrame con tutte le ore ai dati raggruppati
                    df_final = pd.merge(df_all_hours, df_grouped_double, on=['id', 'hour'], how='left')
                    
                    # Percorso del nuovo file nella cartella di output
                    percorso_nuovo_file = os.path.join(cartella_output, nome_file)
                    
                    # Salva il nuovo file CSV
                    df_final.to_csv(percorso_nuovo_file, index=False, sep=';')
                    print(f"Creato file riassunto: {percorso_nuovo_file}")
            except Exception as e:
                print(f"Errore durante la lettura del file {nome_file}: {e}")

# Esempio di utilizzo
cartella = ".\\assets\\dati-agugliano"  # Sostituisci con il percorso della tua cartella
cartella_output = ".\\assets\\dati-riassunti"  # Sostituisci con il percorso della cartella di output
verifica_e_riassumi_file_csv(cartella, cartella_output)