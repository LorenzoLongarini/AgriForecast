import os
import pandas as pd

def rinomina_file_csv(cartella):
    """
    Rinomina i file CSV nella cartella in base al valore della colonna 'datetimeV',
    utilizzando il formato aaaammdd. Effettua un controllo per verificare che i valori
    nella colonna esistano.

    :param cartella: Percorso della cartella contenente i file CSV.
    """
    # Crea una cartella per i file con nomi duplicati
    cartella_duplicati = os.path.join(cartella, "duplicati")
    os.makedirs(cartella_duplicati, exist_ok=True)

    # Itera attraverso tutti i file nella cartella
    for nome_file in os.listdir(cartella):
        if nome_file.endswith(".csv"):
            percorso_file = os.path.join(cartella, nome_file)
            try:
                # Legge il file CSV con separatore ';'
                df = pd.read_csv(percorso_file, sep=';')
                
                # Controlla se la colonna 'datetimeV' esiste
                if 'datetimeV' in df.columns:
                    # Filtra i valori validi nella colonna 'datetimeV'
                    valori_validi = df['datetimeV'].dropna()
                    
                    # Verifica che ci siano valori validi
                    if not valori_validi.empty:
                        # Estrai il primo valore valido della colonna 'datetimeV'
                        prima_data = valori_validi.iloc[0]
                        
                        # Estrai i primi 10 caratteri per ottenere la data in formato aaaammdd
                        data_formattata = prima_data[:10]#replace("-", "")
                        
                        # Nuovo nome per il file
                        nuovo_nome_file = f"{data_formattata}.csv"
                        percorso_nuovo_file = os.path.join(cartella, nuovo_nome_file)
                        
                        # Rinomina il file o spostalo nella cartella dei duplicati se esiste già
                        if os.path.exists(percorso_nuovo_file):
                            percorso_nuovo_file_duplicato = os.path.join(cartella_duplicati, nuovo_nome_file)
                            os.rename(percorso_file, percorso_nuovo_file_duplicato)
                            print(f"File duplicato spostato: {nome_file} -> {percorso_nuovo_file_duplicato}")
                        else:
                            os.rename(percorso_file, percorso_nuovo_file)
                            print(f"Rinominato: {nome_file} -> {nuovo_nome_file}")
                    else:
                        print(f"Nessun valore valido nella colonna 'datetimeV' in {nome_file}")
                else:
                    print(f"Colonna 'datetimeV' non trovata in {nome_file}")
            except Exception as e:
                print(f"Errore durante la lettura del file {nome_file}: {e}")

# Esempio di utilizzo
cartella = ".\\assets\\dati-agugliano"  # Sostituisci con il percorso della tua cartella
rinomina_file_csv(cartella)
