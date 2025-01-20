import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Utilizza il backend Agg per evitare problemi con Tkinter
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Percorsi dei dataset
dataset_paths = [
    '..//..//assets//datimerged//A8404148E18859AD_merged.csv',
    '..//..//assets//datimerged//A84041A9018859AC_merged.csv',
    '..//..//assets//datimerged//A8404136718859A6_merged.csv'
]

def process_dataset(file_path):
    # Carica il dataset
    data = pd.read_csv(file_path)

    # Rimuovi colonne non numeriche
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    numeric_data = numeric_data.drop(columns=['dateandtime', 'hout', 'id', 'hour'], errors='ignore')

    # Rimuovi righe con valori mancanti
    numeric_data = numeric_data.dropna()

    # Calcolo delle matrici di correlazione
    correlation_methods = ['pearson', 'spearman']
    correlation_matrices = {method: numeric_data.corr(method=method) for method in correlation_methods}

    # Identifica feature fortemente correlate
    threshold = 0.9
    initial_features = set(numeric_data.columns)
    for method, corr_matrix in correlation_matrices.items():
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if upper_triangle[i, j] and abs(corr_matrix.loc[col1, col2]) > threshold:
                    if col2 in numeric_data.columns:
                        numeric_data.drop(columns=[col2], inplace=True)

    # Filtra feature con bassa varianza
    selector = VarianceThreshold(threshold=0.1)
    filtered_data = selector.fit_transform(numeric_data)
    filtered_columns = numeric_data.columns[selector.get_support()]

    # Creazione del dataset filtrato
    filtered_numeric_data = numeric_data[filtered_columns]

    # Split dei dati in X (feature) e y (target ritardato)
    data_shifted = filtered_numeric_data.shift(-1)
    data_shifted = data_shifted.dropna()
    X = filtered_numeric_data[:-1]
    y = data_shifted

    # Suddivisione in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Addestramento della Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Calcolo delle importanze delle feature
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return correlation_matrices, feature_importance_df

# Funzione per estrarre il nome del sensore
import os
def get_sensor_name(file_path):
    return os.path.basename(file_path).replace("_merged.csv", "")

# Loop per processare ogni dataset
correlation_methods = ['pearson', 'spearman']
for method in correlation_methods:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, dataset_path in enumerate(dataset_paths):
        correlation_matrices, _ = process_dataset(dataset_path)

        # Plot delle heatmap delle correlazioni
        sns.heatmap(
            correlation_matrices[method], 
            annot=False, 
            cmap='coolwarm', 
            cbar=True, 
            ax=axes[idx]
        )
        sensor_name = get_sensor_name(dataset_path)
        axes[idx].set_title(f"{method.capitalize()} Heatmap del sensore: {sensor_name}")

    plt.tight_layout()
    plt.savefig(f"..//..//assets//dataanalysisplot//heatmap_{method}_all_sensors.png")
    plt.close()

# Plot per importanza delle feature
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, dataset_path in enumerate(dataset_paths):
    _, feature_importance_df = process_dataset(dataset_path)

    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feature_importance_df, 
        ax=axes[idx]
    )
    sensor_name = get_sensor_name(dataset_path)
    axes[idx].set_title(f"Feature Importance del sensore: {sensor_name}")

plt.tight_layout()
plt.savefig("..//..//assets//dataanalysisplot//feature_importance_all_sensors.png")
plt.close()
