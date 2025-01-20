import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Utilizza il backend Agg per evitare problemi con Tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np

# Carica il dataset
file_path = '.\\assets\\datimerged\\A84041A9018859AC_merged.csv'  # Inserisci il percorso corretto
data = pd.read_csv(file_path)

# Rimuovi colonne non numeriche (es. dateandtime)
numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_data = numeric_data.dropna()
# Calcolo delle matrici di correlazione
correlation_methods = ['pearson', 'spearman']
correlation_matrices = {method: numeric_data.corr(method=method) for method in correlation_methods}

# Visualizzazione e salvataggio delle heatmap
for method, corr_matrix in correlation_matrices.items():
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title(f"Heatmap delle Correlazioni ({method.capitalize()})")
    plt.savefig(f"heatmap_{method}.png")  # Salva ogni heatmap
    plt.close()

# Identifica feature fortemente correlate
threshold = 0.8
strong_correlations = []
for method, corr_matrix in correlation_matrices.items():
    upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    strong_corr = [(col1, col2, corr_matrix.loc[col1, col2]) for i, col1 in enumerate(corr_matrix.columns)
                   for j, col2 in enumerate(corr_matrix.columns) if upper_triangle[i, j] and abs(corr_matrix.loc[col1, col2]) > threshold]
    strong_correlations.append((method, strong_corr))

# Filtra feature con bassa varianza
selector = VarianceThreshold(threshold=0.1)
filtered_data = selector.fit_transform(numeric_data)
filtered_columns = numeric_data.columns[selector.get_support()]

# Calcola l'importanza delle feature rispetto all'intero dataset
model = RandomForestRegressor(random_state=42)
model.fit(numeric_data, numeric_data.mean(axis=1))  # Usa la media delle colonne come target fittizio
importances = model.feature_importances_

# Crea un DataFrame per visualizzare l'importanza delle feature
importance_df = pd.DataFrame({
    'Feature': numeric_data.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Seleziona le feature che spiegano il 95% dell'importanza cumulativa
importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()
selected_features = importance_df[importance_df['Cumulative Importance'] <= 0.95]['Feature']

# Riduzione della dimensionalità con PCA sulle feature selezionate
pca = PCA(n_components=0.95)  # Mantiene il 95% della varianza
X_pca = pca.fit_transform(numeric_data[selected_features])

# Risultati finali
print("Feature selezionate tramite Random Forest:", list(selected_features))
print("Componenti principali dopo PCA:", X_pca.shape)