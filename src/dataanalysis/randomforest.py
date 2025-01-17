import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Utilizza il backend Agg per evitare problemi con Tkinter
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Carica il dataset
file_path = '..//..//assets//datimerged//merged_total.csv'  # Inserisci il percorso corretto
data = pd.read_csv(file_path)

# Rimuovi colonne non numeriche
numeric_data = data.select_dtypes(include=['float64', 'int64'])
numeric_data = numeric_data.drop(columns=['dateandtime', 'hout', 'id', 'hour'], errors='ignore')
# Rimuovi righe con valori mancanti
numeric_data = numeric_data.dropna()

# Calcolo delle matrici di correlazione
correlation_methods = ['pearson', 'spearman', 'kendall']
correlation_matrices = {method: numeric_data.corr(method=method) for method in correlation_methods}

# Visualizzazione e salvataggio delle heatmap
for method, corr_matrix in correlation_matrices.items():
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title(f"Heatmap delle Correlazioni ({method.capitalize()})")
    plt.savefig(f"heatmap_{method}.png")  # Salva ogni heatmap
    plt.close()

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

# Confronta le feature rimosse
final_features = set(numeric_data.columns)
removed_features = initial_features - final_features

# Salva le feature rimosse
with open("removed_features.txt", "w") as f:
    f.write("\n".join(removed_features))

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
X = X.dropna()
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

# Visualizzazione delle feature più importanti
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Importanza delle Feature calcolata dalla Random Forest")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.show()

# Salvataggio dei risultati
feature_importance_df.to_csv("feature_importance.csv", index=False)
