import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Cargar el dataset
file_path = "P2_1/Practica2_1/iris.csv"
df = pd.read_csv(file_path)

# Renombrar columnas si es necesario (depende del formato del archivo)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Normalización de datos (excepto la columna de clase)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Análisis exploratorio: Visualización de pares de atributos
sns.pairplot(df, hue='species', diag_kind='kde')
plt.show()

# Reducción de dimensiones con PCA a 2 componentes
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled.iloc[:, :-1])
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
df_pca['species'] = df['species']

# Visualización de los datos en el espacio 2D reducido
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='species', palette='viridis')
plt.title('Proyección PCA del dataset Iris')
plt.show()
