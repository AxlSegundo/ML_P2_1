import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score
from minisom import MiniSom

# Cargar el data set como un csv
file_path = "P2_1/Practica2_1/iris.csv"
df = pd.read_csv(file_path)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Estandarizar los datos
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

# Se crea una visualización para encontrar relaciones entre variables primero con los datos normales y despúes aplicando una reducción a dos dimensiones con el uso de PCA
sns.pairplot(df, hue='species', diag_kind='kde')
plt.show()


pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled.iloc[:, :-1])
df_pca = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
df_pca['species'] = df['species']

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='species', palette='viridis')
plt.title('Proyección PCA del dataset Iris')
plt.show()

# Se crea un dendograma utilizando el método de ward
Z = linkage(df_scaled.iloc[:, :-1], method='ward')


plt.figure(figsize=(10, 6))
dendrogram(Z, labels=df['species'].values, leaf_rotation=90)
plt.title('Dendrograma - Agrupamiento Jerárquico')
plt.xlabel('Muestras')
plt.ylabel('Distancia')
plt.show()
# Evaluación mediante coeficiente de silueta y el índice Davies-Bouldin
clusters = fcluster(Z, t=3, criterion='maxclust')


sil_score = silhouette_score(df_scaled.iloc[:, :-1], clusters)
db_index = davies_bouldin_score(df_scaled.iloc[:, :-1], clusters)

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Davies-Bouldin Index: {db_index:.3f}")

# Creación y entrenamiento de SOM(8x8)
X = df_scaled.iloc[:, :-1].values
y = df_scaled['species'].values


som = MiniSom(x=8, y=8, input_len=4, sigma=1.0, learning_rate=0.5, neighborhood_function='gaussian', random_seed=10)
som.train_random(X, 1000)

# Mapa de distancias (U-Matrix)
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar(label='Distancia entre neuronas')
plt.title('Mapa de distancias (U-Matrix)')
plt.show()


win_map = som.win_map(X)


from collections import defaultdict
bmu_labels = defaultdict(list)

for i, x in enumerate(X):
    w = som.winner(x)
    bmu_labels[w].append(df['species'].iloc[i])


plt.figure(figsize=(8, 8))
for x in range(8):
    for y in range(8):
        if (x, y) in bmu_labels:
            labels = bmu_labels[(x, y)]
            label = max(set(labels), key=labels.count)
            plt.text(x + 0.5, y + 0.5, label[0], color='black', ha='center', va='center')

plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.6)
plt.title('Distribución de clases en el SOM')
plt.gca().invert_yaxis()
plt.show()

