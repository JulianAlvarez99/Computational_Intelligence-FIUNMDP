# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 18:22:12 2024

@author: asusx
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:\\kmeansdata.txt',delimiter='\t',names=["Set_1","Set_2","Set_3","Set_4"])
print(df.head())
# sns.pairplot(data=df);

df_np = df.to_numpy()

# Número de clusters
num_clusters = 3 #VARIO LA CANTIDAD DE CENTROS DE CLUSTERS QUE QUIERO

# Inicialización aleatoria de centroides
np.random.seed(0)  # Fijar la semilla para reproducibilidad // SI NO ESTABLEZCO UNA SEMILLA
                    # LA GENERACION DE CENTROIDES INCIALES SERIA RANDOM
initial_centroids = np.random.permutation(df_np)[:num_clusters]


#Almacenar la trayectoria de los centroides
centroid_trajectory = [initial_centroids]

# Criterio de finalización
tolerance = 1e-4
max_iter = 100
converged = False

# Iteraciones de K-means
for i in range(max_iter):
    # Aplicar K-means con inicialización aleatoria
    kmeans = KMeans(n_clusters=num_clusters,n_init=1).fit(df_np)
    
    # Obtener los centroides actuales
    current_centroids = kmeans.cluster_centers_
    
    # Verificar convergencia
    if np.linalg.norm(current_centroids - centroid_trajectory[-1]) < tolerance:
        converged = True
        break
    
    # Agregar los centroides actuales a la trayectoria
    centroid_trajectory.append(current_centroids)
    
# Obtener etiquetas de los clusters y los centroides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Graficar los puntos y los centroides
plt.figure(2)
plt.scatter(df_np[:, 0],df_np[:, 1], c=labels, cmap='viridis', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, label='Centroides')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clustering con K-means')
plt.legend()
plt.show()
