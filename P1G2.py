# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 19:26:13 2024

@author: julia
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split

matriz = np.array([[1,1.5,2,4.5,5.5,6,6,5],[2,2,3,4,6,4.5,6,7]])

matriz = np.transpose(matriz)

# Número de clusters
num_clusters = 2 #VARIO LA CANTIDAD DE CENTROS DE CLUSTERS QUE QUIERO

# Inicialización aleatoria de centroides
np.random.seed(0)  # Fijar la semilla para reproducibilidad // SI NO ESTABLEZCO UNA SEMILLA
                    # LA GENERACION DE CENTROIDES INCIALES SERIA RANDOM
initial_centroids = np.random.permutation(matriz)[:num_clusters]
initial_centroids = sorted(initial_centroids,key=lambda x:x[0]) #Ordeno la lista de centroides


# # Aplicar K-means con inicialización aleatoria
# kmeans = KMeans(n_clusters=num_clusters, init=initial_centroids, n_init=1)
# kmeans.fit(matriz)

# # Obtener etiquetas de los clusters y los centroides
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_


# # b) Imprimir la matriz de pertenencia
# print("Matriz de pertenencia (estado inicial):")
# for i in range(num_clusters):
#     cluster_points_indices = np.where(labels == i)[0]
#     cluster_points = matriz[cluster_points_indices]
#     print(f"Cluster {i + 1}: {list(zip(cluster_points[:, 0], cluster_points[:, 1]))}")

# # Graficar los puntos y los centroides
# plt.scatter(matriz[:, 0], matriz[:, 1], c=labels, cmap='viridis', edgecolor='k')
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, label='Centroides')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Clustering con K-means')
# plt.legend()
# plt.show()


#c) Almacenar la trayectoria de los centroides
centroid_trajectory = [initial_centroids]

# Criterio de finalización
tolerance = 1e-4
max_iter = 100
converged = False

# Iteraciones de K-means
for i in range(max_iter):
    # Aplicar K-means con inicialización aleatoria
    kmeans = KMeans(n_clusters=num_clusters, init=centroid_trajectory[-1], n_init=1)
    kmeans.fit(matriz)
    
    # Obtener los centroides actuales
    current_centroids = kmeans.cluster_centers_
    
    # Verificar convergencia
    if np.linalg.norm(current_centroids - centroid_trajectory[-1]) < tolerance:
        converged = True
        break
    
    # Agregar los centroides actuales a la trayectoria
    centroid_trajectory.append(current_centroids)

# Imprimir la trayectoria de los centroides
print("Trayectoria de los centroides:")
for i, centroids in enumerate(centroid_trajectory):
    print(f"Iteración {i + 1}: {centroids}")

# Gráfico de la trayectoria de los centroides
centroid_trajectory = np.array(centroid_trajectory)
plt.plot(centroid_trajectory[:, :, 0], centroid_trajectory[:, :, 1], marker='o')
plt.scatter(matriz[:, 0], matriz[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trayectoria de los centroides')
plt.grid(True)
plt.show()

# Imprimir el estado final
print("Estado final:")
print(f"Centroides finales: {centroid_trajectory[-1]}")
print("Etiquetas finales de los clusters:")
for i, label in enumerate(kmeans.labels_):
    print(f"Punto {i + 1} pertenece al Cluster {label+1}")


