# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:42:29 2024

@author: julia
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        # Inicialización aleatoria de los centros de los clusters
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        self.labels = np.zeros(X.shape[0])

        for _ in range(self.max_iter):
            # Asignar cada punto al cluster más cercano
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2)) #EUCLIDEAN DISTANCE
            new_labels = np.argmin(distances, axis=0)

            # Reubicar los centros de los clusters
            new_centroids = np.array([X[new_labels == k].mean(axis=0) for k in range(self.n_clusters)])

            # Comprobar si los centros de los clusters han convergido
            if np.allclose(self.centroids, new_centroids):
                break

            # Actualizar los centros de los clusters
            self.centroids = new_centroids
            self.labels = new_labels

            # Visualizar el progreso de reubicación de los centros de los clusters
            plt.figure()
            sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=self.labels, palette='pastel', alpha=0.7)
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x')
            plt.title(f"Iteration {_+1}")
            plt.show()

        return self.labels, X


# Generar datos de ejemplo
np.random.seed(0)
df = pd.read_csv('C:\\Users\\julia\\Desktop\\Inteligencia_computacional\\P2G2\\kmeansdata.txt',delimiter='\t',names=["Set_1","Set_2","Set_3","Set_4"])
print(df.head())
#sns.pairplot(data=df);

df_np = df.to_numpy()

# Inicializar y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=3,max_iter=25)
labels, features= kmeans.fit(df_np)

coefSilhouette = metrics.silhouette_score(df_np, labels, metric='euclidean')
coeficienteDaviesBouldin = metrics.davies_bouldin_score(df_np,labels)

print("Davies-Bouldin coeficient has to be lowest possible:\t", coeficienteDaviesBouldin)
print("Silhouette coeficient has to be best aproximation to 1:\t", coefSilhouette)

 # Visualizar los clusters finales
plt.figure()
sns.scatterplot(x=df_np[:, 0], y=df_np[:, 1], hue=labels, palette='pastel', alpha=0.7)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x')
plt.title("Final Clustering")
plt.show()
