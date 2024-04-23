# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 00:46:57 2024

@author: julia
"""

import numpy as np
import matplotlib.pyplot as plt

# Definir el rango de x y calcular y
x = np.arange(0, 10.1, 0.1)
y = np.power(np.e, x/5)

# Dividir el rango de x en tres intervalos
intervalos_x = np.array_split(x, 3)
intervalos_y = np.array_split(y, 3)

# Aproximar la función con rectas para cada intervalo
rectas = []
for i in range(3):
    coeficientes = np.polyfit(intervalos_x[i], intervalos_y[i], 1)  # Ajustar una recta (polinomio de grado 1)
    rectas.append(np.poly1d(coeficientes))  # Convertir los coeficientes en una función polinómica

# Graficar la función original y las rectas aproximadas
plt.plot(x, y, marker='.', linestyle='None', label='Función original')  # '.' para puntos, None para no conectar

for i, recta in enumerate(rectas):
    plt.plot(intervalos_x[i], recta(intervalos_x[i]), label=f'Recta {i+1}')

plt.title('Aproximación de la función con tres rectas')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

def clustering_substractivo(data, density, radius):
    """
    Implementación del algoritmo de Clustering Substractivo.

    Parámetros:
    - data: Matriz de datos de entrada, donde cada fila representa un dato.
    - density: Densidad deseada de los clusters.
    - radius: Radio de influencia para la identificación de los clusters.

    Retorna:
    - centros: Array con los centroides de los clusters.
    - membresia: Matriz de grado de membresía de cada dato respecto a los clusters.
    """

    # Inicialización de variables
    n_samples, n_features = data.shape
    memb_degree = np.zeros(n_samples)
    centros = []

    # Paso 1: Calcular las distancias entre cada par de puntos
    distances = np.linalg.norm(data[:, np.newaxis, :] - data[np.newaxis, :, :], axis=2)

    # Paso 2: Calcular la densidad de los puntos
    density_matrix = np.exp(-density * np.square(distances))

    # Paso 3: Calcular el grado de pertenencia a los clusters
    memb_degree = np.max(density_matrix, axis=1)

    # Paso 4: Identificar los puntos centrales
    central_points = np.argmax(memb_degree)

    # Paso 5: Calcular la influencia de los puntos centrales
    central_influence = np.exp(-np.square(distances[central_points, :]) / (2 * radius ** 2))

    # Paso 6: Calcular los centroides de los clusters
    for i in range(n_samples):
        centros.append(np.sum(data.T * central_influence[i], axis=1) / np.sum(central_influence[i]))

    return np.array(centros), memb_degree

def plot_clusters(data, centros):
    # Función para graficar los clusters
    plt.scatter(data[:, 0], data[:, 1], c='blue', label='Datos')
    plt.scatter(centros[:, 0], centros[:, 1], c='red', marker='*', s=200, label='Centroides')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Clusters resultantes')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# data = np.vstack((x,y))
# # Ejemplo de uso
# density = 31  # Densidad deseada de los clusters
# radius = 2  # Radio de influencia

# centros, membresia = clustering_substractivo(data, density, radius)
# print("Centros de los clusters:")
# print(centros)

# plot_clusters(data, centros)


