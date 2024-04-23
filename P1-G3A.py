# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:42:02 2024

@author: julia
"""

import numpy as np
import matplotlib.pyplot as plt

# Definir rangos y parámetros
x = np.linspace(0, 40, 100)  # Rango de temperatura interior
umbral_alta = 25  # Umbral para temperatura "alta"
umbral_baja = 0  # Umbral para temperatura "baja"

# Función de pertenencia para "Alta"
alta = np.maximum(0, 1 - np.abs(x - umbral_alta) / 15)

# Función de pertenencia para "Baja"
baja = np.maximum(0, 1 - np.abs(x - umbral_baja) / 15)

# Función de pertenencia para "Temperatura NO Alta"
no_alta = np.where(x <= 15, 1, np.where(x >= 20, 0, 1 - (x - 15) / 5))

# Función de pertenencia para "Temperatura MUY Alta"
muy_alta = np.where(x <= 30, 0, np.where(x <= 35, (x - 30) / 5, 1))

# Graficar las funciones de pertenencia
plt.figure(figsize=(10, 8))
plt.plot(x, alta, label='Alta')
plt.plot(x, baja, label='Baja')
plt.plot(x, no_alta, label='NO Alta')
plt.plot(x, muy_alta, label='MUY Alta')
plt.title('Funciones de pertenencia para la temperatura interior')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Pertenencia')
plt.legend()
plt.grid(True)
plt.show()


