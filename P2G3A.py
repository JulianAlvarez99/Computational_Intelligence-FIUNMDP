# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:22:58 2024

@author: julia
"""
from PIL import Image
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Paso 1: Cargar la imagen y convertirla a RGB
image = Image.open('C:\\Users\\julia\\Desktop\\Inteligencia_computacional\\pikachu.jpg')
image_rgb = image.convert('RGB')
image_array = np.array(image_rgb)

# Paso 2: Definir conjuntos difusos para los componentes de color
r = np.arange(0, 256, 1)
g = np.arange(0, 256, 1)
b = np.arange(0, 256, 1)

# Definir funciones de pertenencia para el color amarillo en los canales RGB
sigma = 255  # Ajusta este valor según sea necesario
r_yellow = fuzz.gaussmf(r, 255, sigma)
g_yellow = fuzz.gaussmf(g, 255, sigma)
b_yellow = fuzz.gaussmf(b, 0, sigma)

# Paso 3: Evaluar la pertenencia de cada píxel al color amarillo
membership_yellow = np.minimum(r_yellow[image_array[:,:,0]], 
                                np.minimum(g_yellow[image_array[:,:,1]], b_yellow[image_array[:,:,2]]))

# Paso 4: Postprocesamiento (opcional)
# Aquí podrías aplicar alguna técnica de segmentación para resaltar las áreas de amarillo.

# Paso 5: Visualización o análisis de resultados

plt.imshow(membership_yellow, cmap='jet')
plt.colorbar(label='Pertenencia al color amarillo')
plt.title('Pertenencia al color amarillo')
plt.show()