# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:08:18 2024

@author: asusx
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ruta de tu archivo (EL ARCHIVO TIENE QUE SER JPG PORQUE EN PNG LO LEVANTA EN ESCALA DE GRISES)
ruta_archivo = "C:\\Users\\asusx\\OneDrive\\Escritorio\\Inteligencia_Computacional\\image_test.jpg"

# Abre la imagen
imagen = Image.open(ruta_archivo)

# Convierte la imagen a un arreglo NumPy
arreglo_imagen = np.array(imagen)

# Calcula los histogramas para cada canal
hist_R, bins_R = np.histogram(arreglo_imagen[:,:,0].flatten(), bins=256, range=[0,256])
hist_G, bins_G = np.histogram(arreglo_imagen[:,:,1].flatten(), bins=256, range=[0,256])
hist_B, bins_B = np.histogram(arreglo_imagen[:,:,2].flatten(), bins=256, range=[0,256])

# Trazar la imagen
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(arreglo_imagen)
plt.axis('off')
plt.title('Imagen')
plt.show()

# Trazar los histogramas
plt.plot(hist_R, color='red', label='Rojo')
plt.plot(hist_G, color='green', label='Verde')
plt.plot(hist_B, color='blue', label='Azul')
plt.xlabel('Intensidad de píxeles')
plt.ylabel('Cantidad de píxeles')
plt.title('Histograma de la imagen en escala de colores RGB')
plt.legend()
plt.tight_layout()
plt.show()

print(np.var(arreglo_imagen))
print(np.mean(arreglo_imagen))
print(arreglo_imagen.shape)
