# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:38:10 2024

@author: asusx
"""

import os

def renombrar_y_convertir_imagenes(ruta_carpeta):
    """
    Renombra y convierte todas las imágenes .jpeg a .jpg en la carpeta especificada.

    :param ruta_carpeta: Ruta absoluta a la carpeta donde se encuentran las imágenes.
    """
    archivos = [f for f in os.listdir(ruta_carpeta) if f.lower().endswith('.jpg')]
    for i, archivo in enumerate(archivos, start=1):
        nombre_viejo = os.path.join(ruta_carpeta, archivo)
        nombre_nuevo = os.path.join(ruta_carpeta, f'Iris Marcelo {i}.jpg')
        
        try:
            os.rename(nombre_viejo, nombre_nuevo)
            print(f"Renombrado: {archivo} a Iris Marcelo {i}.jpg")
        except Exception as e:
            print(f"Error al renombrar {archivo}: {e}")

# Ejemplo de uso
ruta_carpeta = '/Users/asusx/OneDrive/Escritorio/Iris Marcelo'
renombrar_y_convertir_imagenes(ruta_carpeta)
