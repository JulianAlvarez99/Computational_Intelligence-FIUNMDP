# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 00:10:10 2024

@author: julia
"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

# GENERACION DE DOMINIO DEL UNIVERSO DE LAS VARIABLES A TRABAJAR
x_in = np.arange(-25, 25, 1)
x_out  = np.arange(-5, 20, 1)

# GENERACION DE FUNCIONES DE PERTENENCIA 

# SE DEFINEN LAS FUNCIONES DE PERTENENCIA PARA LA VARIABLE  DE ENTRADA
x_in_peq = fuzz.trapmf(x_in, [-20, -15, -6, -3])
x_in_med = fuzz.trapmf(x_in, [-6, -3, 3, 6])
x_in_grd = fuzz.trapmf(x_in, [3, 6, 15, 20])

# SE DEFINEN LAS FUNCIONES DE PERTENENCIA PARA LA VARIABLE DE SALIDA
x_out_peq = fuzz.trapmf(x_out, [-2.46, -1.46, 1.46, 2.46])
x_out_med = fuzz.trapmf(x_out, [1.46, 2.46, 5, 7])
x_out_grd = fuzz.trapmf(x_out, [5, 7, 13, 15])


#VISUALIZACION DE TODAS LAS FUNCIONES DE PERTENENCIA
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20, 10))

ax0.plot(x_in, x_in_peq, 'b', linewidth=1.5, label='Pequeño')
ax0.plot(x_in, x_in_med, 'g', linewidth=1.5, label='Mediano')
ax0.plot(x_in, x_in_grd, 'r', linewidth=1.5, label='Grande')
ax0.set_title('Entrada')
ax0.legend()

ax1.plot(x_out, x_out_peq, 'b', linewidth=1.5, label='Pequeño')
ax1.plot(x_out, x_out_med, 'g', linewidth=1.5, label='Mediano')
ax1.plot(x_out, x_out_grd, 'r', linewidth=1.5, label='Grande')
ax1.set_title('Salida')
ax1.legend()

# Turn off top/right axes
for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

x1=3.7  #ENTRADA
        # EN CASO DE MAS ENTRADAS Y REGLAS SE EXTIENDE LA SINTESIS
        # DE LA ACTIVACION DE LAS REGLAS
        
# ACTIVACION DE REGLAS SOBRE LAS FUNCIONES DE PERTENENCIA DE LA VARIBALE
# DE ENTRADA   
x_level_lo = fuzz.interp_membership(x_in, x_in_peq, x1)
x_level_md = fuzz.interp_membership(x_in, x_in_med, x1)
x_level_hi = fuzz.interp_membership(x_in, x_in_grd, x1)


# ACTIVACION DE REGLAS SOBRE LAS FUNCIONES DE PERTENENCIA DE LA VARIBALE
# DE SALIDA
y_activation_lo = np.fmin(x_level_lo, x_out_peq)  
y_activation_md = np.fmin(x_level_md, x_out_med)
y_activation_hi = np.fmin(x_level_hi, x_out_grd)

y0 = np.zeros_like(x_out)


fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_out, y0, y_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_out, x_out_peq, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_out, y0, y_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_out, x_out_med, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_out, y0, y_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_out, x_out_grd, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Resultado grafico de aplicacion de reglas segun entrada')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


# AGREGACION DE TODAS LAS FUNCIONES DE PERTENENCIA ACTIVADAS
aggregated = np.fmax(y_activation_lo,
                     np.fmax(y_activation_md, y_activation_hi))

# CALCULO DEL VALOR MEDIANTE DEFUZZIFICACION
y = fuzz.defuzz(x_out, aggregated, 'centroid')
y_activation = fuzz.interp_membership(x_out, aggregated, y)  # for plot


fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_out, x_out_peq, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_out, x_out_med, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_out, x_out_grd, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_out, y0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([y, y], [0, y_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Funcion de agregacion con resultado (linea)')
print("Valor resultante = ", y )
print("\n")

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()






