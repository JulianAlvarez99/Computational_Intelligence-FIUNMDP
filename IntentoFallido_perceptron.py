# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 00:39:40 2024

@author: julia
"""
import numpy as np


def v_calculation(weight, entries, umbral):
    v_result = np.dot(weight, entries.T)
    v_result -= umbral
    return v_result

def Next_weightCalculation(weight, entries, eta, error):
    return weight + eta * error * entries

def Perceptron_simple(data_set, umbral, targets, eta):
    
    if len(data_set) != len(targets):
        raise ValueError("Diferent size dimension between data_set and targets")
    
    rows, cols = data_set.shape
    umbral_col = np.full((rows, 1), umbral)
    data_set = np.concatenate((data_set, umbral_col), axis=1)
  
    length = len(data_set)
    weight = np.zeros(cols+1)  # +1 for the bias weight
    
    weight_historial = [np.zeros(cols + 1)] * rows
    v_historial = [0] * length
    function_historial = [0] * length
    error_historial = [0] * length
    
    weight_historial.append(weight)
        
        
    for i in enumerate(data_set):
        v_result = v_calculation(weight_historial[i], data_set[i], umbral)
        if v_result >= 0:
            function_historial[i] = 1
        else:
            function_historial[i] = -1
            
        v_historial[i] = v_result 
        error_historial[i] = (targets[i] - function_historial[i])
        new_weight = Next_weightCalculation(weight_historial[-1], data_set[i], eta, error_historial[i])
        weight_historial[i] = new_weight
        
    
    return weight_historial, v_historial, function_historial, error_historial

  

data_set = np.array([[1,2],
                     [2,3],
                     [3,1],
                     [6,5],
                     [7,7],
                     [8,6]])

target = np.array([0,0,0,1,1,1])

weight_historial, v_historial, function_hitorial, error_historial = Perceptron_simple(data_set, -1, target, 0.5)
