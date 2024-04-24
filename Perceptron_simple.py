# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 02:59:58 2024

@author: julia
"""

import numpy as np

class Perceptron:
    def __init__(self, input_size, bias = 1, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.bias = bias
        self.epochs = epochs

    def activation_function(self, x):
        # Step function
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) - self.bias# Dot product + bias
        return self.activation_function(summation)

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            prev_weights = np.copy(self.weights)
            print("Epoch:", epoch + 1)
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error
                print("Inputs:", inputs, "Weights:", self.weights[1:], "Learning rate:", self.learning_rate, "Target:", label, "Prediction:", prediction, "Error:", error)
            # Verificar convergencia después de la primera época
            if epoch > 0 and np.allclose(prev_weights, self.weights):
                print("Convergence reached at epoch", epoch + 1)
                break

# Ejemplo de uso
training_inputs = np.array([[1,2],
                     [2,3],
                     [3,1],
                     [6,5],
                     [7,7],
                     [8,6]])

labels = np.array([0,0,0,1,1,1])

perceptron = Perceptron(input_size=2, learning_rate=0.5, epochs= 40)
perceptron.train(training_inputs, labels)

test_input = np.array([1, 1])
print(perceptron.predict(test_input))  # Salida esperada: 1
