# Funcion de Activacion en Redes Neuronales

# Las funciones de activacion permite que una red neuronal decida que valores deben Activarse Ejemplo:
# -Sigmoid -> Usada para problemas de clasificacion binaria
# -> ReLU -> Activacion solo Valores Positivos, usada en redes neuronales

# Ejemplo de Funciones Sigmoid en Python

"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 /(1 + np.exp(-x))


x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x,y)
plt.title("Funcion Sigmoid")
plt.show()

"""

# Ejemplo con la libreria pandas

"""

import pandas as pd

data = {'Edad': [20, 25, 30, 35, 40], 'Ingresos': [2000, 2500, 3000, 3500, 4000]}
df = pd.DataFrame(data)

# Media de ingresos
print(df['Ingresos'].mean())

"""