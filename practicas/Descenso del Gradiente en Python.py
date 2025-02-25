import numpy as np
import matplotlib.pyplot as plt

# Generamos datos de ejemplo (y = 2x + 1)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Forma: (100, 1)
y = 2 * X + 1 + np.random.randn(100, 1)  # Agregamos algo de ruido, forma: (100, 1)

# Inicializamos los parámetros
w = np.random.randn(1, 1)  # Forma corregida: (1, 1)
b = np.random.randn(1)  # Forma: (1,)
learning_rate = 0.01
epochs = 100
m = len(X)

# Función de costo (MSE)
def compute_cost(X, y, w, b):
    predictions = X.dot(w) + b  # Forma: (100, 1)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

# Descenso del gradiente
for epoch in range(epochs):
    # Predicción
    y_pred = X.dot(w) + b  # Forma: (100, 1)
    
    # Cálculo de gradientes
    dw = (1/m) * np.dot(X.T, (y_pred - y))  # Forma: (1, 1)
    db = (1/m) * np.sum(y_pred - y)  # Forma: (1,)
    
    # Actualización de parámetros
    w -= learning_rate * dw  # Forma: (1, 1)
    b -= learning_rate * db  # Forma: (1,)
    
    # Imprimir el costo cada 100 iteraciones
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Cost {compute_cost(X, y, w, b)}')

# Mostrar la recta ajustada
plt.scatter(X, y)
plt.plot(X, X.dot(w) + b, color='red')
plt.title('Regresión Lineal con Descenso del Gradiente')
plt.xlabel('X')
plt.ylabel('y')
plt.show()