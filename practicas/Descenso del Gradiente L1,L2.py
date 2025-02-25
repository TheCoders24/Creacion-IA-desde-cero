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
epochs = 1000
m = len(X)

# Hiperparámetros de regularización
lambda_l1 = 10.1  # Para L1
lambda_l2 = 6.1  # Para L2

# Función de costo (MSE) con L1 y L2
def compute_cost(X, y, w, b, lambda_l1, lambda_l2):
    predictions = X.dot(w) + b  # Forma: (100, 1)
    mse = (1/(2*m)) * np.sum((predictions - y) ** 2)
    
    # Términos de regularización
    l1_penalty = lambda_l1 * np.sum(np.abs(w))  # L1
    l2_penalty = lambda_l2 * np.sum(w ** 2)     # L2
    
    # Costo total
    cost = mse + l1_penalty + l2_penalty
    return cost

# Descenso del gradiente con L1 y L2
for epoch in range(epochs):
    # Predicción
    y_pred = X.dot(w) + b  # Forma: (100, 1)
    
    # Cálculo de gradientes
    dw = (1/m) * np.dot(X.T, (y_pred - y))  # Gradiente del MSE
    db = (1/m) * np.sum(y_pred - y)         # Gradiente del sesgo
    
    # Gradientes de regularización
    dw_l1 = lambda_l1 * np.sign(w)  # Gradiente de L1
    dw_l2 = 2 * lambda_l2 * w       # Gradiente de L2
    
    # Actualización de parámetros
    w -= learning_rate * (dw + dw_l1 + dw_l2)  # Actualización con L1 y L2
    b -= learning_rate * db
    
    # Imprimir el costo cada 100 iteraciones
    if epoch % 100 == 0:
        cost = compute_cost(X, y, w, b, lambda_l1, lambda_l2)
        print(f'Epoch {epoch}: Cost {cost}')

# Mostrar la recta ajustada
plt.scatter(X, y)
plt.plot(X, X.dot(w) + b, color='red')
plt.title('Regresión Lineal con Descenso del Gradiente y Regularización L1/L2')
plt.xlabel('X')
plt.ylabel('y')
plt.show()