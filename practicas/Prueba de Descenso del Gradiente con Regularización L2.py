import numpy as np

# Generar algunos datos simples
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Añadir una columna de unos para el término de sesgo (intercepto)
X_b = np.c_[np.ones((100, 1)), X]

# Parámetros
alpha = 0.1  # Tasa de aprendizaje
lambda_val = 0.1  # Coeficiente de regularización L2
epochs = 1000
m = len(X_b)

# Inicializar theta (pesos)
theta = np.random.randn(2, 1)

# Implementar descenso del gradiente con L2
for epoch in range(epochs):
    gradients = (1/m) * X_b.T.dot(X_b.dot(theta) - y) + (lambda_val/m) * theta
    theta -= alpha * gradients

print(f"Coeficientes después del entrenamiento: {theta}")
