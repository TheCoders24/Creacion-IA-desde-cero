import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generamos un conjunto de datos de regresión
X, y = make_regression(n_samples=10000, n_features=1, noise=10)

# Creamos un modelo de regresión Ridge (L2)
lambda_values = [0.1, 1, 10, 100]
plt.figure(figsize=(10, 6))

for lambda_val in lambda_values:
    model = Ridge(alpha=lambda_val)  # alpha es el valor de lambda
    model.fit(X, y)
    plt.plot(X, model.predict(X), label=f'λ = {lambda_val}')

plt.scatter(X, y, color='black', label='Datos')
plt.legend()
plt.title("Efecto de la regularización L2 en la regresión lineal")
plt.show()
