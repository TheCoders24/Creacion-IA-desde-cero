from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# Generamos un conjunto de datos de clasificación
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)

# Creamos un modelo de regresión logística con L2
lambda_values = [0.1, 1, 10, 100]
plt.figure(figsize=(15, 10))

for i, lambda_val in enumerate(lambda_values):
    # Crear el modelo de regresión logística con regularización L2
    model = LogisticRegression(C=1/lambda_val, penalty='l2', solver='liblinear')
    model.fit(X, y)
    
    # Crear una malla para visualizar la frontera de decisión
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    # Predecir sobre la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Graficar la frontera de decisión y los puntos
    plt.subplot(2, 2, i + 1)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    plt.title(f'λ = {lambda_val} (C = {1/lambda_val:.2f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()