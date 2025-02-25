¡Genial! Vamos a profundizar más en cómo implementar la regularización L2 en diferentes modelos y en cómo hacer las pruebas prácticas con descenso del gradiente.

---

### **1. Implementación de Regularización L2 en Modelos**
La regularización L2 se puede implementar en distintos modelos de Machine Learning. A continuación, te explicaré cómo hacerlo en algunos modelos comunes y cómo ajustar el parámetro \( \lambda \) (o **coeficiente de regularización**) para controlar su efecto.

#### **Regresión Lineal con L2 (Ridge)**
En la regresión lineal con regularización L2, el objetivo es minimizar la siguiente función de costo:
\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
\]
Donde:
- \( h_\theta(x) \) es la predicción del modelo (modelo lineal: \( \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n \)).
- \( \lambda \) es el parámetro de regularización (también llamado **coeficiente de penalización**).
- La segunda parte de la función de costo es el término de regularización, que penaliza los grandes valores de \( \theta \).

##### **Ejemplo con `scikit-learn` en Python**:
```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generamos un conjunto de datos de regresión
X, y = make_regression(n_samples=100, n_features=1, noise=10)

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
```

En este ejemplo, verás cómo el valor de \( \lambda \) (representado como `alpha` en `Ridge`) afecta a la pendiente de la recta ajustada. A medida que aumentas \( \lambda \), la regularización se vuelve más fuerte, forzando a los coeficientes a ser más pequeños.

#### **Regresión Logística con L2**
La regularización L2 también se puede aplicar a modelos de clasificación como la **regresión logística**. Aquí, la función de costo incluye el término de regularización L2, que ayuda a evitar el sobreajuste de los parámetros.

##### **Ejemplo con `scikit-learn`**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generamos un conjunto de datos de clasificación
X, y = make_classification(n_samples=100, n_features=2, n_classes=2)

# Creamos un modelo de regresión logística con L2
lambda_values = [0.1, 1, 10, 100]
plt.figure(figsize=(10, 6))

for lambda_val in lambda_values:
    model = LogisticRegression(C=1/lambda_val)  # C es el inverso de lambda
    model.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), label=f'λ = {lambda_val}')

plt.title("Efecto de la regularización L2 en la regresión logística")
plt.legend()
plt.show()
```

Aquí, **`C`** es el inverso de \( \lambda \), por lo que un valor más pequeño de \( C \) significa una mayor regularización.

---

### **2. Prueba de Descenso del Gradiente con Regularización L2**

El descenso del gradiente con regularización L2 se realiza de manera similar al descenso del gradiente básico, pero añadiendo un término de regularización en la actualización de los parámetros.

#### **Función de costo con regularización L2**:
La función de costo se modifica así:
\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
\]
Para implementar el descenso del gradiente con regularización L2, el gradiente de los parámetros se modifica agregando un término de penalización. El paso de actualización de los parámetros sería:
\[
\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m} \theta_j \right)
\]
Donde:
- \( \alpha \) es la tasa de aprendizaje.
- \( \lambda \) es el coeficiente de regularización.

#### **Implementación en Python**:
```python
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
```

---

### **3. Comparar los Resultados con y sin Regularización**

La comparación entre usar y no usar regularización L2 se puede hacer al observar cómo cambia el modelo cuando entrenamos sin el término de regularización y con él. En general, al **no usar regularización**, el modelo puede ajustarse demasiado a los datos de entrenamiento (sobreajuste), mientras que al **usar regularización L2**, el modelo será más robusto y evitará que los pesos sean excesivamente grandes.

#### **Sin regularización**:
- El modelo puede ajustarse demasiado a los ruidos en los datos de entrenamiento.
- El desempeño en los datos de test puede ser malo.

#### **Con regularización L2**:
- El modelo buscará soluciones más simples (pesos pequeños) y será menos propenso al sobreajuste.
- El desempeño en los datos de test generalmente mejora.

---

### **4. Experimentar con Tasa de Aprendizaje y Épocas**

La tasa de aprendizaje controla la velocidad con que el modelo ajusta sus parámetros. Si la tasa de aprendizaje es muy grande, el modelo puede sobrepasar el mínimo de la función de costo, mientras que una tasa demasiado pequeña hará que el entrenamiento sea muy lento.

Las épocas indican cuántas veces el modelo pasa por todo el conjunto de datos durante el entrenamiento. Si el número de épocas es muy bajo, el modelo puede no haber tenido tiempo de aprender lo suficiente, y si es muy alto, puede sobreajustarse.

---

### Resumen de lo que debes hacer:

1. **Implementar la regularización L2** en regresión lineal y logística, y observar cómo afecta el modelo al cambiar el valor de \( \lambda \).
2. **Prueba de descenso del gradiente** con y sin regularización, y experimenta con diferentes tasas de aprendizaje y épocas.
3. **Comparar los resultados** con y sin regularización L2 y ver cómo mejora la capacidad de generalización del modelo.

---

📢 ¿Te gustaría que sigamos con más ejemplos prácticos o una explicación más detallada sobre alguno de los modelos o técnicas mencionadas?