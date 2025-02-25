¡Genial! Ahora que tienes la tarea clara, vamos a abordar un tema clave para mejorar nuestros modelos y evitar que se ajusten demasiado a los datos de entrenamiento: **la regularización**.

---

## **📚 Clase 4: Regularización en Modelos de IA (L1 y L2)**

### **¿Por qué necesitamos regularización?**
El **sobreajuste** (overfitting) ocurre cuando un modelo se ajusta demasiado a los datos de entrenamiento, capturando detalles irrelevantes o ruido, en lugar de aprender patrones generales. Esto puede hacer que el modelo tenga un desempeño pobre en nuevos datos (datos de prueba o datos reales).

La **regularización** es una técnica que penaliza los valores grandes de los parámetros del modelo para evitar el sobreajuste, promoviendo modelos más simples que generalicen mejor.

### **1. Regularización L2 (Ridge)**
La regularización **L2** agrega una penalización basada en el cuadrado de los parámetros (pesos del modelo). El término penaliza grandes valores de \( w \) (los pesos) para evitar que el modelo se sobreajuste. Es útil cuando hay una cantidad significativa de datos y se quiere evitar que el modelo se enfoque demasiado en características irrelevantes.

La **función de pérdida** con L2 es:
\[
J(w) = \text{MSE} + \lambda \cdot \sum_{i=1}^{n} w_i^2
\]
donde:
- \( \lambda \) es el **hiperparámetro de regularización**, que controla la fuerza de la penalización.
- \( w_i \) son los pesos del modelo.
- \( \text{MSE} \) es el error cuadrático medio.

### **2. Regularización L1 (Lasso)**
La regularización **L1** agrega una penalización basada en el valor absoluto de los parámetros. Esto tiene el efecto de **eliminar** completamente algunas características al poner sus pesos en cero, lo que resulta en un modelo más esparso. Es útil cuando se cree que algunas características no son relevantes y se desea una selección automática de características.

La **función de pérdida** con L1 es:
\[
J(w) = \text{MSE} + \lambda \cdot \sum_{i=1}^{n} |w_i|
\]

### **3. Regularización ElasticNet**
La **regularización ElasticNet** es una combinación de L1 y L2, que intenta aprovechar lo mejor de ambos mundos. Es útil cuando se quiere tanto una penalización de los pesos grandes (L2) como la eliminación de características irrelevantes (L1).

La **función de pérdida** de ElasticNet es:
\[
J(w) = \text{MSE} + \lambda_1 \cdot \sum_{i=1}^{n} |w_i| + \lambda_2 \cdot \sum_{i=1}^{n} w_i^2
\]
donde \( \lambda_1 \) y \( \lambda_2 \) son los hiperparámetros de regularización.

---

### **Implementación de Regularización en Descenso del Gradiente (Python)**

Aquí tienes un ejemplo de cómo implementar la **regularización L2 (Ridge)** en el descenso del gradiente para una regresión lineal:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo (y = 2x + 1)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1)  # Agregar algo de ruido

# Inicializar los parámetros
w = np.random.randn(1)
b = np.random.randn(1)
learning_rate = 0.01
epochs = 1000
lambda_reg = 0.1  # Hiperparámetro de regularización L2
m = len(X)

# Función de costo (MSE + Regularización L2)
def compute_cost(X, y, w, b, lambda_reg):
    predictions = X.dot(w) + b
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2) + (lambda_reg / 2) * np.sum(w ** 2)
    return cost

# Descenso del gradiente con regularización L2
for epoch in range(epochs):
    # Predicción
    y_pred = X.dot(w) + b
    
    # Cálculo de gradientes
    dw = (1/m) * np.dot(X.T, (y_pred - y)) + lambda_reg * w  # Término de regularización en dw
    db = (1/m) * np.sum(y_pred - y)
    
    # Actualización de parámetros
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Imprimir el costo cada 100 iteraciones
    if epoch % 100 == 0:
        print(f'Epoch {epoch}: Cost {compute_cost(X, y, w, b, lambda_reg)}')

# Mostrar la recta ajustada
plt.scatter(X, y)
plt.plot(X, X.dot(w) + b, color='red')
plt.title('Regresión Lineal con Regularización L2')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

---

### **📌 Tarea para la próxima clase:**
1. **Investiga sobre la regularización L1, L2 y ElasticNet** y cómo se ajustan en diferentes modelos.
2. **Prueba el código de descenso del gradiente** con regularización L2, cambiando el valor de \( \lambda \) y observando cómo afecta al modelo.
3. **Compara los resultados** entre usar o no regularización, y experimenta con diferentes tasas de aprendizaje y épocas.

---
***TAREA DE LA PROXIMA CLASE***

### 1. **Investigar sobre la regularización L1, L2 y ElasticNet y cómo se ajustan en diferentes modelos**

La **regularización** es una técnica fundamental para mejorar la capacidad de generalización de los modelos de IA y evitar el sobreajuste (overfitting). Existen diferentes tipos de regularización:

#### **L1 (Lasso)**
- **¿Qué hace?**: Agrega una penalización a la suma de los valores absolutos de los pesos del modelo.
- **¿Cómo ayuda?**: L1 tiene la propiedad de "eliminar" características, ya que algunos de los pesos pueden volverse exactamente cero, lo que lleva a un modelo más simple y más interpretativo.
- **¿Dónde se usa?**: L1 es útil cuando se cree que solo un pequeño subconjunto de las características es relevante, ya que automáticamente realiza una selección de características.

#### **L2 (Ridge)**
- **¿Qué hace?**: Agrega una penalización a la suma de los cuadrados de los pesos del modelo.
- **¿Cómo ayuda?**: L2 penaliza los pesos grandes, pero no los lleva exactamente a cero. Esto ayuda a evitar que el modelo se enfoque demasiado en características con valores extremos y hace que el modelo sea más general.
- **¿Dónde se usa?**: L2 es útil cuando hay muchas características y se desea evitar que alguna de ellas tenga un peso excesivo, pero no se quiere eliminar ninguna.

#### **ElasticNet**
- **¿Qué hace?**: Combina tanto L1 como L2. La idea es usar lo mejor de ambos mundos: la selección automática de características de L1 y la estabilidad de L2.
- **¿Cómo ayuda?**: ElasticNet es útil cuando hay muchas características correlacionadas entre sí. L1 puede eliminar algunas de estas características, pero L2 mantiene el modelo estable y evita el sobreajuste.
- **¿Dónde se usa?**: Se utiliza cuando se sospecha que tanto la selección de características (L1) como la penalización de grandes pesos (L2) son necesarias.

---

### 2. **Prueba el código de descenso del gradiente con regularización L2, cambiando el valor de \( \lambda \) y observando cómo afecta al modelo**

- **Objetivo**: El valor de \( \lambda \) controla la fuerza de la penalización en la regularización L2.
  - **Si \( \lambda \) es grande**: El modelo se verá más penalizado por los grandes pesos, lo que reducirá la complejidad del modelo y posiblemente lo hará menos preciso.
  - **Si \( \lambda \) es pequeño**: El modelo tendrá más libertad para ajustarse a los datos, lo que puede llevar a un sobreajuste si el valor de \( \lambda \) es muy bajo.
  
- **Tarea práctica**: Ejecutar el código de descenso del gradiente con regularización L2 y variar el valor de \( \lambda \) para ver cómo cambia el modelo. Observa cómo afecta el costo y el comportamiento de la recta ajustada a los datos.

---

### 3. **Compara los resultados entre usar o no regularización, y experimenta con diferentes tasas de aprendizaje y épocas**

- **Comparar resultados con y sin regularización**: Esto te permitirá observar si la regularización mejora la generalización del modelo o si en realidad lo hace menos preciso al intentar forzar los pesos a ser pequeños.
  - **Sin regularización**: El modelo puede ajustarse demasiado a los datos de entrenamiento y tener un desempeño pobre en datos nuevos.
  - **Con regularización**: El modelo debería tener un mejor desempeño general en datos no vistos, pero podría perder algo de precisión en los datos de entrenamiento debido a la penalización.

- **Tasas de aprendizaje**: La tasa de aprendizaje controla cuán rápido ajusta el modelo sus parámetros. Si es demasiado alta, el modelo puede no converger; si es demasiado baja, el entrenamiento será lento. Experimenta con diferentes valores de la tasa de aprendizaje para ver cómo afecta la velocidad de convergencia y la calidad del modelo.

- **Épocas**: Son el número de veces que el modelo pasa por todos los datos de entrenamiento. Si tienes demasiadas épocas, el modelo puede sobreajustarse. Si tienes pocas épocas, el modelo podría no entrenarse lo suficiente. Experimenta con distintas cantidades de épocas para encontrar el punto donde el modelo converge de forma óptima.

---

### **Resumen de lo que debes hacer:**
1. **Investigar** los tres tipos de regularización (L1, L2 y ElasticNet) y cómo se utilizan en distintos modelos.
2. **Probar el código de descenso del gradiente con regularización L2**, modificando \( \lambda \) y observando su efecto en el modelo.
3. **Comparar los resultados** entre usar regularización y no usarla, ajustando también la tasa de aprendizaje y las épocas.

---
