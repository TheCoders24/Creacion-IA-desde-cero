¡Perfecto! Vamos a seguir con la **Clase 5**. Esta vez, nos vamos a centrar en un tema clave para avanzar en tus conocimientos: **Redes Neuronales y Deep Learning**. Estas redes son la base de muchos de los avances más recientes en IA, especialmente en áreas como visión por computadora, procesamiento de lenguaje natural, y más.

---

## **📚 Clase 5: Redes Neuronales y Deep Learning**

### **1. ¿Qué es una Red Neuronal?**
Una **red neuronal** es un modelo de aprendizaje automático inspirado en el cerebro humano. Está compuesta por capas de nodos (o neuronas) que transforman los datos de entrada en salidas de acuerdo con una serie de parámetros aprendidos durante el entrenamiento.

#### **Estructura Básica de una Red Neuronal:**
- **Entrada (Input Layer)**: Son los datos que alimentan la red, como imágenes, texto o números.
- **Capas Ocultas (Hidden Layers)**: Realizan los cálculos internos. Las redes profundas tienen múltiples capas ocultas.
- **Salida (Output Layer)**: La predicción final o resultado de la red.

#### **Ejemplo de Red Neuronal Simple:**
Una red neuronal simple para clasificación puede tener:
- 1 capa de entrada con 3 neuronas (representando 3 características del dato).
- 1 capa oculta con 5 neuronas.
- 1 capa de salida con 2 neuronas (para clasificación binaria).

---

### **2. Funcionamiento de las Redes Neuronales**

Cada neurona en una capa realiza una operación de suma ponderada y luego pasa el resultado a través de una **función de activación**. Esto permite que la red aprenda patrones complejos.

#### **Fórmula Básica para una Neurona:**
\[ 
z = \sum_{i=1}^{n} w_i x_i + b
\]
\[ 
a = f(z)
\]
- \( x_i \) son las entradas.
- \( w_i \) son los pesos que se ajustan durante el entrenamiento.
- \( b \) es el sesgo o bias.
- \( f(z) \) es la función de activación que transforma la salida.

#### **Funciones de Activación:**
- **Sigmoid**: Para problemas de clasificación binaria.
- **ReLU** (Rectified Linear Unit): Muy utilizada en redes profundas, activa solo valores positivos.
- **Tanh**: Una función sigmoide que va de -1 a 1.

---

### **3. Propagación hacia Adelante y Retropropagación**

- **Propagación hacia Adelante (Forward Propagation)**: Los datos pasan desde la capa de entrada hasta la capa de salida, realizando cálculos en cada neurona.
  
- **Retropropagación (Backpropagation)**: Es el proceso en el que la red ajusta sus pesos a partir del error en la salida. La red calcula el gradiente del error con respecto a cada peso y lo ajusta usando el **descenso del gradiente**.

#### **Proceso de Retropropagación:**
1. **Cálculo del error**: La red calcula el error en la salida (por ejemplo, la diferencia entre la salida predicha y la salida real).
2. **Cálculo del gradiente**: Se calcula el gradiente del error con respecto a los pesos y sesgos.
3. **Ajuste de pesos y sesgos**: Los pesos y sesgos se actualizan en función del gradiente, usando una tasa de aprendizaje.

---

### **4. Implementación en Python con `TensorFlow` / `Keras`**

Ahora que tenemos una buena base teórica, veamos cómo implementar una red neuronal simple utilizando **TensorFlow** y **Keras**.

#### **Ejemplo de una Red Neuronal para Clasificación:**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generar un conjunto de datos de clasificación
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un modelo secuencial
model = Sequential([
    Dense(32, activation='relu', input_shape=(20,)),  # Capa oculta con 32 neuronas
    Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
```

#### **Explicación del Código:**
- **`Dense(32, activation='relu')`**: Crea una capa oculta con 32 neuronas y una función de activación ReLU.
- **`Dense(1, activation='sigmoid')`**: Crea la capa de salida con 1 neurona y una función de activación Sigmoid para clasificación binaria.
- **`optimizer='adam'`**: Utiliza el optimizador Adam, que es eficiente para redes neuronales profundas.
- **`loss='binary_crossentropy'`**: Usamos esta función de pérdida para problemas de clasificación binaria.

---

### **5. ¿Por qué Usar Redes Neuronales?**
Las redes neuronales son muy poderosas para capturar patrones complejos, especialmente cuando tienes grandes volúmenes de datos. Se utilizan en diversas aplicaciones como:
- **Visión por Computadora**: Clasificación de imágenes, detección de objetos.
- **Procesamiento de Lenguaje Natural (NLP)**: Traducción automática, análisis de sentimientos.
- **Reconocimiento de Voz**: Asistentes de voz, transcripción de audio.

---

### **Tarea para la próxima clase:**
1. **Practica entrenando diferentes tipos de redes neuronales** con **Keras**: Por ejemplo, con más capas ocultas o usando otras funciones de activación.
2. **Explora redes neuronales convolucionales (CNNs)** para clasificación de imágenes. Puedes probar un modelo simple para clasificar dígitos utilizando el conjunto de datos **MNIST**.

📢 ¿Te gustaría que profundicemos en la implementación de redes neuronales convolucionales (CNNs) o que sigamos con otros modelos más avanzados?