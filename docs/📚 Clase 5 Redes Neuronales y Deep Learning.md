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


***TAREA PARA LA PROXIMA CLASE***


¡Con gusto! Vamos a desglosar la tarea para que tengas una mejor idea de lo que necesitas hacer:

---

### **Tarea para la próxima clase:**

1. **Practica entrenando diferentes tipos de redes neuronales con Keras**:
   - **Lo que necesitas hacer**: Experimenta con redes neuronales simples, pero trata de modificarlas para que tengan más capas ocultas o utiliza diferentes funciones de activación. Por ejemplo:
     - Prueba **más capas ocultas** (en lugar de solo una).
     - Usa diferentes **funciones de activación** como **Tanh**, **ReLU** o **Sigmoid** para ver cómo afectan el rendimiento.
     - Juega con los **número de neuronas** en cada capa.
     - Ajusta el **tamaño del batch** o la **tasa de aprendizaje**.

   - **Objetivo**: El objetivo es familiarizarte con la estructura de las redes neuronales y cómo diferentes configuraciones afectan el rendimiento. Al hacerlo, puedes observar cómo las redes más profundas o las funciones de activación diferentes tienen impactos distintos en los resultados.

2. **Explora redes neuronales convolucionales (CNNs)** para clasificación de imágenes:
   - **Lo que necesitas hacer**: Las **redes neuronales convolucionales (CNNs)** son muy utilizadas en la clasificación de imágenes. Una tarea popular es clasificar dígitos escritos a mano en el conjunto de datos **MNIST**. Este conjunto de datos contiene imágenes de dígitos (0 a 9), y el objetivo es construir un modelo que pueda reconocer esos dígitos.

   - **Pasos**:
     1. **Carga el conjunto de datos MNIST**: Este conjunto de datos está disponible en Keras, por lo que puedes cargarlo fácilmente.
     2. **Construye una CNN**: Las redes convolucionales tienen capas especiales como **convolutional layers** (capas convolucionales) y **pooling layers** (capas de agrupamiento), que permiten que el modelo extraiga características de las imágenes de manera más eficiente.
     3. **Entrena el modelo**: Utiliza los datos de entrenamiento para enseñar a la red a reconocer los dígitos. Puedes ajustar parámetros como la cantidad de capas, el tamaño de las imágenes, y la cantidad de épocas.
     4. **Evalúa el modelo**: Después de entrenar el modelo, evalúa su desempeño utilizando los datos de prueba.

   - **Objetivo**: Te ayudará a entender cómo las CNNs funcionan específicamente en el ámbito de la visión por computadora y la clasificación de imágenes.

---

### **¿Qué puedes aprender con esta tarea?**

1. **Redes Neuronales Avanzadas**: Experimentar con diferentes configuraciones de redes neuronales te permitirá entender mejor cómo las redes profundas o las diferentes funciones de activación afectan el rendimiento. Te ayudará a afinar la intuición para elegir la mejor arquitectura para diferentes problemas.

2. **Redes Neuronales Convolucionales (CNNs)**: Al trabajar con imágenes y construir una CNN, aprenderás los principios fundamentales detrás de las CNNs y cómo estas redes son más efectivas que las redes neuronales tradicionales cuando se trata de imágenes y otras entradas estructuradas espacialmente.

---

### **Pasos Sugeridos para la Tarea:**

1. **Entrenar Redes Neuronales con Keras**:
   - Experimenta con el código básico de redes neuronales y cambia los parámetros como el número de capas, la función de activación, el número de neuronas, etc.
   - Aquí tienes un ejemplo de cómo podría lucir un modelo más complejo en Keras:
   
     ```python
     model = Sequential([
         Dense(64, activation='relu', input_shape=(20,)),  # Capa oculta
         Dense(64, activation='relu'),                      # Otra capa oculta
         Dense(1, activation='sigmoid')                     # Capa de salida
     ])
     ```

2. **Explorar CNNs con MNIST**:
   - Carga el conjunto de datos de MNIST con Keras:
   
     ```python
     from tensorflow.keras.datasets import mnist
     (X_train, y_train), (X_test, y_test) = mnist.load_data()
     X_train = X_train.reshape(-1, 28, 28, 1) / 255.0  # Normalizar
     X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
     ```

   - Construye una red CNN con capas convolucionales y de agrupamiento (pooling):
   
     ```python
     from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
     
     model = Sequential([
         Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
         MaxPooling2D(pool_size=(2, 2)),
         Flatten(),
         Dense(128, activation='relu'),
         Dense(10, activation='softmax')  # Para clasificación de 10 clases (0-9)
     ])
     ```

   - Entrena el modelo:
   
     ```python
     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     model.fit(X_train, y_train, epochs=5, batch_size=64)
     ```

   - Evalúa el rendimiento del modelo:
   
     ```python
     loss, accuracy = model.evaluate(X_test, y_test)
     print(f"Loss: {loss}, Accuracy: {accuracy}")
     ```

---

### **Conclusión:**

Al completar esta tarea, estarás más familiarizado con cómo construir y ajustar redes neuronales, y cómo las CNNs son un componente esencial en el aprendizaje profundo para tareas de visión por computadora. Si te sientes cómodo con estos conceptos, estarás listo para abordar tareas aún más complejas en el campo de la IA.

¿Listo para comenzar? ¡Estoy aquí si necesitas ayuda con algún paso o código! 🚀