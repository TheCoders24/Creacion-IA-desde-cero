¡Perfecto! Continuemos con la **Clase 6** sobre **Redes Neuronales Recurrentes (RNN)** y **Transformers**.

### **📚 Clase 6: Redes Neuronales Recurrentes (RNN) y Transformers**

#### **1. Redes Neuronales Recurrentes (RNN)**

Las **RNN** son una clase de redes neuronales diseñadas para trabajar con datos secuenciales, como texto, audio, series temporales, etc. A diferencia de las redes neuronales tradicionales (como las redes feed-forward), las RNN pueden "recordar" información de pasos anteriores de la secuencia gracias a su estructura de bucles.

**Características clave de las RNN**:
- **Secuencialidad**: Las RNN procesan los datos en secuencias, lo que les permite capturar dependencias temporales o espaciales en los datos.
- **Celdas de memoria**: Las RNN tienen una "memoria" que les permite recordar información de pasos anteriores.
  
**Estructura básica de una RNN**:
- Cada celda de una RNN toma un input en un paso temporal, además de un valor de memoria (que es la salida de la celda del paso anterior).
- La salida de la red en cada paso temporal depende tanto del input en ese paso como de la memoria interna.

**Ejemplo de implementación con Keras:**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Crear un modelo secuencial
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),  # Capa RNN
    Dense(1, activation='sigmoid')  # Capa de salida
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

**RNNs más avanzadas**:
- **LSTM (Long Short-Term Memory)** y **GRU (Gated Recurrent Units)** son versiones mejoradas de las RNN tradicionales. Son capaces de recordar información durante más tiempo sin el problema del "desvanecimiento del gradiente", que afecta a las RNN simples.
  
**Ejemplo de una LSTM**:
```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),  # Capa LSTM
    Dense(1, activation='sigmoid')  # Capa de salida
])

# Compilar y entrenar el modelo de manera similar al anterior.
```

#### **2. Transformers**

Los **Transformers** son un tipo de arquitectura más moderna y poderosa para trabajar con secuencias, que ha superado a las RNN en tareas como procesamiento de lenguaje natural (NLP). La clave de los transformers es su capacidad para procesar todos los pasos de la secuencia de manera paralela, a diferencia de las RNN que lo hacen de manera secuencial.

**Características clave de los Transformers**:
- **Auto-atención**: Permite a un modelo decidir qué partes de la secuencia son más relevantes para predecir una salida.
- **Paralelización**: A diferencia de las RNN, los Transformers procesan las secuencias completas en paralelo, lo que los hace mucho más rápidos para entrenamiento.
- **Capacidades de captura de dependencias a largo plazo**: Los transformers pueden modelar relaciones a largo plazo de manera más efectiva.

**La arquitectura básica de un Transformer**:
- **Capa de auto-atención**: Mide la relevancia de cada palabra en una secuencia con respecto a las otras.
- **Capa de alimentación hacia adelante**: Después de la capa de atención, cada paso de la secuencia es procesado por una capa totalmente conectada.

**Ejemplo de implementación de un Transformer** (usando una biblioteca como `transformers` de Hugging Face):
```python
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from tensorflow.keras.optimizers import Adam

# Cargar un modelo preentrenado y un tokenizador
model_name = 'bert-base-uncased'
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenizar los textos
inputs = tokenizer(["Hello, how are you?", "I am fine, thank you!"], padding=True, truncation=True, return_tensors='tf')

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo (ejemplo con datos de entrenamiento)
model.fit(inputs['input_ids'], labels, epochs=3, batch_size=16)
```

#### **3. Comparación entre RNNs y Transformers**:
- **RNNs** son efectivas para secuencias más simples y cuando la información a recordar no es muy extensa. Son ideales para tareas donde la secuencia tiene una estructura temporal o secuencial simple.
- **Transformers** son mucho más poderosos para tareas complejas de secuencias largas, especialmente en NLP. Son ideales para modelos grandes como BERT, GPT, etc., y pueden procesar toda la secuencia en paralelo, lo que los hace mucho más rápidos para entrenamiento.

---

### **📌 Tarea para la próxima clase**
1. Investiga más a fondo sobre **LSTM y GRU** y sus diferencias con las RNN tradicionales.
2. Implementa una RNN o LSTM para una tarea sencilla (por ejemplo, predicción de series temporales).
3. Experimenta con un modelo Transformer en una tarea de NLP (como clasificación de sentimientos).

---



