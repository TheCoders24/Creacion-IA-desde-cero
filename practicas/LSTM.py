from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Supongamos que tenemos un conjunto de datos X (textos) y y (etiquetas de sentimiento)
# Preprocesamiento: convierte las secuencias de texto en secuencias numéricas
X = pad_sequences(X, padding='post', maxlen=100)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear el modelo LSTM
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dense(1, activation='sigmoid')  # Capa de salida para clasificación binaria
])

# Compilar y entrenar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)
