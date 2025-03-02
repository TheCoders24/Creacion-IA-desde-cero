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
