from tensorflow.keras.datasets import mnist

# Cargar el conjunto de datos MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocesar los datos (normalizar y reshape para que se ajusten a la entrada de la CNN)
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0  # Normalizar imágenes de 0 a 1 y cambiar forma
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0  # Normalizar imágenes de 0 a 1


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Crear el modelo secuencial
model = Sequential([
    # Capa convolucional con 32 filtros y tamaño de kernel (3x3)
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    
    # Capa de agrupamiento (pooling) para reducir la dimensionalidad
    MaxPooling2D(pool_size=(2, 2)),
    
    # Aplanar la salida para pasar a la capa densa
    Flatten(),
    
    # Capa densa con 128 neuronas
    Dense(128, activation='relu'),
    
    # Capa de salida con 10 neuronas (una por cada clase) y activación softmax
    Dense(10, activation='softmax')  # Para clasificación de 10 clases (0-9)
])

# Compilar el modelo con el optimizador Adam y la función de pérdida sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=5, batch_size=64)


# Evaluar el rendimiento del modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test)

# Imprimir la pérdida y la precisión
print(f"Loss: {loss}, Accuracy: {accuracy}")
