import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np

# Verificar si TensorFlow está utilizando la GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"GPU detectada: {physical_devices[0].name}")
    # Configurar para uso dinámico de memoria en la GPU
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No se detectaron GPUs. Se usará la CPU.")

# 1. Cargar dataset IMDb
dataset = load_dataset("imdb")
texts = dataset['train']['text'][:5000]
labels = dataset['train']['label'][:5000]

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 3. Cargar el tokenizer de BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_texts(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="tf")

# Tokenizar los textos
train_encodings = tokenize_texts(X_train)
test_encodings = tokenize_texts(X_test)

# 4. Cargar el modelo BERT preentrenado
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

# 5. Construir el modelo con BERT
input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")

inputs = [input_ids, attention_mask]

# ✅ Encapsulamos BERT dentro de Lambda para evitar el error de KerasTensor
bert_output = Lambda(
    lambda x: bert_model(x, training=False),
    output_shape=(None, 128, 768)  # Especificamos la forma de salida
)(inputs)

pooled_output = bert_output[1]  # pooler_output está en la posición [1]

output = Dense(1, activation="sigmoid")(pooled_output)

model = Model(inputs=inputs, outputs=output)

# 6. Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# 7. Entrenar el modelo
history = model.fit(
    x={'input_ids': train_encodings["input_ids"], 'attention_mask': train_encodings["attention_mask"]},
    y=np.array(y_train),
    validation_data=(
        {'input_ids': test_encodings["input_ids"], 'attention_mask': test_encodings["attention_mask"]},
        np.array(y_test)
    ),
    epochs=3,
    batch_size=16
)

# 8. Evaluar el modelo
loss, accuracy = model.evaluate(
    {'input_ids': test_encodings["input_ids"], 'attention_mask': test_encodings["attention_mask"]},
    np.array(y_test)
)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
