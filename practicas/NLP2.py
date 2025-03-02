from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Cargar modelo y tokenizador BERT
model_name = 'bert-base-uncased'
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Datos de ejemplo más grandes (reemplazar con tus propios datos)
texts = [
    "que es ciberseguridad"
    "I love this product!",
    "This is the worst experience ever.",
    "The service was excellent.",
    "I'm very disappointed.",
    "It's a great value for the money.",
    "This is a complete waste of time.",
    "I'm so happy with my purchase.",
    "This is absolutely terrible.",
    "The food was delicious.",
    "I'm very angry about this."
]
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 1 = positivo, 0 = negativo

# Dividir datos en entrenamiento y validación
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Tokenizar textos de entrenamiento y validación
train_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors='tf')
val_inputs = tokenizer(val_texts, padding=True, truncation=True, return_tensors='tf')

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Entrenar el modelo
history = model.fit(
    x={'input_ids': train_inputs['input_ids'], 'attention_mask': train_inputs['attention_mask']},
    y=train_labels,
    validation_data=(
        {'input_ids': val_inputs['input_ids'], 'attention_mask': val_inputs['attention_mask']},
        val_labels
    ),
    epochs=10,
    batch_size=2
)

# Evaluar el modelo en el conjunto de validación
val_predictions = model.predict({'input_ids': val_inputs['input_ids'], 'attention_mask': val_inputs['attention_mask']})
val_predicted_labels = tf.argmax(val_predictions.logits, axis=1).numpy()
print(classification_report(val_labels, val_predicted_labels))

# Guardar el modelo entrenado
model.save_pretrained("./sentiment_model/")

# Ejemplo de predicción con el modelo guardado.
loaded_model = TFAutoModelForSequenceClassification.from_pretrained("./sentiment_model/")
new_texts = ["This is wonderful!", "This is terrible!"]
new_inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors='tf')
predictions = loaded_model.predict({'input_ids': new_inputs['input_ids'], 'attention_mask': new_inputs['attention_mask']})
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()
print(f"Predictions: {predicted_labels}")