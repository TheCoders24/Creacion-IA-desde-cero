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
