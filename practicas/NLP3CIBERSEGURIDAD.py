import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import confusion_matrix, classification_report

# ✅ Verificar GPU
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# ✅ Cargar modelo BERT preentrenado
model_name = 'distilbert-base-uncased'
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Datos de entrenamiento: Logs de seguridad (Ejemplos)
texts = [
    "Failed SSH login attempt from 192.168.1.10",  # 1 (Malicioso)
    "User admin logged in successfully",          # 0 (Benigno)
    "Suspicious connection detected on port 445", # 1 (Malicioso)
    "System rebooted by user",                    # 0 (Benigno)
    "Brute force attack detected on admin account", # 1
    "Firewall blocked an unauthorized request",   # 1
    "New software update installed",              # 0
    "Multiple login failures detected",           # 1
    "Antivirus scan completed successfully",      # 0
    "Malware detected in email attachment",       # 1
     "Failed SSH login attempt from 192.168.1.10",  # 1 (Malicioso)
    "User admin logged in successfully",          # 0 (Benigno)
    "Suspicious connection detected on port 445", # 1 (Malicioso)
    "System rebooted by user",                    # 0 (Benigno)
    "Brute force attack detected on admin account", # 1
    "Firewall blocked an unauthorized request",   # 1
    "New software update installed",              # 0
    "Multiple login failures detected",           # 1
    "Antivirus scan completed successfully",      # 0
    "Malware detected in email attachment",       # 1,
    "High CPU usage detected on server",         # 1
    "User attempted to escalate privileges",     # 1
    "Database accessed from unknown IP",        # 1,
    "Routine system maintenance",               # 0,
]
labels = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0])  # 1 = Malicioso, 0 = Benigno

# ✅ Tokenizar textos
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='tf')

# ✅ Compilar el modelo con métricas adicionales
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# ✅ Entrenar el modelo y guardar el historial
history = model.fit(
    x={'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
    y=labels,
    epochs=15,
    batch_size=5,
    verbose=1
)

# ✅ Gráfica de precisión y pérdida
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # 🔹 Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # 🔹 Precisión
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.show()

plot_training_history(history)

# ✅ Probar con nuevos logs
new_texts = [
    "Unauthorized access detected from unknown IP",
    "User logged out successfully",
    "Detected network intrusion attempt",
    "Routine system update applied"
]
new_labels = np.array([1, 0, 1, 0])  # Etiquetas reales para la prueba

new_inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors='tf')
predictions = model.predict({'input_ids': new_inputs['input_ids'], 'attention_mask': new_inputs['attention_mask']})
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

# ✅ Imprimir clasificación y reporte
print(f"\nPredictions: {predicted_labels}")
print("\nClassification Report:\n", classification_report(new_labels, predicted_labels, target_names=["Benigno", "Malicioso"]))

# ✅ Matriz de confusión
conf_matrix = confusion_matrix(new_labels, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=["Benigno", "Malicioso"], yticklabels=["Benigno", "Malicioso"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
