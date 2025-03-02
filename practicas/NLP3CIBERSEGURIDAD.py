# Importaciones necesarias
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, AdamWeightDecay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from googletrans import Translator
import shap
import requests
import json
from fastapi import FastAPI
import uvicorn
import re

# ✅ Verificar GPU
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# ✅ Cargar modelo BERT preentrenado
model_name = 'distilbert-base-uncased'
model = TFAutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Configurar dropout manualmente
for layer in model.layers:
    if hasattr(layer, 'dropout'):
        layer.dropout.rate = 0.1

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Inicializar textos y etiquetas
texts = []
labels = np.array([])

# ✅ Función para obtener logs de la API de LLaMA
def fetch_logs_from_api(api_url, prompt, headers=None, num_samples=5):
    """Obtener múltiples muestras de la API"""
    all_logs = []
    for _ in range(num_samples):
        data = {
            "model": "deepseek-r1-distill-qwen-7b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0.7,
        }
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            if 'choices' in response.json() and len(response.json()['choices']) > 0:
                all_logs.append(response.json()['choices'][0]['message']['content'])
        except Exception as e:
            print(f"Error: {e}")
    return all_logs

# ✅ Obtener logs
api_url = "http://192.168.193.1:1234/v1/chat/completions"
headers = {"Content-Type": "application/json"}
logs = fetch_logs_from_api(api_url, "Genera un log de seguridad de acceso no autorizado", headers)

# ✅ Añadir datos al conjunto
if logs:
    texts += logs
    labels = np.concatenate([labels, np.ones(len(logs))])
    print(f"Se añadieron {len(logs)} muestras nuevas")
else:
    print("No se obtuvieron logs. Usando datos de ejemplo.")
    texts += ["Una entrada de ejemplo de fallo de seguridad", "Intento de acceso no autorizado detectado"]
    labels = np.concatenate([labels, [1, 1]])

# ✅ Preprocesamiento y aumentación
def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text.strip().lower()

texts = [clean_text(text) for text in texts]

# ✅ Aumentación de datos con sincronización de etiquetas
if len(texts) < 10:
    augmented_texts = []
    augmented_labels = []
    
    for text, label in zip(texts.copy(), labels.copy()):
        try:
            # Traducción a español
            trans_es = back_translate(text, 'en', 'es')
            augmented_texts.append(trans_es)
            augmented_labels.append(label)
            
            # Traducción a francés
            trans_fr = back_translate(text, 'en', 'fr')
            augmented_texts.append(trans_fr)
            augmented_labels.append(label)
        except Exception as e:
            print(f"Error en aumentación para texto: {text[:50]}... - {e}")
    
    texts += augmented_texts
    labels = np.concatenate([labels, augmented_labels])

# ✅ Verificación de consistencia
assert len(texts) == len(labels), f"Inconsistencia detectada: {len(texts)} textos vs {len(labels)} etiquetas"

# ✅ División de datos adaptativa
MIN_SAMPLES = 5

if len(texts) >= MIN_SAMPLES:
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
else:
    print(f"Solo {len(texts)} muestras disponibles. Usando todos los datos para entrenamiento.")
    train_texts, train_labels = texts, labels
    val_texts, test_texts, val_labels, test_labels = [], [], [], []

# ✅ Tokenización con manejo de conjuntos vacíos
def safe_tokenization(texts_list, tokenizer):
    return tokenizer(texts_list, padding=True, truncation=True, return_tensors='tf') if texts_list else None

train_inputs = safe_tokenization(train_texts, tokenizer)
val_inputs = safe_tokenization(val_texts, tokenizer)
test_inputs = safe_tokenization(test_texts, tokenizer)

# ✅ Entrenamiento condicional
if train_inputs and len(train_texts) >= 2:
    model.compile(
        optimizer=AdamWeightDecay(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    history = model.fit(
        {'input_ids': train_inputs['input_ids'], 'attention_mask': train_inputs['attention_mask']},
        train_labels,
        validation_data=(
            {'input_ids': val_inputs['input_ids'], 'attention_mask': val_inputs['attention_mask']}, 
            val_labels
        ) if val_inputs else None,
        epochs=10,
        batch_size=8,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=1)
        ],
        verbose=1
    )
else:
    print("Datos insuficientes para entrenamiento. Se requiere al menos 2 muestras.")

# ✅ Evaluación segura
if test_inputs and len(test_texts) > 0:
    predictions = model.predict(test_inputs)
    predicted_labels = np.argmax(predictions.logits, axis=1)
    print(classification_report(test_labels, predicted_labels))
    sns.heatmap(confusion_matrix(test_labels, predicted_labels), annot=True)
else:
    print("No hay datos de prueba para evaluar")

# ✅ Despliegue condicional
if len(texts) >= MIN_SAMPLES:
    app = FastAPI()
    @app.post("/predict")
    def predict(text: str):
        inputs = tokenizer(text, return_tensors='tf')
        return {"prediction": int(np.argmax(model(inputs).logits))}
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    print("No se despliega la API por falta de datos")