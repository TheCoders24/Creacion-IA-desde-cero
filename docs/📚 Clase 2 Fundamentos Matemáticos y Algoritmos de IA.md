¡Excelente! Vamos a revisar brevemente lo que aprendiste y avanzamos con la siguiente clase.  

📌 **Resumen rápido de la tarea:**  
1. **Aprendizaje supervisado**: El modelo aprende con datos etiquetados. Ejemplo: Clasificación de correos en "spam" o "no spam".  
2. **Aprendizaje no supervisado**: El modelo encuentra patrones sin etiquetas. Ejemplo: Agrupar clientes en categorías según su comportamiento.  
3. **Aprendizaje por refuerzo**: El modelo aprende mediante prueba y error, recibiendo recompensas o castigos. Ejemplo: IA jugando ajedrez.  

---

## **📚 Clase 2: Fundamentos Matemáticos y Algoritmos de IA**  

Antes de empezar con la práctica, es crucial entender las bases matemáticas que sostienen la IA. No necesitas ser un experto en matemáticas, pero sí comprender los conceptos básicos.  

### **1. Álgebra Lineal y Representación de Datos**
Las IA trabajan con datos en forma de **vectores y matrices**. Ejemplo:  
- Una imagen se representa como una **matriz de píxeles**.  
- Un texto se convierte en **vectores numéricos** (word embeddings).  

📌 **Conceptos clave:**  
- **Vectores**: Representan características de los datos (ejemplo: [altura, peso, edad]).  
- **Matrices**: Conjuntos de vectores organizados en filas y columnas.  
- **Operaciones básicas**: Multiplicación de matrices, trasposición, determinantes.  

Ejemplo en Python con `numpy`:  
```python
import numpy as np

# Crear una matriz (2x2)
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])

# Multiplicación de matrices
C = np.dot(A, B)
print(C)
```

---

### **2. Estadística y Probabilidad en IA**
Los modelos de IA hacen predicciones basadas en **probabilidades**.  
📌 **Conceptos clave:**  
- **Distribución de datos**: Cómo se organizan los valores en un conjunto (normal, uniforme, etc.).  
- **Media y varianza**: Nos dicen el centro y la dispersión de los datos.  
- **Regresión y correlación**: Cómo se relacionan dos variables.  

Ejemplo en Python con `pandas`:  
```python
import pandas as pd

data = {'Edad': [20, 25, 30, 35, 40], 'Ingresos': [2000, 2500, 3000, 3500, 4000]}
df = pd.DataFrame(data)

# Media de ingresos
print(df['Ingresos'].mean())
```

---

### **3. Función de Activación en Redes Neuronales**
Las funciones de activación permiten que una red neuronal decida qué valores deben "activarse".  
Ejemplo:  
- **Sigmoid** → Usada para problemas de clasificación binaria.  
- **ReLU** → Activa solo valores positivos, usada en redes profundas.  

Ejemplo de función Sigmoid en Python:  
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title("Función Sigmoid")
plt.show()
```

---

## **📌 Tarea para la próxima clase**
1. Investiga sobre **descenso del gradiente** y cómo se usa para entrenar modelos.  
2. Prueba los códigos en Python y modifica valores para entender mejor.  

📢 ¿Te gustaría que en la próxima clase expliquemos el descenso del gradiente con más matemáticas o pasamos a implementarlo en código? 🚀