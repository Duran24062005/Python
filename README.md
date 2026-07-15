# Python para Machine Learning y Visión por Computadora

> **Viaje de aprendizaje desde los fundamentos de Python hasta aplicaciones avanzadas en IA y Visión por Computadora**

---

## 📚 Tabla de Contenidos

1. [¿Qué es Python?](#qué-es-python)
2. [Por qué Python para ML y IA](#por-qué-python-para-ml-e-ia)
3. [Conceptos Fundamentales](#conceptos-fundamentales)
4. [Roadmap de Aprendizaje](#roadmap-de-aprendizaje)
5. [Teoría de Machine Learning](#teoría-de-machine-learning)
6. [Visión por Computadora](#visión-por-computadora)
7. [Bibliotecas Clave](#bibliotecas-clave)
8. [Recursos de Aprendizaje](#recursos-de-aprendizaje)

---

## ¿Qué es Python?

### Definición

**Python** es un lenguaje de programación de alto nivel, interpretado y dinámicamente tipado, creado en 1991 por Guido van Rossum. Se caracteriza por su sintaxis clara, legible y similar al lenguaje natural.

### Características Principales

- **Legibilidad**: El código es fácil de leer y comprender
- **Interpretado**: Se ejecuta línea por línea sin compilación previa
- **Versátil**: Soporta múltiples paradigmas (imperativo, orientado a objetos, funcional)
- **Comunidad Activa**: Millones de desarrolladores y extensa documentación
- **Ecosistema Rico**: Miles de librerías disponibles mediante `pip`

### Ejemplo Básico

```python
# Tipos de datos fundamentales
numero = 42                          # Entero
decimales = 3.14                     # Flotante
texto = "Python es increíble"        # String
booleano = True                      # Booleano
lista = [1, 2, 3, 4, 5]            # Lista
diccionario = {"nombre": "Juan", "edad": 25}  # Diccionario

# Operaciones básicas
print(f"Texto: {texto}")
print(f"Suma: {numero + decimales}")

# Estructura condicional
if numero > 30:
    print("El número es mayor a 30")

# Bucle
for i in lista:
    print(i ** 2)  # Cuadrado de cada número
```

---

## Por qué Python para ML e IA

### Razones Técnicas

1. **Rendimiento en Ciencia de Datos**: Python cuenta con NumPy, que realiza operaciones numéricas eficientemente
2. **Comunidad Especializada**: La mayoría de investigadores en IA usan Python
3. **Librerías Dominantes**: 
   - TensorFlow y PyTorch para Deep Learning
   - Scikit-learn para Machine Learning clásico
   - OpenCV para Visión por Computadora
4. **Prototipado Rápido**: Desarrollo ágil y experimentación iterativa
5. **Integración**: Fácil integración con C/C++ para operaciones críticas

### Estadísticas (2024-2025)

- <cite index="1-1">Python es el lenguaje predeterminado para muchos tipos de proyectos de machine learning, desde visión por computadora hasta procesamiento de lenguaje natural</cite>
- <cite index="2-1">Las cinco librerías principales de visión por computadora que cubren la mayoría de necesidades son: Supervision para utilidades de detección agnósticas del modelo, OpenCV para algoritmos clásicos de procesamiento de imágenes, Torchvision para construcción de modelos de deep learning, Transformers para arquitecturas preentrenadas de última generación, y Timm para clasificación de imágenes</cite>

---

## Conceptos Fundamentales

### 1. Estructuras de Datos

Python maneja estructuras de datos esenciales:

```python
import numpy as np

# Listas - Colecciones ordenadas y mutables
datos = [1, 2, 3, 4, 5]
datos.append(6)  # Agregar elemento

# Tuplas - Colecciones inmutables
coordenadas = (x=10, y=20)

# Diccionarios - Pares clave-valor
estudiante = {
    "nombre": "Carlos",
    "edad": 20,
    "calificaciones": [8.5, 9.0, 7.8]
}

# Arrays NumPy - Optimizados para operaciones numéricas
matriz = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(matriz.shape)      # Dimensiones: (3, 3)
print(matriz.sum())      # Suma de todos elementos: 45
```

### 2. Funciones y Módulos

```python
# Definir una función
def calcular_promedio(calificaciones):
    """Calcula el promedio de una lista de calificaciones"""
    return sum(calificaciones) / len(calificaciones)

# Importar módulos
import math
from datetime import datetime

# Usar funciones de módulos
print(math.sqrt(16))           # 4.0
print(datetime.now())          # Fecha y hora actual
```

### 3. Comprensión de Listas

```python
# Forma tradicional
numeros_pares = []
for i in range(1, 11):
    if i % 2 == 0:
        numeros_pares.append(i)

# Forma con comprensión (más Pythónica)
numeros_pares = [i for i in range(1, 11) if i % 2 == 0]
print(numeros_pares)  # [2, 4, 6, 8, 10]
```

---

## Roadmap de Aprendizaje

### Fase 1: Python Fundamental (Curso 1: Python Total)

**Temas a Cubrir:**
- Variables, tipos de datos y operadores
- Estructuras de control (if, for, while)
- Funciones y módulos
- Manejo de excepciones (try-except)
- Programación orientada a objetos
- Lectura/escritura de archivos

**Objetivo**: Dominar la sintaxis y lógica de programación en Python

### Fase 2: Python para Data Science (Curso 2: Python para Data Science)

**Temas a Cubrir:**
- NumPy: Operaciones con arrays
- Pandas: Manipulación de datos
- Matplotlib y Seaborn: Visualización
- Estadística básica

**Objetivo**: Aprender a trabajar con datos y visualizarlos

### Fase 3: Machine Learning (Posterior)

**Temas:**
- Scikit-learn
- Algoritmos supervisados (regresión, clasificación)
- Algoritmos no supervisados (clustering)
- Validación de modelos

### Fase 4: Deep Learning y Visión por Computadora

**Temas:**
- Redes Neuronales Convolucionales (CNN)
- TensorFlow/PyTorch
- OpenCV
- Modelos preentrenados

---

## Teoría de Machine Learning

### ¿Qué es Machine Learning?

**Machine Learning (ML)** es un subcampo de la Inteligencia Artificial que permite a las máquinas aprender patrones de datos sin ser programadas explícitamente para cada tarea.

### Tipos de Aprendizaje

#### 1. Aprendizaje Supervisado
El modelo se entrena con datos etiquetados (X, y) donde y es la respuesta conocida.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
iris = load_iris()
X, y = iris.data, iris.target

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100)
modelo.fit(X_train, y_train)

# Evaluar
precision = modelo.score(X_test, y_test)
print(f"Precisión del modelo: {precision:.2%}")
```

**Aplicaciones:**
- Clasificación: ¿Es spam este email?
- Regresión: ¿Cuál será el precio de la casa?

#### 2. Aprendizaje No Supervisado
El modelo encuentra patrones en datos sin etiquetas.

```python
from sklearn.cluster import KMeans
import numpy as np

# Datos sin etiquetas
datos = np.random.randn(300, 2)

# Clustering K-means
kmeans = KMeans(n_clusters=3, random_state=42)
etiquetas = kmeans.fit_predict(datos)

# Las muestras se agrupan en 3 clusters automáticamente
```

**Aplicaciones:**
- Clustering: Segmentación de clientes
- Reducción de dimensionalidad

#### 3. Aprendizaje Reforzado
El modelo aprende mediante interacción con un entorno (recompensas/penalizaciones).

**Aplicaciones:**
- Juegos (AlphaGo)
- Robots autónomos

### El Ciclo de Machine Learning

```
1. RECOLECCIÓN DE DATOS
    ↓
2. LIMPIEZA Y PREPROCESAMIENTO
    ↓
3. ANÁLISIS EXPLORATORIO (EDA)
    ↓
4. INGENIERÍA DE CARACTERÍSTICAS (Feature Engineering)
    ↓
5. SELECCIÓN DE MODELO
    ↓
6. ENTRENAMIENTO
    ↓
7. VALIDACIÓN Y EVALUACIÓN
    ↓
8. AJUSTE DE HIPERPARÁMETROS (si es necesario)
    ↓
9. PREDICCIÓN EN DATOS NUEVOS
```

---

## Visión por Computadora

### ¿Qué es Visión por Computadora?

**Computer Vision (CV)** es el campo que capacita a las máquinas para "ver" e interpretar el mundo visual, extrayendo información significativa de imágenes y videos.

### Conceptos Clave

#### 1. Imágenes Digitales

Una imagen digital es una matriz de píxeles. En imágenes en color (RGB), cada píxel tiene 3 valores (Rojo, Verde, Azul) entre 0-255.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Crear una imagen simple
imagen = np.zeros((100, 100, 3), dtype=np.uint8)
imagen[25:75, 25:75] = [255, 0, 0]  # Cuadrado rojo

# Mostrar
plt.imshow(imagen)
plt.title("Imagen Simple")
plt.show()
```

#### 2. Operaciones Básicas con OpenCV

```python
import cv2

# Leer una imagen
imagen = cv2.imread('imagen.jpg')

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Redimensionar
redimensionada = cv2.resize(imagen, (300, 300))

# Aplicar filtro Gaussiano
suavizado = cv2.GaussianBlur(imagen, (5, 5), 0)

# Detección de bordes (Canny)
bordes = cv2.Canny(gris, 100, 200)

# Guardar resultado
cv2.imwrite('bordes.jpg', bordes)
```

#### 3. Redes Neuronales Convolucionales (CNN)

Las CNN son especializadas en procesar imágenes. Su arquitectura permite capturar características visuales jerarquizadas.

**Estructura Típica:**
```
Imagen Input
    ↓
[Convolución] → Extrae características locales
    ↓
[ReLU] → Activación no lineal
    ↓
[Pooling] → Reduce dimensionalidad
    ↓
[Convolución] → Extrae características más complejas
    ↓
[Fully Connected Layers] → Clasificación final
    ↓
Output (Clases predichas)
```

**Ejemplo básico con PyTorch:**

```python
import torch
import torch.nn as nn

class CNN_Simple(nn.Module):
    def __init__(self):
        super(CNN_Simple, self).__init__()
        # Capa convolucional: 3 canales de entrada, 32 filtros, kernel 3x3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Capa completamente conectada
        self.fc = nn.Linear(32 * 16 * 16, 10)  # 10 clases
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Crear modelo
modelo = CNN_Simple()
print(modelo)
```

### Tareas Comunes en Visión por Computadora

| Tarea | Descripción | Aplicación |
|-------|-------------|-----------|
| **Clasificación** | Asignar categoría a una imagen | Reconocimiento de objetos |
| **Detección** | Localizar y clasificar objetos | Detección de rostros, vehículos |
| **Segmentación** | Etiquetar cada píxel | Segmentación de órganos médicos |
| **Seguimiento** | Rastrear objetos en video | Vigilancia, análisis deportivo |
| **Generación** | Crear nuevas imágenes | Síntesis, edición de imágenes |

---

## Bibliotecas Clave

### NumPy
Computación numérica y operaciones con arrays.
```python
import numpy as np

array = np.array([1, 2, 3, 4, 5])
media = np.mean(array)
desv_std = np.std(array)
```

### Pandas
Manipulación y análisis de datos tabulares.
```python
import pandas as pd

df = pd.read_csv('datos.csv')
df.describe()  # Estadísticas descriptivas
df.groupby('categoría').mean()  # Agrupar datos
```

### Matplotlib y Seaborn
Visualización de datos.
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Gráfico Simple')
plt.show()

sns.scatterplot(data=df, x='x', y='y', hue='categoría')
```

### Scikit-learn
Machine Learning clásico.
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

modelo = LogisticRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

### OpenCV
Procesamiento de imágenes y visión por computadora.
```python
import cv2

imagen = cv2.imread('imagen.jpg')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
```

### TensorFlow / PyTorch
Deep Learning y redes neuronales.
```python
import torch
import torch.nn as nn

modelo = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

---

## Relación entre Python, ML, IA y Visión por Computadora

```
┌─────────────────────────────────────────────────┐
│            INTELIGENCIA ARTIFICIAL              │
│  (Campo general de máquinas inteligentes)       │
└──────────────┬──────────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐   ┌─────▼────────────┐
│  NLP        │   │  COMPUTER VISION │
│(Lenguaje)   │   │  (Imágenes)      │
└─────────────┘   └─────┬────────────┘
                        │
              ┌─────────┴──────────┐
              │                    │
         ┌────▼─────┐      ┌──────▼──────┐
         │ OpenCV   │      │   CNNs      │
         │(Classic) │      │(Deep Learn.)│
         └──────────┘      └─────────────┘

TODO SE IMPLEMENTA EN: PYTHON
     NumPy, Pandas, TensorFlow, PyTorch, Scikit-learn, OpenCV...
```

---

## Recursos de Aprendizaje

### Cursos a Seguir

1. **Python Total** (Udemy)
   - Fundamentos sólidos de Python
   - Duración: Variable según ritmo de aprendizaje
   
2. **Python para Data Science** (Udemy)
   - NumPy, Pandas, Matplotlib
   - Análisis de datos práctico

3. **Machine Learning y Visión por Computadora** (Posterior)
   - Recomendado: Combinación de TensorFlow y OpenCV cursos

### Documentación Oficial

- **Python**: https://docs.python.org/3/
- **NumPy**: https://numpy.org/doc/
- **Pandas**: https://pandas.pydata.org/docs/
- **Scikit-learn**: https://scikit-learn.org/stable/documentation.html
- **TensorFlow**: https://www.tensorflow.org/learn
- **PyTorch**: https://pytorch.org/tutorials/
- **OpenCV**: https://docs.opencv.org/

### Comunidades

- Stack Overflow
- GitHub
- Kaggle (Datasets y Competiciones)
- Reddit: r/MachineLearning, r/computervision

---

## 📂 Estructura del Repositorio

```
Python/
├── README.md                          # Este archivo
├── 01_fundamentos/                    # Fase 1: Python Básico
│   ├── 01_variables_tipos.py
│   ├── 02_estructuras_control.py
│   ├── 03_funciones.py
│   └── 04_orientado_objetos.py
├── 02_data_science/                   # Fase 2: Data Science
│   ├── 01_numpy_arrays.py
│   ├── 02_pandas_dataframes.py
│   ├── 03_visualizacion.py
│   └── 04_estadistica.py
├── 03_machine_learning/               # Fase 3: ML Clásico
│   ├── 01_regresion.py
│   ├── 02_clasificacion.py
│   ├── 03_clustering.py
│   └── 04_validacion_modelos.py
├── 04_vision_computadora/             # Fase 4: Computer Vision
│   ├── 01_opencv_basico.py
│   ├── 02_procesamiento_imagenes.py
│   ├── 03_deteccion_caracteristicas.py
│   └── 04_redes_convolucionales.py
└── datasets/                          # Datos para prácticas
```

---

## 💡 Tips para Aprender Efectivamente

1. **Practica regularmente**: Dedica tiempo cada día a programar
2. **Lee y entiende el código**: No solo copies y pegues
3. **Experimenta**: Modifica ejemplos, prueba variaciones
4. **Resuelve problemas**: Usa Codewars, LeetCode
5. **Aprende en proyectos**: Construye cosas reales
6. **Documenta tu progreso**: Toma notas, crea apuntes
7. **Enseña otros**: Explicar refuerza el aprendizaje

---

## 🚀 Próximos Pasos

Después de completar este roadmap:

- Especializarse en un área específica
- Participar en Kaggle competitions
- Contribuir a proyectos open source
- Crear portfolio de proyectos
- Buscar oportunidades laborales o académicas

---

## 📝 Licencia

Este repositorio es de código abierto. Siéntete libre de usar, modificar y compartir.

---

**Creado**: Julio 2026  
**Última actualización**: Julio 2026

---

> "Los que saben programar, crean el futuro. Los que no lo saben, viven en el que otros crearon."  
> — Anónimo