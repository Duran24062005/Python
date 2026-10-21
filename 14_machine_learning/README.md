## ¿Qué es Machine Learning?

**Machine Learning (ML) o aprendizaje automático** es una rama de la inteligencia artificial que permite que un sistema **aprenda patrones a partir de datos y utilice esos patrones para hacer predicciones o tomar decisiones**, sin que tengamos que programar explícitamente cada regla.

Una forma sencilla de verlo:

> **Programación tradicional:**
> Datos + Reglas → Resultado
>
> **Machine Learning:**
> Datos + Resultados conocidos → Modelo aprendido → Predicción sobre nuevos datos

### Ejemplo sencillo

Imagina que quieres crear un sistema que determine si un correo es **spam**.

Con programación tradicional tendrías que escribir reglas:

```text
SI contiene "gana dinero"
O contiene "premio"
O contiene demasiados enlaces
→ SPAM
```

El problema es que aparecerán nuevos tipos de spam que tus reglas no conocen.

Con Machine Learning puedes darle miles de correos:

```text
Correo 1 → SPAM
Correo 2 → NORMAL
Correo 3 → SPAM
Correo 4 → NORMAL
...
```

El algoritmo encuentra patrones por sí mismo.

Después puedes darle un correo nuevo:

```text
Nuevo correo
      ↓
   Modelo ML
      ↓
  94% SPAM
```

---

# ¿Cómo funciona?

Normalmente tenemos este flujo:

```text
                 DATOS
                   │
                   ▼
          ┌─────────────────┐
          │ Preprocesamiento │
          └────────┬────────┘
                   │
                   ▼
             Datos de
             entrenamiento
                   │
                   ▼
          ┌─────────────────┐
          │ Algoritmo de ML │
          └────────┬────────┘
                   │
                   ▼
                 MODELO
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Datos nuevos       Predicción
```

Por ejemplo, para predecir el precio de una casa:

| Habitaciones |   Área | Ciudad      | Precio |
| -----------: | -----: | ----------- | -----: |
|            2 |  60 m² | Bucaramanga |  $180M |
|            3 |  80 m² | Bucaramanga |  $240M |
|            4 | 120 m² | Bucaramanga |  $350M |
|            3 | 100 m² | Bucaramanga |  $290M |

El modelo intenta encontrar relaciones entre las variables:

```text
Área
  +
Habitaciones
  +
Ubicación
  ↓
Precio
```

Luego:

```text
Nueva casa
3 habitaciones
90 m²
Bucaramanga
       ↓
    MODELO
       ↓
Predicción: $265M
```

No significa que el modelo **"sepa"** que una casa cuesta $265M. Ha aprendido una relación matemática aproximada a partir de los ejemplos.

---

# ¿Qué significa "aprender"?

Esta es probablemente la parte más importante.

Un modelo tiene **parámetros** que se van ajustando durante el entrenamiento.

Simplificando muchísimo:

```text
Datos
  ↓
Predicción
  ↓
Comparar con respuesta real
  ↓
Error
  ↓
Ajustar parámetros
  ↓
Nueva predicción
  ↓
...
```

Por ejemplo:

```text
Precio real:       $300M
Predicción:        $250M

Error:             $50M
```

El algoritmo modifica sus parámetros para intentar reducir ese error.

Después de muchas iteraciones:

```text
Predicción: $250M
       ↓
Predicción: $275M
       ↓
Predicción: $292M
       ↓
Predicción: $298M
       ↓
Predicción: $301M
```

Esto es una simplificación, pero representa la idea fundamental del **entrenamiento**.

---

# ¿Qué tipos de Machine Learning existen?

Principalmente puedes dividirlo en tres grandes categorías.

### 1. Supervised Learning

Tienes datos y la respuesta correcta.

```text
Entrada                Respuesta

Edad = 25              Compra = Sí
Edad = 42              Compra = No
Edad = 31              Compra = Sí
```

El modelo aprende:

```text
Entrada → Respuesta
```

Se utiliza para:

* Predecir precios
* Detectar spam
* Detectar fraude
* Clasificar imágenes
* Predecir abandono de clientes
* Diagnóstico asistido
* Predicción de demanda

Dos tareas muy comunes:

**Clasificación**

```text
Imagen → gato
Imagen → perro
```

**Regresión**

```text
Características → $235.000
```

---

### 2. Unsupervised Learning

Aquí no tienes la respuesta correcta.

Por ejemplo:

```text
Cliente A → compra mucho
Cliente B → compra poco
Cliente C → compra mucho
Cliente D → compra ocasionalmente
...
```

El algoritmo intenta encontrar estructuras o grupos.

Por ejemplo:

```text
             CLIENTES

       ┌───────────────┐
       │ Compradores   │
       │ frecuentes    │
       └───────────────┘

       ┌───────────────┐
       │ Compradores   │
       │ ocasionales   │
       └───────────────┘

       ┌───────────────┐
       │ Poco activos  │
       └───────────────┘
```

Un algoritmo famoso para esto es **K-Means**.

Se utiliza para:

* Segmentación de clientes
* Detección de patrones
* Agrupación de documentos
* Análisis exploratorio
* Detección de anomalías

---

### 3. Reinforcement Learning

Aquí un agente aprende mediante **acciones, recompensas y penalizaciones**.

Por ejemplo, un robot:

```text
             ROBOT
               │
               ▼
          Observa entorno
               │
               ▼
            Acción
               │
        ┌──────┴──────┐
        ▼             ▼
     Buena          Mala
     acción         acción
        │             │
        ▼             ▼
    recompensa     penalización
```

Con suficiente experiencia aprende qué acciones producen mejores resultados.

Se utiliza, entre otras cosas, en:

* Robótica
* Juegos
* Optimización
* Control de sistemas
* Investigación de agentes autónomos

---

# ¿Y Deep Learning?

**Deep Learning es una subárea de Machine Learning.**

La relación sería aproximadamente:

```text
Artificial Intelligence
│
└── Machine Learning
    │
    ├── Classical ML
    │   ├── Linear Regression
    │   ├── Decision Trees
    │   ├── Random Forest
    │   └── K-Means
    │
    └── Deep Learning
        ├── Neural Networks
        ├── CNN
        ├── RNN
        └── Transformers
```

Las **redes neuronales profundas** son especialmente útiles cuando tenemos grandes cantidades de datos complejos.

Por ejemplo:

```text
Imagen
  ↓
Neural Network
  ↓
Características
  ↓
Clasificación
  ↓
"Perro"
```

Esto conecta directamente con **visión computacional**.

---

# ¿Para qué sirve en una empresa?

Aquí es donde ML se vuelve realmente interesante para ti como desarrollador.

Imagina una empresa de logística:

### Problema

Tiene miles de entregas históricas.

Tiene datos como:

```text
distancia
peso
hora
día
ciudad
clima
tráfico
tipo de vehículo
tiempo de entrega
```

Puede entrenar un modelo:

```text
              DATOS HISTÓRICOS
                     │
                     ▼
              MACHINE LEARNING
                     │
                     ▼
           MODELO DE PREDICCIÓN
                     │
              ┌──────┴──────┐
              ▼             ▼
        Tiempo estimado   Riesgo de
        de entrega       retraso
```

Entonces su software podría decir:

> "Esta entrega tiene un 82% de probabilidad de retrasarse."

Eso puede permitir tomar acciones antes de que ocurra el problema.

---

# ML + tu stack

Y aquí hay algo importante para ti.

**No necesitas abandonar tu stack de desarrollo para aprender Machine Learning.**

Puedes construir sistemas así:

```text
                    FRONTEND
                 React / Next.js
                       │
                       ▼
                  BACKEND API
                NestJS / FastAPI
                       │
              ┌────────┴────────┐
              ▼                 ▼
           Database          ML Model
       PostgreSQL/MongoDB      Python
                                │
                       scikit-learn
                       PyTorch
                       TensorFlow
```

Por ejemplo:

```text
React
  ↓
NestJS
  ↓
Python ML Service
  ↓
scikit-learn
  ↓
Predicción
  ↓
NestJS
  ↓
React
```

Eso es perfectamente válido en una arquitectura empresarial.

---

# Un ejemplo que te puede servir

Supongamos que quieres construir para una empresa:

**Sistema de predicción de abandono de clientes.**

Tienes:

```text
Cliente
├── edad
├── compras
├── frecuencia
├── última compra
├── valor promedio
├── número de reclamos
└── tiempo como cliente
```

Y sabes cuáles clientes abandonaron.

Entrenas:

```text
Datos históricos
      ↓
Machine Learning
      ↓
Modelo
```

Después:

```text
Cliente actual
      ↓
Modelo ML
      ↓
87% probabilidad de abandono
```

Tu backend podría entonces:

```text
if probability > 0.80:
    create_alert()
```

Y tu aplicación podría mostrar:

```text
⚠ Cliente con alto riesgo

Probabilidad de abandono: 87%

Motivos principales:
- 90 días sin comprar
- 3 reclamos recientes
- Disminución del 60% en compras
```

Ahí ya estás combinando:

**Software + Backend + Database + Machine Learning + IA.**

---

## Y algo importante

Machine Learning **no es simplemente "usar IA"**.

Un sistema que hace:

```text
Usuario → ChatGPT → Respuesta
```

no necesariamente implica que tú estés haciendo Machine Learning.

En cambio, cuando tú:

```text
recopilas datos
      ↓
limpias datos
      ↓
seleccionas características
      ↓
entrenas un modelo
      ↓
evalúas el modelo
      ↓
lo despliegas
      ↓
lo utilizas para predecir nuevos casos
```

estás construyendo un sistema de **Machine Learning**.

Para tu perfil de desarrollo, yo lo estudiaría en este orden:

**Python → NumPy/Pandas → estadística básica → scikit-learn → regresión/clasificación → árboles/Random Forest → evaluación de modelos → APIs con FastAPI → Docker → despliegue → después PyTorch/Deep Learning.**

Eso te permitiría pasar progresivamente de **"soy desarrollador que consume modelos de IA"** a **"soy desarrollador capaz de construir e integrar sistemas de Machine Learning"**.

<br>
<br>
<br>

---

# Pasos para entrenar un modelo predictivo en Jupyter Notebook

Objetivo: entrenar un modelo de Machine Learning con Python, utilizando `pandas` para manipular datos, `matplotlib` para visualizarlos y `scikit-learn` para entrenar y evaluar el modelo.

## Flujo de trabajo

| Paso                  | Objetivo                           | Descripción                                                                                                          |
| --------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 1. Importar librerías | Preparar el entorno                | Importar `pandas`, `matplotlib` y `sklearn`.                                                                         |
| 2. Cargar datos       | Obtener información                | Leer un archivo CSV o Excel con `pd.read_csv()`.                                                                     |
| 3. Explorar datos     | Conocer el dataset                 | Usar `head()`, `info()`, `describe()` y `isnull().sum()`.                                                            |
| 4. Limpiar datos      | Mejorar su calidad                 | Tratar valores nulos, duplicados y datos incorrectos.                                                                |
| 5. Visualizar datos   | Identificar patrones               | Crear gráficos con `matplotlib` para detectar tendencias y relaciones.                                               |
| 6. Definir variables  | Establecer qué aprenderá el modelo | Separar las características `X` de la variable objetivo `y`.                                                         |
| 7. Dividir los datos  | Medir la generalización            | Separar datos de entrenamiento y prueba con `train_test_split()`.                                                    |
| 8. Entrenar el modelo | Aprender patrones                  | Elegir un algoritmo, como `LinearRegression`, `RandomForestRegressor` o `RandomForestClassifier`, según el problema. |
| 9. Evaluar resultados | Medir el rendimiento               | Usar métricas como MAE y RMSE para regresión, o accuracy, precision y recall para clasificación.                     |
| 10. Guardar el modelo | Reutilizar lo aprendido            | Exportarlo con `joblib` para realizar predicciones futuras.                                                          |

## Estructura práctica del Notebook

Organiza las celdas en este orden:

```
# 1. Importar libreríasimport pandas as pdimport matplotlib.pyplot as pltimport joblibfrom sklearn.model_selection import train_test_splitfrom sklearn.ensemble import RandomForestClassifierfrom sklearn.metrics import accuracy_score# 2. Cargar datosdf = pd.read_csv("datos.csv")# 3. Explorar y limpiarprint(df.head())print(df.info())print(df.isnull().sum())# 4. Visualizardf.hist(figsize=(10, 6))plt.tight_layout()plt.show()# 5. Separar características y objetivoX = df.drop(columns=["objetivo"])y = df["objetivo"]# 6. Dividir datos (80 % entrenamiento, 20 % prueba)X_train, X_test, y_train, y_test = train_test_split(    X, y, test_size=0.2, random_state=42)# 7. Entrenarmodelo = RandomForestClassifier(random_state=42)modelo.fit(X_train, y_train)# 8. Predecir y evaluarpredicciones = modelo.predict(X_test)print("Accuracy:", accuracy_score(y_test, predicciones))# 9. Guardarjoblib.dump(modelo, "modelo.pkl")
```

Este ejemplo supone un problema de clasificación con variables compatibles con el algoritmo. Debes reemplazar `"objetivo"` por la columna que deseas predecir.

## Conceptos clave

- Clasificación: predice categorías, como fraude/no fraude.
- Regresión: predice valores numéricos, como ventas o precios.
- Entrenamiento (`fit`): el modelo aprende patrones a partir de los datos.
- Predicción (`predict`): genera resultados para datos nuevos.
- Evaluación: mide si el modelo generaliza a datos que no utilizó para entrenar.&#x20;

  ![image](https://www.google.com/s2/favicons?domain=https://scikit-learn.org\&sz=32)

  scikit-learn 1.9.1 documentation

  +1



Recomendación: primero define qué quieres predecir y selecciona un dataset adecuado. Luego, realiza la limpieza y el entrenamiento; evita que los datos de prueba intervengan en el aprendizaje o la preparación del modelo.&#x20;

![image](https://www.google.com/s2/favicons?domain=https://scikit-learn.org\&sz=32)

scikit-learn 1.9.1 documentation

+1
