# PhotosManager

Programa de Python para detectar y reconocer caras en fotos usando OpenCV y
DeepFace. El detector predeterminado es `yunet`, más apropiado para fotos con
varias personas, caras pequeñas u ocultas parcialmente.

## Estructura de datos

La estructura de datos utilizada por este proyecto es:

```text
15_facial_recognition/
├── assets/
│   └── carpetas/
│       ├── conocidos/
│       │   ├── monica.png
│       │   └── ...
│       ├── pruebas/
│       │   ├── imagen01.png
│       │   └── ...
│       └── __MACOSX/       # se ignora
└── PhotosManager/
    ├── friendsRecogniser.py
    ├── evaluation.py
    └── evaluation_annotations.template.json
```

Cada imagen de `conocidos` debe mostrar una sola persona y su nombre se obtiene
del nombre del archivo (`ross.png` se etiqueta como `ross`). Las imágenes de
`pruebas` pueden tener varias caras o ninguna. También se ignoran los archivos
de metadatos de macOS (`__MACOSX`, `._*` y `.DS_Store`).

## Instalación

El proyecto requiere Python 3.14 o una versión compatible con las dependencias
de `pyproject.toml`.

Desde la raíz del repositorio:

```bash
uv sync
```

La primera ejecución de DeepFace puede descargar los pesos del modelo elegido.

## Ejecución

Desde la raíz del repositorio:

```bash
uv run python PhotosManager/friendsRecogniser.py
```

Por cada foto de prueba se abre una ventana con las caras anotadas:

- verde + nombre: coincidencia con una persona conocida;
- rojo + `unknown`: no existe una coincidencia verificada.

El programa no anota la región completa de la imagen que DeepFace devuelve como
respaldo cuando no detecta caras. También descarta detecciones menores de 40
píxeles por lado y cajas desconocidas sin landmarks de ambos ojos; estas
validaciones evitan cajas rojas sobre ropa, cabello o elementos del fondo.
Una coincidencia conocida sí puede conservarse aunque tenga los ojos cubiertos.

Presiona cualquier tecla para continuar. `q` o `Esc` terminan la ejecución y
conservan el resumen de las fotos procesadas hasta ese momento.

También se pueden cambiar las carpetas y el modelo:

```bash
uv run python PhotosManager/friendsRecogniser.py \
  --known-dir assets/carpetas/conocidos \
  --test-dir assets/carpetas/pruebas \
  --model-name VGG-Face \
  --detector-backend yunet \
  --distance-metric cosine
```

Si una imagen contiene caras muy pequeñas, se puede reducir el filtro, por
ejemplo con `--min-face-size 24`. Para usar el detector Haar legado se puede
indicar `--detector-backend opencv`, aunque es más sensible a falsos positivos
y a caras tapadas. La primera ejecución con `yunet` puede descargar el modelo
`face_detection_yunet_2023mar.onnx`.

El parámetro `--threshold` permite probar un umbral calibrado explícitamente;
si se omite, DeepFace aplica su umbral preajustado para el modelo y la métrica.

Para comprobar el recorrido en un entorno sin interfaz gráfica se puede usar
`--no-display`; esta opción no sustituye la validación visual del programa.

Al final se imprime un resumen como este (los valores dependen de las
coincidencias reales):

```text
Resumen
Fotos analizadas: 16
Personas conocidas distintas encontradas: 6
Nombres encontrados: chandler, joey, monica, phoebe, rachel, ross
```

El programa no mueve ni copia archivos: procesa, muestra y resume.

## Pruebas

Las pruebas unitarias no necesitan cargar los modelos de DeepFace:

```bash
python3 -m unittest discover -s PhotosManager -p 'test_*.py' -v
```

La prueba real de reconocimiento requiere instalar las dependencias, ejecutar
el comando principal y revisar las anotaciones en la ventana de OpenCV.

## Evaluación cuantitativa

El evaluador separado no modifica el flujo interactivo. Requiere un archivo de
ground truth manual; no supone que el nombre de una fotografía ni el contenido
de `conocidos` sean etiquetas correctas. Usa cajas `[x, y, ancho, alto]`,
emparejamiento uno-a-uno por IoU y registra cada comparación con distancia,
umbral, decisión y tiempo.

Copia [evaluation_annotations.template.json](evaluation_annotations.template.json)
a un archivo de trabajo y completa, por ejemplo:

```bash
cp PhotosManager/evaluation_annotations.template.json \
   PhotosManager/evaluation_annotations.json
```

```json
{
  "version": 1,
  "images": [
    {
      "id": "imagen07",
      "path": "assets/carpetas/pruebas/imagen07.png",
      "split": "test",
      "notes": "dos rostros visibles",
      "faces": [
        {
          "identity": "ross",
          "registered": true,
          "bbox": [378, 63, 208, 208]
        },
        {
          "identity": "rachel",
          "registered": true,
          "bbox": [151, 214, 168, 168]
        }
      ]
    }
  ]
}
```

Ejecuta una comparación reproducible de configuraciones seleccionadas:

```bash
uv run python PhotosManager/evaluation.py \
  --annotations PhotosManager/evaluation_annotations.json \
  --detector-backends opencv,retinaface,yunet \
  --models VGG-Face,Facenet \
  --metrics cosine \
  --iou-threshold 0.5 \
  --output /tmp/photosmanager-evaluation.json
```

El resultado contiene métricas globales de detección (`precision`, `recall`,
`f1`), métricas de identidad, resultados por imagen y una recomendación de
umbral basada únicamente en muestras positivas y negativas anotadas. El
umbral recomendado no se aplica automáticamente al reconocedor; se puede
probar explícitamente con `--threshold` después de revisar el resultado.

Los modelos o detectores seleccionados pueden descargar sus propios pesos en
la primera ejecución. El evaluador no entrena modelos ni descarga variantes no
solicitadas. El repositorio incluye anotaciones manuales iniciales para las 16
imágenes de prueba; si se modifica el dataset, hay que actualizar el JSON
antes de interpretar las métricas.

Las configuraciones iniciales previstas son `VGG-Face`, `Facenet`, `Facenet512`
y `ArcFace`, junto con detectores como `opencv`, `retinaface` y `yunet`, siempre
que la versión instalada de DeepFace y sus pesos sean compatibles. Una variante
no disponible queda registrada como error y no como una métrica inventada.
