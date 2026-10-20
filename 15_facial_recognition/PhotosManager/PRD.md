# PRD: reconocimiento facial para `PhotosManager`

## Problema y objetivo

El proyecto necesita recorrer fotos de prueba, detectar todas las caras visibles y
compararlas con una colección de fotos conocidas. Cada cara debe quedar
identificada visualmente y el proceso debe terminar con un resumen de resultados.

## Alcance

- Leer personas conocidas desde `assets/carpetas/conocidos/*.png`.
- Usar el nombre del archivo, sin extensión, como nombre de la persona.
- Leer fotos de prueba desde `assets/carpetas/pruebas`.
- Ignorar `__MACOSX`, archivos `._*` y `.DS_Store`.
- Detectar cero, una o varias caras en cada foto de prueba.
- Marcar coincidencias con rectángulo verde y el nombre de la persona.
- Marcar caras sin coincidencia con rectángulo rojo y `unknown`.
- Mostrar cada foto y esperar una tecla antes de avanzar.
- Imprimir cantidad de fotos analizadas y personas conocidas distintas encontradas.

El programa no moverá, copiará ni sobrescribirá las fotos. La organización
solicitada se limita a reconocer y anotar visualmente durante la ejecución.

## Enfoque de implementación

1. Resolver las carpetas por defecto relativas a este archivo, permitiendo
   reemplazarlas con argumentos de línea de comandos.
2. Detectar y extraer una única cara de cada imagen conocida usando `yunet` por
   defecto. Si una imagen no contiene exactamente una cara, se detiene con un
   mensaje accionable.
3. Detectar todas las caras de cada imagen de prueba con
   `DeepFace.extract_faces(..., enforce_detection=False)`.
4. Ignorar la región de respaldo de tamaño completo que DeepFace devuelve con
   confianza cero cuando no encuentra caras y descartar detecciones menores de
   40 píxeles por lado. Las cajas desconocidas sin landmarks de ambos ojos no
   se anotan para evitar falsos positivos del detector Haar; las coincidencias
   conocidas sí se conservan aunque estén parcialmente ocultas.
5. Comparar cada recorte facial alineado con los recortes conocidos usando
   `DeepFace.verify`. Se elige la coincidencia verificada con menor distancia.
6. Dibujar las anotaciones con OpenCV y usar `cv2.waitKey(0)` para esperar al
   usuario. `q` y `Esc` permiten terminar anticipadamente sin perder el resumen.

## Actores y reglas

- La persona que ejecuta el script mantiene la colección de referencia y decide
  cuándo pasar a la siguiente foto.
- Cada archivo conocido representa a una sola persona y debe contener una sola
  cara detectable.
- Los nombres se comparan sin distinguir mayúsculas para detectar duplicados,
  pero la etiqueta conserva el nombre original del archivo.
- Una cara solo cuenta como encontrada si DeepFace devuelve `verified=True`.

## Impacto, riesgos y validación

- DeepFace puede descargar pesos de modelos durante la primera ejecución.
- La calidad depende del modelo, detector, iluminación, pose y calidad de las
  fotos de referencia. Se pueden ajustar `--model-name`, `--detector-backend`,
  `--distance-metric` y `--min-face-size` sin cambiar el código. `yunet`
  requiere descargar su modelo ONNX en la primera ejecución; `opencv` queda
  disponible como alternativa local.
- El funcionamiento local completo requiere una sesión gráfica para abrir las
  ventanas de OpenCV. La opción `--no-display` permite validar el recorrido en
  un entorno sin interfaz gráfica, pero no reemplaza la revisión visual.
- Se validarán las rutas y las imágenes ilegibles antes de procesarlas. Las
  pruebas unitarias cubren el filtrado de archivos, la eliminación de regiones
  de respaldo/detecciones pequeñas, la selección de la mejor coincidencia y
  la supresión de falsos positivos sin landmarks, sin descargar modelos.

## Evaluación reproducible

La evaluación cuantitativa vive en `PhotosManager/evaluation.py` y no cambia la
ejecución interactiva de `friendsRecogniser.py`.

### Anotaciones

`evaluation_annotations.template.json` define un formato JSON manual con:

- `id`, `path`, `split` y `notes` por imagen.
- `identity`, `registered` y `bbox: [x, y, ancho, alto]` por rostro.
- `identity: null` y `registered: false` para personas desconocidas.

El cargador valida identificadores duplicados, cajas no finitas y la coherencia
entre identidad y registro. No infiere ground truth a partir de los nombres de
archivos. La plantilla vacía no produce métricas: primero debe completarse con
anotaciones reales y dividirse en `validation`/`test` cuando se calibren
umbrales.

### Métricas y trazabilidad

- Las cajas se emparejan una sola vez mediante IoU; el umbral predeterminado es
  `0.5` y se puede cambiar con `--iou-threshold`.
- Por imagen se registran cajas, confianza solo si DeepFace la devuelve,
  tiempos de detección y reconocimiento, comparaciones contra cada persona,
  distancia, umbral devuelto por DeepFace, decisión y errores.
- Globalmente se calculan verdaderos positivos, falsos positivos, falsos
  negativos, precisión, recall y F1 de detección.
- En reconocimiento se separan aciertos, falsas aceptaciones, falsos rechazos
  y clasificaciones erróneas. La exactitud se informa sobre rostros reales
  emparejados, sin ocultar los fallos de detección.
- La calibración busca el umbral que maximiza F1 sobre las distancias anotadas.
  Se informa como recomendación y no reemplaza automáticamente los umbrales
  preajustados de DeepFace por modelo y métrica.

### Comparación de configuraciones

El comando acepta listas separadas por comas de detectores, modelos y métricas;
evalúa el producto cartesiano y guarda un JSON con resultados individuales y
configuración. Solo se cargan las variantes solicitadas. La falta de pesos o
compatibilidad de una variante queda registrada como error de esa configuración
sin inventar métricas.
