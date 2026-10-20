# PRD: reconocimiento facial para `PhotosManager`

## Problema y objetivo

El proyecto necesita recorrer fotos de prueba, detectar todas las caras visibles y
compararlas con una colección de fotos conocidas. Cada cara debe quedar
identificada visualmente y el proceso debe terminar con un resumen de resultados.

## Alcance

- Leer personas conocidas desde `carpetas/conocidos/*.png`.
- Usar el nombre del archivo, sin extensión, como nombre de la persona.
- Leer fotos de prueba desde `carpetas/pruebas`.
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
2. Detectar y extraer una única cara de cada imagen conocida. Si una imagen no
   contiene exactamente una cara, se detiene con un mensaje accionable.
3. Detectar todas las caras de cada imagen de prueba con
   `DeepFace.extract_faces(..., enforce_detection=False)`.
4. Comparar cada recorte facial alineado con los recortes conocidos usando
   `DeepFace.verify`. Se elige la coincidencia verificada con menor distancia.
5. Dibujar las anotaciones con OpenCV y usar `cv2.waitKey(0)` para esperar al
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
  fotos de referencia. Se pueden ajustar `--model-name`,
  `--detector-backend` y `--distance-metric` sin cambiar el código.
- El funcionamiento local completo requiere una sesión gráfica para abrir las
  ventanas de OpenCV. La opción `--no-display` permite validar el recorrido en
  un entorno sin interfaz gráfica, pero no reemplaza la revisión visual.
- Se validarán las rutas y las imágenes ilegibles antes de procesarlas. Las
  pruebas unitarias cubren el filtrado de archivos y la elección de la mejor
  coincidencia sin descargar modelos.
