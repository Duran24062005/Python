# PhotosManager

Programa de Python para detectar y reconocer caras en fotos usando OpenCV y
DeepFace.

## Estructura de datos

La estructura incluida en este proyecto es:

```text
PhotosManager/
├── friendsRecogniser.py
└── carpetas/
    ├── conocidos/
    │   ├── monica.png
    │   └── ...
    ├── pruebas/
    │   ├── imagen01.png
    │   └── ...
    └── __MACOSX/       # se ignora
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

Presiona cualquier tecla para continuar. `q` o `Esc` terminan la ejecución y
conservan el resumen de las fotos procesadas hasta ese momento.

También se pueden cambiar las carpetas y el modelo:

```bash
uv run python PhotosManager/friendsRecogniser.py \
  --known-dir PhotosManager/carpetas/conocidos \
  --test-dir PhotosManager/carpetas/pruebas \
  --model-name VGG-Face \
  --detector-backend opencv \
  --distance-metric cosine
```

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
