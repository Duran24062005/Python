
from pathlib import Path

import cv2
from deepface import DeepFace

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
reference_path = BASE_DIR / "assets" / "capture.jpg"
capture_path = BASE_DIR / "assets" / "capture2.jpg"

# Comprobar que exista la imagen de referencia
if not reference_path.exists():
    raise FileNotFoundError(
        f"No se encontró la imagen de referencia: {reference_path}"
    )

# Capturar una fotografía con la webcam
camera = cv2.VideoCapture(0)

try:
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    # Leer algunos frames para estabilizar la captura
    for _ in range(3):
        success, photo = camera.read()

    if not success:
        raise RuntimeError("No se pudo capturar la fotografía.")

finally:
    camera.release()

# Guardar la fotografía capturada
if not cv2.imwrite(str(capture_path), photo):
    raise RuntimeError("No se pudo guardar la fotografía.")

# Comparar el rostro capturado con la imagen de referencia
result = DeepFace.verify(
    img1_path=str(capture_path),
    img2_path=str(reference_path),
    enforce_detection=True
)

if result["verified"]:
    text = "Alexi"
    color = (0, 255, 0)  # Verde
else:
    text = "Unknown"
    color = (0, 0, 255)  # Rojo

# Detectar los rostros en la fotografía capturada
faces = DeepFace.extract_faces(
    img_path=str(capture_path),
    enforce_detection=True
)

# Dibujar el bounding box alrededor de cada rostro
for face in faces:
    area = face["facial_area"]

    x = area["x"]
    y = area["y"]
    width = area["w"]
    height = area["h"]

    # Rectángulo alrededor del rostro
    cv2.rectangle(
        photo,
        (x, y),
        (x + width, y + height),
        color,
        2
    )

    # Nombre encima del rectángulo
    cv2.putText(
        photo,
        text,
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

# Mostrar el resultado
cv2.imshow("Face Recognition", photo)

cv2.waitKey(0)
cv2.destroyAllWindows()
q