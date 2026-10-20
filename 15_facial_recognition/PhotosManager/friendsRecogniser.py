"""Reconocimiento de personas conocidas en una colección de fotos.

El script está pensado para ejecutarse desde la raíz del repositorio:

    uv run python PhotosManager/friendsRecogniser.py

Las imágenes conocidas se extraen una sola vez y cada cara de una foto de
prueba se compara con esos recortes mediante DeepFace. OpenCV se encarga de
leer las imágenes, dibujar las cajas y mostrar la ventana interactiva.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import cv2 as cv
except ImportError as exc:  # permite ejecutar --help y pruebas sin dependencias
    cv = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR: ImportError | None = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    from deepface import DeepFace
except ImportError as exc:  # permite importar helpers sin descargar modelos
    DeepFace = None  # type: ignore[assignment,misc]
    _DEEPFACE_IMPORT_ERROR: ImportError | None = exc
else:
    _DEEPFACE_IMPORT_ERROR = None


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
PHOTO_DATA_DIR = PROJECT_DIR / "assets" / "carpetas"
DEFAULT_KNOWN_DIR = PHOTO_DATA_DIR / "conocidos"
DEFAULT_TEST_DIR = PHOTO_DATA_DIR / "pruebas"
DEFAULT_DETECTOR_BACKEND = "yunet"
DEFAULT_MIN_FACE_SIZE = 40
KNOWN_EXTENSIONS = frozenset({".png"})
TEST_EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
WINDOW_NAME = "PhotosManager - reconocimiento facial"


@dataclass(frozen=True)
class KnownPerson:
    """Nombre y recorte facial de una persona conocida."""

    name: str
    face: Any


@dataclass(frozen=True)
class FaceMatch:
    """Resultado de comparar una cara de prueba contra las conocidas."""

    name: str
    distance: float | None = None
    threshold: float | None = None
    matched: bool = False

    @property
    def is_known(self) -> bool:
        return self.matched


@dataclass(frozen=True)
class ProcessingSummary:
    """Resumen de una ejecución del reconocedor."""

    photos_analyzed: int
    recognized_people: tuple[str, ...]
    interrupted: bool = False


@dataclass(frozen=True)
class FaceComparison:
    """Comparación individual contra una persona conocida."""

    known_name: str
    distance: float | None
    threshold: float | None
    verified: bool
    error: str | None = None


def _is_ignored_path(path: Path) -> bool:
    """Indica si una ruta pertenece a metadatos de macOS que no son fotos."""

    return "__MACOSX" in path.parts or path.name == ".DS_Store" or path.name.startswith("._")


def iter_image_files(directory: Path, extensions: Iterable[str]) -> tuple[Path, ...]:
    """Devuelve archivos de imagen directos, ordenados y sin metadatos basura."""

    if not directory.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {directory}")

    normalized_extensions = {extension.lower() for extension in extensions}
    files = (
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in normalized_extensions
        and not _is_ignored_path(path)
    )
    return tuple(sorted(files, key=lambda path: path.name.casefold()))


def _require_opencv() -> Any:
    """Comprueba que OpenCV esté instalado."""

    if cv is None:
        raise RuntimeError(
            "OpenCV no está instalado. Ejecuta 'uv sync' y vuelve a intentarlo."
        ) from _CV2_IMPORT_ERROR
    return cv


def _require_deepface() -> Any:
    """Comprueba que DeepFace esté instalado."""

    if DeepFace is None:
        raise RuntimeError(
            "DeepFace no está instalado. Ejecuta 'uv sync' y vuelve a intentarlo."
        ) from _DEEPFACE_IMPORT_ERROR
    return DeepFace


def _require_dependencies() -> tuple[Any, Any]:
    """Comprueba que OpenCV y DeepFace estén instalados antes de procesar."""

    return _require_opencv(), _require_deepface()


def extract_faces(
    image: Any,
    *,
    detector_backend: str,
    enforce_detection: bool,
    min_face_size: int = DEFAULT_MIN_FACE_SIZE,
) -> list[dict[str, Any]]:
    """Extrae caras válidas y conserva su caja junto con el recorte alineado.

    DeepFace devuelve una región que ocupa toda la imagen cuando se usa
    ``enforce_detection=False`` y no encuentra ninguna cara. Esa región tiene
    confianza cero y no debe entrar al reconocimiento. También se descartan
    detecciones demasiado pequeñas, que suelen ser falsos positivos del
    detector Haar/OpenCV sobre el fondo.
    """

    if min_face_size < 0:
        raise ValueError("min_face_size no puede ser negativo")

    deepface = _require_deepface()
    detected_faces = deepface.extract_faces(
        img_path=image,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=True,
        color_face="rgb",
        normalize_face=False,
    )

    if isinstance(detected_faces, dict):
        detected_faces = [detected_faces]
    valid_faces: list[dict[str, Any]] = []
    for face in detected_faces:
        if (
            not isinstance(face, dict)
            or "face" not in face
            or "facial_area" not in face
        ):
            continue

        area = face["facial_area"]
        if not isinstance(area, dict):
            continue

        width = max(0, int(area.get("w", 0)))
        height = max(0, int(area.get("h", 0)))
        confidence = _as_distance(face.get("confidence"))
        if width < min_face_size or height < min_face_size:
            continue
        if confidence is not None and confidence <= 0:
            continue
        valid_faces.append(face)

    return valid_faces


def load_known_people(
    paths: Sequence[Path],
    *,
    detector_backend: str,
    min_face_size: int = DEFAULT_MIN_FACE_SIZE,
) -> tuple[KnownPerson, ...]:
    """Valida y extrae exactamente una cara de cada imagen conocida."""

    if not paths:
        raise ValueError("No se encontraron imágenes .png en la carpeta de conocidos.")

    people: list[KnownPerson] = []
    names_seen: set[str] = set()

    for path in paths:
        name = path.stem.strip()
        normalized_name = name.casefold()
        if not name:
            raise ValueError(f"La imagen conocida tiene un nombre vacío: {path.name}")
        if normalized_name in names_seen:
            raise ValueError(f"Hay nombres conocidos duplicados: {name}")
        names_seen.add(normalized_name)

        faces = extract_faces(
            str(path),
            detector_backend=detector_backend,
            enforce_detection=True,
            min_face_size=min_face_size,
        )
        if len(faces) != 1:
            raise ValueError(
                f"La imagen conocida '{path.name}' debe contener exactamente una cara; "
                f"se detectaron {len(faces)}."
            )
        people.append(KnownPerson(name=name, face=faces[0]["face"]))

    return tuple(people)


def _as_distance(value: Any) -> float | None:
    """Convierte una distancia de DeepFace a float sin romper el flujo."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_better_match(candidate: FaceMatch, current: FaceMatch | None) -> bool:
    """Compara dos coincidencias verificadas priorizando la menor distancia."""

    if current is None:
        return True
    if candidate.distance is None:
        return current.distance is None
    return current.distance is None or candidate.distance < current.distance


def find_best_match(
    face: Any,
    known_people: Sequence[KnownPerson],
    *,
    model_name: str,
    distance_metric: str,
    threshold: float | None = None,
) -> FaceMatch:
    """Devuelve la mejor persona verificada o ``unknown`` si no hay coincidencia."""

    best_match: FaceMatch | None = None

    for comparison in compare_face_to_known_people(
        face,
        known_people,
        model_name=model_name,
        distance_metric=distance_metric,
        threshold=threshold,
    ):
        if comparison.error is not None or not comparison.verified:
            continue
        if comparison.distance is None:
            continue

        candidate = FaceMatch(
            name=comparison.known_name,
            distance=comparison.distance,
            threshold=comparison.threshold,
            matched=True,
        )
        if _is_better_match(candidate, best_match):
            best_match = candidate

    return best_match or FaceMatch(name="unknown")


def compare_face_to_known_people(
    face: Any,
    known_people: Sequence[KnownPerson],
    *,
    model_name: str,
    distance_metric: str,
    threshold: float | None = None,
    emit_warnings: bool = True,
) -> tuple[FaceComparison, ...]:
    """Registra todas las comparaciones, incluidas las rechazadas o fallidas."""

    deepface = _require_deepface()
    comparisons: list[FaceComparison] = []

    for person in known_people:
        try:
            result = deepface.verify(
                img1_path=person.face,
                img2_path=face,
                model_name=model_name,
                detector_backend="skip",
                distance_metric=distance_metric,
                enforce_detection=False,
                align=False,
                silent=True,
                threshold=threshold,
            )
        except Exception as exc:  # una comparación fallida no invalida las demás
            if emit_warnings:
                print(
                    f"Advertencia: no se pudo comparar una cara con '{person.name}': {exc}",
                    file=sys.stderr,
                )
            comparisons.append(
                FaceComparison(
                    known_name=person.name,
                    distance=None,
                    threshold=None,
                    verified=False,
                    error=str(exc),
                )
            )
            continue

        comparisons.append(
            FaceComparison(
                known_name=person.name,
                distance=_as_distance(result.get("distance")),
                threshold=_as_distance(result.get("threshold")),
                verified=bool(result.get("verified", False)),
            )
        )

    return tuple(comparisons)


def _box_from_area(area: dict[str, Any], image_shape: Sequence[int]) -> tuple[int, int, int, int]:
    """Normaliza y recorta la caja facial a los límites de la imagen."""

    height, width = int(image_shape[0]), int(image_shape[1])
    x = max(0, min(width, int(area.get("x", 0))))
    y = max(0, min(height, int(area.get("y", 0))))
    box_width = max(0, int(area.get("w", 0)))
    box_height = max(0, int(area.get("h", 0)))
    right = max(x, min(width, x + box_width))
    bottom = max(y, min(height, y + box_height))
    return x, y, right, bottom


def draw_face_annotation(image: Any, area: dict[str, Any], match: FaceMatch) -> None:
    """Dibuja la caja y la etiqueta verde/roja sobre una imagen BGR de OpenCV."""

    opencv, _ = _require_dependencies()
    x, y, right, bottom = _box_from_area(area, image.shape)
    color = (0, 255, 0) if match.is_known else (0, 0, 255)
    label = match.name
    thickness = 2
    opencv.rectangle(image, (x, y), (right, bottom), color, thickness)

    font = opencv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    (text_width, text_height), baseline = opencv.getTextSize(
        label, font, font_scale, thickness
    )
    label_left = x
    label_top = y - text_height - baseline - 6
    if label_top < 0:
        label_top = min(bottom, y + 2)
    label_bottom = label_top + text_height + baseline + 6
    label_right = min(image.shape[1], label_left + text_width + 8)
    opencv.rectangle(
        image,
        (label_left, label_top),
        (label_right, label_bottom),
        color,
        opencv.FILLED,
    )
    opencv.putText(
        image,
        label,
        (label_left + 4, label_bottom - baseline - 3),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        opencv.LINE_AA,
    )


def _has_eye_landmarks(area: dict[str, Any]) -> bool:
    """Indica si el detector encontró ambos ojos en la región facial."""

    return area.get("left_eye") is not None and area.get("right_eye") is not None


def _should_annotate_face(area: dict[str, Any], match: FaceMatch) -> bool:
    """Evita pintar falsos positivos sin landmarks, salvo si fueron reconocidos."""

    # Una cara conocida puede estar parcialmente tapada y aun así producir un
    # embedding fiable. Para una cara desconocida exigimos la señal geométrica
    # mínima del detector; esto elimina cajas que Haar suele colocar sobre
    # ropa, cabello o elementos del fondo.
    return match.is_known or _has_eye_landmarks(area)


def process_photo(
    path: Path,
    known_people: Sequence[KnownPerson],
    *,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    threshold: float | None = None,
    min_face_size: int = DEFAULT_MIN_FACE_SIZE,
) -> tuple[Any, set[str]]:
    """Procesa una foto y devuelve la imagen anotada y los nombres encontrados."""

    opencv = _require_opencv()
    image = opencv.imread(str(path), opencv.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen de prueba: {path}")

    faces = extract_faces(
        image,
        detector_backend=detector_backend,
        enforce_detection=False,
        min_face_size=min_face_size,
    )
    recognized_people: set[str] = set()
    for face in faces:
        match = find_best_match(
            face["face"],
            known_people,
            model_name=model_name,
            distance_metric=distance_metric,
            threshold=threshold,
        )
        if not _should_annotate_face(face["facial_area"], match):
            continue
        draw_face_annotation(image, face["facial_area"], match)
        if match.is_known:
            recognized_people.add(match.name)

    return image, recognized_people


def process_test_images(
    known_people: Sequence[KnownPerson],
    test_paths: Sequence[Path],
    *,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    threshold: float | None = None,
    min_face_size: int = DEFAULT_MIN_FACE_SIZE,
    show: bool = True,
) -> ProcessingSummary:
    """Procesa las pruebas en orden y muestra cada resultado si corresponde."""

    opencv = _require_opencv()
    photos_analyzed = 0
    recognized_people: set[str] = set()
    interrupted = False

    try:
        for path in test_paths:
            print(f"Procesando: {path.name}")
            try:
                annotated_image, photo_people = process_photo(
                    path,
                    known_people,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    threshold=threshold,
                    min_face_size=min_face_size,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"Advertencia: se omite '{path.name}': {exc}", file=sys.stderr)
                continue

            photos_analyzed += 1
            recognized_people.update(photo_people)

            if not show:
                continue

            opencv.imshow(WINDOW_NAME, annotated_image)
            key = opencv.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                interrupted = True
                break
    finally:
        if show:
            opencv.destroyAllWindows()

    return ProcessingSummary(
        photos_analyzed=photos_analyzed,
        recognized_people=tuple(sorted(recognized_people, key=str.casefold)),
        interrupted=interrupted,
    )


def print_summary(summary: ProcessingSummary) -> None:
    """Imprime el resumen solicitado por el programa."""

    print("\nResumen")
    print(f"Fotos analizadas: {summary.photos_analyzed}")
    print(
        "Personas conocidas distintas encontradas: "
        f"{len(summary.recognized_people)}"
    )
    if summary.recognized_people:
        print("Nombres encontrados: " + ", ".join(summary.recognized_people))
    else:
        print("Nombres encontrados: ninguno")
    if summary.interrupted:
        print("Procesamiento detenido por el usuario.")


def build_parser() -> argparse.ArgumentParser:
    """Construye la interfaz de línea de comandos."""

    parser = argparse.ArgumentParser(
        description="Detecta y reconoce caras en assets/carpetas/pruebas."
    )
    parser.add_argument(
        "--known-dir",
        type=Path,
        default=DEFAULT_KNOWN_DIR,
        help="carpeta con imágenes .png de una persona por archivo",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="carpeta con imágenes que se van a procesar",
    )
    parser.add_argument(
        "--model-name",
        default="VGG-Face",
        help="modelo de DeepFace (por defecto: VGG-Face)",
    )
    parser.add_argument(
        "--detector-backend",
        default=DEFAULT_DETECTOR_BACKEND,
        help="detector de DeepFace (por defecto: yunet)",
    )
    parser.add_argument(
        "--distance-metric",
        default="cosine",
        help="métrica de distancia de DeepFace (por defecto: cosine)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "umbral explícito de verificación; por defecto DeepFace usa el "
            "umbral calibrado para el modelo y la métrica"
        ),
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=DEFAULT_MIN_FACE_SIZE,
        help=(
            "tamaño mínimo de ancho/alto para una detección en píxeles "
            f"(por defecto: {DEFAULT_MIN_FACE_SIZE})"
        ),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="procesa sin abrir ventanas; útil para una validación sin GUI",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del programa."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.threshold is not None and args.threshold < 0:
            raise ValueError("threshold no puede ser negativo")
        known_paths = iter_image_files(args.known_dir, KNOWN_EXTENSIONS)
        test_paths = iter_image_files(args.test_dir, TEST_EXTENSIONS)
        if not test_paths:
            raise ValueError("No se encontraron imágenes de prueba para procesar.")

        known_people = load_known_people(
            known_paths,
            detector_backend=args.detector_backend,
            min_face_size=args.min_face_size,
        )
        summary = process_test_images(
            known_people,
            test_paths,
            model_name=args.model_name,
            detector_backend=args.detector_backend,
            distance_metric=args.distance_metric,
            threshold=args.threshold,
            min_face_size=args.min_face_size,
            show=not args.no_display,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
