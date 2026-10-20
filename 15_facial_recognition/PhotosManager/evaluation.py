"""Evaluación reproducible de detección y reconocimiento facial.

El módulo no infiere etiquetas a partir del nombre de una foto. Solo calcula
métricas cuando las imágenes tienen anotaciones explícitas en JSON.

Ejemplo:

    uv run python PhotosManager/evaluation.py \
        --annotations PhotosManager/evaluation_annotations.json \
        --detector-backends opencv,yunet \
        --models VGG-Face,Facenet \
        --metrics cosine \
        --output /tmp/photosmanager-evaluation.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from PhotosManager import friendsRecogniser as recogniser
except ModuleNotFoundError:  # ejecución directa: python PhotosManager/evaluation.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from PhotosManager import friendsRecogniser as recogniser


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_KNOWN_DIR = recogniser.DEFAULT_KNOWN_DIR
DEFAULT_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class GroundTruthFace:
    """Anotación manual de una cara dentro de una imagen."""

    identity: str | None
    registered: bool
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class AnnotatedImage:
    """Imagen y sus anotaciones de referencia."""

    image_id: str
    path: Path
    split: str
    faces: tuple[GroundTruthFace, ...]
    notes: str | None = None


@dataclass(frozen=True)
class DetectionPair:
    """Emparejamiento entre una detección y una anotación real."""

    prediction_index: int
    ground_truth_index: int
    iou: float


@dataclass(frozen=True)
class CalibrationSample:
    """Distancia etiquetada para calibrar un umbral concreto."""

    distance: float
    positive: bool


def _parse_bbox(value: Any, *, context: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{context}: bbox debe ser [x, y, w, h]")
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: bbox contiene valores no numéricos") from exc
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError(f"{context}: bbox contiene valores no finitos")
    if bbox[2] < 0 or bbox[3] < 0:
        raise ValueError(f"{context}: el ancho y el alto no pueden ser negativos")
    return bbox


def load_annotations(path: Path, *, project_dir: Path = PROJECT_DIR) -> tuple[AnnotatedImage, ...]:
    """Carga y valida anotaciones JSON sin inventar ground truth."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        template = path.with_name("evaluation_annotations.template.json")
        hint = (
            f" Crea uno a partir de la plantilla con: "
            f"cp {template} {path}"
            if template.exists()
            else ""
        )
        raise FileNotFoundError(
            f"No se pudo leer el archivo de anotaciones: {path}.{hint}"
        ) from exc
    except OSError as exc:
        raise FileNotFoundError(f"No se pudo leer el archivo de anotaciones: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"El archivo de anotaciones no es JSON válido: {path}") from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("El JSON debe contener una lista raíz llamada 'images'")

    annotations: list[AnnotatedImage] = []
    ids_seen: set[str] = set()
    for index, raw_image in enumerate(payload["images"]):
        context = f"images[{index}]"
        if not isinstance(raw_image, dict):
            raise ValueError(f"{context} debe ser un objeto")
        image_id = str(raw_image.get("id", "")).strip()
        raw_path = str(raw_image.get("path", "")).strip()
        split = str(raw_image.get("split", "test")).strip() or "test"
        if not image_id or not raw_path:
            raise ValueError(f"{context} requiere 'id' y 'path'")
        if image_id.casefold() in ids_seen:
            raise ValueError(f"Identificador de imagen duplicado: {image_id}")
        ids_seen.add(image_id.casefold())

        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = project_dir / image_path

        raw_faces = raw_image.get("faces", [])
        if not isinstance(raw_faces, list):
            raise ValueError(f"{context}.faces debe ser una lista")
        faces: list[GroundTruthFace] = []
        for face_index, raw_face in enumerate(raw_faces):
            face_context = f"{context}.faces[{face_index}]"
            if not isinstance(raw_face, dict):
                raise ValueError(f"{face_context} debe ser un objeto")
            identity_value = raw_face.get("identity")
            identity = None if identity_value in (None, "", "unknown") else str(identity_value)
            registered_value = raw_face.get("registered", identity is not None)
            if not isinstance(registered_value, bool):
                raise ValueError(f"{face_context}: 'registered' debe ser booleano")
            registered = registered_value
            if registered and identity is None:
                raise ValueError(
                    f"{face_context}: una cara registrada requiere 'identity'"
                )
            faces.append(
                GroundTruthFace(
                    identity=identity,
                    registered=registered,
                    bbox=_parse_bbox(raw_face.get("bbox"), context=face_context),
                )
            )

        notes_value = raw_image.get("notes")
        notes = None if notes_value is None else str(notes_value)
        annotations.append(
            AnnotatedImage(
                image_id=image_id,
                path=image_path,
                split=split,
                faces=tuple(faces),
                notes=notes,
            )
        )

    return tuple(annotations)


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Calcula IoU para cajas ``[x, y, w, h]``."""

    first_x, first_y, first_w, first_h = first
    second_x, second_y, second_w, second_h = second
    first_right, first_bottom = first_x + first_w, first_y + first_h
    second_right, second_bottom = second_x + second_w, second_y + second_h
    intersection_width = max(0.0, min(first_right, second_right) - max(first_x, second_x))
    intersection_height = max(0.0, min(first_bottom, second_bottom) - max(first_y, second_y))
    intersection = intersection_width * intersection_height
    union = first_w * first_h + second_w * second_h - intersection
    return 0.0 if union <= 0 else intersection / union


def match_detections(
    predicted_boxes: Sequence[Sequence[float]],
    ground_truth_boxes: Sequence[Sequence[float]],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[DetectionPair, ...]:
    """Empareja cajas por IoU descendente sin reutilizar ninguna caja."""

    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold debe estar entre 0 y 1")

    candidates = [
        (bbox_iou(predicted, truth), prediction_index, truth_index)
        for prediction_index, predicted in enumerate(predicted_boxes)
        for truth_index, truth in enumerate(ground_truth_boxes)
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    pairs: list[DetectionPair] = []
    for iou, prediction_index, truth_index in candidates:
        if iou < iou_threshold:
            break
        if prediction_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(prediction_index)
        used_truth.add(truth_index)
        pairs.append(DetectionPair(prediction_index, truth_index, iou))
    return tuple(pairs)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def detection_metrics(*, true_positives: int, predicted: int, ground_truth: int) -> dict[str, Any]:
    false_positives = predicted - true_positives
    false_negatives = ground_truth - true_positives
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _normalise_identity(identity: str | None, registered: bool) -> str | None:
    if not registered or identity is None or identity.casefold() == "unknown":
        return None
    return identity.casefold()


def _select_comparison(
    comparisons: Sequence[recogniser.FaceComparison],
) -> recogniser.FaceComparison | None:
    valid = [
        comparison
        for comparison in comparisons
        if comparison.error is None
        and comparison.verified
        and comparison.distance is not None
    ]
    return min(valid, key=lambda comparison: comparison.distance) if valid else None


def calibrate_threshold(samples: Sequence[CalibrationSample]) -> dict[str, Any]:
    """Busca el umbral que maximiza F1 sobre muestras anotadas."""

    usable = tuple(sample for sample in samples if math.isfinite(sample.distance))
    positives = sum(sample.positive for sample in usable)
    negatives = len(usable) - positives
    if not usable or positives == 0 or negatives == 0:
        return {
            "available": False,
            "reason": "se requieren muestras positivas y negativas",
            "samples": len(usable),
            "positive_samples": positives,
            "negative_samples": negatives,
        }

    candidates = sorted({sample.distance for sample in usable})
    best: tuple[float, float, int, int, int] | None = None
    for threshold in candidates:
        true_positive = sum(sample.positive and sample.distance <= threshold for sample in usable)
        false_positive = sum(not sample.positive and sample.distance <= threshold for sample in usable)
        false_negative = positives - true_positive
        precision = _ratio(true_positive, true_positive + false_positive) or 0.0
        recall = _ratio(true_positive, true_positive + false_negative) or 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        candidate = (f1, -threshold, true_positive, false_positive, false_negative)
        if best is None or candidate > best:
            best = candidate

    assert best is not None
    return {
        "available": True,
        "recommended_threshold": -best[1],
        "f1": best[0],
        "true_positives": best[2],
        "false_positives": best[3],
        "false_negatives": best[4],
        "samples": len(usable),
        "positive_samples": positives,
        "negative_samples": negatives,
    }


def evaluate_dataset(
    annotations: Sequence[AnnotatedImage],
    known_people: Sequence[recogniser.KnownPerson],
    *,
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    threshold: float | None = None,
    min_face_size: int = recogniser.DEFAULT_MIN_FACE_SIZE,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Ejecuta una configuración y devuelve resultados por imagen y globales."""

    opencv = recogniser._require_opencv()
    image_results: list[dict[str, Any]] = []
    calibration_samples: list[CalibrationSample] = []
    total_tp = total_predicted = total_ground_truth = 0
    recognition = {
        "matched_ground_truth_faces": 0,
        "correct": 0,
        "false_acceptances": 0,
        "false_rejections": 0,
        "misclassifications": 0,
    }

    for annotation in annotations:
        image_result: dict[str, Any] = {
            "id": annotation.image_id,
            "path": str(annotation.path),
            "split": annotation.split,
            "ground_truth_faces": len(annotation.faces),
            "detections": [],
        }
        image = opencv.imread(str(annotation.path), opencv.IMREAD_COLOR)
        if image is None:
            image_result["error"] = "no se pudo leer la imagen"
            image_result["detection_time_ms"] = None
            image_results.append(image_result)
            total_ground_truth += len(annotation.faces)
            recognition["false_rejections"] += sum(
                face.registered for face in annotation.faces
            )
            continue

        detection_started = time.perf_counter()
        try:
            faces = recogniser.extract_faces(
                image,
                detector_backend=detector_backend,
                enforce_detection=False,
                min_face_size=min_face_size,
            )
        except Exception as exc:
            image_result["error"] = str(exc)
            image_result["detection_time_ms"] = round((time.perf_counter() - detection_started) * 1000, 3)
            image_results.append(image_result)
            total_ground_truth += len(annotation.faces)
            recognition["false_rejections"] += sum(
                face.registered for face in annotation.faces
            )
            continue

        image_result["detection_time_ms"] = round((time.perf_counter() - detection_started) * 1000, 3)
        image_result["image_size"] = {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }
        predicted_boxes: list[tuple[float, float, float, float]] = []
        for face in faces:
            area = face["facial_area"]
            bbox = tuple(float(area.get(key, 0)) for key in ("x", "y", "w", "h"))
            predicted_boxes.append(bbox)
            recognition_started = time.perf_counter()
            comparisons = recogniser.compare_face_to_known_people(
                face["face"],
                known_people,
                model_name=model_name,
                distance_metric=distance_metric,
                threshold=threshold,
                emit_warnings=False,
            )
            selected = _select_comparison(comparisons)
            recognition_time_ms = round((time.perf_counter() - recognition_started) * 1000, 3)
            image_result["detections"].append(
                {
                    "bbox": list(bbox),
                    "confidence": face.get("confidence"),
                    "identity_predicted": selected.known_name if selected else "unknown",
                    "distance": selected.distance if selected else None,
                    "threshold": selected.threshold if selected else threshold,
                    "verified": selected is not None,
                    "recognition_time_ms": recognition_time_ms,
                    "comparisons": [
                        {
                            "identity_candidate": comparison.known_name,
                            "distance": comparison.distance,
                            "threshold": comparison.threshold,
                            "verified": comparison.verified,
                            "error": comparison.error,
                        }
                        for comparison in comparisons
                    ],
                }
            )

        pairs = match_detections(
            predicted_boxes,
            [face.bbox for face in annotation.faces],
            iou_threshold=iou_threshold,
        )
        pair_by_prediction = {pair.prediction_index: pair for pair in pairs}
        total_tp += len(pairs)
        total_predicted += len(predicted_boxes)
        total_ground_truth += len(annotation.faces)

        for prediction_index, detection in enumerate(image_result["detections"]):
            pair = pair_by_prediction.get(prediction_index)
            if pair is None:
                detection.update({"ground_truth_index": None, "iou": None})
                continue
            truth = annotation.faces[pair.ground_truth_index]
            truth_identity = _normalise_identity(truth.identity, truth.registered)
            predicted_identity = _normalise_identity(
                detection["identity_predicted"],
                detection["identity_predicted"] != "unknown",
            )
            detection.update(
                {
                    "ground_truth_index": pair.ground_truth_index,
                    "iou": pair.iou,
                    "identity_real": truth.identity,
                    "identity_correct": predicted_identity == truth_identity,
                    "false_acceptance": truth_identity is None and predicted_identity is not None,
                    "false_rejection": truth_identity is not None and predicted_identity is None,
                }
            )
            recognition["matched_ground_truth_faces"] += 1
            if predicted_identity == truth_identity:
                recognition["correct"] += 1
            elif truth_identity is None and predicted_identity is not None:
                recognition["false_acceptances"] += 1
            elif truth_identity is not None and predicted_identity is None:
                recognition["false_rejections"] += 1
            else:
                recognition["misclassifications"] += 1

            for comparison in detection["comparisons"]:
                candidate_identity = _normalise_identity(
                    comparison["identity_candidate"],
                    True,
                )
                is_true_identity = (
                    truth_identity is not None
                    and candidate_identity == truth_identity
                )
                comparison.update(
                    {
                        "identity_real": truth.identity,
                        "is_true_identity": is_true_identity,
                        "decision_correct": comparison["verified"] == is_true_identity,
                        "false_acceptance": (
                            comparison["verified"] and not is_true_identity
                        ),
                        "false_rejection": (
                            is_true_identity and not comparison["verified"]
                        ),
                    }
                )
                if comparison["distance"] is not None:
                    calibration_samples.append(
                        CalibrationSample(
                            distance=float(comparison["distance"]),
                            positive=(
                                truth_identity is not None
                                and comparison["identity_candidate"].casefold() == truth_identity
                            ),
                        )
                    )

        image_result["matched_faces"] = len(pairs)
        matched_truth_indices = {pair.ground_truth_index for pair in pairs}
        image_result["unmatched_detection_indices"] = [
            index
            for index in range(len(predicted_boxes))
            if index not in pair_by_prediction
        ]
        image_result["unmatched_ground_truth_indices"] = [
            index
            for index in range(len(annotation.faces))
            if index not in matched_truth_indices
        ]
        recognition["false_rejections"] += sum(
            face.registered
            for index, face in enumerate(annotation.faces)
            if index not in matched_truth_indices
        )
        image_results.append(image_result)

    matched = recognition["matched_ground_truth_faces"]
    recognition["accuracy_on_matched_faces"] = _ratio(recognition["correct"], matched)
    return {
        "configuration": {
            "model_name": model_name,
            "detector_backend": detector_backend,
            "distance_metric": distance_metric,
            "threshold": threshold,
            "min_face_size": min_face_size,
            "iou_threshold": iou_threshold,
        },
        "metrics": {
            "detection": detection_metrics(
                true_positives=total_tp,
                predicted=total_predicted,
                ground_truth=total_ground_truth,
            ),
            "recognition": recognition,
            "threshold_calibration": calibrate_threshold(calibration_samples),
        },
        "images": image_results,
    }


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("La lista de configuración no puede estar vacía")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa detección y reconocimiento facial.")
    parser.add_argument("--annotations", type=Path, required=True, help="archivo JSON con ground truth")
    parser.add_argument("--known-dir", type=Path, default=DEFAULT_KNOWN_DIR)
    parser.add_argument("--detector-backends", default="yunet", help="lista separada por comas")
    parser.add_argument("--models", default="VGG-Face", help="lista separada por comas")
    parser.add_argument("--metrics", default="cosine", help="lista separada por comas")
    parser.add_argument("--split", default="test", help="split a evaluar o 'all'")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-face-size", type=int, default=recogniser.DEFAULT_MIN_FACE_SIZE)
    parser.add_argument("--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--output", type=Path, required=True, help="JSON de resultados")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.threshold is not None and args.threshold < 0:
            raise ValueError("threshold no puede ser negativo")
        if args.min_face_size < 0:
            raise ValueError("min-face-size no puede ser negativo")
        if not 0 <= args.iou_threshold <= 1:
            raise ValueError("iou-threshold debe estar entre 0 y 1")

        all_annotations = load_annotations(args.annotations)
        annotations = (
            all_annotations
            if args.split.casefold() == "all"
            else tuple(item for item in all_annotations if item.split.casefold() == args.split.casefold())
        )
        if not annotations:
            raise ValueError("No hay imágenes anotadas para el split solicitado")

        known_paths = recogniser.iter_image_files(args.known_dir, recogniser.KNOWN_EXTENSIONS)
        results: list[dict[str, Any]] = []
        for detector_backend in _csv_values(args.detector_backends):
            try:
                known_people = recogniser.load_known_people(
                    known_paths,
                    detector_backend=detector_backend,
                    min_face_size=args.min_face_size,
                )
            except Exception as exc:
                for model_name in _csv_values(args.models):
                    for distance_metric in _csv_values(args.metrics):
                        results.append(
                            {
                                "configuration": {
                                    "model_name": model_name,
                                    "detector_backend": detector_backend,
                                    "distance_metric": distance_metric,
                                },
                                "error": str(exc),
                            }
                        )
                continue

            for model_name in _csv_values(args.models):
                for distance_metric in _csv_values(args.metrics):
                    results.append(
                        evaluate_dataset(
                            annotations,
                            known_people,
                            model_name=model_name,
                            detector_backend=detector_backend,
                            distance_metric=distance_metric,
                            threshold=args.threshold,
                            min_face_size=args.min_face_size,
                            iou_threshold=args.iou_threshold,
                        )
                    )

        payload = {
            "schema_version": 1,
            "annotations": str(args.annotations),
            "split": args.split,
            "annotated_images": len(annotations),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resultados escritos en: {args.output}")
        return 0
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
