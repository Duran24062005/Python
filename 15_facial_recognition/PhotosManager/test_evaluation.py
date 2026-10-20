import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PhotosManager import evaluation
from PhotosManager import friendsRecogniser as recogniser


class EvaluationTests(unittest.TestCase):
    def test_bbox_iou_and_one_to_one_matching(self):
        self.assertAlmostEqual(
            evaluation.bbox_iou((0, 0, 10, 10), (5, 0, 10, 10)),
            1 / 3,
        )
        pairs = evaluation.match_detections(
            [(0, 0, 10, 10), (20, 20, 10, 10)],
            [(0, 0, 10, 10), (20, 20, 10, 10)],
        )
        self.assertEqual(
            {(pair.prediction_index, pair.ground_truth_index) for pair in pairs},
            {(0, 0), (1, 1)},
        )

    def test_load_annotations_validates_identity_and_bbox(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "annotations.json"
            path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "images": [
                            {
                                "id": "photo-1",
                                "path": "photo.png",
                                "split": "test",
                                "faces": [
                                    {
                                        "identity": "ross",
                                        "registered": True,
                                        "bbox": [1, 2, 30, 40],
                                    },
                                    {
                                        "identity": None,
                                        "registered": False,
                                        "bbox": [40, 2, 20, 20],
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            annotations = evaluation.load_annotations(path, project_dir=Path(temporary_directory))

        self.assertEqual(annotations[0].faces[0].identity, "ross")
        self.assertIsNone(annotations[0].faces[1].identity)
        self.assertEqual(annotations[0].path.name, "photo.png")

    def test_calibrate_threshold_requires_both_classes_and_returns_best(self):
        unavailable = evaluation.calibrate_threshold(
            [evaluation.CalibrationSample(distance=0.2, positive=True)]
        )
        self.assertFalse(unavailable["available"])

        calibrated = evaluation.calibrate_threshold(
            [
                evaluation.CalibrationSample(distance=0.2, positive=True),
                evaluation.CalibrationSample(distance=0.4, positive=True),
                evaluation.CalibrationSample(distance=0.3, positive=False),
                evaluation.CalibrationSample(distance=0.8, positive=False),
            ]
        )
        self.assertTrue(calibrated["available"])
        self.assertEqual(calibrated["recommended_threshold"], 0.4)

    def test_evaluate_dataset_records_detection_and_comparisons(self):
        class FakeImage:
            shape = (80, 80, 3)

        class FakeOpenCV:
            IMREAD_COLOR = 1

            @staticmethod
            def imread(_path, _flag):
                return FakeImage()

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "photo.png"
            image_path.touch()
            annotations = (
                evaluation.AnnotatedImage(
                    image_id="photo-1",
                    path=image_path,
                    split="test",
                    faces=(
                        evaluation.GroundTruthFace(
                            identity="ross",
                            registered=True,
                            bbox=(10, 10, 30, 30),
                        ),
                    ),
                ),
            )
            detected_face = {
                "face": "test-face",
                "facial_area": {
                    "x": 10,
                    "y": 10,
                    "w": 30,
                    "h": 30,
                    "left_eye": (15, 18),
                    "right_eye": (25, 18),
                },
                "confidence": 0.95,
            }
            comparison = recogniser.FaceComparison(
                known_name="ross",
                distance=0.2,
                threshold=0.4,
                verified=True,
            )
            with (
                patch.object(recogniser, "_require_opencv", return_value=FakeOpenCV()),
                patch.object(recogniser, "extract_faces", return_value=[detected_face]),
                patch.object(
                    recogniser,
                    "compare_face_to_known_people",
                    return_value=(comparison,),
                ),
            ):
                result = evaluation.evaluate_dataset(
                    annotations,
                    (recogniser.KnownPerson(name="ross", face="known-face"),),
                    model_name="VGG-Face",
                    detector_backend="opencv",
                    distance_metric="cosine",
                )

        self.assertEqual(result["metrics"]["detection"]["true_positives"], 1)
        self.assertEqual(result["metrics"]["recognition"]["correct"], 1)
        detection = result["images"][0]["detections"][0]
        self.assertEqual(detection["identity_real"], "ross")
        self.assertEqual(len(detection["comparisons"]), 1)
        self.assertTrue(detection["comparisons"][0]["decision_correct"])
        self.assertFalse(detection["comparisons"][0]["false_acceptance"])
        self.assertFalse(detection["comparisons"][0]["false_rejection"])


if __name__ == "__main__":
    unittest.main()
