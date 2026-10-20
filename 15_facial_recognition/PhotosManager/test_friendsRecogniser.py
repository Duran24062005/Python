import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PhotosManager import friendsRecogniser as recogniser


class FakeDeepFace:
    @staticmethod
    def verify(*, img1_path, **_kwargs):
        distances = {
            "joey-face": (True, 0.35),
            "ross-face": (True, 0.18),
            "unknown-face": (False, 0.99),
        }
        verified, distance = distances[img1_path]
        return {"verified": verified, "distance": distance}

    @staticmethod
    def extract_faces(**_kwargs):
        return []


class FakeDeepFaceWithInvalidDetections:
    @staticmethod
    def extract_faces(**_kwargs):
        return [
            {
                "face": "whole-image-placeholder",
                "facial_area": {"x": 0, "y": 0, "w": 640, "h": 480},
                "confidence": 0,
            },
            {
                "face": "tiny-false-positive",
                "facial_area": {"x": 100, "y": 100, "w": 24, "h": 24},
                "confidence": 0.99,
            },
            {
                "face": "valid-face",
                "facial_area": {"x": 120, "y": 80, "w": 120, "h": 120},
                "confidence": 0.92,
            },
        ]


class FriendsRecogniserTests(unittest.TestCase):
    def test_iter_image_files_ignores_metadata_and_non_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "ross.png").touch()
            (directory / "notes.txt").touch()
            (directory / ".DS_Store").touch()
            (directory / "._ross.png").touch()

            macosx_directory = directory / "__MACOSX"
            macosx_directory.mkdir()
            (macosx_directory / "hidden.png").touch()

            files = recogniser.iter_image_files(directory, {".png"})

        self.assertEqual(files, (directory / "ross.png",))

    def test_find_best_match_uses_lowest_verified_distance(self):
        known_people = (
            recogniser.KnownPerson(name="joey", face="joey-face"),
            recogniser.KnownPerson(name="ross", face="ross-face"),
            recogniser.KnownPerson(name="unknown", face="unknown-face"),
        )

        with patch.object(recogniser, "DeepFace", FakeDeepFace):
            match = recogniser.find_best_match(
                "test-face",
                known_people,
                model_name="VGG-Face",
                distance_metric="cosine",
            )

        self.assertEqual(match.name, "ross")
        self.assertEqual(match.distance, 0.18)

    def test_find_best_match_returns_unknown_without_verified_result(self):
        known_people = (recogniser.KnownPerson(name="unknown", face="unknown-face"),)

        with patch.object(recogniser, "DeepFace", FakeDeepFace):
            match = recogniser.find_best_match(
                "test-face",
                known_people,
                model_name="VGG-Face",
                distance_metric="cosine",
            )

        self.assertEqual(match.name, "unknown")
        self.assertIsNone(match.distance)

    def test_extract_faces_discards_placeholder_and_tiny_detections(self):
        with patch.object(recogniser, "DeepFace", FakeDeepFaceWithInvalidDetections):
            faces = recogniser.extract_faces(
                "test-image",
                detector_backend="opencv",
                enforce_detection=False,
            )

        self.assertEqual([face["face"] for face in faces], ["valid-face"])

    def test_unknown_without_eye_landmarks_is_not_annotated(self):
        unknown = recogniser.FaceMatch(name="unknown")
        known = recogniser.FaceMatch(name="ross", matched=True, distance=0.4)
        without_landmarks = {"left_eye": None, "right_eye": None}
        with_landmarks = {"left_eye": (10, 10), "right_eye": (20, 10)}

        self.assertFalse(
            recogniser._should_annotate_face(without_landmarks, unknown)
        )
        self.assertTrue(recogniser._should_annotate_face(with_landmarks, unknown))
        self.assertTrue(recogniser._should_annotate_face(without_landmarks, known))

    def test_parser_uses_yunet_by_default_and_exposes_face_size_filter(self):
        args = recogniser.build_parser().parse_args([])

        self.assertEqual(args.detector_backend, "yunet")
        self.assertEqual(args.min_face_size, 40)


if __name__ == "__main__":
    unittest.main()
