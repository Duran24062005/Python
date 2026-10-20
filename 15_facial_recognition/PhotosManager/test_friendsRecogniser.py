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


if __name__ == "__main__":
    unittest.main()
