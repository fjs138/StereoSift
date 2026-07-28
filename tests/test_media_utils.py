import tempfile
import unittest
from pathlib import Path

from media_utils import collect_images, collect_videos, detect_input_kind


class TestMediaUtils(unittest.TestCase):
    def test_detect_input_kind_for_files_and_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "photo.JPG"
            video = root / "clip.mp4"
            note = root / "note.txt"
            image.write_bytes(b"fake")
            video.write_bytes(b"fake")
            note.write_text("no")

            self.assertEqual(detect_input_kind(str(image)), "image")
            self.assertEqual(detect_input_kind(str(video)), "video")
            self.assertEqual(detect_input_kind(str(note)), "unknown")
            self.assertEqual(detect_input_kind(str(root)), "mixed")

    def test_collectors_return_non_recursive_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "photo.jpg"
            video = root / "clip.MOV"
            nested = root / "nested"
            nested.mkdir()
            image.write_bytes(b"fake")
            video.write_bytes(b"fake")
            (nested / "ignored.jpg").write_bytes(b"fake")

            self.assertEqual(collect_images(str(root)), [str(image)])
            self.assertEqual(collect_videos(str(root)), [str(video)])


if __name__ == "__main__":
    unittest.main()
