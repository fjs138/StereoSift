import tempfile
import unittest
from pathlib import Path

from media_utils import (
    collect_images,
    collect_videos,
    detect_input_kind,
    relative_output_subdir,
)


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

    def test_recursive_collectors_include_nested_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            image = nested / "photo.jpg"
            video = nested / "clip.mp4"
            image.write_bytes(b"fake")
            video.write_bytes(b"fake")

            self.assertEqual(collect_images(str(root), recursive=True), [str(image)])
            self.assertEqual(collect_videos(str(root), recursive=True), [str(video)])
            self.assertEqual(detect_input_kind(str(root), recursive=True), "mixed")
            self.assertEqual(relative_output_subdir(str(root), str(image)), "nested")


if __name__ == "__main__":
    unittest.main()
