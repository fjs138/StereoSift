import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from upscaler import collect_videos, upscale_video


class FakeVideoUpscaler:
    def upscale(self, image, control=None):
        if control:
            control()
        return image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST)


def _write_video(path: str, *, width: int = 8, height: int = 6, frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (width, height),
    )
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = index * 50
        frame[:, :, 1] = 120
        writer.write(frame)
    writer.release()


class TestVideoUpscaler(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_collect_videos_accepts_file_and_non_recursive_folder(self):
        mp4 = Path(self.tmpdir) / "clip.mp4"
        mov = Path(self.tmpdir) / "movie.MOV"
        mp4.write_bytes(b"fake")
        mov.write_bytes(b"fake")
        (Path(self.tmpdir) / "photo.jpg").write_text("no")

        self.assertEqual(collect_videos(str(mp4)), [str(mp4)])
        self.assertEqual(collect_videos(str(self.tmpdir)), [str(mp4), str(mov)])

    def test_upscale_video_writes_resized_video_with_fake_upscaler(self):
        source = os.path.join(self.tmpdir, "tiny.mp4")
        output_dir = os.path.join(self.tmpdir, "out")
        _write_video(source)

        def fake_mux(args, **_kwargs):
            Path(args[-1]).write_bytes(Path(args[3]).read_bytes())
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("upscaler.subprocess.run", side_effect=fake_mux):
            destination = upscale_video(
                source,
                output_dir,
                FakeVideoUpscaler(),
                long_edge=16,
                log=lambda _msg: None,
            )

        capture = cv2.VideoCapture(destination)
        self.assertTrue(capture.isOpened())
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 16)
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 12)
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
        capture.release()

    def test_upscale_video_reports_frame_progress(self):
        source = os.path.join(self.tmpdir, "progress.mp4")
        output_dir = os.path.join(self.tmpdir, "out")
        _write_video(source, frames=3)
        updates = []

        def fake_mux(args, **_kwargs):
            Path(args[-1]).write_bytes(Path(args[3]).read_bytes())
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("upscaler.subprocess.run", side_effect=fake_mux):
            upscale_video(
                source,
                output_dir,
                FakeVideoUpscaler(),
                long_edge=16,
                log=lambda _msg: None,
                progress=lambda done, total: updates.append((done, total)),
            )

        self.assertEqual(updates, [(1, 3), (2, 3), (3, 3)])


if __name__ == "__main__":
    unittest.main()
