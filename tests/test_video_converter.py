import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from video_converter import (
    _normalise_depth,
    _reset_streaming_state,
    _target_resolution,
    convert_video_to_sbs,
)


class FakeStreamingModel:
    def __init__(self):
        self.transform = "stale"
        self.frame_height = 999
        self.frame_width = 999
        self.frame_id_list = [1, 2, 3]
        self.frame_cache_list = ["cached"]
        self.id = 42
        self.calls = 0
        self.first_call_was_reset = False

    def infer_video_depth_one(self, frame, **_kwargs):
        if self.calls == 0:
            self.first_call_was_reset = (
                self.transform is None
                and self.frame_id_list == []
                and self.frame_cache_list == []
                and self.id == -1
            )
        self.calls += 1
        self.id += 1
        self.transform = "active"
        self.frame_height, self.frame_width = frame.shape[:2]
        h, w = frame.shape[:2]
        return np.linspace(0.0, 1.0, max((h // 2) * (w // 2), 1), dtype=np.float32).reshape(
            max(h // 2, 1),
            max(w // 2, 1),
        )


class RampDepthModel:
    def infer_video_depth_one(self, frame, **_kwargs):
        h, w = frame.shape[:2]
        return np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))


def _write_test_video(path: str, *, width: int = 8, height: int = 6, frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (width, height),
    )
    for idx in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (idx * 40) % 256
        frame[:, :, 1] = 80
        writer.write(frame)
    writer.release()


class TestVideoConverter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_reset_streaming_state_clears_video_cache(self):
        model = FakeStreamingModel()

        _reset_streaming_state(model)

        self.assertIsNone(model.transform)
        self.assertEqual(model.frame_id_list, [])
        self.assertEqual(model.frame_cache_list, [])
        self.assertEqual(model.id, -1)

    def test_target_resolution_downscales_and_keeps_even_dimensions(self):
        self.assertEqual(_target_resolution(101, 51, 50), (50, 24))

    def test_normalise_depth_handles_nan_and_metric_inversion(self):
        depth = _normalise_depth(np.array([[0.0, np.nan], [2.0, 4.0]]), is_metric=True)

        self.assertTrue(np.isfinite(depth).all())
        self.assertAlmostEqual(float(depth[0, 0]), 1.0)
        self.assertAlmostEqual(float(depth[1, 1]), 0.0)

    def test_convert_video_to_sbs_writes_side_by_side_video_with_fake_model(self):
        input_path = os.path.join(self.tmpdir, "clip.mp4")
        output_dir = os.path.join(self.tmpdir, "out")
        _write_test_video(input_path)
        model = FakeStreamingModel()

        def fake_sbs(base_image, *_args, **_kwargs):
            return torch.cat([base_image, base_image], dim=2)

        def fake_mux(args, **_kwargs):
            shutil.copyfile(args[3], args[-1])
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("video_converter.process_image_sbs", side_effect=fake_sbs), patch(
            "video_converter.subprocess.run",
            side_effect=fake_mux,
        ):
            ok = convert_video_to_sbs(
                input_path,
                output_dir,
                model,
                torch.device("cpu"),
                torch.float32,
                False,
                max_len=2,
                max_res=-1,
                temporal_smoothing=0.0,
                log=lambda _msg: None,
            )

        out_path = os.path.join(output_dir, "clip_SBS_LR.mp4")
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(out_path))
        self.assertEqual(model.calls, 2)
        self.assertTrue(model.first_call_was_reset)

        cap = cv2.VideoCapture(out_path)
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 16)
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 6)
        cap.release()

    def test_convert_video_to_sbs_left_and_right_eyes_are_not_identical(self):
        input_path = os.path.join(self.tmpdir, "bars.mp4")
        output_dir = os.path.join(self.tmpdir, "out")
        writer = cv2.VideoWriter(
            input_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (128, 72),
        )
        frame = np.zeros((72, 128, 3), dtype=np.uint8)
        for x in range(128):
            frame[:, x] = [(x * 3) % 256, (x * 7) % 256, (x * 13) % 256]
        writer.write(frame)
        writer.release()

        def fake_mux(args, **_kwargs):
            shutil.copyfile(args[3], args[-1])
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("video_converter.subprocess.run", side_effect=fake_mux):
            ok = convert_video_to_sbs(
                input_path,
                output_dir,
                RampDepthModel(),
                torch.device("cpu"),
                torch.float32,
                False,
                depth_scale=70,
                max_len=1,
                max_res=-1,
                temporal_smoothing=0.0,
                log=lambda _msg: None,
            )

        self.assertTrue(ok)
        cap = cv2.VideoCapture(os.path.join(output_dir, "bars_SBS_LR.mp4"))
        self.assertTrue(cap.isOpened())
        ok, sbs_frame = cap.read()
        cap.release()
        self.assertTrue(ok)
        height, width = sbs_frame.shape[:2]
        self.assertEqual((height, width), (72, 256))
        left = sbs_frame[:, : width // 2].astype(np.float32)
        right = sbs_frame[:, width // 2 :].astype(np.float32)
        self.assertGreater(float(np.mean(np.abs(left - right))), 5.0)

    def test_convert_video_to_sbs_can_limit_by_seconds(self):
        input_path = os.path.join(self.tmpdir, "longer.mp4")
        output_dir = os.path.join(self.tmpdir, "out")
        _write_test_video(input_path, width=8, height=6, frames=10)
        model = FakeStreamingModel()

        def fake_sbs(base_image, *_args, **_kwargs):
            return torch.cat([base_image, base_image], dim=2)

        def fake_mux(args, **_kwargs):
            shutil.copyfile(args[3], args[-1])
            return type("Result", (), {"returncode": 0, "stderr": ""})()

        with patch("video_converter.process_image_sbs", side_effect=fake_sbs), patch(
            "video_converter.subprocess.run",
            side_effect=fake_mux,
        ):
            ok = convert_video_to_sbs(
                input_path,
                output_dir,
                model,
                torch.device("cpu"),
                torch.float32,
                False,
                max_seconds=0.4,
                max_res=-1,
                temporal_smoothing=0.0,
                log=lambda _msg: None,
            )

        self.assertTrue(ok)
        self.assertEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
