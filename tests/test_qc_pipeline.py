import json
import os
import shutil
import tempfile
import unittest

from qc_pipeline import QCSettings, _route_image, collect_images, run_qc


class TestQCPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.tmpdir, "input")
        os.makedirs(self.input_dir, exist_ok=True)
        from PIL import Image

        Image.new("RGB", (256, 256), color=(255, 255, 255)).save(os.path.join(self.input_dir, "good.png"))
        Image.new("RGB", (256, 256), color=(20, 20, 20)).save(os.path.join(self.input_dir, "dark.png"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_collect_images(self):
        images = collect_images(self.input_dir)
        self.assertEqual(len(images), 2)

    def test_run_qc_creates_report_and_folders(self):
        output_dir = os.path.join(self.tmpdir, "out")
        results = run_qc(
            self.input_dir,
            output_dir,
            settings=QCSettings(use_yolo=False, use_deep_scan=False),
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(os.path.exists(os.path.join(self.input_dir, "good.png")))
        self.assertTrue(os.path.exists(os.path.join(self.input_dir, "dark.png")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "report.json")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "pass")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "fail")))

        with open(os.path.join(output_dir, "report.json"), "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(len(payload), 2)

    def test_run_qc_strict_offline_ignores_backend_url(self):
        output_dir = os.path.join(self.tmpdir, "out")
        results = run_qc(
            self.input_dir,
            output_dir,
            backend_url="http://127.0.0.1:8000/v1",
            settings=QCSettings(strict_offline=True, use_yolo=False, use_deep_scan=False),
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["status"] in {"pass", "warning", "fail"} for r in results))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "report.json")))

    def test_rerouting_removes_stale_status_copy(self):
        output_dir = os.path.join(self.tmpdir, "out")
        image_path = os.path.join(self.input_dir, "good.png")

        old = _route_image(image_path, output_dir, "pass", False)
        self.assertTrue(os.path.exists(old))
        new = _route_image(image_path, output_dir, "fail", False)

        self.assertFalse(os.path.exists(old))
        self.assertTrue(os.path.exists(new))


if __name__ == "__main__":
    unittest.main()
