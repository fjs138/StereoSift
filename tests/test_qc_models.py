import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from qc_pipeline import QCSettings, _run_moondream, classify_image


class TestQCModelDecisions(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.image_path = os.path.join(self.tmpdir, "person.png")
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(self.image_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo")
    def test_severe_moondream_structure_result_fails(self, mock_yolo, mock_moondream):
        mock_yolo.return_value = {"person_count": 1, "detections": ["person"]}
        mock_moondream.return_value = {
            "verdict": "fail",
            "structure_ok": False,
            "structure_note": "FAIL: The subject has a duplicate head and fused arm.",
            "raw": "FAIL: The subject has a duplicate head and fused arm.",
        }

        result = classify_image(
            self.image_path,
            os.path.join(self.tmpdir, "out"),
            settings=QCSettings(use_yolo=True, use_deep_scan=True),
        )

        self.assertEqual(result["status"], "fail")
        self.assertIn("duplicate head", result["structure_note"])
        self.assertTrue(any("major structure defect" in issue for issue in result["issues"]))

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo")
    def test_person_only_mode_skips_non_person_images(self, mock_yolo, mock_moondream):
        mock_yolo.return_value = {"person_count": 0, "detections": ["bottle"]}

        result = classify_image(
            self.image_path,
            os.path.join(self.tmpdir, "out"),
            settings=QCSettings(
                use_yolo=True,
                use_deep_scan=True,
                deep_scan_persons_only=True,
            ),
        )

        mock_moondream.assert_not_called()
        self.assertEqual(result["structure_note"], "")

    @patch("qc_pipeline._get_moondream")
    def test_moondream_no_prefix_means_structure_is_ok(self, mock_get_moondream):
        model = Mock()
        # query() now returns a dict with an "answer" key
        model.query.return_value = {"answer": "PASS"}
        mock_get_moondream.return_value = model

        result = _run_moondream(self.image_path)

        self.assertTrue(result["structure_ok"])
        self.assertEqual(result["verdict"], "pass")

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo")
    def test_exposure_does_not_fail_a_structurally_valid_image(self, mock_yolo, mock_moondream):
        mock_yolo.return_value = {"person_count": 1, "detections": ["person"]}
        mock_moondream.return_value = {
            "verdict": "pass", "structure_ok": True,
            "structure_note": "PASS", "raw": "PASS",
        }
        Image.new("RGB", (64, 64), color=(0, 0, 0)).save(self.image_path)

        result = classify_image(
            self.image_path,
            os.path.join(self.tmpdir, "out"),
            settings=QCSettings(use_yolo=True, use_deep_scan=True),
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["score"], 100.0)

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo")
    def test_uncertain_structure_enters_failure_review_queue(self, mock_yolo, mock_moondream):
        mock_yolo.return_value = {"person_count": 1, "detections": ["person"]}
        mock_moondream.return_value = {
            "verdict": "uncertain", "structure_ok": False,
            "structure_note": "UNCERTAIN: the torso is partly obscured.",
            "raw": "UNCERTAIN: the torso is partly obscured.",
        }

        result = classify_image(
            self.image_path,
            os.path.join(self.tmpdir, "out"),
            settings=QCSettings(use_yolo=True, use_deep_scan=True),
        )

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["score"], 25.0)


if __name__ == "__main__":
    unittest.main()
