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
    def test_uncertain_structure_routes_to_warning(self, mock_yolo, mock_moondream):
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

        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["score"], 50.0)

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo_pose")
    @patch("qc_pipeline._run_yolo")
    def test_pose_duplicate_torso_fails_and_skips_moondream(self, mock_yolo, mock_pose, mock_moondream):
        import numpy as np
        mock_yolo.return_value = {"person_count": 2, "detections": ["person", "person"]}
        
        # Scale = (200 + 200) / 2 = 200.
        # Hips at same location (100, 200). Shoulders separated: Person 0 at (60, 100), Person 1 at (140, 100).
        # Noses separated: Person 0 at (60, 50), Person 1 at (140, 50).
        xy0 = np.zeros((17, 2))
        conf0 = np.zeros(17)
        xy0[0] = [60, 50]    # nose
        conf0[0] = 1.0
        xy0[5] = [60, 100]   # left shoulder
        conf0[5] = 1.0
        xy0[6] = [60, 100]   # right shoulder
        conf0[6] = 1.0
        xy0[11] = [100, 200] # left hip
        conf0[11] = 1.0
        xy0[12] = [100, 200] # right hip
        conf0[12] = 1.0

        xy1 = np.zeros((17, 2))
        conf1 = np.zeros(17)
        xy1[0] = [140, 50]
        conf1[0] = 1.0
        xy1[5] = [140, 100]
        conf1[5] = 1.0
        xy1[6] = [140, 100]
        conf1[6] = 1.0
        xy1[11] = [100, 200]
        conf1[11] = 1.0
        xy1[12] = [100, 200]
        conf1[12] = 1.0

        mock_pose.return_value = {
            "keypoints": [
                {"xy": xy0, "conf": conf0},
                {"xy": xy1, "conf": conf1}
            ],
            "boxes": [
                [50, 20, 150, 350],
                [50, 20, 150, 350]
            ]
        }

        result = classify_image(
            self.image_path,
            os.path.join(self.tmpdir, "out"),
            settings=QCSettings(use_yolo=True, use_deep_scan=True),
        )

        # Should fail due to duplicate torso
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["score"], 0.0)
        self.assertTrue(any("duplicate torso" in issue for issue in result["issues"]))
        
        # Moondream should be skipped because YOLO pose failed it
        mock_moondream.assert_not_called()

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo_pose")
    @patch("qc_pipeline._run_yolo")
    def test_pose_duplicate_head_fails(self, mock_yolo, mock_pose, mock_moondream):
        import numpy as np
        mock_yolo.return_value = {"person_count": 2, "detections": ["person", "person"]}
        
        # Hips at same location (100, 200). Shoulders close (100, 100).
        # Noses separated: Person 0 at (60, 50), Person 1 at (140, 50).
        xy0 = np.zeros((17, 2))
        conf0 = np.zeros(17)
        xy0[0] = [60, 50]    # nose
        conf0[0] = 1.0
        xy0[5] = [100, 100]  # left shoulder
        conf0[5] = 1.0
        xy0[6] = [100, 100]  # right shoulder
        conf0[6] = 1.0
        xy0[11] = [100, 200] # left hip
        conf0[11] = 1.0
        xy0[12] = [100, 200] # right hip
        conf0[12] = 1.0

        xy1 = np.zeros((17, 2))
        conf1 = np.zeros(17)
        xy1[0] = [140, 50]
        conf1[0] = 1.0
        xy1[5] = [100, 100]
        conf1[5] = 1.0
        xy1[6] = [100, 100]
        conf1[6] = 1.0
        xy1[11] = [100, 200]
        conf1[11] = 1.0
        xy1[12] = [100, 200]
        conf1[12] = 1.0

        mock_pose.return_value = {
            "keypoints": [
                {"xy": xy0, "conf": conf0},
                {"xy": xy1, "conf": conf1}
            ],
            "boxes": [
                [50, 20, 150, 350],
                [50, 20, 150, 350]
            ]
        }

        result = classify_image(
            self.image_path,
            os.path.join(self.tmpdir, "out"),
            settings=QCSettings(use_yolo=True, use_deep_scan=True),
        )

        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("duplicate head" in issue for issue in result["issues"]))
        mock_moondream.assert_not_called()

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo_pose")
    @patch("qc_pipeline._run_yolo")
    def test_pose_normal_two_people_passes(self, mock_yolo, mock_pose, mock_moondream):
        import numpy as np
        mock_yolo.return_value = {"person_count": 2, "detections": ["person", "person"]}
        mock_moondream.return_value = {
            "verdict": "pass", "structure_ok": True,
            "structure_note": "PASS", "raw": "PASS",
        }
        
        # Two people side by side, completely separated.
        xy0 = np.zeros((17, 2))
        conf0 = np.zeros(17)
        xy0[0] = [100, 50]
        conf0[0] = 1.0
        xy0[5] = [100, 100]
        conf0[5] = 1.0
        xy0[11] = [100, 200]
        conf0[11] = 1.0

        xy1 = np.zeros((17, 2))
        conf1 = np.zeros(17)
        xy1[0] = [300, 50]
        conf1[0] = 1.0
        xy1[5] = [300, 100]
        conf1[5] = 1.0
        xy1[11] = [300, 200]
        conf1[11] = 1.0

        mock_pose.return_value = {
            "keypoints": [
                {"xy": xy0, "conf": conf0},
                {"xy": xy1, "conf": conf1}
            ],
            "boxes": [
                [50, 20, 150, 350],
                [250, 20, 350, 350]
            ]
        }

        result = classify_image(
            self.image_path,
            os.path.join(self.tmpdir, "out"),
            settings=QCSettings(use_yolo=True, use_deep_scan=True),
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["score"], 100.0)
        self.assertEqual(len(result["issues"]), 0)

    @patch("qc_pipeline._get_moondream")
    def test_two_step_moondream_pass(self, mock_get_moondream):
        model = Mock()
        # Step 1 returns "NO"
        model.query.side_effect = [{"answer": "NO"}]
        mock_get_moondream.return_value = model

        result = _run_moondream(self.image_path)
        self.assertTrue(result["structure_ok"])
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["structure_note"], "PASS")

    @patch("qc_pipeline._get_moondream")
    def test_two_step_moondream_fail(self, mock_get_moondream):
        model = Mock()
        # Step 1 returns "YES", Step 2 returns "two heads detected"
        model.query.side_effect = [
            {"answer": "YES"},
            {"answer": "two heads detected"}
        ]
        mock_get_moondream.return_value = model

        result = _run_moondream(self.image_path)
        self.assertFalse(result["structure_ok"])
        self.assertEqual(result["verdict"], "fail")
        self.assertEqual(result["structure_note"], "FAIL: two heads detected")


if __name__ == "__main__":
    unittest.main()
