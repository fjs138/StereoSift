import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from qc_pipeline import (
    QCSettings, _chat_completions_url, _is_strong_structure_defect,
    _parse_structure_verdict, _run_moondream,
    classify_image, classify_image_with_backend, classify_image_with_labels,
)


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
        # Multiple people is OK if they're not twins/duplicates
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

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo_pose")
    @patch("qc_pipeline._run_yolo")
    def test_twins_detected_as_fail(self, mock_yolo, mock_pose, mock_moondream):
        import numpy as np
        # Twins/similar people should be flagged as fail
        mock_yolo.return_value = {"person_count": 2, "detections": ["person", "person"]}
        mock_moondream.return_value = {
            "verdict": "uncertain",
            "structure_ok": False,
            "structure_note": "UNCERTAIN: The image shows two girls with similar body proportions, suggesting they may be twins or very similar.",
            "raw": "UNCERTAIN: The image shows two girls with similar body proportions, suggesting they may be twins or very similar.",
        }

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

        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["score"], 0.0)
        self.assertTrue(any("major structure defect" in issue for issue in result["issues"]))

    @patch("qc_pipeline._get_moondream")
    def test_moondream_pass(self, mock_get_moondream):
        model = Mock()
        model.query.return_value = {"answer": "PASS"}
        mock_get_moondream.return_value = model

        result = _run_moondream(self.image_path)
        self.assertTrue(result["structure_ok"])
        self.assertEqual(result["verdict"], "pass")

    @patch("qc_pipeline._get_moondream")
    def test_moondream_fail(self, mock_get_moondream):
        model = Mock()
        model.query.return_value = {"answer": "FAIL: two heads detected"}
        mock_get_moondream.return_value = model

        result = _run_moondream(self.image_path)
        self.assertFalse(result["structure_ok"])
        self.assertEqual(result["verdict"], "fail")

    @patch("qc_pipeline._get_moondream")
    def test_moondream_uncertain(self, mock_get_moondream):
        model = Mock()
        model.query.return_value = {"answer": "UNCERTAIN: torso partially obscured"}
        mock_get_moondream.return_value = model

        result = _run_moondream(self.image_path)
        self.assertFalse(result["structure_ok"])
        self.assertEqual(result["verdict"], "uncertain")

    def test_verdict_parser_uses_leading_token_not_words_in_explanation(self):
        self.assertEqual(
            _parse_structure_verdict("FAIL: this image does not pass QC"),
            "fail",
        )
        self.assertEqual(
            _parse_structure_verdict("PASS: no fail condition is visible"),
            "pass",
        )
        self.assertEqual(
            _parse_structure_verdict("The image probably passes"),
            "uncertain",
        )

    def test_strong_defect_detector_only_matches_clear_duplicate_structure(self):
        self.assertTrue(_is_strong_structure_defect("duplicate torso: hips vertically aligned"))
        self.assertTrue(_is_strong_structure_defect("two heads visible on one body"))
        self.assertFalse(_is_strong_structure_defect("uncertain structure: torso partly obscured"))

    @patch("qc_pipeline._run_moondream")
    @patch("qc_pipeline._run_yolo_pose")
    @patch("qc_pipeline._run_yolo")
    def test_strict_offline_uses_pixel_only_checks(self, mock_yolo, mock_pose, mock_moondream):
        result = classify_image(
            self.image_path,
            os.path.join(self.tmpdir, "out"),
            settings=QCSettings(strict_offline=True),
        )

        mock_yolo.assert_not_called()
        mock_pose.assert_not_called()
        mock_moondream.assert_not_called()
        self.assertEqual(result["status"], "pass")
        self.assertIn("Strict offline mode", " ".join(result["detector_notes"]))

    def test_backend_url_accepts_base_or_full_endpoint(self):
        endpoint = "http://127.0.0.1:8000/v1/chat/completions"
        self.assertEqual(_chat_completions_url("http://127.0.0.1:8000"), endpoint)
        self.assertEqual(_chat_completions_url("http://127.0.0.1:8000/v1"), endpoint)
        self.assertEqual(_chat_completions_url(endpoint), endpoint)

    @patch("qc_pipeline._http_post")
    def test_backend_uses_openai_multimodal_message_format(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": (
            '{"status":"fail","score":0,"issues":["duplicate torso"]}'
        )}}]}

        result = classify_image_with_backend(
            self.image_path, "http://127.0.0.1:8000/v1",
            os.path.join(self.tmpdir, "out"), model_name="vision-model",
            api_key="secret-token",
        )

        url, payload = mock_post.call_args.args[:2]
        self.assertEqual(url, "http://127.0.0.1:8000/v1/chat/completions")
        content = payload["messages"][1]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"))
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(mock_post.call_args.kwargs["api_key"], "secret-token")
        self.assertEqual(result["status"], "fail")

        user_prompt = payload["messages"][1]["content"][0]["text"]
        self.assertIn("Use FAIL for severe problems", user_prompt)
        self.assertIn("Use WARNING for minor problems", user_prompt)
        self.assertIn("Use PASS when the body structure looks normal", user_prompt)
        self.assertNotIn("counting each visible head", user_prompt)

    @patch("qc_pipeline._http_post")
    def test_backend_does_not_write_human_readable_log(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": (
            '{"status":"warning","score":50,"issues":["cropped body"]}'
        )}}]}

        output_dir = os.path.join(self.tmpdir, "out")
        classify_image_with_backend(
            self.image_path, "http://127.0.0.1:8000/v1",
            output_dir, model_name="vision-model",
        )

        self.assertFalse(os.path.exists(os.path.join(output_dir, "model_responses.log")))

    @patch("qc_pipeline._http_post")
    def test_backend_uses_parsed_status_as_single_source_of_truth(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": (
            '{"status":"pass","score":95,"issues":["No fused bodies are visible."]}'
        )}}]}

        result = classify_image_with_backend(
            self.image_path, "http://127.0.0.1:8000/v1",
            os.path.join(self.tmpdir, "out"), model_name="vision-model",
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["score"], 95.0)

    @patch("qc_pipeline._http_post")
    def test_backend_routes_violation_language_to_violations_folder(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": (
            '{"status":"pass","score":95,"issues":["The image may be a violation of the stated policy."]}'
        )}}]}

        output_dir = os.path.join(self.tmpdir, "out")
        result = classify_image_with_backend(
            self.image_path,
            "http://127.0.0.1:8000/v1",
            output_dir,
            model_name="vision-model",
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["route_folder"], "unscored")
        self.assertTrue(result["destination"].startswith(os.path.join(output_dir, "unscored")))
        self.assertTrue(os.path.exists(result["destination"]))

    @patch("qc_pipeline._http_post")
    def test_organizer_routes_exact_allowed_label(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": (
            '{"label":"indoors","confidence":91,"reason":"Interior room"}'
        )}}]}
        output_dir = os.path.join(self.tmpdir, "organized")

        result = classify_image_with_labels(
            self.image_path,
            "http://127.0.0.1:8000/v1",
            ["outdoors", "indoors"],
            output_dir,
            model_name="vision-model",
        )

        self.assertEqual(result["label"], "indoors")
        self.assertTrue(result["destination"].startswith(os.path.join(output_dir, "indoors")))
        self.assertTrue(os.path.exists(result["destination"]))
        schema = mock_post.call_args.args[1]["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["label"]["enum"], ["outdoors", "indoors"])

    @patch("qc_pipeline._http_post")
    def test_organizer_does_not_route_invented_label(self, mock_post):
        mock_post.return_value = {"choices": [{"message": {"content": (
            '{"label":"somewhere","confidence":50,"reason":"Unclear"}'
        )}}]}
        output_dir = os.path.join(self.tmpdir, "organized")

        with self.assertRaisesRegex(ValueError, "invented a label"):
            classify_image_with_labels(
                self.image_path,
                "http://127.0.0.1:8000/v1",
                ["outdoors", "indoors"],
                output_dir,
                model_name="vision-model",
            )

        self.assertFalse(os.path.exists(os.path.join(output_dir, "outdoors", "person.png")))
        self.assertFalse(os.path.exists(os.path.join(output_dir, "indoors", "person.png")))
        self.assertFalse(os.path.exists(os.path.join(output_dir, "model_responses.log")))


if __name__ == "__main__":
    unittest.main()
