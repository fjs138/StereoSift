import math
import unittest
from unittest.mock import MagicMock, patch

from qwen_structure import QwenStructureJudge, _extract_json


class TestQwenStructureResponse(unittest.TestCase):
    def test_extracts_json_from_markdown_fence(self):
        result = _extract_json(
            '```json\n{"status":"pass","defect_type":"none","confidence":0.9,"evidence":"ok","review":false}\n```'
        )
        self.assertEqual(result["status"], "pass")

    def test_rejects_response_without_json(self):
        with self.assertRaisesRegex(ValueError, "did not return a JSON object"):
            _extract_json("PASS")

    def test_uses_first_complete_object(self):
        result = _extract_json(
            '{"status":"pass"}\n{"status":"fail"}'
        )
        self.assertEqual(result["status"], "pass")

    def test_non_finite_confidence_is_sanitized(self):
        judge = object.__new__(QwenStructureJudge)
        judge.device = "cpu"
        judge.backend_url = None
        judge.processor = MagicMock()
        judge.processor.apply_chat_template.return_value = "prompt"
        inputs = MagicMock()
        inputs.to.return_value = inputs
        inputs.input_ids.shape = (1, 1)
        judge.processor.return_value = inputs
        judge.processor.batch_decode.return_value = [
            '{"status":"pass","defect_type":"none","evidence":"ok","confidence":NaN,"review":false}'
        ]
        judge.model = MagicMock()
        judge.model.generate.return_value = MagicMock()
        judge.model.generate.return_value.__getitem__.return_value = MagicMock()
        with patch("qwen_structure.Image.open") as opened:
            opened.return_value.convert.return_value = MagicMock()
            decision = judge.judge("image.png")
        self.assertFalse(math.isnan(decision.confidence))
        self.assertEqual(decision.confidence, 0.0)
        self.assertFalse(decision.review)

    def test_backend_mode_uses_openai_compatible_api(self):
        judge = object.__new__(QwenStructureJudge)
        judge.backend_url = "http://127.0.0.1:8000/v1"
        judge.model_name = "vision-model"
        judge.api_key = "secret"
        judge.processor = None
        judge.model = None
        judge.device = None

        payload = {
            "choices": [{
                "message": {
                    "content": '{"status":"warning","defect_type":"suspect","confidence":0.6,"evidence":"unclear overlap","review":true}'
                }
            }]
        }

        with patch("qwen_structure._http_post", return_value=payload) as mock_post, \
             patch("qwen_structure.Image.open") as opened:
            opened.return_value.__enter__.return_value.convert.return_value = MagicMock()
            decision = judge.judge("image.png")

        self.assertEqual(decision.verdict, "warning")
        self.assertEqual(decision.defect, "suspect")
        self.assertEqual(decision.review, True)
        mock_post.assert_called_once()
        self.assertIn("/chat/completions", mock_post.call_args.args[0])
        self.assertEqual(mock_post.call_args.kwargs["api_key"], "secret")


if __name__ == "__main__":
    unittest.main()
