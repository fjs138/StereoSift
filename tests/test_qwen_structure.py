import math
import unittest
from unittest.mock import MagicMock, patch

from qwen_structure import QwenStructureJudge, _extract_json


class TestQwenStructureResponse(unittest.TestCase):
    def test_extracts_json_from_markdown_fence(self):
        result = _extract_json(
            '```json\n{"verdict":"PASS","defect":"none","confidence":0.9}\n```'
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_rejects_response_without_json(self):
        with self.assertRaisesRegex(ValueError, "did not return a JSON object"):
            _extract_json("PASS")

    def test_uses_first_complete_object(self):
        result = _extract_json(
            '{"verdict":"PASS"}\n{"verdict":"FAIL"}'
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_non_finite_confidence_is_sanitized(self):
        judge = object.__new__(QwenStructureJudge)
        judge.device = "cpu"
        judge.processor = MagicMock()
        judge.processor.apply_chat_template.return_value = "prompt"
        inputs = MagicMock()
        inputs.to.return_value = inputs
        inputs.input_ids.shape = (1, 1)
        judge.processor.return_value = inputs
        judge.processor.batch_decode.return_value = [
            '{"verdict":"PASS","defect":"none","evidence":"ok","confidence":NaN}'
        ]
        judge.model = MagicMock()
        judge.model.generate.return_value = MagicMock()
        judge.model.generate.return_value.__getitem__.return_value = MagicMock()
        with patch("qwen_structure.Image.open") as opened:
            opened.return_value.convert.return_value = MagicMock()
            decision = judge.judge("image.png")
        self.assertFalse(math.isnan(decision.confidence))
        self.assertEqual(decision.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
