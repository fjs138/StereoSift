import json
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_structure import _load_labels, _metrics


class TestStructureBenchmark(unittest.TestCase):
    def test_load_labels_normalizes_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({"one.png": "PASS", "two.png": "fail"}))
            self.assertEqual(_load_labels(path), {"one.png": "pass", "two.png": "fail"})

    def test_metrics_report_missed_failures(self):
        metrics = _metrics([
            {"expected": "pass", "actual": "pass"},
            {"expected": "fail", "actual": "pass"},
            {"expected": "fail", "actual": "fail"},
            {"expected": "pass", "error": "broken"},
        ])
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["missed_fail"], 1)
        self.assertEqual(metrics["errors"], 1)
        self.assertAlmostEqual(metrics["fail_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
