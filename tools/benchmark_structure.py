#!/usr/bin/env python3
"""Benchmark the local structure judge against an ignored label manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make direct execution (``python tools/benchmark_structure.py``) resolve the
# project module just like ``python -m tools.benchmark_structure`` does.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qwen_structure import DEFAULT_BACKEND_MODEL, DEFAULT_BACKEND_URL, MODEL_ID, QwenStructureJudge


def _load_labels(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("labels must be a non-empty JSON object")
    labels = {str(name): str(label).strip().lower() for name, label in value.items()}
    invalid = {label for label in labels.values() if label not in {"pass", "fail"}}
    if invalid:
        raise ValueError(f"labels must be pass or fail, got: {sorted(invalid)}")
    return labels


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    completed = [row for row in rows if "actual" in row]
    tp = sum(row["expected"] == row["actual"] == "fail" for row in completed)
    tn = sum(row["expected"] == row["actual"] == "pass" for row in completed)
    fp = sum(row["expected"] == "pass" and row["actual"] == "fail" for row in completed)
    fn = sum(row["expected"] == "fail" and row["actual"] == "pass" for row in completed)
    warnings = sum(row["actual"] == "warning" for row in completed)
    total = len(completed)
    return {
        "completed": total,
        "errors": len(rows) - total,
        "correct": tp + tn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "fail_precision": tp / (tp + fp) if tp + fp else 0.0,
        "fail_recall": tp / (tp + fn) if tp + fn else 0.0,
        "true_fail": tp,
        "true_pass": tn,
        "false_fail": fp,
        "missed_fail": fn,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--model-name", default=DEFAULT_BACKEND_MODEL)
    parser.add_argument("--api-key")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    labels = _load_labels(args.labels)
    missing = [name for name in labels if not (args.input / name).is_file()]
    if missing:
        parser.error(f"labeled images not found under {args.input}: {', '.join(missing)}")

    judge = QwenStructureJudge(
        args.model,
        backend_url=args.backend_url,
        model_name=args.model_name,
        api_key=args.api_key,
    )
    rows: list[dict[str, object]] = []
    for filename, expected in labels.items():
        started = time.perf_counter()
        row: dict[str, object] = {"filename": filename, "expected": expected}
        try:
            decision = judge.judge(str(args.input / filename))
            row.update(
                actual=decision.verdict,
                defect=decision.defect,
                confidence=decision.confidence,
                evidence=decision.evidence,
                review=decision.review,
            )
            print(
                f"{filename}\n"
                f"  expected={expected} actual={decision.verdict} review={decision.review} "
                f"defect={decision.defect} confidence={decision.confidence:.2f}\n"
                f"  evidence={decision.evidence}\n"
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{filename}\n  ERROR {row['error']}\n", file=sys.stderr)
        row["seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)

    metrics = _metrics(rows)
    print(
        f"accuracy={metrics['correct']}/{metrics['completed']} "
        f"({metrics['accuracy']:.1%}) | fail precision={metrics['fail_precision']:.1%} "
        f"recall={metrics['fail_recall']:.1%} | warnings={metrics['warnings']} | errors={metrics['errors']}"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"model": args.model, "metrics": metrics, "results": rows}, indent=2)
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
