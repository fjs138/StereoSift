#!/usr/bin/env python3
"""Image quality-control pipeline.

Two complementary local checks, both running through PyTorch with no server:

1. Pixel heuristics (always)
   Exposure, brightness, contrast — fast, zero model dependencies.

2. YOLO object detection (optional, enabled by default)
   ``yolo11n.pt`` (~6 MB, auto-downloaded to ``models/yolo/``)
   Detects persons and objects. Honest scope: reports person count and
   detected object classes. Does NOT claim to detect fused figures —
   YOLO pose produces one skeleton per person instance and cannot reliably
   flag two heads on one body.

3. Moondream2 deep scan (optional, off by default)
   ``vikhyatk/moondream2`` (~2 GB, cached by HuggingFace on first use)
   A small vision-language model that looks at the whole image and reasons
   about it holistically. Catches what YOLO cannot: fused figures, extra
   limbs, doubled heads on one torso, malformed hands, visual glitches.
   Only runs on images that contain people (per YOLO) or when
   ``deep_scan_all=True``.

Optional fourth path: OpenAI-compatible vision backend (LM Studio, Ollama).
Pass ``backend_url`` to replace the local moondream2 scan with an API call.

Output
------
``<output_dir>/pass/``, ``warning/``, ``fail/``, ``report.json``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

_YOLO_DIR          = Path(__file__).parent / "models" / "yolo"
_YOLO_DETECT_FILE  = "yolo11n.pt"
_MOONDREAM_REPO    = "vikhyatk/moondream2"
_MOONDREAM_REVISION = "2025-01-09"   # pinned for reproducibility

# Structure-check prompt sent to moondream2.
_STRUCTURE_PROMPT = (
    "Judge only major duplicated or incorrectly joined human body structure. "
    "FAIL only for a clearly extra head, a second torso sharing one lower body, "
    "an extra arm or leg beyond a plausible human body, or visibly fused people/bodies. "
    "Do not fail for hidden or cropped limbs, hands or fingers, pose, clothing, lighting, "
    "blur, artistic style, or minor visual imperfections. "
    "Reply exactly as one of: PASS; FAIL: <brief reason>; UNCERTAIN: <brief reason>."
)


# ---------------------------------------------------------------------------
# Model caches (process-level singletons)
# ---------------------------------------------------------------------------

_yolo_detect_cache: Any   = None
_moondream_cache: Any = None   # model singleton


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

def _get_yolo_detect():
    global _yolo_detect_cache
    if _yolo_detect_cache is None:
        _yolo_detect_cache = _load_yolo(_YOLO_DETECT_FILE)
    return _yolo_detect_cache


def _load_yolo(filename: str):
    """Load YOLO, downloading to ``models/yolo/`` if not present."""
    from ultralytics import YOLO
    _YOLO_DIR.mkdir(parents=True, exist_ok=True)
    path = _YOLO_DIR / filename
    if not path.exists():
        print(f"Downloading {filename} → {_YOLO_DIR} …")
        tmp = YOLO(filename)
        shutil.copy2(Path(tmp.ckpt_path), path)
        print(f"Saved {path}")
    return YOLO(str(path))


def _run_yolo(image_path: str) -> Dict[str, Any]:
    """Run YOLO detection and return honest findings.

    Returns person_count and a list of all detected class names.
    Does NOT make structure claims beyond object presence/count.
    """
    model       = _get_yolo_detect()
    results     = model(image_path, verbose=False)[0]
    class_names = results.names
    boxes       = results.boxes

    detected: list[str] = []
    person_count = 0

    if boxes is not None and len(boxes):
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            if conf < 0.35:
                continue
            name = class_names.get(int(cls_id), str(int(cls_id)))
            detected.append(name)
            if name == "person":
                person_count += 1

    return {"person_count": person_count, "detections": detected}


# ---------------------------------------------------------------------------
# Moondream2 deep scan
# ---------------------------------------------------------------------------

def _get_moondream():
    """Return the moondream2 model, downloading on first call (~2 GB)."""
    global _moondream_cache
    if _moondream_cache is None:
        import torch
        from transformers import AutoModelForCausalLM
        print(f"Loading moondream2 ({_MOONDREAM_REPO} @ {_MOONDREAM_REVISION}) …")
        model = AutoModelForCausalLM.from_pretrained(
            _MOONDREAM_REPO, revision=_MOONDREAM_REVISION,
            trust_remote_code=True, dtype=torch.float32)
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
        model = model.to(device).eval()
        print(f"moondream2 ready on {device}")
        _moondream_cache = model
    return _moondream_cache


def _run_moondream(image_path: str) -> Dict[str, Any]:
    """Ask moondream2 whether the image has structure defects.

    Returns:
        {
          "structure_ok": bool,   # True = no defects found
          "structure_note": str,  # model's description
          "raw": str,           # full model response
        }
    """
    if Image is None:
        raise RuntimeError("Pillow required for moondream2 inference")

    model = _get_moondream()
    img = Image.open(image_path).convert("RGB")

    # Use the query() API directly — cleaner than encode_image + answer_question.
    result = model.query(img, _STRUCTURE_PROMPT)  # type: ignore
    answer: str = result["answer"].strip() if isinstance(result, dict) else str(result).strip()

    first_word = answer.split()[0].upper().rstrip(".,:") if answer else "UNCERTAIN"
    verdict = first_word if first_word in {"PASS", "FAIL", "UNCERTAIN"} else "UNCERTAIN"

    return {
        "verdict":      verdict.lower(),
        "structure_ok":   verdict == "PASS",
        "structure_note": answer,
        "raw":          answer,
    }


# ---------------------------------------------------------------------------
# Pixel heuristics
# ---------------------------------------------------------------------------

def _pixel_stats(image_path: str) -> Dict[str, float]:
    if Image is None:
        return {"brightness": 0.5, "contrast": 0.2}
    img  = Image.open(image_path).convert("L")
    # getdata() is deprecated in Pillow 14; use tobytes() instead.
    raw  = img.tobytes()
    vals = [v / 255.0 for v in raw]
    mean = sum(vals) / len(vals) if vals else 0.5
    std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 0.2
    return {"brightness": mean, "contrast": std}


# ---------------------------------------------------------------------------
# QC settings dataclass
# ---------------------------------------------------------------------------

class QCSettings:
    """Tunable parameters for the local QC pipeline.

    All thresholds can be adjusted via the GUI settings panel or CLI flags.
    """

    def __init__(
        self,
        # Exposure thresholds
        brightness_dark_fail: float = 0.08,
        brightness_dark_warn: float = 0.20,
        brightness_bright_fail: float = 0.92,
        brightness_bright_warn: float = 0.85,
        contrast_low_fail: float = 0.08,
        contrast_high_warn: float = 0.48,
        # Score thresholds for pass/warning/fail
        score_pass: float = 80.0,
        score_warn: float = 50.0,
        # Feature toggles
        use_yolo: bool = True,
        use_deep_scan: bool = True,   # moondream2
        deep_scan_persons_only: bool = False,
        # Deep-scan strictness
        # "strict"  → any structure note = fail
        # "balanced"→ any structure note = warning, strong language = fail
        # "relaxed" → only fail on very explicit problems
        deep_scan_strictness: str = "relaxed",
    ):
        self.brightness_dark_fail    = brightness_dark_fail
        self.brightness_dark_warn    = brightness_dark_warn
        self.brightness_bright_fail  = brightness_bright_fail
        self.brightness_bright_warn  = brightness_bright_warn
        self.contrast_low_fail       = contrast_low_fail
        self.contrast_high_warn      = contrast_high_warn
        self.score_pass              = score_pass
        self.score_warn              = score_warn
        self.use_yolo                = use_yolo
        self.use_deep_scan           = use_deep_scan
        self.deep_scan_persons_only  = deep_scan_persons_only
        self.deep_scan_strictness    = deep_scan_strictness


_DEFAULT_SETTINGS = QCSettings()


# ---------------------------------------------------------------------------
# Main local classifier
# ---------------------------------------------------------------------------

def classify_image(
    image_path: str,
    output_dir: str,
    move_files: bool = False,
    settings: QCSettings | None = None,
) -> Dict[str, Any]:
    """Classify one image and route it to pass/warning/fail.

    Steps
    -----
    1. Pixel heuristics: exposure and contrast checks (always).
    2. YOLO detection: person count, object classes (if ``settings.use_yolo``).
    3. Moondream2 structure scan: holistic visual reasoning about defects
       (if ``settings.use_deep_scan``).  Only fires on person-containing images
       unless ``deep_scan_persons_only`` is False.

    Returns a result dict with: filename, status, score, issues, brightness,
    contrast, person_count, detections, structure_note, destination.
    """
    if settings is None:
        settings = _DEFAULT_SETTINGS

    issues: list[str] = []
    person_count = 0
    detections: list[str] = []
    structure_note = ""

    # ── 1. Pixel checks ───────────────────────────────────────────────────────
    stats      = _pixel_stats(image_path)
    brightness = stats["brightness"]
    contrast   = stats["contrast"]

    # Exposure and contrast are recorded for diagnostics only. They do not
    # influence structural-structure judgment.

    # ── 2. YOLO detection ─────────────────────────────────────────────────────
    if settings.use_yolo:
        try:
            yolo = _run_yolo(image_path)
            person_count = yolo["person_count"]
            detections   = yolo["detections"]
        except Exception:
            pass

    # ── 3. Moondream2 deep scan ───────────────────────────────────────────────
    run_deep = (
        settings.use_deep_scan
        and (not settings.deep_scan_persons_only or person_count > 0)
    )
    if run_deep:
        try:
            scan = _run_moondream(image_path)
            structure_note = scan["structure_note"]

            verdict = scan.get("verdict", "pass" if scan["structure_ok"] else "uncertain")
            if verdict == "fail":
                issues.append(f"major structure defect: {structure_note}")
            elif verdict == "uncertain":
                issues.append(f"uncertain structure: {structure_note}")
        except Exception as exc:
            issues.append(f"deep scan skipped ({exc})")

    # ── Final verdict ─────────────────────────────────────────────────────────
    if any("major structure defect" in issue for issue in issues):
        status = "fail"
        score = 0.0
    elif any("uncertain structure" in issue or "deep scan skipped" in issue for issue in issues):
        # Conservative routing: anything not confidently PASS belongs in the
        # single review queue. This favors catching defects over precision.
        status = "fail"
        score = 25.0
    else:
        status = "pass"
        score = 100.0

    os.makedirs(output_dir, exist_ok=True)
    for folder in ("pass", "warning", "fail"):
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    destination = _route_image(image_path, output_dir, status, move_files)

    return {
        "filename":     os.path.basename(image_path),
        "status":       status,
        "score":        round(score, 1),
        "issues":       issues,
        "brightness":   round(brightness, 3),
        "contrast":     round(contrast, 3),
        "person_count": person_count,
        "detections":   detections,
        "structure_note": structure_note,
        "destination":  destination,
    }


# ---------------------------------------------------------------------------
# Optional: OpenAI-compatible vision backend
# ---------------------------------------------------------------------------

def _serve_image_url(image_path: str, max_side: int = 512) -> tuple:
    if Image is None:
        raise RuntimeError("Pillow required")
    tmp = tempfile.mkdtemp(prefix="qc_")
    name = quote(os.path.basename(image_path))
    out  = os.path.join(tmp, name)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        img.save(out, format="JPEG", quality=85)
    handler = partial(SimpleHTTPRequestHandler, directory=tmp)
    server  = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/{name}", tmp


def _stop_server(server, tmp: str) -> None:
    try:
        server.shutdown()
    finally:
        server.server_close()
    shutil.rmtree(tmp, ignore_errors=True)


def _http_post(url: str, payload: dict, timeout: float = 120.0) -> dict:
    if requests is not None:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    import urllib.request, urllib.error
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Backend {e.code}: {e.read().decode()}") from e


def _parse_llm_response(text: str) -> Dict[str, Any]:
    """Extract status/score/issues from free-form LLM text."""
    t = text.strip()
    # Try embedded JSON first.
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s:e+1])
        except json.JSONDecodeError:
            pass
    lower = t.lower()
    status = (
        "fail"    if re.search(r"\b(fail|reject|unacceptable)\b", lower)
                     and not re.search(r"\bnot\s+fail\b", lower)
        else "warning" if re.search(r"\b(warning|minor|some issues)\b", lower)
        else "pass"    if re.search(r"\b(pass|good|acceptable|fine|ok)\b", lower)
        else "warning"
    )
    issues = [label for kw, label in [
        ("duplicate", "duplicate structure"), ("extra head", "extra head"),
        ("fused", "fused figures"), ("blur", "blurry"),
        ("dark", "dark"), ("bright", "bright"),
        ("artifact", "artifact"), ("noise", "noise"),
    ] if kw in lower]
    return {"status": status, "score": None, "issues": issues}


def classify_image_with_backend(
    image_path: str,
    backend_url: str,
    output_dir: str,
    timeout: float = 120.0,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
) -> Dict[str, Any]:
    """Classify one image via an OpenAI-compatible vision backend."""
    server, img_url, tmp = _serve_image_url(image_path)
    try:
        prompt = (
            "Inspect this image carefully for quality issues. "
            "Look for: duplicated/fused figures, extra or missing limbs, "
            "malformed hands, exposure problems, blur, and noise. "
            "Reply with a JSON object: {status, score, issues}. "
            "status: pass/warning/fail. score: 0-100. issues: list of strings."
        )
        raw = _http_post(backend_url, {
            "model": model_name, "temperature": 0.1,
            "image_url": img_url,
            "messages": [
                {"role": "system", "content": "You are an image QC assistant."},
                {"role": "user",   "content": prompt},
            ],
        }, timeout=timeout)
    finally:
        _stop_server(server, tmp)

    choices = raw.get("choices") or []
    text = ""
    if choices:
        msg  = choices[0].get("message") or {}
        text = msg.get("content") or choices[0].get("text") or ""
    if not text:
        raise ValueError("Backend returned no content")

    parsed = _parse_llm_response(text)
    status = parsed.get("status", "warning")
    score  = float(parsed.get("score") or
                   (100 if status == "pass" else 65 if status == "warning" else 30))
    issues = parsed.get("issues") or []

    os.makedirs(output_dir, exist_ok=True)
    for folder in ("pass", "warning", "fail"):
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    destination = _route_image(image_path, output_dir, status, move_files)

    return {
        "filename":     os.path.basename(image_path),
        "status":       status,
        "score":        round(score, 1),
        "issues":       issues,
        "structure_note": text[:200],
        "destination":  destination,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def collect_images(input_path: str) -> List[str]:
    input_path = os.path.expanduser(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, n) for n in os.listdir(input_path)
            if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(f"Not found: {input_path}")


def _route_image(image_path: str, output_dir: str,
                 status: str, move_files: bool) -> str:
    dest = os.path.join(output_dir, status, os.path.basename(image_path))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    (shutil.move if move_files else shutil.copy2)(image_path, dest)
    return dest


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_qc(
    input_path: str,
    output_dir: str,
    backend_url: Optional[str] = None,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
    settings: QCSettings | None = None,
) -> List[Dict[str, Any]]:
    """Run QC on all images in ``input_path``, write report.json."""
    if settings is None:
        settings = _DEFAULT_SETTINGS
    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    results = []
    for image_path in images:
        if backend_url:
            r = classify_image_with_backend(
                image_path, backend_url, output_dir,
                model_name=model_name, move_files=move_files)
        else:
            r = classify_image(image_path, output_dir,
                               move_files=move_files, settings=settings)
        results.append(r)

    os.makedirs(output_dir, exist_ok=True)
    report = os.path.join(output_dir, "report.json")
    with open(report, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("pass", "warning", "fail")}
    print(f"QC done — {len(results)} images | "
          f"pass {counts['pass']}  warning {counts['warning']}  "
          f"fail {counts['fail']}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="StereoSift image QC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",        required=True)
    p.add_argument("--output-dir",   default="output/qc")
    p.add_argument("--backend-url",  default=None)
    p.add_argument("--model",        default="llama-3.2-11b-vision-instruct")
    p.add_argument("--deep-scan", dest="deep_scan", action="store_true", default=True,
                   help="Enable moondream2 structure scan (default)")
    p.add_argument("--no-deep-scan", dest="deep_scan", action="store_false",
                   help="Disable moondream2 and run detection metadata only")
    p.add_argument("--strictness",   default="balanced",
                   choices=["relaxed", "balanced", "strict"])
    p.add_argument("--no-yolo",      action="store_true")
    p.add_argument("--move",         action="store_true")
    args = p.parse_args()

    settings = QCSettings(
        use_yolo=not args.no_yolo,
        use_deep_scan=args.deep_scan,
        deep_scan_persons_only=False,
        deep_scan_strictness=args.strictness,
    )
    run_qc(args.input, args.output_dir,
           backend_url=args.backend_url,
           model_name=args.model,
           move_files=args.move,
           settings=settings)


if __name__ == "__main__":
    main()
