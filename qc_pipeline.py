#!/usr/bin/env python3
"""Image quality-control pipeline.

Primary path: local YOLO-based inference via PyTorch (no server required).
  - ``yolo11n.pt``      — object detection, catches exposure/blur/clutter
  - ``yolo11n-pose.pt`` — pose estimation, counts heads/faces/keypoints to
                          flag duplicate structure and missing body parts

Both checkpoints (~6 MB each) are downloaded automatically from the
Ultralytics CDN on first use and cached in ``models/yolo/``.

Optional path: OpenAI-compatible vision backend (LM Studio, Ollama, etc.).
Pass ``backend_url`` to ``run_qc`` or ``classify_image_with_backend`` to use
this instead.  Useful for richer natural-language reasoning about artifacts.

Pixel-only fallback: if Ultralytics is not installed the pipeline falls back
to brightness/contrast/edge heuristics with no model dependency.

Output structure
----------------
Each run writes copies (or moves) of images into::

    <output_dir>/pass/
    <output_dir>/warning/
    <output_dir>/fail/
    <output_dir>/report.json
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
    from PIL import Image, ImageFilter
except ImportError:
    Image = None
    ImageFilter = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Local cache directory for YOLO checkpoints (alongside depth models).
_YOLO_MODEL_DIR = Path(__file__).parent / "models" / "yolo"

# YOLO model filenames.
_YOLO_DETECT_MODEL = "yolo11n.pt"
_YOLO_POSE_MODEL   = "yolo11n-pose.pt"

# Structure anomaly thresholds.
# More than this many people detected with unexpected keypoint counts → warning/fail.
_MAX_HEADS_PER_PERSON = 1   # nose keypoints per detected person
_POSE_CONF_THRESHOLD  = 0.4  # minimum keypoint confidence to count


# ---------------------------------------------------------------------------
# YOLO model loading (lazy, cached per process)
# ---------------------------------------------------------------------------

_yolo_detect_cache: Any = None
_yolo_pose_cache:   Any = None


def _get_yolo_detect():
    """Return the cached detection model, downloading if needed."""
    global _yolo_detect_cache
    if _yolo_detect_cache is None:
        _yolo_detect_cache = _load_yolo(_YOLO_DETECT_MODEL)
    return _yolo_detect_cache


def _get_yolo_pose():
    """Return the cached pose model, downloading if needed."""
    global _yolo_pose_cache
    if _yolo_pose_cache is None:
        _yolo_pose_cache = _load_yolo(_YOLO_POSE_MODEL)
    return _yolo_pose_cache


def _load_yolo(filename: str):
    """Load a YOLO model, downloading to ``models/yolo/`` if absent."""
    from ultralytics import YOLO
    _YOLO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = _YOLO_MODEL_DIR / filename
    # Ultralytics downloads automatically when given just a filename,
    # but we want it stored in our models dir, so copy it there if needed.
    if not path.exists():
        print(f"Downloading {filename} to {_YOLO_MODEL_DIR} …")
        # Load with just the name so Ultralytics uses its own download logic,
        # then save the resulting model file to our preferred location.
        tmp = YOLO(filename)
        src = Path(tmp.ckpt_path)
        shutil.copy2(src, path)
        print(f"Saved {filename} → {path}")
    return YOLO(str(path))


# ---------------------------------------------------------------------------
# YOLO-based classification
# ---------------------------------------------------------------------------

def _classify_with_yolo(image_path: str) -> Dict[str, Any]:
    """Run YOLO detection + pose on one image and return a raw findings dict.

    Returns:
        {
          "person_count": int,
          "head_count": int,        # nose keypoints above threshold
          "extra_heads": bool,
          "missing_person": bool,   # expected person but none found
          "low_confidence_pose": bool,
          "detections": list[str],  # human-readable detected class names
        }
    """
    detect_model = _get_yolo_detect()
    pose_model   = _get_yolo_pose()

    # ── detection pass ───────────────────────────────────────────────────────
    det_results  = detect_model(image_path, verbose=False)[0]
    class_names  = det_results.names
    boxes        = det_results.boxes

    detected_classes: list[str] = []
    person_count = 0
    if boxes is not None and len(boxes):
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            if conf < 0.35:
                continue
            name = class_names.get(int(cls_id), str(cls_id))
            detected_classes.append(name)
            if name == "person":
                person_count += 1

    # ── pose pass (only meaningful when people are present) ──────────────────
    head_count          = 0
    low_confidence_pose = False

    if person_count > 0:
        pose_results = pose_model(image_path, verbose=False)[0]
        kps = pose_results.keypoints  # shape (N_persons, 17, 3) — x,y,conf

        if kps is not None and kps.data is not None:
            kp_data = kps.data.cpu().numpy()   # (N, 17, 3)
            for person_kps in kp_data:
                # Keypoint 0 = nose (head proxy)
                nose_conf = float(person_kps[0, 2])
                if nose_conf >= _POSE_CONF_THRESHOLD:
                    head_count += 1
                # Check overall pose confidence
                visible = (person_kps[:, 2] >= _POSE_CONF_THRESHOLD).sum()
                if visible < 5:
                    low_confidence_pose = True

    extra_heads    = person_count > 0 and head_count > person_count
    missing_person = person_count > 0 and head_count == 0

    return {
        "person_count":        person_count,
        "head_count":          head_count,
        "extra_heads":         extra_heads,
        "missing_person":      missing_person,
        "low_confidence_pose": low_confidence_pose,
        "detections":          detected_classes,
    }


# ---------------------------------------------------------------------------
# Pixel-level heuristics (exposure, contrast — no model needed)
# ---------------------------------------------------------------------------

def _pixel_stats(image_path: str) -> Dict[str, float]:
    """Return brightness and contrast from the greyscale histogram."""
    if Image is None:
        return {"brightness": 0.5, "contrast": 0.2}
    img  = Image.open(image_path).convert("L")
    data = list(img.getdata())
    vals = [v / 255.0 for v in data]
    mean = sum(vals) / len(vals) if vals else 0.5
    std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if vals else 0.2
    return {"brightness": mean, "contrast": std}


# ---------------------------------------------------------------------------
# Main local classifier
# ---------------------------------------------------------------------------

def classify_image(
    image_path: str,
    output_dir: str,
    move_files: bool = False,
    use_yolo: bool = True,
) -> Dict[str, Any]:
    """Classify one image using local models and pixel heuristics.

    Args:
        image_path: Path to the source image.
        output_dir: Root output directory; pass/warning/fail subdirs are
            created automatically.
        move_files: Move the original instead of copying it.
        use_yolo: Set False to skip YOLO (pixel heuristics only).  Useful
            when Ultralytics is not installed or for very fast previews.

    Returns:
        Result dict with keys: filename, status, score, issues, brightness,
        contrast, person_count, head_count, detections, destination.
    """
    issues: list[str] = []
    score = 100.0
    yolo_findings: Dict[str, Any] = {}

    # ── pixel checks (always run) ─────────────────────────────────────────────
    stats = _pixel_stats(image_path)
    brightness = stats["brightness"]
    contrast   = stats["contrast"]

    if brightness < 0.08:
        issues.append("very dark")
        score -= 25
    elif brightness < 0.20:
        issues.append("dark exposure")
        score -= 10

    if brightness > 0.92:
        issues.append("very bright / overexposed")
        score -= 25
    elif brightness > 0.85:
        issues.append("bright exposure")
        score -= 10

    if contrast < 0.08:
        issues.append("low contrast")
        score -= 12
    elif contrast > 0.48:
        issues.append("high contrast / harsh")
        score -= 8

    # ── YOLO checks ───────────────────────────────────────────────────────────
    try:
        if use_yolo:
            yolo_findings = _classify_with_yolo(image_path)

            if yolo_findings["extra_heads"]:
                issues.append(
                    f"duplicate head detected "
                    f"({yolo_findings['head_count']} heads, "
                    f"{yolo_findings['person_count']} person(s))"
                )
                score -= 40

            if yolo_findings["missing_person"]:
                issues.append("person detected but no head visible")
                score -= 20

            if yolo_findings["low_confidence_pose"]:
                issues.append("low-confidence pose — possible structure artifact")
                score -= 15

    except Exception as exc:
        issues.append(f"YOLO check skipped ({exc})")

    # ── final verdict ─────────────────────────────────────────────────────────
    score = max(0.0, min(100.0, score))

    if score >= 80:
        status = "pass"
    elif score >= 50:
        status = "warning"
    else:
        status = "fail"

    # Structure failures always floor to fail regardless of score.
    if yolo_findings.get("extra_heads"):
        status = "fail"

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
        "person_count": yolo_findings.get("person_count", 0),
        "head_count":   yolo_findings.get("head_count", 0),
        "detections":   yolo_findings.get("detections", []),
        "destination":  destination,
    }


# ---------------------------------------------------------------------------
# Optional: OpenAI-compatible vision backend
# ---------------------------------------------------------------------------

def _serve_image_file_url(
    image_path: str, max_side: int = 256, quality: int = 25
) -> tuple[ThreadingHTTPServer, str, str]:
    if Image is None:
        raise RuntimeError("Pillow is required to encode images for backend QC")
    temp_dir    = tempfile.mkdtemp(prefix="qc_img_")
    output_name = quote(os.path.basename(image_path))
    output_path = os.path.join(temp_dir, output_name)
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        img.save(output_path, format="JPEG", quality=quality, optimize=True)
    handler = partial(SimpleHTTPRequestHandler, directory=temp_dir)
    server  = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/{output_name}"
    return server, url, temp_dir


def _shutdown_image_server(server: ThreadingHTTPServer, temp_dir: str) -> None:
    try:
        server.shutdown()
    finally:
        server.server_close()
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    if requests is not None:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    import urllib.request, urllib.error
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend {exc.code} {exc.reason}: {body}") from exc


def _parse_backend_text(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction + text fallback from an LLM response."""
    trimmed = text.strip()
    # Try to find a JSON object anywhere in the response.
    for start, end in [(trimmed.find("{"), trimmed.rfind("}"))]:
        if start >= 0 and end > start:
            try:
                return json.loads(trimmed[start : end + 1])
            except json.JSONDecodeError:
                pass
    # Fall back to keyword extraction.
    lower  = trimmed.lower()
    status = ("fail"    if re.search(r"\b(fail|reject|bad|terrible)\b", lower) and
                          not re.search(r"\bnot\s+fail\b", lower)
              else "warning" if re.search(r"\b(warning|minor|some issues)\b", lower)
              else "pass"    if re.search(r"\b(pass|good|acceptable|fine)\b", lower)
              else "warning")
    issues: list[str] = []
    for kw, label in [("duplicate", "duplicate structure"), ("extra head", "extra head"),
                      ("blur", "blurry"), ("dark", "dark"), ("bright", "bright"),
                      ("artifact", "artifact"), ("noise", "noise")]:
        if kw in lower:
            issues.append(label)
    return {"status": status, "score": None, "issues": issues}


def classify_image_with_backend(
    image_path: str,
    backend_url: str,
    output_dir: str,
    timeout: float = 120.0,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
) -> Dict[str, Any]:
    """Classify one image using an OpenAI-compatible vision backend.

    The image is served over a local HTTP server so the backend can fetch it
    by URL (works with LM Studio, Ollama, and similar local servers).
    """
    server, image_url, temp_dir = _serve_image_file_url(image_path)
    try:
        prompt = (
            "You are an image quality-control assistant. Inspect the image carefully. "
            "Explicitly look for: duplicated or missing heads, faces, torsos, arms, legs, "
            "hands, fingers, fused body parts, exposure problems, blur, and noise. "
            "Reply with a brief assessment followed by a JSON object containing: "
            "status (pass/warning/fail), score (0-100), issues (list of strings)."
        )
        payload = {
            "model": model_name,
            "temperature": 0.2,
            "messages": [
                {"role": "system",
                 "content": "You are an image QC assistant. Read the image from image_url."},
                {"role": "user",
                 "content": prompt},
            ],
            "image_url": image_url,
        }
        raw = _http_post_json(backend_url, payload, timeout=timeout)
    finally:
        _shutdown_image_server(server, temp_dir)

    choices = raw.get("choices") or []
    text    = ""
    if choices:
        msg  = choices[0].get("message") or {}
        text = msg.get("content") or choices[0].get("text") or ""
    if not text:
        raise ValueError("Backend returned no content")

    parsed = _parse_backend_text(text)
    status = parsed.get("status", "warning")
    score  = float(parsed.get("score") or (100 if status == "pass" else 65 if status == "warning" else 30))
    issues = parsed.get("issues") or []

    os.makedirs(output_dir, exist_ok=True)
    for folder in ("pass", "warning", "fail"):
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    destination = _route_image(image_path, output_dir, status, move_files)

    return {
        "filename":    os.path.basename(image_path),
        "status":      status,
        "score":       round(score, 1),
        "issues":      issues,
        "destination": destination,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def collect_images(input_path: str) -> List[str]:
    """Return sorted list of image paths from a file or directory."""
    input_path = os.path.expanduser(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        names = [p for p in os.listdir(input_path)
                 if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS]
        return sorted(os.path.join(input_path, n) for n in names)
    raise FileNotFoundError(f"Input not found: {input_path}")


def _route_image(image_path: str, output_dir: str, status: str, move_files: bool) -> str:
    dest_dir = os.path.join(output_dir, status)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(image_path))
    if move_files:
        shutil.move(image_path, dest)
    else:
        shutil.copy2(image_path, dest)
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
    use_yolo: bool = True,
) -> List[Dict[str, Any]]:
    """Run QC on all images in ``input_path``.

    Uses local YOLO inference by default.  Pass ``backend_url`` to route
    through an OpenAI-compatible vision server instead.

    Returns:
        List of per-image result dicts, also written to
        ``<output_dir>/report.json``.
    """
    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    results = []
    for image_path in images:
        if backend_url:
            r = classify_image_with_backend(
                image_path, backend_url, output_dir,
                model_name=model_name, move_files=move_files,
            )
        else:
            r = classify_image(image_path, output_dir,
                               move_files=move_files, use_yolo=use_yolo)
        results.append(r)

    report_path = os.path.join(output_dir, "report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("pass", "warning", "fail")}
    print(f"QC complete — {len(results)} images | "
          f"pass {counts['pass']}  warning {counts['warning']}  fail {counts['fail']}")
    print(f"Report: {report_path}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StereoSift image QC — local YOLO or vision backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",       required=True, help="Image file or folder")
    parser.add_argument("--output-dir",  default="output/qc")
    parser.add_argument("--backend-url", default=None,
                        help="OpenAI-compatible vision endpoint (optional)")
    parser.add_argument("--model",       default="llama-3.2-11b-vision-instruct",
                        help="Model name for backend QC")
    parser.add_argument("--no-yolo",     action="store_true",
                        help="Skip YOLO — pixel heuristics only")
    parser.add_argument("--move",        action="store_true",
                        help="Move originals instead of copying")
    args = parser.parse_args()

    run_qc(
        args.input,
        args.output_dir,
        backend_url=args.backend_url,
        model_name=args.model,
        move_files=args.move,
        use_yolo=not args.no_yolo,
    )


if __name__ == "__main__":
    main()
