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

3. Pose / structure scan (optional, off by default)
   YOLO pose checks the people in the image for obvious duplicate torsos or
   heads. If enabled, a local LLM scan can still act as a fallback for tricky
   cases, but it is no longer the only structure judge.

Optional fourth path: OpenAI-compatible vision backend (LM Studio, Ollama).
Pass ``backend_url`` to replace the local moondream2 scan with an API call.

Output
------
``<output_dir>/pass/``, ``warning/``, ``fail/``, ``report.json``
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
_YOLO_POSE_FILE    = "yolo11n-pose.pt"
_MOONDREAM_REPO    = "vikhyatk/moondream2"
_MOONDREAM_REVISION = "2025-01-09"   # pinned for reproducibility


# ---------------------------------------------------------------------------
# Model caches (process-level singletons)
# ---------------------------------------------------------------------------

_yolo_detect_cache: Any   = None
_yolo_pose_cache: Any     = None
_moondream_cache: Any = None   # model singleton


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

def _get_yolo_detect():
    global _yolo_detect_cache
    if _yolo_detect_cache is None:
        _yolo_detect_cache = _load_yolo(_YOLO_DETECT_FILE)
    return _yolo_detect_cache


def _get_yolo_pose():
    global _yolo_pose_cache
    if _yolo_pose_cache is None:
        _yolo_pose_cache = _load_yolo(_YOLO_POSE_FILE)
    return _yolo_pose_cache


def _load_yolo(filename: str):
    """Load YOLO, downloading to ``models/yolo/`` if not present."""
    from ultralytics import YOLO
    _YOLO_DIR.mkdir(parents=True, exist_ok=True)
    path = _YOLO_DIR / filename
    if not path.exists():
        root_path = Path(__file__).parent / filename
        if root_path.exists():
            shutil.copy2(root_path, path)
            print(f"Copied {filename} from workspace root → {_YOLO_DIR}")
        else:
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


def _run_yolo_pose(image_path: str) -> Dict[str, Any]:
    """Run YOLO pose detection to get keypoints and boxes for duplicate structure check."""
    model = _get_yolo_pose()
    results = model(image_path, verbose=False)[0]
    boxes = results.boxes
    keypoints = results.keypoints

    kp_list = []
    box_list = []
    if keypoints is not None and len(keypoints) and keypoints.conf is not None:
        xy = keypoints.xy.cpu().numpy() if hasattr(keypoints.xy, "cpu") else keypoints.xy
        conf = keypoints.conf.cpu().numpy() if hasattr(keypoints.conf, "cpu") else keypoints.conf
        box_coords = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy

        for i in range(len(xy)):
            kp_list.append({
                "xy": xy[i],
                "conf": conf[i]
            })
            box_list.append(box_coords[i])

    return {"keypoints": kp_list, "boxes": box_list}


def _check_pose_anomalies(keypoints_list: List[Dict[str, Any]], boxes_list: List[Any]) -> List[str]:
    """Analyze keypoints and boxes of detected persons to find duplicates.

    Checks two cases:
    1. Single skeleton with corrupted geometry (e.g., eyes too far apart = split head)
    2. Two skeletons that share lower body but have separated upper body (fused figures)
    """
    issues = []
    num_persons = len(keypoints_list)

    # ── Check single skeleton for internal corruption ───────────────────────────
    if num_persons == 1:
        xy = keypoints_list[0]['xy']
        conf = keypoints_list[0]['conf']
        box = boxes_list[0]
        scale = max(box[2] - box[0], box[3] - box[1])

        # Eyes too far apart = likely head duplication/split
        if conf[1] > 0.35 and conf[2] > 0.35:
            eye_dist = ((xy[1][0] - xy[2][0])**2 + (xy[1][1] - xy[2][1])**2)**0.5 / scale
            if eye_dist > 0.13:  # Normal ~0.05-0.10, corrupted >0.13
                issues.append("duplicate head: eye keypoints too far apart (split head)")

        # Nose off-center from eyes = one nose serving two heads
        if conf[0] > 0.35 and conf[1] > 0.35 and conf[2] > 0.35:
            nose = xy[0]
            l_eye = xy[1]
            r_eye = xy[2]
            eye_center_x = (l_eye[0] + r_eye[0]) / 2
            nose_offset = abs(nose[0] - eye_center_x) / scale
            if nose_offset > 0.12:  # Nose significantly off-center
                issues.append("duplicate head: nose offset from eye centerline (fused figures)")

        # Ears too far apart relative to eye distance
        if conf[1] > 0.35 and conf[3] > 0.35 and conf[4] > 0.35:
            l_eye = xy[1]
            r_eye = xy[2]
            l_ear = xy[3]
            r_ear = xy[4]
            ear_spread = ((l_ear[0] - r_ear[0])**2 + (l_ear[1] - r_ear[1])**2)**0.5 / scale
            if ear_spread > 0.35:
                issues.append("duplicate head: ear keypoints too far apart")

    if num_persons < 2:
        return issues

    T_close = 0.10  # Relaxed from 0.08 to catch more duplicates
    T_far = 0.12    # Relaxed from 0.15 for better separation detection
    T_vertical_overlap = 0.25  # Hips in similar Y-range = likely same person

    for i in range(num_persons):
        for j in range(i + 1, num_persons):
            box_i = boxes_list[i]
            box_j = boxes_list[j]

            w_i, h_i = box_i[2] - box_i[0], box_i[3] - box_i[1]
            w_j, h_j = box_j[2] - box_j[0], box_j[3] - box_j[1]

            scale = (max(w_i, h_i) + max(w_j, h_j)) / 2.0
            if scale <= 0:
                continue

            xy_i = keypoints_list[i]['xy']
            conf_i = keypoints_list[i]['conf']
            xy_j = keypoints_list[j]['xy']
            conf_j = keypoints_list[j]['conf']

            def kp_dist(k):
                if conf_i[k] > 0.35 and conf_j[k] > 0.35:
                    return ((xy_i[k][0] - xy_j[k][0])**2 + (xy_i[k][1] - xy_j[k][1])**2)**0.5 / scale
                return None

            # Gather lower body distances
            hips = [d for d in [kp_dist(11), kp_dist(12)] if d is not None]
            knees = [d for d in [kp_dist(13), kp_dist(14)] if d is not None]
            ankles = [d for d in [kp_dist(15), kp_dist(16)] if d is not None]

            # Gather upper body distances
            shoulders = [d for d in [kp_dist(5), kp_dist(6)] if d is not None]
            head_pts = [d for d in [kp_dist(0), kp_dist(1), kp_dist(2), kp_dist(3), kp_dist(4)] if d is not None]

            # Check if bounding boxes overlap/nearly touch (suggests one person split)
            boxes_overlapping = False
            x1_min, y1_min, x1_max, y1_max = box_i
            x2_min, y2_min, x2_max, y2_max = box_j
            # Check X-axis overlap: if gaps between boxes is small relative to box width
            x_gap = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
            y_gap = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
            box_width = (box_i[2] - box_i[0] + box_j[2] - box_j[0]) / 2
            if x_gap < box_width * 0.25 and y_gap < scale * 0.5:  # Boxes nearly touch
                boxes_overlapping = True

            # Check if hips are vertically aligned (same person, split horizontally)
            hips_vertically_aligned = False
            hips_horizontally_separated = False
            if conf_i[11] > 0.35 and conf_j[11] > 0.35:  # L_hip
                hip_y_diff = abs(xy_i[11][1] - xy_j[11][1]) / scale
                hip_x_diff = abs(xy_i[11][0] - xy_j[11][0]) / scale
                if hip_y_diff < T_vertical_overlap:
                    hips_vertically_aligned = True
                    if hip_x_diff > 0.10 and boxes_overlapping:  # Only flag if boxes also overlap
                        hips_horizontally_separated = True
            if conf_i[12] > 0.35 and conf_j[12] > 0.35:  # R_hip
                hip_y_diff = abs(xy_i[12][1] - xy_j[12][1]) / scale
                hip_x_diff = abs(xy_i[12][0] - xy_j[12][0]) / scale
                if hip_y_diff < T_vertical_overlap:
                    hips_vertically_aligned = True
                    if hip_x_diff > 0.10 and boxes_overlapping:
                        hips_horizontally_separated = True

            # Assess sharing lower body
            shares_lower = False
            if hips_vertically_aligned and boxes_overlapping:  # Same person, split horizontally
                shares_lower = True
            elif hips and sum(hips)/len(hips) < T_close:
                shares_lower = True
            elif knees and sum(knees)/len(knees) < T_close:
                shares_lower = True
            elif ankles and sum(ankles)/len(ankles) < T_close:
                shares_lower = True
            elif hips and min(hips) < 0.06:
                shares_lower = True

            # Assess shoulder & head separation (only meaningful if hips NOT aligned)
            shoulders_separated = len(shoulders) > 0 and sum(shoulders)/len(shoulders) > T_far
            head_separated = len(head_pts) > 0 and sum(head_pts)/len(head_pts) > T_far

            # Assess shoulder & head closeness
            shoulders_close = len(shoulders) > 0 and sum(shoulders)/len(shoulders) < T_close

            # Flag if hips vertically aligned (same person) but split horizontally
            if shares_lower and hips_horizontally_separated:
                issues.append("duplicate torso: hips vertically aligned but horizontally separated (split person)")
            elif shares_lower:
                if shoulders_separated:
                    issues.append("duplicate torso: sharing a lower body but having separated torsos")
                elif head_separated and shoulders_close:
                    issues.append("duplicate head: sharing a torso but having separated heads")
                elif head_separated:
                    issues.append("duplicate head: sharing a lower body but having separated heads")

    return issues


# ---------------------------------------------------------------------------
# Structure scanning
# ---------------------------------------------------------------------------

def _run_structure_scan(image_path: str) -> Dict[str, Any]:
    """Run pose detection and convert obvious duplicate structure into issues.

    This is the primary structure gate. The local LLM scan is only used as a
    fallback for ambiguous cases after this pass.
    """
    person_count = 0
    detections: list[str] = []
    issues: list[str] = []

    yolo = _run_yolo(image_path)
    person_count = yolo["person_count"]
    detections = yolo["detections"]

    pose_data = _run_yolo_pose(image_path)
    pose_issues = _check_pose_anomalies(pose_data["keypoints"], pose_data["boxes"])
    for issue in pose_issues:
        issues.append(f"major structure defect: {issue}")

    return {
        "person_count": person_count,
        "detections": detections,
        "issues": issues,
    }

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
          "verdict": str,       # "pass", "fail", "uncertain"
          "structure_ok": bool,   # True = no defects found
          "structure_note": str,  # model's description
          "raw": str,           # full model response
        }
    """
    if Image is None:
        raise RuntimeError("Pillow required for moondream2 inference")

    model = _get_moondream()
    img = Image.open(image_path).convert("RGB")

    prompt = (
        "Judge only major duplicated or incorrectly joined human body structure. "
        "FAIL only for a clearly extra head, a second torso sharing one lower body, "
        "an extra arm or leg beyond a plausible human body, or visibly fused people/bodies. "
        "Do not fail for hidden or cropped limbs, hands or fingers, pose, clothing, lighting, "
        "blur, artistic style, or minor visual imperfections. "
        "Reply exactly as one of: PASS; FAIL: <brief reason>; UNCERTAIN: <brief reason>."
    )
    result = model.query(img, prompt)  # type: ignore
    answer: str = result["answer"].strip() if isinstance(result, dict) else str(result).strip()

    verdict = _parse_structure_verdict(answer)

    return {
        "verdict": verdict,
        "structure_ok": verdict == "pass",
        "structure_note": answer,
        "raw": answer,
    }


def _parse_structure_verdict(answer: str) -> str:
    """Parse the required leading verdict without guessing from prose.

    Checking for PASS anywhere before checking for FAIL made responses such as
    ``FAIL: this does not pass`` become passes.  Requiring the verdict at the
    start also prevents explanatory text from accidentally changing a result.
    """
    match = re.match(r"^\s*(PASS|FAIL|UNCERTAIN)\b", answer, re.IGNORECASE)
    if not match:
        return "uncertain"
    return match.group(1).lower()


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
        deep_scan_persons_only: bool = True,
        strict_offline: bool = False,
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
        self.strict_offline          = strict_offline
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
    detector_notes: list[str] = []

    # ── 1. Pixel checks ───────────────────────────────────────────────────────
    stats      = _pixel_stats(image_path)
    brightness = stats["brightness"]
    contrast   = stats["contrast"]

    # Exposure and contrast are recorded for diagnostics only. They do not
    # influence structural-structure judgment.

    # ── 2. Pose structure gate ──────────────────────────────────────────────────
    if settings.strict_offline:
        detector_notes.append(
            "Strict offline mode: pixel-only QC; YOLO pose and fallback scan disabled"
        )
    elif settings.use_yolo:
        try:
            pose_scan = _run_structure_scan(image_path)
            person_count = pose_scan["person_count"]
            detections = pose_scan["detections"]
            issues.extend(pose_scan["issues"])
        except Exception as exc:
            detector_notes.append(f"YOLO structure gate unavailable: {exc}")

    # ── 3. Fallback structure scan ──────────────────────────────────────────────
    # Only use the local LLM if the primary pose gate did not already find a
    # clear structural defect.
    run_deep = (
        (not settings.strict_offline)
        and settings.use_deep_scan
        and (not settings.deep_scan_persons_only or person_count > 0)
        and not any("major structure defect" in issue for issue in issues)
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
        status = "warning"
        score = 50.0
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
        "detector_notes": detector_notes,
        "destination":  destination,
    }


# ---------------------------------------------------------------------------
# Optional: OpenAI-compatible vision backend
# ---------------------------------------------------------------------------

def _image_data_url(image_path: str, max_side: int = 1536) -> str:
    """Return a standard inline image URL understood by LM Studio and oMLX."""
    if Image is None:
        raise RuntimeError("Pillow required")
    from io import BytesIO

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _chat_completions_url(backend_url: str) -> str:
    """Accept a server root, OpenAI base URL, or full chat endpoint."""
    url = backend_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _post_with_requests(url, payload, headers, timeout, max_retries):
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Backend unreachable: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
            continue
        if r.ok:
            return r.json()
        detail = r.text.strip()[:500]
        if r.status_code not in _RETRYABLE_STATUS or attempt == max_retries:
            raise RuntimeError(f"Backend HTTP {r.status_code}: {detail or r.reason}")
        time.sleep(min(2 ** attempt, 8))


def _post_with_urllib(url, payload, headers, timeout, max_retries):
    import urllib.request, urllib.error
    data = json.dumps(payload).encode()
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_STATUS or attempt == max_retries:
                raise RuntimeError(f"Backend {exc.code}: {exc.read().decode()}") from exc
        except urllib.error.URLError as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Backend unreachable: {exc}") from exc
        time.sleep(min(2 ** attempt, 8))


def _http_post(
    url: str,
    payload: dict,
    timeout: float = 120.0,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> dict:
    """POST with retry/backoff for transient failures.

    Retries connection errors, timeouts, and 429/5xx responses (exponential
    backoff capped at 8s). Client errors (4xx besides 429) fail immediately
    since retrying them won't help.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if requests is not None:
        return _post_with_requests(url, payload, headers, timeout, max_retries)
    return _post_with_urllib(url, payload, headers, timeout, max_retries)


_QC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pass", "warning", "fail"]},
        "score": {"type": "number", "minimum": 0, "maximum": 100},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "score", "issues"],
    "additionalProperties": False,
}


def _parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse the structured QC verdict enforced via response_format.

    The request constrains the backend to this JSON schema, so content should
    be pure JSON. Some backends still wrap it in prose or code fences, so fall
    back to extracting the outermost {...} block before giving up — no
    keyword-guessing from prose, since that misreads phrasing like
    "does not fail" as a fail.
    """
    t = text.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Backend response was not valid JSON: {t[:200]!r}")


def classify_image_with_backend(
    image_path: str,
    backend_url: str,
    output_dir: str,
    timeout: float = 120.0,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify one image via an OpenAI-compatible vision backend."""
    img_url = _image_data_url(image_path)
    prompt = (
        "Inspect the human body structure in this image, counting each visible head, "
        "torso, arm, leg, and complete person before deciding. FAIL only for a "
        "clear major structural defect: duplicate heads, two torsos joined to one "
        "lower body, extra limbs, or visibly fused bodies. PASS normal separate "
        "people and do not penalize cropping, occlusion, pose, hands, clothing, "
        "lighting, blur, or artistic style. If genuinely ambiguous use WARNING. "
        "Return only JSON: {\"status\":\"pass|warning|fail\","
        "\"score\":0-100,\"issues\":[\"brief evidence\"]}."
    )
    raw = _http_post(_chat_completions_url(backend_url), {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 250,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "qc_verdict", "strict": True, "schema": _QC_RESPONSE_SCHEMA},
        },
        "messages": [
            {"role": "system", "content": (
                "You are a conservative image QC judge. Base the verdict only "
                "on structure actually visible in the supplied image."
            )},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": img_url, "detail": "high"
                }},
            ]},
        ],
    }, timeout=timeout, api_key=api_key)

    choices = raw.get("choices") or []
    text = ""
    if choices:
        msg  = choices[0].get("message") or {}
        text = msg.get("content") or choices[0].get("text") or ""
        if isinstance(text, list):
            text = "".join(
                item.get("text", "") for item in text if isinstance(item, dict)
            )
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
    # A rerun may produce a different verdict. Remove stale copies so one image
    # can never appear in both pass and fail (or warning) at the same time.
    filename = os.path.basename(image_path)
    source = os.path.abspath(image_path)
    for folder in ("pass", "warning", "fail"):
        old = os.path.abspath(os.path.join(output_dir, folder, filename))
        if old != source and os.path.isfile(old):
            os.unlink(old)
    dest = os.path.join(output_dir, status, filename)
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
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run QC on all images in ``input_path``, write report.json."""
    if settings is None:
        settings = _DEFAULT_SETTINGS
    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    results = []
    for image_path in images:
        if backend_url and not settings.strict_offline:
            r = classify_image_with_backend(
                image_path, backend_url, output_dir,
                model_name=model_name, move_files=move_files,
                api_key=api_key)
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
    p.add_argument("--api-key", default=None,
                   help="Optional Bearer token for the vision backend")
    p.add_argument("--model",        default="llama-3.2-11b-vision-instruct")
    p.add_argument("--deep-scan", dest="deep_scan", action="store_true", default=True,
                   help="Enable moondream2 structure scan (default)")
    p.add_argument("--no-deep-scan", dest="deep_scan", action="store_false",
                   help="Disable moondream2 and run detection metadata only")
    p.add_argument("--strictness",   default="balanced",
                   choices=["relaxed", "balanced", "strict"])
    p.add_argument("--strict-offline", action="store_true",
                   help="Disable backend, YOLO, and model downloads; run pixel-only QC")
    p.add_argument("--no-yolo",      action="store_true")
    p.add_argument("--move",         action="store_true")
    args = p.parse_args()

    settings = QCSettings(
        use_yolo=not args.no_yolo and not args.strict_offline,
        use_deep_scan=args.deep_scan and not args.strict_offline,
        deep_scan_persons_only=False,
        strict_offline=args.strict_offline,
        deep_scan_strictness=args.strictness,
    )
    run_qc(args.input, args.output_dir,
           backend_url=None if args.strict_offline else args.backend_url,
           model_name=args.model,
           move_files=args.move,
           settings=settings,
           api_key=None if args.strict_offline else args.api_key)


if __name__ == "__main__":
    main()
