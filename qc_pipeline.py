#!/usr/bin/env python3
"""Image quality-control pipeline.

Two mutually exclusive judging paths. Which one runs is decided by whether a
backend URL is supplied (see ``run_qc``):

A. Vision backend (the default in the GUI)
   An OpenAI-compatible server — LM Studio, oMLX, Ollama — judges each image in
   a single request. This is the primary path; the local models below are not
   loaded or consulted at all when it is active. Malformed responses degrade to
   a "warning" so a bad reply cannot stall a batch.

B. Local models (used when no backend URL is set, or in strict offline mode)
   1. Pixel heuristics (always)
      Exposure, brightness, contrast — fast, zero model dependencies.
   2. YOLO object detection (``use_yolo``, on by default)
      ``yolo11n.pt`` (~6 MB, auto-downloaded to ``models/yolo/``). Reports
      person count and detected object classes. Honest scope: it does NOT
      detect fused figures — YOLO pose emits one skeleton per person instance
      and cannot flag two heads on one body.
   3. Deep scan (``use_deep_scan``, OFF by default)
      moondream2 (~2 GB) actually reasons about duplicated or incorrectly
      joined structure. Gated by ``deep_scan_persons_only`` so it only runs on
      images where a person was found.

Strict offline mode forces path B and additionally disables YOLO and the deep
scan, leaving pixel-only QC and no network-capable behaviour.

Output
------
``<output_dir>/pass/``, ``warning/``, ``fail/``, ``unscored/``

Originals are copied by default; pass ``move_files=True`` to move instead.
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

from media_utils import (
    collect_images as _collect_images,
    relative_output_subdir,
)

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

_YOLO_DIR          = Path(__file__).parent / "models" / "yolo"
_YOLO_DETECT_FILE  = "yolo11n.pt"
_YOLO_POSE_FILE    = "yolo11n-pose.pt"
_MOONDREAM_REPO    = "vikhyatk/moondream2"
_MOONDREAM_REVISION = "2025-01-09"   # pinned for reproducibility

_STRONG_STRUCTURE_PATTERNS = (
    "duplicate head",
    "two heads",
    "extra head",
    "twin",
    "twins",
    "duplicate torso",
    "two torsos",
    "fused bodies",
    "fused body",
    "fused person",
    "sharing a lower body",
    "sharing one lower body",
    "extra arm",
    "extra leg",
    "split person",
    "split head",
)

_SUSPECT_STRUCTURE_PATTERNS = (
    "two distinct heads",
    "two people",
    "similar body proportions",
    "appears to be twins",
    "twins",
    "twin",
    "overlapping limbs",
    "shared body",
    "shared torso",
)

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
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM

        model_path = snapshot_download(
            _MOONDREAM_REPO,
            revision=_MOONDREAM_REVISION,
            local_files_only=True,
        )
        print(f"Loading moondream2 ({_MOONDREAM_REPO} @ {_MOONDREAM_REVISION}) …")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float32,
        )
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
        model = model.to(device).eval()
        print(f"moondream2 ready on {device}")
        _moondream_cache = model
    return _moondream_cache


def _run_moondream(image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
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
    _append_human_readable_response(
        output_dir,
        image_path,
        f"moondream2 ({_MOONDREAM_REPO} @ {_MOONDREAM_REVISION})",
        answer,
    )

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


def _append_human_readable_response(
    output_dir: Optional[str],
    image_path: str,
    source: str,
    text: str,
    final_status: Optional[str] = None,
    final_score: Optional[float] = None,
    final_issues: Optional[List[str]] = None,
    final_field: str = "status",
) -> None:
    """Append a readable model response to a text log for later review."""
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "model_responses.log")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {os.path.basename(image_path)} | {source}\n")
        fh.write(text.rstrip() + "\n\n")
        if final_status is not None:
            fh.write(
                f"Final verdict: {final_field}={final_status} score="
                f"{final_score if final_score is not None else 'n/a'}\n"
            )
            if final_issues:
                fh.write(f"Final issues: {' | '.join(final_issues)}\n")
            fh.write("\n")


def _reset_human_readable_log(output_dir: Optional[str]) -> None:
    """Start a fresh per-run response log."""
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "model_responses.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("")


def _append_run_event(output_dir: Optional[str], text: str) -> None:
    """Append a run-level event to the human-readable log."""
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "model_responses.log")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {text}\n")


def _remove_stale_audit_artifacts(output_dir: Optional[str]) -> None:
    if not output_dir:
        return
    log_path = os.path.join(output_dir, "model_responses.log")
    if os.path.exists(log_path):
        os.unlink(log_path)


def _is_strong_structure_defect(note: str) -> bool:
    lowered = note.lower()
    return any(pattern in lowered for pattern in _STRONG_STRUCTURE_PATTERNS)


def _is_suspect_structure(note: str) -> bool:
    lowered = note.lower()
    return any(pattern in lowered for pattern in _SUSPECT_STRUCTURE_PATTERNS)


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
        use_deep_scan: bool = False,   # moondream2
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
    relative_dir: str = "",
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
            scan = _run_moondream(image_path, output_dir)
            structure_note = scan["structure_note"]

            verdict = scan.get("verdict", "pass" if scan["structure_ok"] else "uncertain")
            if verdict == "fail":
                if _is_strong_structure_defect(structure_note):
                    issues.append(f"major structure defect: {structure_note}")
                else:
                    issues.append(f"uncertain structure: {structure_note}")
            elif verdict == "uncertain":
                # Escalate only clearly structural uncertainty to fail.
                if _is_strong_structure_defect(structure_note):
                    issues.append(f"major structure defect: {structure_note}")
                else:
                    issues.append(f"uncertain structure: {structure_note}")
            elif _is_suspect_structure(structure_note) and person_count <= 1:
                issues.append(f"uncertain structure: {structure_note}")
        except Exception as exc:
            issues.append(f"deep scan skipped ({exc})")

    # ── Final verdict ─────────────────────────────────────────────────────────
    if any("major structure defect" in issue for issue in issues):
        if person_count > 2:
            status = "warning"
            score = 50.0
        else:
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
    destination = _route_image(
        image_path, output_dir, status, move_files, relative_dir=relative_dir
    )

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
        "note": {"type": "string"},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "score", "note", "issues"],
    "additionalProperties": False,
}

_LABEL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"},
    },
    "required": ["label", "confidence", "reason"],
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


def _normalize_choice_label(label: str, labels: List[str]) -> str:
    candidate = re.sub(r"[\s_\-]+", "", str(label)).lower()
    if not candidate:
        raise ValueError("Backend returned an empty organizer label")
    for option in labels:
        normalized = re.sub(r"[\s_\-]+", "", option).lower()
        if candidate == normalized:
            return option
    raise ValueError(f"Backend invented a label outside the allowed choices: {label}")


def validate_organizer_labels(labels: List[str]) -> List[str]:
    """Return safe, unique folder labels or raise a user-facing error."""
    clean_labels = [str(label).strip() for label in labels if str(label).strip()]
    if not clean_labels:
        raise ValueError("At least one label is required")

    reserved = {"report.json", "model_responses.log"}
    invalid_chars = set('<>:"/\\|?*')
    seen = set()
    for label in clean_labels:
        folded = label.casefold()
        if folded in seen:
            raise ValueError(f"Duplicate organizer label: {label}")
        if label in {".", ".."} or folded in reserved:
            raise ValueError(f"Organizer label cannot be used as a folder: {label}")
        if len(label) > 100:
            raise ValueError(f"Organizer label is too long (100 characters max): {label}")
        if any(char in invalid_chars or ord(char) < 32 for char in label):
            raise ValueError(
                f"Organizer label contains a character that is unsafe in a folder name: {label}"
            )
        seen.add(folded)
    return clean_labels


_REFUSAL_MARKERS = (
    "violation",
    "cannot assist",
    "can't assist",
    "unable to provide",
    "unable to assist",
    "i cannot",
    "i can't",
    "against policy",
    "content policy",
)


def _looks_like_model_refusal(text: str) -> bool:
    """True when the backend declined to return a verdict.

    Matches phrasing the model itself emits when it will not answer, so the
    image can be routed to ``unscored/`` rather than scored as a failure.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def classify_image_with_backend(
    image_path: str,
    backend_url: str,
    output_dir: str,
    timeout: float = 120.0,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
    api_key: Optional[str] = None,
    relative_dir: str = "",
) -> Dict[str, Any]:
    """Classify one image via an OpenAI-compatible vision backend."""
    img_url = _image_data_url(image_path)
    prompt = (
        "Review this image for human body structure issues.\n"
        "Use FAIL for severe problems like heads in the wrong place, "
        "duplicate heads, multiple torsos that look fused, people merged into "
        "one body, or limbs attached in impossible ways.\n"
        "Use WARNING for minor problems like small proportion issues, awkward "
        "poses, mild weirdness, or other hiccups that are not clearly severe.\n"
        "Use PASS when the body structure looks normal and no real issue is "
        "detected, and do not be overly picky.\n"
        "Return only JSON: {\"status\":\"pass|warning|fail\","
        "\"score\":0-100,\"note\":\"one concise natural-language summary\","
        "\"issues\":[\"brief evidence\"]}."
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
                "You are an image review judge. Be practical and fail only for "
                "clearly severe structure issues."
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

    try:
        parsed = _parse_llm_response(text)
        status = parsed.get("status", "warning")
        score = float(parsed.get("score") or
                      (100 if status == "pass" else 65 if status == "warning" else 30))
        note = str(parsed.get("note") or "").strip()
        issues = parsed.get("issues") or []
    except Exception as exc:
        # Keep the batch moving even if the backend emits malformed JSON.
        status = "warning"
        score = 50.0
        note = ""
        issues = [f"backend response parse failed: {exc}"]

    if not note:
        note = " ".join(str(issue) for issue in issues).strip()

    issue_text = " ".join(str(issue) for issue in issues)
    route_folder = status
    if _looks_like_model_refusal(f"{text} {issue_text}"):
        route_folder = "unscored"

    _append_human_readable_response(
        output_dir,
        image_path,
        f"backend model={model_name} url={backend_url}",
        text,
        final_status=status,
        final_score=round(score, 1),
        final_issues=[str(issue) for issue in issues],
    )

    os.makedirs(output_dir, exist_ok=True)
    for folder in ("pass", "warning", "fail", "unscored"):
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    destination = _route_image_to_folder(
        image_path,
        output_dir,
        route_folder,
        move_files,
        sibling_folders=["pass", "warning", "fail", "unscored"],
        relative_dir=relative_dir,
    )

    return {
        "filename":     os.path.basename(image_path),
        "status":       status,
        "score":        round(score, 1),
        "issues":       issues,
        "structure_note": note or text[:200],
        "route_folder": route_folder,
        "destination":  destination,
    }


def classify_image_with_labels(
    image_path: str,
    backend_url: str,
    labels: List[str],
    output_dir: str,
    timeout: float = 120.0,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
    api_key: Optional[str] = None,
    relative_dir: str = "",
) -> Dict[str, Any]:
    """Classify one image into one of the provided labels via a vision backend."""
    clean_labels = validate_organizer_labels(labels)

    img_url = _image_data_url(image_path)
    label_list = "\n".join(f"- {label}" for label in clean_labels)
    response_schema = {
        **_LABEL_RESPONSE_SCHEMA,
        "properties": {
            **_LABEL_RESPONSE_SCHEMA["properties"],
            "label": {"type": "string", "enum": clean_labels},
        },
    }
    prompt = (
        "Choose the single best label for this image from the allowed list.\n"
        "Allowed labels:\n"
        f"{label_list}\n\n"
        "Pick exactly one label from the list. Use your best judgment if the "
        "image is ambiguous. Do not invent new labels.\n"
        "Return only JSON: {\"label\":\"one_allowed_label\","
        "\"confidence\":0-100,\"reason\":\"brief reason\"}."
    )
    raw = _http_post(_chat_completions_url(backend_url), {
        "model": model_name,
        "temperature": 0,
        "max_tokens": 250,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "label_choice", "strict": True, "schema": response_schema},
        },
        "messages": [
            {"role": "system", "content": (
                "You are a careful image organizer. Choose only from the provided labels."
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
        msg = choices[0].get("message") or {}
        text = msg.get("content") or choices[0].get("text") or ""
        if isinstance(text, list):
            text = "".join(
                item.get("text", "") for item in text if isinstance(item, dict)
            )
    if not text:
        raise ValueError("Backend returned no content")

    try:
        parsed = _parse_llm_response(text)
        label = _normalize_choice_label(parsed.get("label", ""), clean_labels)
        confidence = float(parsed.get("confidence") or 0.0)
        reason = str(parsed.get("reason") or "").strip() or "No reason provided."
    except Exception as exc:
        reason = f"backend response parse failed: {exc}"
        _append_human_readable_response(
            output_dir,
            image_path,
            f"organizer model={model_name} url={backend_url}",
            text,
            final_status="error",
            final_score=0.0,
            final_issues=[reason],
            final_field="label",
        )
        raise ValueError(reason) from exc

    _append_human_readable_response(
        output_dir,
        image_path,
        f"organizer model={model_name} url={backend_url}",
        text,
        final_status=label,
        final_score=round(confidence, 1),
        final_issues=[reason],
        final_field="label",
    )

    destination = _route_image_to_folder(
        image_path,
        output_dir,
        label,
        move_files,
        sibling_folders=clean_labels,
        relative_dir=relative_dir,
    )

    return {
        "filename": os.path.basename(image_path),
        "label": label,
        "status": label,
        "confidence": round(confidence, 1),
        "reason": reason,
        "organize_note": text[:200],
        "destination": destination,
        "labels": clean_labels,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def collect_images(input_path: str, *, recursive: bool = False) -> List[str]:
    try:
        return _collect_images(input_path, recursive=recursive)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Not found: {input_path}") from exc


def _route_image_to_folder(
    image_path: str,
    output_dir: str,
    folder: str,
    move_files: bool,
    sibling_folders: Optional[List[str]] = None,
    relative_dir: str = "",
) -> str:
    # A rerun may produce a different verdict. Remove stale copies so one image
    # can never appear in both pass and fail (or warning) at the same time.
    filename = os.path.basename(image_path)
    relative_dir = relative_dir.strip().strip(os.sep)
    destination_dir = os.path.join(output_dir, folder, relative_dir)
    source = os.path.abspath(image_path)
    folders = sibling_folders or []
    for sibling in folders:
        old = os.path.abspath(os.path.join(output_dir, sibling, relative_dir, filename))
        if old != source and os.path.isfile(old):
            os.unlink(old)
    dest = os.path.join(destination_dir, filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    (shutil.move if move_files else shutil.copy2)(image_path, dest)
    return dest


def _route_image(image_path: str, output_dir: str,
                 status: str, move_files: bool, *, relative_dir: str = "") -> str:
    return _route_image_to_folder(
        image_path,
        output_dir,
        status,
        move_files,
        sibling_folders=["pass", "warning", "fail"],
        relative_dir=relative_dir,
    )


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
    recursive: bool = False,
) -> List[Dict[str, Any]]:
    """Run QC on all images in ``input_path``."""
    if settings is None:
        settings = _DEFAULT_SETTINGS
    _remove_stale_audit_artifacts(output_dir)
    _reset_human_readable_log(output_dir)
    started_at = time.perf_counter()
    run_mode = (
        f"backend model={model_name} url={backend_url}"
        if backend_url and not settings.strict_offline
        else (
            f"local deep_scan={settings.use_deep_scan} "
            f"strict_offline={settings.strict_offline} yolo={settings.use_yolo}"
        )
    )
    _append_run_event(output_dir, f"Run started: qc input={input_path} mode={run_mode}")
    images = collect_images(input_path, recursive=recursive)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    results = []
    for image_path in images:
        relative_dir = relative_output_subdir(input_path, image_path)
        if backend_url and not settings.strict_offline:
            r = classify_image_with_backend(
                image_path, backend_url, output_dir,
                model_name=model_name, move_files=move_files,
                api_key=api_key, relative_dir=relative_dir)
        else:
            r = classify_image(image_path, output_dir,
                               move_files=move_files, settings=settings,
                               relative_dir=relative_dir)
        results.append(r)

    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("pass", "warning", "fail")}
    elapsed = time.perf_counter() - started_at
    _append_run_event(
        output_dir,
        f"Run finished: qc images={len(results)} elapsed={elapsed:.2f}s",
    )
    print(f"QC done — {len(results)} images | "
          f"pass {counts['pass']}  warning {counts['warning']}  "
          f"fail {counts['fail']}")
    return results


def run_organize(
    input_path: str,
    output_dir: str,
    labels: List[str],
    backend_url: str,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
    api_key: Optional[str] = None,
    recursive: bool = False,
) -> List[Dict[str, Any]]:
    """Run image organization without emitting audit artifacts."""
    clean_labels = validate_organizer_labels(labels)

    _remove_stale_audit_artifacts(output_dir)
    _reset_human_readable_log(output_dir)
    started_at = time.perf_counter()
    _append_run_event(
        output_dir,
        f"Run started: organize input={input_path} labels={', '.join(clean_labels)} "
        f"model={model_name} url={backend_url}",
    )
    images = collect_images(input_path, recursive=recursive)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    results = []
    for image_path in images:
        relative_dir = relative_output_subdir(input_path, image_path)
        r = classify_image_with_labels(
            image_path,
            backend_url,
            clean_labels,
            output_dir,
            model_name=model_name,
            move_files=move_files,
            api_key=api_key,
            relative_dir=relative_dir,
        )
        results.append(r)

    elapsed = time.perf_counter() - started_at
    _append_run_event(
        output_dir,
        f"Run finished: organize images={len(results)} elapsed={elapsed:.2f}s",
    )
    print(f"Organize done — {len(results)} images | labels: {', '.join(clean_labels)}")
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
    p.add_argument("--deep-scan", dest="deep_scan", action="store_true", default=False,
                   help="Enable moondream2 structure scan")
    p.add_argument("--no-deep-scan", dest="deep_scan", action="store_false",
                   help="Disable moondream2 and run detection metadata only")
    p.add_argument("--strictness",   default="balanced",
                   choices=["relaxed", "balanced", "strict"])
    p.add_argument("--strict-offline", action="store_true",
                   help="Disable backend, YOLO, and model downloads; run pixel-only QC")
    p.add_argument("--no-yolo",      action="store_true")
    p.add_argument("--move",         action="store_true")
    p.add_argument("--recursive", action="store_true",
                   help="Process images in subfolders and preserve their structure")
    args = p.parse_args()

    settings = QCSettings(
        use_yolo=not args.no_yolo and not args.strict_offline,
        use_deep_scan=args.deep_scan and not args.strict_offline,
        deep_scan_persons_only=True,
        strict_offline=args.strict_offline,
        deep_scan_strictness=args.strictness,
    )
    run_qc(args.input, args.output_dir,
           backend_url=None if args.strict_offline else args.backend_url,
           model_name=args.model,
           move_files=args.move,
           settings=settings,
           api_key=None if args.strict_offline else args.api_key,
           recursive=args.recursive)


if __name__ == "__main__":
    main()
