#!/usr/bin/env python3
"""Simple image quality-control pipeline for batch image review."""

import argparse
import json
import os
import re
import shutil
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import quote

try:
    import requests
except ImportError:
    requests = None

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageChops
except ImportError:  # pragma: no cover - exercised in lightweight test environments
    Image = None
    ImageDraw = None
    ImageFilter = None
    ImageChops = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(input_path: str) -> List[str]:
    input_path = os.path.expanduser(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend([p for p in os.listdir(input_path) if p.lower().endswith(ext)])
        return sorted({os.path.join(input_path, name) for name in images})
    raise FileNotFoundError(f"Input not found: {input_path}")


def _image_stats(image) -> Dict[str, float]:
    pixels = list(image.getdata())
    if not pixels:
        return {"brightness": 0.0, "contrast": 0.0}

    gray_values = [value / 255.0 for value in pixels]
    mean = sum(gray_values) / len(gray_values)
    variance = sum((value - mean) ** 2 for value in gray_values) / len(gray_values)
    std = variance ** 0.5
    return {"brightness": mean, "contrast": std}


def _edge_component_count(gray_image) -> int:
    if ImageFilter is None:
        return 0

    edge_image = gray_image.filter(ImageFilter.FIND_EDGES)
    width, height = edge_image.size
    pixels = list(edge_image.getdata())
    visited = [False] * len(pixels)
    threshold = 40
    component_count = 0

    for index, value in enumerate(pixels):
        if visited[index] or value < threshold:
            continue

        queue = [index]
        visited[index] = True
        component_size = 0
        while queue:
            current = queue.pop()
            component_size += 1
            x = current % width
            y = current // width
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_index = ny * width + nx
                    if not visited[neighbor_index] and pixels[neighbor_index] >= threshold:
                        visited[neighbor_index] = True
                        queue.append(neighbor_index)

        if component_size >= 6:
            component_count += 1

    return component_count


def _json_from_text(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty backend response text")

    trimmed = text.strip()
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        start = trimmed.find("{")
        end = trimmed.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            candidate = trimmed[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        return _interpret_backend_text(trimmed)


def _extract_status_from_text(text: str) -> Optional[str]:
    lower = text.lower()
    if re.search(r"\b(not\s+)?(fail|failed|unacceptable|poor|bad|reject|unusable|terrible)\b", lower):
        if re.search(r"\bnot\s+(fail|failed|unacceptable|poor|bad|reject|unusable|terrible)\b", lower):
            pass
        else:
            return "fail"
    if re.search(r"\bwarning\b|\bminor\b|\bsome issues\b|\bneeds improvement\b|\bshould fix\b|\bconsider fixing\b", lower):
        return "warning"
    if re.search(r"\bpass\b|\bpassed\b|\bacceptable\b|\bgood\b|\bexcellent\b|\bfine\b|\bok\b|\bokay\b|\bclear\b", lower):
        if re.search(r"\bnot\s+(pass|passed|acceptable|good|excellent|fine|ok|okay|clear)\b", lower):
            return None
        return "pass"
    return None


def _extract_score_from_text(text: str) -> Optional[int]:
    lower = text.lower()
    score_match = re.search(r"\b([0-9]{1,3})\s*(?:/\s*100|percent|%)\b", lower)
    if score_match:
        score = int(score_match.group(1))
        return max(0, min(100, score))
    return None


def _extract_issues_from_text(text: str) -> List[str]:
    lower = text.lower()
    issues = []
    keyword_map = [
        ("very dark", "very dark"),
        ("dark exposure", "dark exposure"),
        ("bright exposure", "bright exposure"),
        ("very bright", "very bright"),
        ("low contrast", "low contrast"),
        ("high contrast", "high contrast"),
        ("blurry", "blurry"),
        ("blur", "blurry"),
        ("noisy", "noise"),
        ("noise", "noise"),
        ("artifact", "artifact"),
        ("cropped", "cropped composition"),
        ("missing structure", "missing structure"),
        ("extra limbs", "extra limbs"),
        ("distorted structure", "distorted structure"),
        ("unnatural pose", "unnatural pose"),
        ("bad crop", "bad crop"),
        ("missing face details", "missing face details"),
        ("missing object parts", "missing object parts"),
        ("overexposed", "overexposed"),
        ("underexposed", "underexposed"),
        ("noise", "noise"),
        ("artifact", "artifact"),
        ("glare", "glare"),
        ("shadow", "shadow"),
    ]
    for phrase, label in keyword_map:
        if phrase in lower and label not in issues:
            issues.append(label)
    if not issues and re.search(r"\b(issue|problem|defect|flaw)\b", lower):
        issues.append("quality issue")
    return issues


def _interpret_backend_text(text: str) -> Dict[str, Any]:
    status = _extract_status_from_text(text)
    issues = _extract_issues_from_text(text)
    score = _extract_score_from_text(text)
    if score is None:
        score = 100 if status == "pass" else 65 if status == "warning" else 30
    if status is None:
        status = "warning" if issues else "pass"
    return {
        "status": status,
        "score": score,
        "issues": issues,
        "brightness": None,
        "contrast": None,
        "edge_components": None,
    }


def _serve_image_file_url(image_path: str, max_side: int = 256, quality: int = 25) -> tuple[ThreadingHTTPServer, str, str]:
    if Image is None:
        raise RuntimeError("Pillow is required to encode images for backend QC")

    temp_dir = tempfile.mkdtemp(prefix="qc_img_")
    output_name = quote(os.path.basename(image_path))
    output_path = os.path.join(temp_dir, output_name)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.LANCZOS)
        image.save(output_path, format="JPEG", quality=quality, optimize=True)

    handler = partial(SimpleHTTPRequestHandler, directory=temp_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

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
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    import urllib.request
    import urllib.error

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Backend request failed: {exc.code} {exc.reason}: {error_body}") from exc


def _parse_backend_response(response_json: Dict[str, Any], raw_text: str) -> Dict[str, object]:
    status = response_json.get("status")
    score = response_json.get("score")
    issues = response_json.get("issues")
    brightness = response_json.get("brightness")
    contrast = response_json.get("contrast")
    edge_components = response_json.get("edge_components")

    if isinstance(issues, str):
        issues = [issues]
    if issues is None:
        issues = []

    def _normalize_number(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, list) and value:
            return _normalize_number(value[0])
        if isinstance(value, str):
            try:
                return int(value) if value.isdigit() else float(value)
            except ValueError:
                return None
        return None

    text_status = _extract_status_from_text(raw_text)
    text_score = _extract_score_from_text(raw_text)
    text_issues = _extract_issues_from_text(raw_text)

    if status not in {"pass", "warning", "fail"}:
        status = text_status or ("warning" if issues else "pass")
    elif text_status and text_status != status:
        if status == "pass" and text_status in {"warning", "fail"}:
            status = text_status
        elif status == "warning" and text_status == "fail":
            status = "fail"
        elif status == "fail" and text_status == "pass" and not issues:
            status = "pass"

    if not issues and text_issues:
        issues = text_issues

    if score is None:
        score = text_score
    if score is None:
        score = 100 if status == "pass" else 65 if status == "warning" else 30

    return {
        "status": status,
        "score": float(_normalize_number(score) or 0.0),
        "issues": issues,
        "brightness": float(_normalize_number(brightness)) if _normalize_number(brightness) is not None else None,
        "contrast": float(_normalize_number(contrast)) if _normalize_number(contrast) is not None else None,
        "edge_components": int(_normalize_number(edge_components)) if _normalize_number(edge_components) is not None else None,
    }


def _route_image(image_path: str, output_dir: str, status: str, move_files: bool) -> str:
    """Copy (default) or move an image into its classification directory."""
    destination_dir = os.path.join(output_dir, status)
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, os.path.basename(image_path))
    if move_files:
        shutil.move(image_path, destination)
    else:
        shutil.copy2(image_path, destination)
    return destination


def classify_image_with_backend(
    image_path: str,
    backend_url: str,
    output_dir: str,
    timeout: float = 120.0,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
) -> Dict[str, object]:
    if not backend_url:
        raise ValueError("backend_url must be provided for backend QC")

    server, image_url, temp_dir = _serve_image_file_url(image_path)
    try:
        prompt = (
            "You are an image quality control assistant for 2D image review. "
            "Look at the image carefully and provide a brief natural-language assessment, followed by a single JSON object at the end. "
            "The JSON object should include at least: status, score, issues. Optional keys are brightness, contrast, edge_components. "
            "Status should be pass, warning, or fail. Score should be 0-100. Issues should be a list of strings describing visible quality concerns. "
            "If the image looks good, say it is acceptable and provide status pass. "
            "If there are only minor problems, say warning. "
            "Explicitly inspect for duplicated or missing heads, faces, torsos, arms, legs, hands, fingers, and fused body parts. "
            "Use fail for unmistakable structural/anatomical defects, warning when uncertain, and pass only when no meaningful defect is visible. "
            "If the JSON is not perfectly formatted, we will still interpret your assessment from the text."
        )
        payload = {
            "model": model_name,
            "image_url": image_url,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an image quality control assistant. The image will be read from the image_url field."
                },
                {"role": "user", "content": prompt},
            ],
        }

        payload_json = _http_post_json(backend_url, payload, timeout=timeout)
    finally:
        _shutdown_image_server(server, temp_dir)

    if not isinstance(payload_json, dict):
        raise ValueError("Unexpected backend response format")

    choices = payload_json.get("choices")
    if not choices or not isinstance(choices, list):
        raise ValueError("Backend did not return any choices")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    text = None
    if message and isinstance(message, dict):
        text = message.get("content")
    elif choices[0].get("text") is not None:
        text = choices[0].get("text")

    if not text:
        raise ValueError("Backend response missing assistant content")

    backend_response = _json_from_text(text)
    backend_result = _parse_backend_response(backend_response, text)

    status = backend_result["status"]
    os.makedirs(output_dir, exist_ok=True)
    for folder_name in ("pass", "warning", "fail"):
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
    destination = _route_image(image_path, output_dir, status, move_files)

    return {
        "filename": os.path.basename(image_path),
        "status": status,
        "score": round(backend_result["score"], 1),
        "issues": backend_result["issues"],
        "brightness": backend_result["brightness"],
        "contrast": backend_result["contrast"],
        "edge_components": backend_result["edge_components"],
        "destination": destination,
    }


def classify_image(image_path: str, output_dir: str, move_files: bool = False) -> Dict[str, object]:
    if Image is None:
        raise RuntimeError("Pillow is required for QC processing")

    image = Image.open(image_path).convert("RGB")
    gray = image.convert("L")
    stats = _image_stats(gray)
    brightness = stats["brightness"]
    contrast = stats["contrast"]
    edge_components = _edge_component_count(gray)

    issues: List[str] = []
    score = 100.0

    if edge_components >= 20:
        issues.append("high visual complexity; vision review recommended")
        score -= 20
    elif edge_components >= 10:
        issues.append("moderate visual complexity; vision review recommended")
        score -= 10

    if brightness < 0.08:
        issues.append("very dark")
        score -= 20
    elif brightness < 0.2:
        issues.append("dark")
        score -= 8

    if brightness > 0.92:
        issues.append("very bright")
        score -= 20
    elif brightness > 0.85:
        issues.append("bright")
        score -= 8

    if contrast < 0.08:
        issues.append("low contrast")
        score -= 10
    elif contrast > 0.45:
        issues.append("high contrast / possibly harsh artifact")
        score -= 8

    if not issues:
        status = "pass"
    else:
        status = "warning"

    score = max(0.0, min(100.0, score))

    os.makedirs(output_dir, exist_ok=True)
    for folder_name in ("pass", "warning", "fail"):
        os.makedirs(os.path.join(output_dir, folder_name), exist_ok=True)
    destination = _route_image(image_path, output_dir, status, move_files)

    return {
        "filename": os.path.basename(image_path),
        "status": status,
        "score": round(score, 1),
        "issues": issues,
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "edge_components": edge_components,
        "destination": destination,
    }


def run_qc(
    input_path: str,
    output_dir: str,
    backend_url: Optional[str] = None,
    model_name: str = "llama-3.2-11b-vision-instruct",
    move_files: bool = False,
) -> List[Dict[str, object]]:
    images = collect_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    results = []
    for image_path in images:
        if backend_url:
            results.append(classify_image_with_backend(
                image_path, backend_url, output_dir,
                model_name=model_name, move_files=move_files,
            ))
        else:
            results.append(classify_image(image_path, output_dir, move_files=move_files))

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Processed {len(results)} image(s). Report: {report_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple image QC pipeline")
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    parser.add_argument("--output-dir", default="output/qc", help="Where to save QC results")
    parser.add_argument(
        "--backend-url",
        default=None,
        help=(
            "Optional OpenAI-compatible local QC backend endpoint. "
            "Example: http://127.0.0.1:1234/v1/chat/completions"
        ),
    )
    parser.add_argument("--model", default="llama-3.2-11b-vision-instruct", help="Vision model served by the backend")
    parser.add_argument("--move", action="store_true", help="Move originals instead of safely copying them")
    args = parser.parse_args()

    run_qc(
        args.input,
        args.output_dir,
        backend_url=args.backend_url,
        model_name=args.model,
        move_files=args.move,
    )


if __name__ == "__main__":
    main()
