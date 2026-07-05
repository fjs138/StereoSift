"""Local Qwen3-VL structural-structure judge.

This module deliberately has no dependency on the GUI or routing policy.  It
returns a structured observation so the caller can benchmark the model before
deciding where a file belongs.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8001/v1"
DEFAULT_BACKEND_MODEL = "Qwen3.6-35B-A3B-MLX-4bit"

STRUCTURE_PROMPT = """You are a strict structure QC gate for stereo image review.

Decide whether the image contains impossible or suspicious human body structure that should
be rejected or reviewed. Be conservative: normal multi-person photos are allowed, but
any sign of shared structure, fused bodies, duplicated heads, duplicated torsos, or
extra limbs should be flagged.

Ignore all non-structure content. Do not comment on nudity, sex, violence, drugs,
weapons, gore, clothing, background objects, style, or scene semantics unless they
directly affect whether the visible human body structure is duplicated, fused, missing, or
otherwise structurally wrong.

Check the image in this order:
1. Count visible people.
2. For each person, trace head -> neck -> shoulders -> torso -> pelvis/lower body.
3. Decide whether every visible person is anatomically separate and complete.
4. If there is any shared torso, merged silhouette, duplicated body part, or uncertain
   boundary, do not call it pass.

Use these labels exactly:
- status: pass, warning, or fail
- defect_type: none, duplicate_head, duplicate_torso, extra_limb, fused_bodies, or suspect
- confidence: a number from 0.0 to 1.0
- evidence: one short sentence
- review: true or false

Decision rules:
- FAIL when you can clearly see impossible structure: twins sharing one body,
  conjoined twins, one body supporting two heads, two torsos fused together,
  duplicated torsos, extra limbs, or visibly merged bodies.
- WARNING when the image is crowded, blurry, cropped, stylized, or ambiguous enough that
  you cannot prove a hard defect, but the structure still looks suspicious.
- PASS only when every visible person looks anatomically normal and separate from head
  through lower body.

Important:
- Do not pass an image just because the scene contains multiple people.
- Do not describe ordinary nearby people as a defect if their bodies are clearly separate.
- If you are uncertain between pass and fail, choose warning.
- If you find duplicated heads, duplicated torsos, or conjoined / twin-like shared bodies,
  treat that as a failure even if the rest of the scene looks plausible.
- Never return content moderation language or safety commentary; only return structure QC.

Examples:
Example 1:
Image: one person standing normally.
Output: {"status":"pass","defect_type":"none","confidence":0.98,"evidence":"One person has a complete, separate structure.","review":false}

Example 2:
Image: two heads share one torso, or conjoined twins / twin-like bodies appear fused together.
Output: {"status":"fail","defect_type":"fused_bodies","confidence":0.96,"evidence":"Two heads are attached to a shared or fused body.","review":true}

Example 3:
Image: a crowded or partially occluded group where separation is unclear.
Output: {"status":"warning","defect_type":"suspect","confidence":0.55,"evidence":"The structure is hard to verify because the bodies overlap or are partly hidden.","review":true}

Return one JSON object only, with exactly these keys and no extra text:
{"status":"pass|warning|fail","defect_type":"none|duplicate_head|duplicate_torso|extra_limb|fused_bodies|suspect","confidence":0.0,"evidence":"brief visible evidence","review":true}
"""


@dataclass(frozen=True)
class StructureDecision:
    verdict: str
    defect: str
    evidence: str
    confidence: float
    review: bool
    raw: str


_STRUCTURE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pass", "warning", "fail"]},
        "defect_type": {
            "type": "string",
            "enum": [
                "none",
                "duplicate_head",
                "duplicate_torso",
                "extra_limb",
                "fused_bodies",
                "suspect",
            ],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
        "review": {"type": "boolean"},
    },
    "required": ["status", "defect_type", "confidence", "evidence", "review"],
    "additionalProperties": False,
}


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"model did not return a JSON object: {text!r}")


def _normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if status in {"pass", "warning", "fail"}:
        return status
    verdict = str(value or "").strip().lower()
    if verdict in {"pass", "warning", "fail"}:
        return verdict
    raise ValueError(f"invalid status: {value!r}")


def _normalize_defect_type(value: Any) -> str:
    defect = str(value or "none").strip().lower()
    valid = {"none", "duplicate_head", "duplicate_torso", "extra_limb", "fused_bodies", "suspect"}
    if defect not in valid:
        return "suspect"
    return defect


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return False


def _image_data_url(image_path: str, max_side: int = 1536) -> str:
    from io import BytesIO

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=92)
    import base64

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _chat_completions_url(backend_url: str) -> str:
    url = backend_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _http_post(
    url: str,
    payload: dict,
    timeout: float = 120.0,
    api_key: str | None = None,
    max_retries: int = 3,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if requests is not None:
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt == max_retries:
                    raise RuntimeError(f"Backend unreachable: {exc}") from exc
                time.sleep(min(2 ** attempt, 8))
                continue
            if response.ok:
                return response.json()
            detail = response.text.strip()[:500]
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
                raise RuntimeError(f"Backend HTTP {response.status_code}: {detail or response.reason}")
            time.sleep(min(2 ** attempt, 8))
        raise RuntimeError("Backend request failed")

    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode()
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == max_retries:
                raise RuntimeError(f"Backend {exc.code}: {exc.read().decode()}") from exc
        except urllib.error.URLError as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Backend unreachable: {exc}") from exc
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError("Backend request failed")


def _parse_backend_response(raw: dict) -> str:
    choices = raw.get("choices") or []
    if not choices:
        raise ValueError("Backend returned no choices")
    msg = choices[0].get("message") or {}
    text = msg.get("content") or choices[0].get("text") or ""
    if isinstance(text, list):
        text = "".join(item.get("text", "") for item in text if isinstance(item, dict))
    if not text:
        raise ValueError("Backend returned no content")
    return str(text).strip()


class QwenStructureJudge:
    """Load Qwen3-VL once and evaluate images locally through PyTorch."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        backend_url: str | None = DEFAULT_BACKEND_URL,
        model_name: str = DEFAULT_BACKEND_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.backend_url = backend_url.strip() if backend_url else None
        self.model_name = model_name.strip() if model_name else DEFAULT_BACKEND_MODEL
        self.api_key = api_key.strip() if api_key else None
        self.model_id = model_id

        if self.backend_url:
            self.processor = None
            self.model = None
            self.device = None
            return

        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = Path(self.model_id).expanduser()
        if not model_path.exists():
            try:
                model_path = Path(snapshot_download(self.model_id, local_files_only=True))
            except Exception as exc:
                raise RuntimeError(
                    f"Qwen model {self.model_id!r} is not fully available in the local "
                    "Hugging Face cache"
                ) from exc

        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
            except Exception:
                model_type = None
            if model_type and "vl" not in str(model_type).lower():
                raise RuntimeError(
                    f"{model_path} looks like a text-only model ({model_type!r}), not a "
                    "vision-language model. The structure judge needs a Qwen-VL checkpoint "
                    "or an OpenAI-compatible backend."
                )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
        ).to(self.device).eval()

    def judge(self, image_path: str) -> StructureDecision:
        if self.backend_url:
            raw = _http_post(
                _chat_completions_url(self.backend_url),
                {
                    "model": self.model_name,
                    "temperature": 0,
                    "max_tokens": 200,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structure_verdict",
                            "strict": True,
                            "schema": _STRUCTURE_RESPONSE_SCHEMA,
                        },
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict image QC judge. Base the verdict only on "
                                "visible human body structure in the supplied image."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": STRUCTURE_PROMPT},
                                {"type": "image_url", "image_url": {"url": _image_data_url(image_path), "detail": "high"}},
                            ],
                        },
                    ],
                },
                api_key=self.api_key,
            )
            raw = _parse_backend_response(raw)
        else:
            image = Image.open(image_path).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": STRUCTURE_PROMPT},
                ],
            }]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[prompt], images=[image], padding=True, return_tensors="pt"
            ).to(self.device)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
            )
            trimmed = generated[:, inputs.input_ids.shape[1]:]
            raw = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        parsed = _extract_json(raw)
        verdict = _normalize_status(parsed.get("status", parsed.get("verdict")))
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        return StructureDecision(
            verdict=verdict,
            defect=_normalize_defect_type(parsed.get("defect_type", parsed.get("defect"))),
            evidence=str(parsed.get("evidence", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            review=_normalize_bool(parsed.get("review", verdict != "pass")),
            raw=raw,
        )
