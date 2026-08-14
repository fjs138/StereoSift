"""Experimental local Qwen3-VL structural-anatomy judge.

WIP / reference-only code: this module is intentionally inert in the app unless
someone imports or runs it directly. It has no dependency on the GUI, converter,
or production routing policy. It returns a structured observation so the caller
can benchmark the model before deciding whether it should ever graduate into the
live QC flow.
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

STRUCTURE_PROMPT = """Act as a conservative structural-anatomy quality inspector.

Your only job is to decide whether visible human bodies have major structural
anatomy defects. Ignore scene meaning, artistic intent, clothing, style, lighting,
and background. Do not mention or classify any non-anatomy subject matter.

Trace every visible head downward through its neck, shoulders, upper torso, and
pelvis or lower body. Do not assume that two faces represent two valid people
merely because two faces are visible. They count as separate people only when each
head connects to its own plausible, visibly separate body chain. Judge the visible
geometry, not the likely artistic intent, and do not explain an impossible merge
away as a composite, collage, portrait effect, or unusual pose.

Send the image to failure review when any of these is clear or reasonably likely:
- two heads connect to one torso or one shared body chain;
- two upper torsos converge into one pelvis or lower body;
- a full extra arm or leg attaches to one body;
- bodies visibly fuse, intersect, or share anatomy.

Normal people who merely overlap must pass when their body chains remain visibly
distinct. Ignore hands, fingers, facial beauty, lighting, clothing details, and
minor image flaws. Cropping alone is not a defect, but a cropped body does not
prove that a suspiciously joined head or torso is a separate person. When
separation versus fusion is genuinely ambiguous, choose fail so a human reviews it.

Use these labels exactly:
- status: pass, warning, or fail
- defect_type: none, duplicate_head, duplicate_torso, extra_limb, fused_bodies, or suspect
- confidence: a number from 0.0 to 1.0
- evidence: one short sentence describing the visible head-to-body connections
- review: true or false

Decision rules:
- PASS only when every visible head/torso pair has its own plausible body chain,
  or when there are no people.
- FAIL for duplicate heads, duplicate torsos, extra full limbs, shared pelvises,
  shared lower bodies, or visibly fused/intersecting bodies.
- WARNING only for low image quality or heavy occlusion where no specific major
  defect is visible. Do not use warning just because a failure is uncomfortable.

Examples:
Output for no people:
{"status":"pass","defect_type":"none","confidence":1.0,"evidence":"No human anatomy is visible.","review":false}

Output for two normal nearby people:
{"status":"pass","defect_type":"none","confidence":0.98,"evidence":"Each visible head connects to a separate neck, torso, and lower body.","review":false}

Output for stacked heads or two heads sharing one body chain:
{"status":"fail","defect_type":"duplicate_head","confidence":0.9,"evidence":"Two visible heads do not connect to separate complete body chains.","review":true}

Output for two torsos merging into one lower body:
{"status":"fail","defect_type":"duplicate_torso","confidence":0.9,"evidence":"Two upper torsos converge into one shared lower body.","review":true}

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
                    "vision-language model. The anatomy judge needs a Qwen-VL checkpoint "
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
                            "name": "anatomy_verdict",
                            "strict": True,
                            "schema": _STRUCTURE_RESPONSE_SCHEMA,
                        },
                    },
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict image QC judge. Base the verdict only on "
                                "visible human anatomy in the supplied image."
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
