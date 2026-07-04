"""Local Qwen3-VL structural-structure judge.

This module deliberately has no dependency on the GUI or routing policy.  It
returns a structured observation so the caller can benchmark the model before
deciding where a file belongs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

STRUCTURE_PROMPT = """Inspect this image only for major impossible duplication or fusion of human body structure.

Trace each visible head through its neck, shoulder line, upper torso, and lower body.
Count distinct people, heads, upper torsos, and lower bodies before deciding. Do not
invent structure outside the frame, and do not excuse an impossible structure merely
because the composition appears intentional or artistic.

FAIL only when clearly visible evidence shows at least one of these:
- multiple heads attached to one torso;
- multiple upper torsos attached to one lower body;
- an extra full arm or leg attached to one body;
- two people visibly fused into one body.

Normal separate or overlapping people, hidden/cropped limbs, hands, fingers, pose, hair,
clothing, lighting, and minor image flaws must PASS.

Return JSON only:
{"verdict":"PASS or FAIL","defect":"none, duplicate_head, duplicate_torso, extra_limb, or fused_bodies","evidence":"brief visible evidence","confidence":0.0}
"""


@dataclass(frozen=True)
class StructureDecision:
    verdict: str
    defect: str
    evidence: str
    confidence: float
    raw: str


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


class QwenStructureJudge:
    """Load Qwen3-VL once and evaluate images locally through PyTorch."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model_path = Path(model_id).expanduser()
        if not model_path.exists():
            try:
                model_path = Path(snapshot_download(model_id, local_files_only=True))
            except Exception as exc:
                raise RuntimeError(
                    f"Qwen model {model_id!r} is not fully available in the local "
                    "Hugging Face cache"
                ) from exc

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
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if verdict not in {"pass", "fail"}:
            raise ValueError(f"invalid verdict: {verdict!r}")
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        return StructureDecision(
            verdict=verdict,
            defect=str(parsed.get("defect", "none")).strip().lower(),
            evidence=str(parsed.get("evidence", "")).strip(),
            confidence=max(0.0, min(1.0, confidence)),
            raw=raw,
        )
