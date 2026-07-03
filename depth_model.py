"""Depth Anything V2 model configuration and loading utilities.

Supported checkpoints are hosted on Hugging Face (``yushan777/DepthAnythingV2``)
and downloaded automatically on first use into ``models/``.

Available model names
---------------------
Each name encodes the encoder size and precision:

- ``vits`` / ``vitb`` / ``vitl`` — Small / Base / Large encoder.
- ``fp16`` — half precision (faster on GPU, not supported on CPU).
- ``fp32`` — full precision (required for CPU inference).
- ``metric_hypersim`` / ``metric_vkitti`` — metric-depth variants trained on
  indoor (Hypersim) or outdoor (Virtual KITTI) data respectively.

For most use cases ``depth_anything_v2_vitl_fp16.safetensors`` gives the best
quality/speed trade-off on a CUDA or MPS device.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from depth_anything_v2.dpt import DepthAnythingV2


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

AVAILABLE_MODELS: list[str] = [
    "depth_anything_v2_vits_fp16.safetensors",
    "depth_anything_v2_vits_fp32.safetensors",
    "depth_anything_v2_vitb_fp16.safetensors",
    "depth_anything_v2_vitb_fp32.safetensors",
    "depth_anything_v2_vitl_fp16.safetensors",
    "depth_anything_v2_vitl_fp32.safetensors",
    "depth_anything_v2_metric_hypersim_vitl_fp32.safetensors",
    "depth_anything_v2_metric_vkitti_vitl_fp32.safetensors",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_depth_model(
    model_name: str,
    device: torch.device,
    models_dir: str = "models",
) -> tuple[DepthAnythingV2, torch.dtype, bool]:
    """Load a Depth Anything V2 checkpoint onto ``device``.

    The checkpoint is downloaded from Hugging Face automatically on first use
    and cached in ``models_dir``.

    Args:
        model_name: Filename of the checkpoint (must be in
            :data:`AVAILABLE_MODELS`).
        device: Torch device to load the model onto.
        models_dir: Local directory where checkpoints are stored.

    Returns:
        ``(model, dtype, is_metric)`` — the loaded model, the dtype it runs
        in (float16 on GPU unless the name says fp32, float32 on CPU), and
        whether this is a metric-depth model.

    Raises:
        ValueError: If ``model_name`` is not in :data:`AVAILABLE_MODELS`.
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unsupported model {model_name!r}. Choose from:\n"
            + "\n".join(f"  {m}" for m in AVAILABLE_MODELS)
        )

    dtype   = torch.float16 if "fp16" in model_name and device.type != "cpu" else torch.float32
    encoder = next(name for name in ("vitl", "vitb", "vits") if name in model_name)
    path    = Path(models_dir).expanduser() / model_name

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model_name}…")
        hf_hub_download(
            repo_id="yushan777/DepthAnythingV2",
            filename=model_name,
            local_dir=str(path.parent),
        )

    print(f"Loading {path}")
    state_dict = load_safetensors(str(path), device="cpu")

    is_metric = "metric" in model_name
    max_depth = 20.0 if "hypersim" in model_name else 80.0

    model = DepthAnythingV2(**MODEL_CONFIGS[encoder], is_metric=is_metric, max_depth=max_depth)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype).eval()

    return model, dtype, is_metric


# Backwards-compatible alias.
load_model = load_depth_model
