"""Depth Anything V2 model configuration and loading utilities."""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

AVAILABLE_MODELS = [
    "depth_anything_v2_vits_fp16.safetensors",
    "depth_anything_v2_vits_fp32.safetensors",
    "depth_anything_v2_vitb_fp16.safetensors",
    "depth_anything_v2_vitb_fp32.safetensors",
    "depth_anything_v2_vitl_fp16.safetensors",
    "depth_anything_v2_vitl_fp32.safetensors",
    "depth_anything_v2_metric_hypersim_vitl_fp32.safetensors",
    "depth_anything_v2_metric_vkitti_vitl_fp32.safetensors",
]


def load_depth_model(model_name: str, device: torch.device, models_dir: str = "models"):
    """Load a supported Depth Anything V2 checkpoint onto ``device``."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unsupported model {model_name!r}. Choose from: {AVAILABLE_MODELS}")

    dtype = torch.float16 if "fp16" in model_name and device.type != "cpu" else torch.float32
    encoder = next(name for name in ("vitl", "vitb", "vits") if name in model_name)
    model_path = Path(models_dir).expanduser() / model_name

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading image depth model: {model_name}")
        hf_hub_download(
            repo_id="yushan777/DepthAnythingV2",
            filename=model_name,
            local_dir=str(model_path.parent),
        )

    print(f"Loading image depth model: {model_path}")
    state_dict = load_safetensors(str(model_path), device="cpu")
    is_metric = "metric" in model_name
    max_depth = 20.0 if "hypersim" in model_name else 80.0
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder], is_metric=is_metric, max_depth=max_depth)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype).eval()
    return model, dtype, is_metric


# Compatibility alias for older imports.
load_model = load_depth_model
