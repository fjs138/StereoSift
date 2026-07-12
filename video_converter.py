"""Video depth estimation and SBS conversion using Video Depth Anything.

This module handles all video-specific processing:

- Loading the official Video Depth Anything streaming model (auto-downloaded
  from Hugging Face on first use).
- Per-frame streaming inference: depth is estimated and the SBS frame is
  rendered and encoded immediately, so memory usage stays constant regardless
  of video length.
- Audio muxing: the original audio track is copied into the output without
  re-encoding.

The streaming approach is the only video path used by ``convert.py``.  The
model's ``infer_video_depth_one`` method processes one frame at a time, which
keeps peak GPU memory low enough to handle long 1080p videos on consumer
hardware.

Supported encoders
------------------
``vits`` (Small) — fastest, least memory, good quality for most content.
``vitb`` (Base)  — balanced.
``vitl`` (Large) — highest quality, requires more VRAM.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import cv2
import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from sbs.sbs import process_image_sbs

# ---------------------------------------------------------------------------
# Video Depth Anything repository path
# ---------------------------------------------------------------------------

_VDA_REPO = Path(__file__).parent / "video_depth_anything_repo"
_VDA_CHECKPOINTS = _VDA_REPO / "checkpoints"

if str(_VDA_REPO) not in sys.path:
    sys.path.insert(0, str(_VDA_REPO))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_video_depth_model(
    encoder: str = "vitl",
    metric: bool = False,
    device: torch.device | str = "cuda",
) -> tuple[object, torch.dtype, bool]:
    """Load the Video Depth Anything streaming model.

    The checkpoint is downloaded from Hugging Face automatically on first use
    and cached in ``video_depth_anything_repo/checkpoints/``.

    Args:
        encoder: Model variant — ``"vits"``, ``"vitb"``, or ``"vitl"``.
        metric: Use the metric-depth variant (absolute scale).
        device: Torch device to load the model onto.

    Returns:
        ``(model, dtype, is_metric)`` — the loaded model, the dtype it runs
        in (float16 on GPU/MPS, float32 on CPU), and whether it is a metric
        model.
    """
    from video_depth_anything.video_depth_stream import VideoDepthAnything

    checkpoint_name = f"{'metric_' if metric else ''}video_depth_anything_{encoder}.pth"
    checkpoint_path = _VDA_CHECKPOINTS / checkpoint_name

    if not checkpoint_path.exists():
        print(f"Downloading Video Depth Anything checkpoint: {checkpoint_name}")
        _VDA_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        scale_label = {"vits": "Small", "vitb": "Base", "vitl": "Large"}[encoder]
        repo_prefix = "Metric-Video-Depth-Anything" if metric else "Video-Depth-Anything"
        hf_hub_download(
            repo_id=f"depth-anything/{repo_prefix}-{scale_label}",
            filename=checkpoint_name,
            local_dir=str(_VDA_CHECKPOINTS),
            local_dir_use_symlinks=False,
        )

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    device = torch.device(device)
    # fp16 on accelerators keeps memory low during temporal window processing.
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    model = VideoDepthAnything(**model_configs[encoder])
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype).eval()

    print(f"Video Depth Anything ({encoder}) loaded on {device}")
    return model, dtype, metric


# Alias used by convert.py imports.
load_video_depth_anything_model = load_video_depth_model


# ---------------------------------------------------------------------------
# Streaming conversion
# ---------------------------------------------------------------------------

def convert_video_to_sbs(
    video_path: str,
    output_dir: str,
    model,
    device: torch.device | str,
    dtype: torch.dtype,
    is_metric: bool,
    *,
    sbs_method: str = "mesh_warping",
    depth_scale: int = 40,
    sbs_mode: str = "parallel",
    sbs_blur: int = 7,
    max_len: int = -1,
    target_fps: int = -1,
    max_res: int = 1280,
    input_size: int = 518,
    temporal_smoothing: float = 0.2,
    depth_only: bool = False,
    log: Callable[[str], None] = print,
    control: Callable[[], None] | None = None,
    **_ignored,
) -> bool:
    """Convert a single video to a side-by-side 3D video.

    Frames are processed one at a time: depth is estimated, the SBS view is
    rendered, and the frame is written to the output encoder immediately.
    Peak memory is therefore proportional to a single frame, not the full
    video duration.

    The source audio track is muxed into the output without re-encoding.
    If no audio track exists or the mux fails, the video is saved without
    audio and a warning is printed.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory where the output file will be written.
        model: Loaded Video Depth Anything streaming model.
        device: Torch device.
        dtype: Model dtype (float16 or float32).
        is_metric: Whether the model produces metric (absolute) depth.
        sbs_method: ``"mesh_warping"`` or ``"grid_sampling"``.
        depth_scale: Stereo strength (see :mod:`sbs.sbs` for a guide).
        sbs_mode: ``"parallel"`` or ``"cross-eyed"``.
        sbs_blur: Depth-map smoothing kernel size (odd number, 3–15).
        max_len: Stop after this many output frames (-1 = no limit).
        target_fps: Encode at this frame rate (-1 = match source).
        max_res: Downscale so the longest edge is at most this many pixels
            (-1 = no limit).
        input_size: Resolution fed to the depth model (default 518).
        temporal_smoothing: Blend factor with the previous frame's depth
            (0 = off, 0.2 = mild). Reduces flicker on noisy sequences.
        depth_only: Write a greyscale depth visualisation instead of SBS.

    Returns:
        ``True`` if at least one frame was written successfully.
    """
    device = torch.device(device)
    name = Path(video_path).stem
    ext  = Path(video_path).suffix

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Could not open video: {video_path}")
        return False

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = src_fps if target_fps < 0 else target_fps
    if out_fps <= 0:
        raise ValueError("target_fps must be > 0 or -1.")
    stride = max(round(src_fps / out_fps), 1)

    # Compute output resolution — keep even dimensions for the encoder.
    if max_res > 0 and max(src_height, src_width) > max_res:
        scale  = max_res / max(src_height, src_width)
        width  = round(src_width  * scale)
        height = round(src_height * scale)
    else:
        width, height = src_width, src_height
    width  -= width  % 2
    height -= height % 2

    selected_total = (src_count + stride - 1) // stride if src_count > 0 else None
    if max_len > 0 and selected_total is not None:
        selected_total = min(selected_total, max_len)

    os.makedirs(output_dir, exist_ok=True)
    suffix       = "depth" if depth_only else "sbs"
    out_path     = os.path.join(output_dir, f"{name}_{suffix}{ext}")
    tmp_video    = out_path + ".video-only.mp4"

    writer = imageio.get_writer(
        tmp_video,
        fps=out_fps,
        macro_block_size=1,
        codec="libx264",
        ffmpeg_params=["-crf", "18"],
    )

    prev_depth:  np.ndarray | None = None
    src_idx  = 0
    out_idx  = 0

    log(f"Streaming {selected_total or '?'} frames at {width}x{height}, {out_fps:.1f} fps")

    try:
        with tqdm(total=selected_total, desc="Depth + SBS", unit="frame") as bar:
            while cap.isOpened():
                if control:
                    control()
                ok, bgr = cap.read()
                if not ok or (max_len > 0 and out_idx >= max_len):
                    break

                if src_idx % stride != 0:
                    src_idx += 1
                    continue
                src_idx += 1

                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # Depth inference (single frame).
                depth = model.infer_video_depth_one(
                    frame,
                    input_size=input_size,
                    device=device.type,
                    fp32=(dtype == torch.float32),
                ).astype(np.float32)

                # Normalise to [0, 1].
                d_min, d_max = float(depth.min()), float(depth.max())
                depth = (depth - d_min) / max(d_max - d_min, 1e-6)

                # Optional temporal smoothing.
                if prev_depth is not None and temporal_smoothing > 0:
                    depth = (1.0 - temporal_smoothing) * depth + temporal_smoothing * prev_depth
                prev_depth = depth

                if depth_only:
                    encoded = np.repeat((depth[..., None] * 255).astype(np.uint8), 3, axis=2)
                else:
                    frame_t = torch.from_numpy(frame).to(device=device, dtype=dtype).div_(255.0)
                    depth_t = torch.from_numpy(depth).to(device=device, dtype=dtype)

                    result = process_image_sbs(
                        frame_t.unsqueeze(0),
                        depth_t.unsqueeze(0).unsqueeze(-1),
                        method=sbs_method,
                        depth_scale=depth_scale,
                        mode=sbs_mode,
                        depth_blur_strength=sbs_blur,
                    )
                    sbs_tensor = result[0] if isinstance(result, tuple) else result
                    encoded = (
                        sbs_tensor.squeeze(0).float().clamp(0, 1).cpu().numpy() * 255
                    ).astype(np.uint8)

                writer.append_data(encoded)
                out_idx += 1
                bar.update(1)

    finally:
        cap.release()
        writer.close()

    # Mux original audio and embed SBS stereo metadata.
    # The stereo_mode flag tells compliant players (Meta Quest, etc.) that
    # this is a left-right side-by-side 3D video without requiring a
    # special filename convention.
    try:
        import imageio_ffmpeg
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                "-i", tmp_video, "-i", video_path,
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                # MKV/MP4 stereo layout: 1 = side-by-side (left eye left)
                "-metadata:s:v:0", "stereo_mode=left_right",
                # Human-readable container metadata
                "-metadata", "comment=SBS 3D left-right",
                out_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip().splitlines()[-1])
        os.remove(tmp_video)
    except Exception as exc:
        log(f"Warning: audio/metadata mux failed ({exc}); saving video-only.")
        os.replace(tmp_video, out_path)

    log(f"Saved: {out_path}")
    return out_idx > 0
