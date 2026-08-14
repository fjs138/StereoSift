"""Stereoscopic side-by-side (SBS) rendering for images and video.

Generates 3D effects by horizontally shifting pixels based on a depth map.
Brighter areas in the depth map appear closer; darker areas appear further away.

Two rendering methods are available:

- ``mesh_warping``: Curved, flow-based warp using a normalised coordinate grid.
  Produces smooth, natural-looking depth with subtle barrel distortion. Generally
  the better choice.
- ``grid_sampling``: Direct pixel-shift using disparity values. Faster and
  simpler; good for quick previews or when the depth map is already clean.

Depth scale reference
---------------------
==========  =============================================================
10–20       Subtle stereo — good for portraits and flat scenes.
21–50       Balanced — recommended starting range for most content.
50–100      Strong — works well with clean depth maps and landscapes,
            but can look artificial if transitions are noisy.
>100        Usually too much; causes ghosting and stretching.
==========  =============================================================

Temporal smoothing reference (video only)
------------------------------------------
====  ====================================================================
0.0   No smoothing — each frame treated independently.
0.1–0.3  Mild to moderate — balances consistency without noticeable lag.
0.5   Strong — maximises stability; may lag on rapid scene changes.
====  ====================================================================
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

# Set to True to tint left/right views red/cyan for debugging eye assignment.
DEBUG_MODE = os.environ.get("SBS_DEBUG", "").lower() in ("1", "true", "yes")

# dtype used for all SBS tensor operations.  float16 is faster on GPU;
# float32 is safer on CPU and avoids precision loss for high-res images.
_STEREO_DTYPE = torch.float16

# Coordinate-grid caches keyed by (height, width, dtype).
_GRID_CACHE_GS: dict = {}
_GRID_CACHE_MW: dict = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_depth_bchw(depth_map: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalise a depth-map tensor to shape ``[B, 1, H, W]``.

    Accepts any of the common layouts produced by depth models:
    ``[B, H, W, 1]``, ``[B, H, W, 3]``, or ``[B, H, W]``.
    """
    if depth_map.ndim == 4 and depth_map.shape[-1] in (1, 3):
        depth_map = depth_map.permute(0, 3, 1, 2)
        if depth_map.shape[1] > 1:
            depth_map = depth_map[:, :1]
    elif depth_map.ndim == 3:
        depth_map = depth_map.unsqueeze(1)
    return depth_map.to(device)


def _blur_depth(depth_map: torch.Tensor, strength: int) -> torch.Tensor:
    """Smooth depth-map transitions with a separable box blur.

    Applies two horizontal + vertical passes, approximating a Gaussian.
    This reduces hard edges in the warp that would otherwise appear as
    visible "seams" in the final SBS image.

    Args:
        depth_map: Tensor of shape ``[B, C, H, W]``.
        strength: Kernel size (will be rounded up to the next odd number).

    Returns:
        Blurred depth-map tensor with the same shape.
    """
    if strength % 2 == 0:
        strength += 1
    h_pad = strength // 2
    v_pad = strength // 2

    out = depth_map
    for _ in range(2):
        out = F.avg_pool2d(out, kernel_size=(1, strength), stride=1, padding=(0, h_pad))
        out = F.avg_pool2d(out, kernel_size=(strength, 1), stride=1, padding=(v_pad, 0))
    return out


def _get_grid_gs(h: int, w: int, dtype: torch.dtype, device: torch.device):
    """Return a cached ``(y, x)`` pixel-coordinate grid for grid-sampling."""
    key = (h, w, dtype)
    if key not in _GRID_CACHE_GS:
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        _GRID_CACHE_GS[key] = (y.unsqueeze(0).unsqueeze(0), x.unsqueeze(0).unsqueeze(0))
    return _GRID_CACHE_GS[key]


def _get_grid_mw(h: int, w: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a cached normalised ``[-1, 1]`` coordinate grid for mesh warping."""
    key = (h, w, dtype)
    if key not in _GRID_CACHE_MW:
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        _GRID_CACHE_MW[key] = torch.stack((x, y), dim=-1)
    return _GRID_CACHE_MW[key]


def _apply_debug_tint(left: torch.Tensor, right: torch.Tensor):
    """Tint left view red and right view cyan for eye-assignment debugging."""
    left = left * torch.tensor([1.0, 0.5, 0.5], device=left.device, dtype=left.dtype).view(1, 3, 1, 1)
    right = right * torch.tensor([0.5, 1.0, 1.0], device=right.device, dtype=right.dtype).view(1, 3, 1, 1)
    return left, right


# ---------------------------------------------------------------------------
# Per-image SBS renderers
# ---------------------------------------------------------------------------

def _sbs_grid_sampling(
    device: torch.device,
    base_image: torch.Tensor,
    depth_map: torch.Tensor,
    depth_scale: int,
    mode: str,
    depth_blur_strength: int,
) -> tuple[torch.Tensor]:
    """Render SBS via direct pixel-shift (grid sampling).

    Computes per-pixel horizontal disparity from the depth map and uses
    ``F.grid_sample`` to pull source pixels into left- and right-eye views.

    Args:
        device: Torch device to run on.
        base_image: ``[B, H, W, C]`` float tensor in ``[0, 1]``.
        depth_map: ``[B, H, W, 1|3]`` or ``[B, H, W]`` depth tensor.
        depth_scale: Controls the magnitude of the stereo shift.
        mode: ``"parallel"`` or ``"cross-eyed"``.
        depth_blur_strength: Kernel size for depth smoothing.

    Returns:
        Tuple of one tensor: ``[B, H, W*2, C]``.
    """
    base_image = base_image.to(device, dtype=_STEREO_DTYPE)
    depth_map = depth_map.to(device, dtype=_STEREO_DTYPE)

    b, h, w, c = base_image.shape
    image = base_image.permute(0, 3, 1, 2)  # BHWC -> BCHW

    depth_map = _ensure_depth_bchw(depth_map, device)
    if depth_map.shape[2:] != (h, w):
        depth_map = F.interpolate(depth_map, size=(h, w), mode="bilinear", align_corners=False)
    depth_map = _blur_depth(depth_map, depth_blur_strength)

    disparity = depth_map * 255.0 * (depth_scale / w)
    y_grid, x_grid = _get_grid_gs(h, w, _STEREO_DTYPE, device)

    if b > 1:
        x_grid = x_grid.expand(b, 1, h, w)
        y_grid = y_grid.expand(b, 1, h, w)

    # Normalise shifted x-coordinates to [-1, 1] for grid_sample
    x_left  = 2.0 * (x_grid - disparity) / (w - 1) - 1.0
    x_right = 2.0 * (x_grid + disparity) / (w - 1) - 1.0
    y_norm  = 2.0 * y_grid / (h - 1) - 1.0

    grid_left  = torch.stack((x_left.squeeze(1),  y_norm.squeeze(1)), dim=-1).float()
    grid_right = torch.stack((x_right.squeeze(1), y_norm.squeeze(1)), dim=-1).float()

    image_f = image.float()
    left  = F.grid_sample(image_f, grid_left,  mode="bilinear", padding_mode="border", align_corners=True)
    right = F.grid_sample(image_f, grid_right, mode="bilinear", padding_mode="border", align_corners=True)
    left, right = left.to(_STEREO_DTYPE), right.to(_STEREO_DTYPE)

    if DEBUG_MODE:
        left, right = _apply_debug_tint(left, right)

    sbs = torch.cat([left, right] if mode == "parallel" else [right, left], dim=3)
    return (sbs.permute(0, 2, 3, 1),)  # BCHW -> BHWC


def _sbs_mesh_warping(
    device: torch.device,
    base_image: torch.Tensor,
    depth_map: torch.Tensor,
    depth_scale: int,
    mode: str,
    depth_blur_strength: int,
) -> torch.Tensor:
    """Render SBS via normalised mesh warping.

    Uses a ``[-1, 1]`` coordinate grid and displaces sample points by a
    fraction of the image width derived from the depth map. Produces smoother,
    more natural-looking depth than grid sampling.

    Args:
        device: Torch device to run on.
        base_image: ``[B, H, W, C]`` float tensor in ``[0, 1]``.
        depth_map: ``[B, H, W, 1|3]`` or ``[B, H, W]`` depth tensor.
        depth_scale: Controls the stereo separation strength.
        mode: ``"parallel"`` or ``"cross-eyed"``.
        depth_blur_strength: Kernel size for depth smoothing.

    Returns:
        Tensor of shape ``[B, H, W*2, C]``.
    """
    base_image = base_image.to(device, dtype=_STEREO_DTYPE)
    depth_map = depth_map.to(device, dtype=_STEREO_DTYPE)

    B, H, W, C = base_image.shape
    eye_separation = depth_scale / (W * 2)

    depth_map = _ensure_depth_bchw(depth_map, device)
    if depth_map.shape[2:] != (H, W):
        depth_map = F.interpolate(depth_map, size=(H, W), mode="bilinear", align_corners=False)
    depth_map = _blur_depth(depth_map, depth_blur_strength)

    base_grid = _get_grid_mw(H, W, _STEREO_DTYPE, device).unsqueeze(0).expand(B, H, W, 2)
    left_grid  = base_grid.clone()
    right_grid = base_grid.clone()

    for b in range(B):
        left_grid[b,  ..., 0] -= eye_separation * depth_map[b, 0]
        right_grid[b, ..., 0] += eye_separation * depth_map[b, 0]

    image_nchw = base_image.permute(0, 3, 1, 2).float()
    left  = F.grid_sample(image_nchw, left_grid.float(),  mode="bilinear", padding_mode="border", align_corners=True)
    right = F.grid_sample(image_nchw, right_grid.float(), mode="bilinear", padding_mode="border", align_corners=True)
    left  = left.to(_STEREO_DTYPE).permute(0, 2, 3, 1)  # BCHW -> BHWC
    right = right.to(_STEREO_DTYPE).permute(0, 2, 3, 1)

    if DEBUG_MODE:
        # Re-permute to BCHW for tinting then back
        left_c  = left.permute(0, 3, 1, 2)
        right_c = right.permute(0, 3, 1, 2)
        left_c, right_c = _apply_debug_tint(left_c, right_c)
        left  = left_c.permute(0, 2, 3, 1)
        right = right_c.permute(0, 2, 3, 1)

    if mode == "parallel":
        return torch.cat([left, right], dim=2)
    return torch.cat([right, left], dim=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_image_sbs(
    base_image: torch.Tensor,
    depth_map: torch.Tensor,
    method: str = "mesh_warping",
    depth_scale: int = 40,
    mode: str = "parallel",
    depth_blur_strength: int = 7,
):
    """Convert a single image to a side-by-side stereoscopic image.

    Args:
        base_image: ``[B, H, W, C]`` float tensor in ``[0, 1]``.
        depth_map: Depth map tensor — any layout accepted by
            :func:`_ensure_depth_bchw`.
        method: ``"mesh_warping"`` (default) or ``"grid_sampling"``.
        depth_scale: Stereo strength. See module docstring for a guide.
        mode: ``"parallel"`` for VR headsets; ``"cross-eyed"`` for
            unaided cross-eyed viewing.
        depth_blur_strength: Depth-map smoothing kernel size (odd, 3–15).

    Returns:
        For ``mesh_warping``: tensor ``[B, H, W*2, C]``.
        For ``grid_sampling``: tuple of one tensor ``[B, H, W*2, C]``.
    """
    device = base_image.device
    if method == "grid_sampling":
        return _sbs_grid_sampling(device, base_image, depth_map, depth_scale, mode, depth_blur_strength)
    return _sbs_mesh_warping(device, base_image, depth_map, depth_scale, mode, depth_blur_strength)


def process_image_anaglyph(
    base_image: torch.Tensor,
    depth_map: torch.Tensor,
    method: str = "mesh_warping",
    depth_scale: int = 40,
    depth_blur_strength: int = 7,
    convergence: float = 0.5,
) -> torch.Tensor:
    """Convert a single image to a red-cyan anaglyph.

    Produces a single composite image where the left-eye view is encoded in
    the red channel and the right-eye view is encoded in the cyan (green +
    blue) channels.  Requires red-cyan 3D glasses to view.

    The ``convergence`` parameter shifts the zero-disparity plane — the depth
    at which objects appear to sit exactly on the screen:

    =========  ============================================================
    0.0        Far objects (depth = 0) appear at screen depth.
    0.5        Mid-depth objects appear at screen depth (balanced default).
    1.0        Near objects (depth = 1) appear at screen depth.
    =========  ============================================================

    Higher convergence values push the 3D effect "into" the screen, lower
    values pull it "out".  Portraits typically benefit from 0.6–0.8;
    landscapes from 0.3–0.5.

    Args:
        base_image: ``[B, H, W, C]`` float tensor in ``[0, 1]``.
        depth_map: Depth map tensor.
        method: ``"mesh_warping"`` or ``"grid_sampling"``.
        depth_scale: Stereo strength.
        depth_blur_strength: Depth-map smoothing kernel size (odd, 3–15).
        convergence: Zero-disparity plane (0.0–1.0+, default 0.5).

    Returns:
        Anaglyph tensor ``[B, H, W, 3]`` with values in ``[0, 1]``.
    """
    device = base_image.device

    # Render separate left and right eye views using the standard SBS path.
    renderer = _sbs_grid_sampling if method == "grid_sampling" else _sbs_mesh_warping
    sbs = renderer(device, base_image, depth_map, depth_scale, "parallel", depth_blur_strength)
    if isinstance(sbs, tuple):
        sbs = sbs[0]
    # sbs: [B, H, W*2, C] — left half is left eye, right half is right eye.
    W2 = sbs.shape[2] // 2
    left  = sbs[:, :, :W2,  :].permute(0, 3, 1, 2)   # [B, C, H, W]
    right = sbs[:, :,  W2:, :].permute(0, 3, 1, 2)

    # Apply convergence offset: shift both views horizontally so that the
    # chosen depth plane aligns at centre.  Positive shift = pull nearer.
    if convergence != 0.5:
        shift_px = int((convergence - 0.5) * depth_scale)
        if shift_px != 0:
            left  = torch.roll(left,  -shift_px, dims=3)
            right = torch.roll(right,  shift_px, dims=3)

    # Convert views to greyscale for luminance-preserving anaglyph.
    # Rec. 709 luma weights.
    weights = torch.tensor([0.2126, 0.7152, 0.0722],
                            device=device, dtype=left.dtype)
    left_gray  = (left  * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    right_gray = (right * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    # Red channel from left eye; green + blue from right eye.
    anaglyph = torch.cat([left_gray, right_gray, right_gray], dim=1)
    return anaglyph.permute(0, 2, 3, 1).clamp(0, 1)   # [B, H, W, 3]


def process_video_sbs(
    frames: torch.Tensor,
    depth_maps: torch.Tensor,
    method: str = "mesh_warping",
    depth_scale: int = 30,
    mode: str = "parallel",
    depth_blur_strength: int = 7,
    temporal_smoothing: float = 0.2,
    batch_size: int = 16,
) -> tuple[torch.Tensor]:
    """Convert a sequence of frames to side-by-side stereoscopic video.

    Processes frames in batches to balance GPU and CPU memory usage.
    Intermediate results are written to a memory-mapped temporary file so
    large videos do not need to fit entirely in RAM.

    Args:
        frames: ``[N, H, W, C]`` float tensor in ``[0, 1]``.
        depth_maps: ``[N, H, W, 1]`` depth tensor aligned with ``frames``.
        method: ``"mesh_warping"`` or ``"grid_sampling"``.
        depth_scale: Stereo strength. See module docstring for a guide.
        mode: ``"parallel"`` or ``"cross-eyed"``.
        depth_blur_strength: Depth-map smoothing kernel size (odd, 3–15).
        temporal_smoothing: Blend factor with the previous frame's disparity
            (0 = off, 0.5 = strong). Reduces flicker on noisy depth sequences.
        batch_size: Number of frames processed per GPU batch.

    Returns:
        Tuple of one tensor: ``[N, H, W*2, C]``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_frames, height, width, channels = frames.shape
    print(f"Processing {num_frames} frames | method={method} batch={batch_size}")

    numpy_dtype = np.float16 if _STEREO_DTYPE == torch.float16 else np.float32
    final_shape = (num_frames, height, width * 2, channels)

    # Use a memory-mapped file so the full output never needs to live in RAM.
    tmp_path = os.path.join(tempfile.gettempdir(), f"stereosift_sbs_{os.getpid()}_{id(object())}.npy")
    print(f"Writing output to temp memmap: {tmp_path}")
    memmap = np.memmap(tmp_path, dtype=numpy_dtype, mode="w+", shape=final_shape)

    processor = _sbs_grid_sampling if method == "grid_sampling" else _sbs_mesh_warping

    previous_disparity: torch.Tensor | None = None

    try:
        for batch_start in range(0, num_frames, batch_size):
            batch_end = min(batch_start + batch_size, num_frames)
            batch_n = batch_end - batch_start
            print(f"  Batch {batch_start // batch_size + 1}/{(num_frames + batch_size - 1) // batch_size}"
                  f"  frames {batch_start + 1}–{batch_end}")

            batch_frames = frames[batch_start:batch_end].to(device, dtype=_STEREO_DTYPE, non_blocking=True)
            batch_depths = depth_maps[batch_start:batch_end].to(device, dtype=_STEREO_DTYPE, non_blocking=True)
            batch_out = torch.zeros((batch_n, height, width * 2, channels), dtype=_STEREO_DTYPE, device=device)

            with torch.no_grad():
                for j in range(batch_n):
                    frame = batch_frames[j : j + 1]
                    depth = batch_depths[j : j + 1]

                    # Normalise depth to [B, 1, H, W] for temporal smoothing.
                    depth_bchw = _ensure_depth_bchw(depth, device)
                    if depth_bchw.shape[2:] != (height, width):
                        depth_bchw = F.interpolate(depth_bchw, size=(height, width), mode="bilinear", align_corners=False)
                    depth_bchw = _blur_depth(depth_bchw, depth_blur_strength)

                    if temporal_smoothing > 0:
                        disparity = depth_bchw * 255.0 * (depth_scale / width)
                        if previous_disparity is not None:
                            disparity = torch.lerp(disparity, previous_disparity.to(disparity.dtype), temporal_smoothing)
                        previous_disparity = disparity.clone()
                        depth_for_frame = disparity / (255.0 * (depth_scale / width))
                    else:
                        depth_for_frame = depth_bchw

                    # Convert back to [B, H, W, 1] for the renderer.
                    depth_bhwc = depth_for_frame.permute(0, 2, 3, 1)

                    result = processor(device, frame, depth_bhwc, depth_scale, mode, depth_blur_strength)
                    # Both renderers return either a tensor or a 1-tuple of a tensor.
                    batch_out[j] = result[0] if isinstance(result, tuple) else result[0:1].squeeze(0) if result.ndim == 4 else result

            memmap[batch_start:batch_end] = batch_out.cpu().numpy().astype(numpy_dtype, copy=False)

            del batch_frames, batch_depths, batch_out
            if device.type == "cuda":
                torch.cuda.empty_cache()

        memmap.flush()
        print("Building final tensor from memmap…")
        final_tensor = torch.from_numpy(np.array(memmap)).to(_STEREO_DTYPE)
    finally:
        # Drop the mapping before unlinking; Windows refuses to remove a mapped file.
        del memmap
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return (final_tensor,)
