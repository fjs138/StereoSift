#!/usr/bin/env python3
"""Tiled Real-ESRGAN x2 image upscaling with aspect-safe target sizing."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path
from typing import Callable

import cv2
import imageio
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)
MODEL_NAME = "RealESRGAN_x2plus.pth"


def collect_images(input_path: str) -> list[str]:
    path = Path(input_path).expanduser()
    if path.is_file():
        return [str(path)] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if path.is_dir():
        return [str(p) for p in sorted(path.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return []


def collect_videos(input_path: str) -> list[str]:
    path = Path(input_path).expanduser()
    if path.is_file():
        return [str(path)] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    if path.is_dir():
        return [str(p) for p in sorted(path.iterdir())
                if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return []


def target_dimensions(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Scale up to a requested long edge without changing aspect ratio."""
    if width <= 0 or height <= 0 or long_edge <= 0:
        raise ValueError("Image and target dimensions must be positive")
    scale = long_edge / max(width, height)
    if scale <= 1:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def fit_dimensions(width: int, height: int, max_width: int,
                   max_height: int) -> tuple[int, int]:
    """Resize an image to fit within a box, preserving its aspect ratio."""
    if min(width, height, max_width, max_height) <= 0:
        raise ValueError("Image and target dimensions must be positive")
    scale = min(max_width / width, max_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pixel_unshuffle(x, scale: int):
    import torch
    b, c, h, w = x.size()
    return (x.view(b, c, h // scale, scale, w // scale, scale)
            .permute(0, 1, 3, 5, 2, 4).reshape(b, c * scale * scale,
                                                h // scale, w // scale))


def _build_rrdbnet():
    """Build the exact RRDBNet architecture used by RealESRGAN_x2plus."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, channels=64, growth=32):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
            self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
            self.conv3 = nn.Conv2d(channels + growth * 2, growth, 3, 1, 1)
            self.conv4 = nn.Conv2d(channels + growth * 3, growth, 3, 1, 1)
            self.conv5 = nn.Conv2d(channels + growth * 4, channels, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1 = ResidualDenseBlock()
            self.rdb2 = ResidualDenseBlock()
            self.rdb3 = ResidualDenseBlock()

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            # x2 uses pixel-unshuffle by 2, then the standard x4 RRDB trunk.
            self.conv_first = nn.Conv2d(12, 64, 3, 1, 1)
            self.body = nn.Sequential(*(RRDB() for _ in range(23)))
            self.conv_body = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(_pixel_unshuffle(x, 2))
            body = self.conv_body(self.body(feat)) + feat
            body = self.lrelu(self.conv_up1(F.interpolate(body, scale_factor=2,
                                                           mode="nearest")))
            body = self.lrelu(self.conv_up2(F.interpolate(body, scale_factor=2,
                                                           mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(body)))

    return RRDBNet()


def ensure_model(
    model_dir: str = "models",
    log: Callable[[str], None] = print,
    control: Callable[[], None] | None = None,
) -> str:
    import requests
    os.makedirs(model_dir, exist_ok=True)
    destination = os.path.join(model_dir, MODEL_NAME)
    if os.path.isfile(destination) and os.path.getsize(destination) > 10_000_000:
        return destination
    partial = destination + ".part"
    log(f"Downloading {MODEL_NAME} (about 67 MB)…")
    if control:
        control()
    with requests.get(MODEL_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(partial, "wb") as fh:
            for chunk in response.iter_content(1024 * 1024):
                if control:
                    control()
                if chunk:
                    fh.write(chunk)
    os.replace(partial, destination)
    return destination


class RealESRGANx2:
    def __init__(self, model_path: str, tile: int = 256):
        import torch
        self.device = _device()
        self.tile = tile
        self.model = _build_rrdbnet()
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)

    def _infer_tile(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        # x2plus starts with a 2x pixel-unshuffle, so odd tile edges need one
        # temporary pixel of padding. Crop that padding back off after inference.
        height, width = rgb.shape[:2]
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
        tensor = tensor.unsqueeze(0).to(self.device) / 255.0
        with torch.inference_mode():
            output = self.model(tensor).clamp_(0, 1)
        result = (output[0].float().cpu().numpy().transpose(1, 2, 0) * 255.0)
        return np.rint(result[:height * 2, :width * 2]).astype(np.uint8)

    def upscale(
        self,
        image: Image.Image,
        control: Callable[[], None] | None = None,
    ) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        tile, pad = self.tile, 16
        output = np.empty((h * 2, w * 2, 3), dtype=np.uint8)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                if control:
                    control()
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(w, x + tile + pad), min(h, y + tile + pad)
                patch = self._infer_tile(rgb[y0:y1, x0:x1])
                crop_x, crop_y = (x - x0) * 2, (y - y0) * 2
                out_x1, out_y1 = min(w, x + tile) * 2, min(h, y + tile) * 2
                output[y * 2:out_y1, x * 2:out_x1] = patch[
                    crop_y:crop_y + out_y1 - y * 2,
                    crop_x:crop_x + out_x1 - x * 2]
        return Image.fromarray(output, "RGB")


def upscale_file(path: str, output_dir: str, upscaler: RealESRGANx2,
                 long_edge: int = 3840, output_format: str = "PNG",
                 log: Callable[[str], None] = print,
                 target_box: tuple[int, int] | None = None,
                 control: Callable[[], None] | None = None) -> str:
    if control:
        control()
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        target = (fit_dimensions(*source.size, *target_box) if target_box
                  else target_dimensions(*source.size, long_edge))
        image = source.convert("RGB")
        if target == image.size:
            log(f"Already {image.width}×{image.height}; copying without enlargement")
        elif target[0] < image.width or target[1] < image.height:
            log(f"Source exceeds target; resizing to {target[0]}×{target[1]}")
            image = image.resize(target, Image.Resampling.LANCZOS)
        else:
            passes = max(1, math.ceil(math.log2(max(target[0] / image.width,
                                                     target[1] / image.height))))
            for index in range(passes):
                if control:
                    control()
                log(f"  Real-ESRGAN x2 pass {index + 1}/{passes}")
                image = upscaler.upscale(image, control=control)
            if image.size != target:
                image = image.resize(target, Image.Resampling.LANCZOS)

        os.makedirs(output_dir, exist_ok=True)
        ext = ".png" if output_format == "PNG" else ".jpg"
        destination = os.path.join(output_dir, f"{Path(path).stem}-upscaled{ext}")
        save_args = {"compress_level": 2} if output_format == "PNG" else {
            "quality": 95, "subsampling": 0}
        image.save(destination, output_format, **save_args)
        log(f"Saved {image.width}×{image.height} → {destination}")
        return destination


def upscale_video(
    path: str,
    output_dir: str,
    upscaler: RealESRGANx2,
    long_edge: int = 3840,
    *,
    target_box: tuple[int, int] | None = None,
    max_seconds: float = -1,
    log: Callable[[str], None] = print,
    control: Callable[[], None] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> str:
    """Upscale a video frame-by-frame and preserve the original audio track."""
    if control:
        control()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if src_fps <= 0:
        src_fps = 25.0
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_width <= 0 or src_height <= 0:
        cap.release()
        raise ValueError(f"Video has invalid dimensions: {path}")

    target = (fit_dimensions(src_width, src_height, *target_box) if target_box
              else target_dimensions(src_width, src_height, long_edge))
    out_width = max(2, target[0] - target[0] % 2)
    out_height = max(2, target[1] - target[1] % 2)
    frame_limit = None
    if max_seconds > 0:
        frame_limit = max(1, math.ceil(max_seconds * src_fps))
    if frame_limit is not None and src_count > 0:
        total = min(src_count, frame_limit)
    else:
        total = src_count if src_count > 0 else None

    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, f"{Path(path).stem}-upscaled{Path(path).suffix}")
    tmp_video = destination + ".video-only.mp4"
    log(f"Upscaling video to {out_width}×{out_height} at {src_fps:.1f} fps")

    writer = None
    frames_written = 0
    try:
        writer = imageio.get_writer(
            tmp_video,
            fps=src_fps,
            macro_block_size=1,
            codec="libx264",
            ffmpeg_params=["-crf", "18"],
        )
        while cap.isOpened():
            if control:
                control()
            if frame_limit is not None and frames_written >= frame_limit:
                break
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb, "RGB")
            if target == image.size:
                out_image = image
            elif target[0] < image.width or target[1] < image.height:
                out_image = image.resize(target, Image.Resampling.LANCZOS)
            else:
                out_image = image
                passes = max(1, math.ceil(math.log2(max(
                    target[0] / image.width,
                    target[1] / image.height,
                ))))
                for _ in range(passes):
                    if control:
                        control()
                    out_image = upscaler.upscale(out_image, control=control)
                if out_image.size != target:
                    out_image = out_image.resize(target, Image.Resampling.LANCZOS)
            if out_image.size != (out_width, out_height):
                out_image = out_image.resize((out_width, out_height), Image.Resampling.LANCZOS)
            writer.append_data(np.asarray(out_image.convert("RGB"), dtype=np.uint8))
            frames_written += 1
            if progress and total:
                progress(frames_written, total)
            if total:
                log(f"  frame {frames_written}/{total}")
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if frames_written == 0:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        raise ValueError(f"No frames were read from video: {path}")

    try:
        import imageio_ffmpeg
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                "-i", tmp_video, "-i", path,
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                destination,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip().splitlines()[-1])
        os.remove(tmp_video)
    except Exception as exc:
        log(f"Warning: audio mux failed ({exc}); saving video-only.")
        os.replace(tmp_video, destination)

    log(f"Saved {frames_written} frame(s) → {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Upscale images or videos with Real-ESRGAN x2plus")
    parser.add_argument("--input", required=True, help="Image/video file or folder")
    parser.add_argument("--output-dir", default="output/upscaled")
    parser.add_argument("--long-edge", type=int, default=3840,
                        help="Target long edge (ignored by --quest-3-sbs)")
    parser.add_argument("--quest-3-sbs", action="store_true",
                        help="Fit each future eye view within 2064x2208")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--format", choices=("PNG", "JPEG"), default="PNG")
    parser.add_argument("--max-seconds", type=float, default=-1,
                        help="For video inputs, process only this many seconds (-1 = full video)")
    args = parser.parse_args()
    video_files = collect_videos(args.input)
    files = video_files or collect_images(args.input)
    if not files:
        parser.error("No supported images or videos found")
    engine = RealESRGANx2(ensure_model(), tile=args.tile)
    target_box = (2064, 2208) if args.quest_3_sbs else None
    for path in files:
        if path in video_files:
            upscale_video(
                path,
                args.output_dir,
                engine,
                args.long_edge,
                target_box=target_box,
                max_seconds=args.max_seconds,
            )
        else:
            upscale_file(path, args.output_dir, engine, args.long_edge, args.format,
                         target_box=target_box)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
