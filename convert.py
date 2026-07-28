#!/usr/bin/env python3
"""Convert 2D images and videos to Quest-ready side-by-side (SBS) 3D."""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from depth_model import AVAILABLE_MODELS, load_depth_model
from media_utils import collect_images, collect_videos
from sbs.sbs import process_image_sbs, process_image_anaglyph


def get_device():
    if torch.cuda.is_available():
        print("CUDA GPU detected. Using GPU.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Apple Silicon GPU detected. Using MPS.")
        return torch.device("mps")
    print("No GPU detected. Using CPU.")
    return torch.device("cpu")


def prompt_yes_no(message, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{message} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter y or n.")


def infer_depth(model, image, device, dtype, is_metric, depth_input_scale, log=print):
    """Run depth estimation. Returns grayscale depth PIL at full input resolution."""
    original = image.convert("RGB")
    working = original

    if depth_input_scale < 1.0:
        w, h = working.size
        new_w = max(1, int(w * depth_input_scale))
        new_h = max(1, int(h * depth_input_scale))
        log(f"Downscaling for depth from {w}x{h} to {new_w}x{new_h} ({depth_input_scale * 100:.0f}%)")
        working = working.resize((new_w, new_h), Image.Resampling.BICUBIC)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(working).unsqueeze(0).to(device=device, dtype=dtype)

    orig_h, orig_w = image_tensor.shape[2:]
    new_h, new_w = orig_h, orig_w
    if new_w % 14 != 0:
        new_w -= new_w % 14
    if new_h % 14 != 0:
        new_h -= new_h % 14

    if new_h != orig_h or new_w != orig_w:
        log(f"Resizing depth input from {orig_w}x{orig_h} to {new_w}x{new_h}")
        image_tensor = F.interpolate(image_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    with torch.no_grad():
        depth = model(image_tensor)

    depth = depth.squeeze(0).squeeze(0)
    depth_range = depth.max() - depth.min()
    depth = (depth - depth.min()) / torch.clamp(depth_range, min=1e-6)

    full_w, full_h = original.size
    final_h = (full_h // 2) * 2
    final_w = (full_w // 2) * 2
    if depth.shape[0] != final_h or depth.shape[1] != final_w:
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(final_h, final_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    depth = torch.clamp(depth, 0, 1)
    if is_metric:
        depth = 1.0 - depth

    depth_np = (depth.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(depth_np)


def make_sbs(original, depth_map, sbs_method, depth_scale, sbs_mode, sbs_blur, device):
    """Build side-by-side 3D image from original photo and depth map."""
    to_tensor = transforms.ToTensor()

    base_image = to_tensor(original.convert("RGB")).permute(1, 2, 0).unsqueeze(0)
    depth_tensor = to_tensor(depth_map).permute(1, 2, 0).unsqueeze(0)

    stereo_dtype = torch.float32 if device.type == "cpu" else torch.float16
    base_image = base_image.to(device=device, dtype=stereo_dtype)
    depth_tensor = depth_tensor.to(device=device, dtype=stereo_dtype)

    if sbs_blur % 2 == 0:
        sbs_blur += 1

    sbs_tensor = process_image_sbs(
        base_image=base_image,
        depth_map=depth_tensor,
        method=sbs_method,
        depth_scale=depth_scale,
        mode=sbs_mode,
        depth_blur_strength=sbs_blur,
    )

    return transforms.ToPILImage()(sbs_tensor.squeeze(0).cpu().permute(2, 0, 1))


def make_anaglyph(original, depth_map, sbs_method, depth_scale, sbs_blur,
                  convergence, device):
    """Build a red-cyan anaglyph from original photo and depth map."""
    to_tensor = transforms.ToTensor()

    base_image   = to_tensor(original.convert("RGB")).permute(1, 2, 0).unsqueeze(0)
    depth_tensor = to_tensor(depth_map).permute(1, 2, 0).unsqueeze(0)

    stereo_dtype = torch.float32 if device.type == "cpu" else torch.float16
    base_image   = base_image.to(device=device, dtype=stereo_dtype)
    depth_tensor = depth_tensor.to(device=device, dtype=stereo_dtype)

    if sbs_blur % 2 == 0:
        sbs_blur += 1

    ana_tensor = process_image_anaglyph(
        base_image=base_image,
        depth_map=depth_tensor,
        method=sbs_method,
        depth_scale=depth_scale,
        depth_blur_strength=sbs_blur,
        convergence=convergence,
    )
    return transforms.ToPILImage()(ana_tensor.squeeze(0).cpu().permute(2, 0, 1))


def convert_one(
    model,
    image_path,
    output_dir,
    device,
    dtype,
    is_metric,
    *,
    depth_only,
    depth_input_scale,
    sbs_method,
    depth_scale,
    sbs_mode,
    sbs_blur,
    output_format="sbs",   # "sbs", "anaglyph", or "both"
    convergence=0.5,
    log=print,
    control=None,
):
    name, ext = os.path.splitext(os.path.basename(image_path))

    if control:
        control()
    try:
        image = Image.open(image_path)
    except Exception as e:
        log(f"Skipping {name}{ext} (could not open): {e}")
        return False

    if control:
        control()
    depth_map = infer_depth(model, image, device, dtype, is_metric, depth_input_scale, log=log)

    os.makedirs(output_dir, exist_ok=True)

    if depth_only:
        if control:
            control()
        out_path = os.path.join(output_dir, f"{name}_depth{ext}")
        depth_map.save(out_path)
        return True

    if output_format in ("sbs", "both"):
        if control:
            control()
        sbs_image = make_sbs(image, depth_map, sbs_method, depth_scale,
                             sbs_mode, sbs_blur, device)
        sbs_image.save(os.path.join(output_dir, f"{name}_sbs{ext}"))

    if output_format in ("anaglyph", "both"):
        if control:
            control()
        ana_image = make_anaglyph(image, depth_map, sbs_method, depth_scale,
                                  sbs_blur, convergence, device)
        ana_image.save(os.path.join(output_dir, f"{name}_anaglyph{ext}"))

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D images and videos to Quest-ready side-by-side (SBS) 3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Image or video file/folder.")
    parser.add_argument("--output-dir", default="output", help="Where to save results.")
    parser.add_argument("--model", default="depth_anything_v2_vitl_fp16.safetensors", choices=AVAILABLE_MODELS)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--depth-only", action="store_true", help="Save depth map only (no SBS).")
    parser.add_argument("--depth-input-scale", type=float, default=0.5,
                        help="Downscale before depth inference (0.5 = half size). Saves memory on large images.")
    parser.add_argument("--depth-scale", type=int, default=40, help="SBS 3D strength (try 30-50).")
    parser.add_argument("--sbs-method", choices=["mesh_warping", "grid_sampling"], default="mesh_warping")
    parser.add_argument("--sbs-mode", choices=["parallel", "cross-eyed"], default="parallel",
                        help="Use parallel for VR headsets like Meta Quest.")
    parser.add_argument("--sbs-blur", type=int, default=7, help="Depth blur (odd number, 3-15).")
    parser.add_argument("--output-format", choices=["sbs", "anaglyph", "both"], default="sbs",
                        help="sbs = side-by-side (VR headsets), anaglyph = red-cyan glasses, both = save both.")
    parser.add_argument("--convergence", type=float, default=0.5,
                        help="Anaglyph zero-disparity plane (0.0–1.0). 0.5 = midpoint, higher = push into screen.")
    # Video-specific options
    parser.add_argument("--video", action="store_true", help="Treat input as video (use per-frame processing).")
    parser.add_argument("--video-encoder", choices=["vits", "vitb", "vitl"], default="vits",
                        help="Official Video Depth Anything model used for videos.")
    parser.add_argument("--video-metric", action="store_true",
                        help="Use the metric Video Depth Anything checkpoint.")
    parser.add_argument("--max-len", type=int, default=-1, help="Max frames to process (-1 = all).")
    parser.add_argument("--max-seconds", type=float, default=-1,
                        help="Max output seconds to process (-1 = all).")
    parser.add_argument("--target-fps", type=int, default=-1, help="Target FPS (-1 = original).")
    parser.add_argument("--max-res", type=int, default=1280, help="Max resolution dimension.")
    parser.add_argument("--video-input-size", type=int, default=518,
                        help="Resolution fed to Video Depth Anything.")
    parser.add_argument("--temporal-smoothing", type=float, default=0.2,
                        help="Temporal smoothing for video (0.0-0.5).")
    parser.add_argument("--batch-size", type=int, default=16, help="Frames per batch.")
    args = parser.parse_args()

    # Shells do not expand ~ when a path comes from an interactive prompt or is
    # quoted. Normalize user paths here so "~" and "~/Videos" work everywhere.
    args.input = os.path.abspath(os.path.expanduser(args.input))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    args.models_dir = os.path.abspath(os.path.expanduser(args.models_dir))

    device = get_device()
    if device.type == "mps" and args.max_res == 1280:
        args.max_res = 720
        print("Apple Silicon safety limit: using --max-res 720 (override explicitly if desired).")

    ok = 0
    try:
        videos = collect_videos(args.input)
        images = [] if args.video else collect_images(args.input)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if not images and not videos:
        print(f"No supported images or videos found in: {args.input}")
        sys.exit(1)

    if images and videos:
        print(f"Found mixed input: {len(images)} image(s), {len(videos)} video(s)")

    if images:
        # Image processing mode
        try:
            model, dtype, is_metric = load_depth_model(args.model, device, args.models_dir)
        except Exception as e:
            print(f"Failed to load image model: {e}")
            sys.exit(1)

        if args.depth_only:
            mode_label = "depth maps only"
        else:
            mode_label = "Quest-ready SBS 3D images"

        print(f"Found {len(images)} image(s)")
        print(f"Mode: {mode_label}")
        print(f"Output: {os.path.abspath(args.output_dir)}")

        if not args.yes and sys.stdin.isatty():
            if not prompt_yes_no("Proceed with image conversion?"):
                print("Cancelled.")
                sys.exit(0)

        with tqdm(
            total=len(images),
            unit="img",
            desc="Converting images",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        ) as pbar:
            for image_path in images:
                pbar.set_postfix_str(os.path.basename(image_path)[:40], refresh=False)
                if convert_one(
                    model,
                    image_path,
                    args.output_dir,
                    device,
                    dtype,
                    is_metric,
                    depth_only=args.depth_only,
                    depth_input_scale=args.depth_input_scale,
                    sbs_method=args.sbs_method,
                    depth_scale=args.depth_scale,
                    sbs_mode=args.sbs_mode,
                    sbs_blur=args.sbs_blur,
                    output_format=args.output_format,
                    convergence=args.convergence,
                    log=pbar.write,
                ):
                    ok += 1
                pbar.update(1)

        if not args.depth_only:
            if args.output_format in ("sbs", "both"):
                print("Load the *_sbs files on your Quest (Skybox, Pigasus, etc.) in side-by-side 3D mode.")
            if args.output_format in ("anaglyph", "both"):
                print("View *_anaglyph files with red-cyan 3D glasses.")

    if videos:
        # Video processing mode
        from video_converter import (
            convert_video_to_sbs,
            load_video_depth_model,
        )

        mode_label = "depth maps only" if args.depth_only else "Quest-ready SBS 3D video"
        print(f"Found {len(videos)} video(s)")
        print(f"Mode: {mode_label}")
        print(f"Output: {os.path.abspath(args.output_dir)}")

        if not args.yes and sys.stdin.isatty():
            if not prompt_yes_no("Proceed with video conversion?"):
                print("Cancelled.")
                sys.exit(0)

        try:
            print(f"Loading official Video Depth Anything ({args.video_encoder})...")
            video_model, video_dtype, video_is_metric = load_video_depth_model(
                encoder=args.video_encoder,
                metric=args.video_metric,
                device=device,
            )
        except Exception as e:
            print(f"Failed to load Video Depth Anything: {e}")
            sys.exit(1)

        video_ok = 0
        with tqdm(
            total=len(videos),
            unit="vid",
            desc="Converting videos",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        ) as pbar:
            for video_path in videos:
                pbar.set_postfix_str(os.path.basename(video_path)[:40], refresh=False)
                try:
                    if convert_video_to_sbs(
                        video_path=video_path,
                        output_dir=args.output_dir,
                        model=video_model,
                        device=device,
                        dtype=video_dtype,
                        is_metric=video_is_metric,
                        sbs_method=args.sbs_method,
                        depth_scale=args.depth_scale,
                        sbs_mode=args.sbs_mode,
                        sbs_blur=args.sbs_blur,
                        max_len=args.max_len,
                        max_seconds=args.max_seconds,
                        target_fps=args.target_fps,
                        max_res=args.max_res,
                        input_size=args.video_input_size,
                        temporal_smoothing=args.temporal_smoothing,
                        batch_size=args.batch_size,
                        depth_only=args.depth_only,
                    ):
                        ok += 1
                        video_ok += 1
                except Exception as e:
                    log_msg = f"Error processing {video_path}: {e}"
                    pbar.write(log_msg)
                pbar.update(1)

        print(f"\nVideo conversion done. {video_ok}/{len(videos)} succeeded.")
        if not args.depth_only:
            print("Load the output videos on your Quest (Skybox, Pigasus, etc.) in side-by-side 3D mode.")

    print(f"\nAll done. {ok}/{len(images) + len(videos)} file(s) succeeded.")


if __name__ == "__main__":
    main()
