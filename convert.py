#!/usr/bin/env python3
"""Convert 2D images and videos to Quest-ready side-by-side (SBS) 3D."""

import argparse
import glob
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from depth_model import AVAILABLE_MODELS, load_depth_model
from sbs.sbs import process_image_sbs

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def get_device():
    if torch.cuda.is_available():
        print("CUDA GPU detected. Using GPU.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Apple Silicon GPU detected. Using MPS.")
        return torch.device("mps")
    print("No GPU detected. Using CPU.")
    return torch.device("cpu")


def collect_images(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            images.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(set(images))

    raise FileNotFoundError(f"Input not found: {input_path}")


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
    log=print,
):
    name, ext = os.path.splitext(os.path.basename(image_path))

    try:
        image = Image.open(image_path)
    except Exception as e:
        log(f"Skipping {name}{ext} (could not open): {e}")
        return False

    depth_map = infer_depth(model, image, device, dtype, is_metric, depth_input_scale, log=log)

    os.makedirs(output_dir, exist_ok=True)

    if depth_only:
        out_path = os.path.join(output_dir, f"{name}_depth{ext}")
        depth_map.save(out_path)
        return True

    sbs_image = make_sbs(image, depth_map, sbs_method, depth_scale, sbs_mode, sbs_blur, device)
    out_path = os.path.join(output_dir, f"{name}_sbs{ext}")
    sbs_image.save(out_path)
    return True


def collect_videos(input_path):
    """Collect video files from a path (file or directory)."""
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        videos = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            videos.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(set(videos))

    raise FileNotFoundError(f"Input not found: {input_path}")


def _convert_video_per_frame(
    video_path,
    output_dir,
    model,
    device,
    dtype,
    is_metric,
    sbs_method="mesh_warping",
    depth_scale=40,
    sbs_mode="parallel",
    sbs_blur=7,
    depth_input_scale=0.5,
    max_len=-1,
    target_fps=-1,
    max_res=1280,
    input_size=518,
    temporal_smoothing=0.2,
    batch_size=16,
    depth_only=False,
    log=print,
):
    """Convert a video to SBS 3D using the image depth model frame-by-frame.

    This is a fallback path used when Video Depth Anything is unavailable.
    It runs the DepthAnythingV2 image model on each frame individually with
    optional temporal smoothing to reduce flicker.

    Args:
        video_path: Path to input video file.
        output_dir: Directory for output files.
        model: DepthAnythingV2 image depth model.
        device: Computation device.
        dtype: Model dtype.
        is_metric: Whether the model produces metric depth.
        sbs_method: ``"mesh_warping"`` or ``"grid_sampling"``.
        depth_scale: SBS stereo strength.
        sbs_mode: ``"parallel"`` or ``"cross-eyed"``.
        sbs_blur: Depth blur strength (odd number).
        depth_input_scale: Downscale factor before depth inference.
        max_len: Maximum frames to process (-1 = all).
        target_fps: Output frame rate (-1 = match source).
        max_res: Longest edge cap in pixels (-1 = no limit).
        input_size: Input size for the depth model.
        temporal_smoothing: Blend factor with previous frame depth (0–0.5).
        batch_size: Frames per SBS processing batch.
        depth_only: Save depth visualisation only (no SBS).
        log: Logging callable.

    Returns:
        ``True`` if successful.
    """
    import imageio

    name, ext = os.path.splitext(os.path.basename(video_path))

    log(f"Extracting frames from: {video_path}")

    # Extract video frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Could not open video: {video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)
    else:
        height, width = original_height, original_width

    # Ensure even dimensions
    if height % 2 != 0:
        height -= 1
    if width % 2 != 0:
        width -= 1

    if original_fps <= 0:
        log("Could not determine the source frame rate; using 25 fps.")
        original_fps = 25.0
    fps = original_fps if target_fps < 0 else target_fps
    if fps <= 0:
        raise ValueError("Target FPS must be greater than zero or -1.")
    stride = max(round(original_fps / fps), 1)

    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_len > 0 and frame_count >= max_len:
            break
        if frame_count % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if max_res > 0 and max(original_height, original_width) > max_res:
                frame = cv2.resize(frame, (width, height))
            frames.append(frame)
        frame_count += 1
    cap.release()

    if not frames:
        log(f"No frames extracted from {video_path}")
        return False

    num_frames = len(frames)
    height, width, _ = frames[0].shape
    log(f"Extracted {num_frames} frames at {width}x{height}, {fps:.1f} fps")

    log("Running the selected DepthAnythingV2 model on video frames...")

    # Process frames in batches for depth estimation
    all_depth_maps = []
    with tqdm(total=num_frames, desc="Depth estimation", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        for i in range(num_frames):
            pil_img = Image.fromarray(frames[i].astype(np.uint8))

            # Infer depth (reuse existing function)
            depth_map = infer_depth(model, pil_img, device, dtype, is_metric,
                                    depth_input_scale, log=log)

            # Convert to tensor format for SBS processing: (H, W, 1)
            depth_np = np.array(depth_map).astype(np.float32) / 255.0
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(-1).float().to(device)
            all_depth_maps.append(depth_tensor)

            pbar.update(1)

    # Stack depth maps: (N, H, W, 1)
    depth_tensor = torch.stack(all_depth_maps)

    # Convert frames to tensor: (N, H, W, C)
    frames_tensor = torch.from_numpy(np.asarray(frames)).float().div_(255.0).to(device)

    # Apply SBS conversion with temporal smoothing
    log(f"Converting {num_frames} frames to SBS 3D...")

    from sbs.sbs import process_video_sbs

    sbs_result = process_video_sbs(
        frames=frames_tensor,  # (N, H, W, C)
        depth_maps=depth_tensor,  # (N, H, W, 1)
        method=sbs_method,
        depth_scale=depth_scale,
        mode=sbs_mode,
        depth_blur_strength=sbs_blur,
        temporal_smoothing=temporal_smoothing,
        batch_size=batch_size,
    )

    sbs_tensor = sbs_result[0]  # (N, H, W*2, C)

    # Encode as video
    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, f"{name}_sbs{ext}")

    log(f"Encoding SBS video to: {out_video_path}")
    writer = imageio.get_writer(out_video_path, fps=fps, macro_block_size=1,
                                 codec='libx264', ffmpeg_params=['-crf', '18'])

    for i in range(num_frames):
        sbs_frame = sbs_tensor[i].cpu().numpy()
        sbs_frame_uint8 = (sbs_frame * 255).astype(np.uint8)
        writer.append_data(sbs_frame_uint8)

    writer.close()
    log(f"SBS video saved to: {out_video_path}")

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
    # Video-specific options
    parser.add_argument("--video", action="store_true", help="Treat input as video (use per-frame processing).")
    parser.add_argument("--video-encoder", choices=["vits", "vitb", "vitl"], default="vits",
                        help="Official Video Depth Anything model used for videos.")
    parser.add_argument("--video-metric", action="store_true",
                        help="Use the metric Video Depth Anything checkpoint.")
    parser.add_argument("--max-len", type=int, default=-1, help="Max frames to process (-1 = all).")
    parser.add_argument("--target-fps", type=int, default=-1, help="Target FPS (-1 = original).")
    parser.add_argument("--max-res", type=int, default=1280, help="Max resolution dimension.")
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

    # Detect input type
    if os.path.isfile(args.input):
        detected_video = os.path.splitext(args.input)[1].lower() in VIDEO_EXTENSIONS
    elif os.path.isdir(args.input):
        detected_video = any(
            os.path.isfile(os.path.join(args.input, name))
            and os.path.splitext(name)[1].lower() in VIDEO_EXTENSIONS
            for name in os.listdir(args.input)
        )
    else:
        detected_video = False
    is_video_input = args.video or detected_video

    ok = 0
    total = 0

    if is_video_input:
        # Video processing mode
        from video_converter import (
            convert_video_to_sbs,
            load_video_depth_model,
        )

        try:
            videos = collect_videos(args.input)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

        if not videos:
            print(f"No video files found in: {args.input}")
            sys.exit(1)

        mode_label = "depth maps only" if args.depth_only else "Quest-ready SBS 3D video"
        print(f"Found {len(videos)} video(s)")
        print(f"Mode: {mode_label}")
        print(f"Output: {os.path.abspath(args.output_dir)}")

        if not args.yes and sys.stdin.isatty():
            if not prompt_yes_no("Proceed?"):
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

        with tqdm(total=len(videos), unit="vid", desc="Converting videos", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
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
                        target_fps=args.target_fps,
                        max_res=args.max_res,
                        temporal_smoothing=args.temporal_smoothing,
                        batch_size=args.batch_size,
                        depth_only=args.depth_only,
                    ):
                        ok += 1
                except Exception as e:
                    log_msg = f"Error processing {video_path}: {e}"
                    pbar.write(log_msg)
                pbar.update(1)

        print(f"\nDone. {ok}/{len(videos)} succeeded.")
        if not args.depth_only:
            print("Load the output videos on your Quest (Skybox, Pigasus, etc.) in side-by-side 3D mode.")
    else:
        # Image processing mode (existing)
        try:
            model, dtype, is_metric = load_depth_model(args.model, device, args.models_dir)
        except Exception as e:
            print(f"Failed to load image model: {e}")
            sys.exit(1)

        try:
            images = collect_images(args.input)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

        if not images:
            print(f"No images found in: {args.input}")
            sys.exit(1)

        if args.depth_only:
            mode_label = "depth maps only"
        else:
            mode_label = "Quest-ready SBS 3D images"

        print(f"Found {len(images)} image(s)")
        print(f"Mode: {mode_label}")
        print(f"Output: {os.path.abspath(args.output_dir)}")

        if not args.yes and sys.stdin.isatty():
            if not prompt_yes_no("Proceed?"):
                print("Cancelled.")
                sys.exit(0)

        with tqdm(total=len(images), unit="img", desc="Converting", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
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
                    log=pbar.write,
                ):
                    ok += 1
                pbar.update(1)

        print(f"\nDone. {ok}/{len(images)} succeeded.")
        if not args.depth_only:
            print("Load the *_sbs.png files on your Quest (Skybox, Pigasus, etc.) in side-by-side 3D mode.")


if __name__ == "__main__":
    main()
