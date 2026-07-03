#!/usr/bin/env python3
"""Video-to-SBS 3D converter using Video-Depth-Anything for depth estimation.

This module extends the existing SBS pipeline to support video files by:
1. Extracting per-frame depth maps using Video-Depth-Anything
2. Applying the existing SBS 3D conversion (mesh warping / grid sampling)
3. Encoding the result as a side-by-side 3D video

Usage:
    python3 video_converter.py --input input/video.mp4 --output-dir output/sbs_video
    python3 video_converter.py --input input/video.mp4 --output-dir output/sbs_video \
        --model depth_anything_v2_vitl_fp16.safetensors --encoder vitl
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms
from tqdm import tqdm

# Local imports from existing pipeline
from depth_model import AVAILABLE_MODELS, load_depth_model
from sbs.sbs import process_image_sbs, process_video_sbs

# ================================================================================
# Video-Depth-Anything integration
# ================================================================================

# Try to import decord (faster video reading), fall back to cv2
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

# Video-Depth-Anything model paths (stored in video_depth_anything_repo/checkpoints/)
VDA_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'video_depth_anything_repo', 'checkpoints')

# Video-Depth-Anything repository path
VDA_REPO_PATH = os.path.join(os.path.dirname(__file__), 'video_depth_anything_repo')

# Add VDA repo to path for imports
if VDA_REPO_PATH not in sys.path:
    sys.path.insert(0, VDA_REPO_PATH)


def load_video_depth_anything_model(encoder='vitl', metric=False, device='cuda', streaming=False):
    """Load Video-Depth-Anything model for video depth estimation.
    
    Args:
        encoder: Model variant ('vits', 'vitb', 'vitl')
        metric: Whether to use metric depth model
        device: Computation device
        
    Returns:
        tuple: (model, dtype, is_metric)
    """
    if streaming:
        from video_depth_anything.video_depth_stream import VideoDepthAnything
    else:
        from video_depth_anything.video_depth import VideoDepthAnything
    
    checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
    checkpoint_path = os.path.join(VDA_CHECKPOINT_DIR, f'{checkpoint_name}_{encoder}.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"Video-Depth-Anything checkpoint not found at {checkpoint_path}")
        print("Downloading from Hugging Face...")
        os.makedirs(VDA_CHECKPOINT_DIR, exist_ok=True)
        # Download from Hugging Face (if available)
        try:
            scale_name = {"vits": "Small", "vitb": "Base", "vitl": "Large"}[encoder]
            repo_prefix = "Metric-Video-Depth-Anything" if metric else "Video-Depth-Anything"
            hf_hub_download(
                repo_id=f"depth-anything/{repo_prefix}-{scale_name}",
                filename=f'{checkpoint_name}_{encoder}.pth',
                local_dir=VDA_CHECKPOINT_DIR,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"Could not download from Hugging Face: {e}")
            print("Please manually download the checkpoint and place it in:", VDA_CHECKPOINT_DIR)
            raise
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    
    # Keep accelerator inference in FP16. Video Depth Anything's temporal
    # windows are memory intensive; FP32 can trigger macOS's OOM killer.
    dtype = torch.float16 if torch.device(device).type in ('cuda', 'mps') else torch.float32
    model_kwargs = model_configs[encoder]
    if not streaming:
        model_kwargs = {**model_kwargs, 'metric': metric}
    model = VideoDepthAnything(**model_kwargs)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    
    print(f"Video-Depth-Anything model ({encoder}) loaded successfully on {device}")
    return model, dtype, metric


def convert_video_streaming(
    video_path,
    output_dir,
    model,
    device,
    dtype,
    sbs_method="mesh_warping",
    depth_scale=40,
    sbs_mode="parallel",
    sbs_blur=7,
    max_len=-1,
    target_fps=-1,
    max_res=1280,
    input_size=518,
    temporal_smoothing=0.2,
    depth_only=False,
):
    """Decode, infer, warp, and encode one frame at a time."""
    name, ext = os.path.splitext(os.path.basename(video_path))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = original_fps if target_fps < 0 else target_fps
    if fps <= 0:
        raise ValueError("Target FPS must be greater than zero or -1.")
    stride = max(round(original_fps / fps), 1)

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        width = round(original_width * scale)
        height = round(original_height * scale)
    else:
        width, height = original_width, original_height
    width -= width % 2
    height -= height % 2

    selected_total = (source_count + stride - 1) // stride if source_count > 0 else None
    if max_len > 0 and selected_total is not None:
        selected_total = min(selected_total, max_len)

    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, f"{name}_{'depth' if depth_only else 'sbs'}{ext}")
    video_only_path = out_video_path + ".video-only.mp4"
    writer = imageio.get_writer(
        video_only_path,
        fps=fps,
        macro_block_size=1,
        codec="libx264",
        ffmpeg_params=["-crf", "18"],
    )

    previous_depth = None
    source_index = 0
    output_index = 0
    device_name = torch.device(device).type
    print(f"Streaming video: {selected_total or '?'} frames at {width}x{height}, {fps:.1f} fps")

    try:
        with tqdm(total=selected_total, desc="Streaming depth + SBS", unit="frame") as progress:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or (max_len > 0 and output_index >= max_len):
                    break
                if source_index % stride != 0:
                    source_index += 1
                    continue
                source_index += 1

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                depth = model.infer_video_depth_one(
                    frame,
                    input_size=input_size,
                    device=device_name,
                    fp32=dtype == torch.float32,
                ).astype(np.float32)
                depth_min, depth_max = float(depth.min()), float(depth.max())
                depth = (depth - depth_min) / max(depth_max - depth_min, 1e-6)
                if previous_depth is not None and temporal_smoothing > 0:
                    depth = (1.0 - temporal_smoothing) * depth + temporal_smoothing * previous_depth
                previous_depth = depth

                if depth_only:
                    encoded = np.repeat((depth[..., None] * 255).astype(np.uint8), 3, axis=2)
                else:
                    frame_tensor = torch.from_numpy(frame).to(device=device, dtype=dtype).div_(255.0)
                    depth_tensor = torch.from_numpy(depth).to(device=device, dtype=dtype)
                    result = process_image_sbs(
                        frame_tensor.unsqueeze(0),
                        depth_tensor.unsqueeze(0).unsqueeze(-1),
                        method=sbs_method,
                        depth_scale=depth_scale,
                        mode=sbs_mode,
                        depth_blur_strength=sbs_blur,
                    )
                    if isinstance(result, tuple):
                        result = result[0]
                    encoded = (result.squeeze(0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

                writer.append_data(encoded)
                output_index += 1
                progress.update(1)
    finally:
        cap.release()
        writer.close()

    # Copy the source audio into the completed SBS movie without re-encoding
    # the video. The optional audio map also handles silent source videos.
    try:
        import imageio_ffmpeg

        mux = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                "-i", video_only_path, "-i", video_path,
                "-map", "0:v:0", "-map", "1:a?",
                "-c:v", "copy", "-c:a", "aac", "-shortest", out_video_path,
            ],
            capture_output=True,
            text=True,
        )
        if mux.returncode != 0:
            raise RuntimeError(mux.stderr.strip().splitlines()[-1])
        os.remove(video_only_path)
    except Exception as exc:
        print(f"Warning: audio could not be copied ({exc}); saving video without audio.")
        os.replace(video_only_path, out_video_path)

    print(f"SBS video saved to: {out_video_path}")
    return output_index > 0


def read_video_frames(video_path, max_len=-1, target_fps=-1, max_res=1280):
    """Read video frames for depth estimation.
    
    Args:
        video_path: Path to input video file
        max_len: Maximum number of frames (-1 for unlimited)
        target_fps: Target FPS (-1 for original)
        max_res: Maximum resolution dimension (default 1280)
        
    Returns:
        tuple: (frames_array, fps) where frames_array is np.ndarray of shape (N, H, W, C)
    """
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            # Ensure even dimensions
            height = round(original_height * scale) if round(original_height * scale) % 2 == 0 else round(original_height * scale) + 1
            width = round(original_width * scale) if round(original_width * scale) % 2 == 0 else round(original_width * scale) + 1
        
        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = max(round(vid.get_avg_fps() / fps), 1)
        frames_idx = list(range(0, len(vid), stride))
        
        if max_len > 0 and max_len < len(frames_idx):
            frames_idx = frames_idx[:max_len]
        
        frames = vid.get_batch(frames_idx).asnumpy()  # Shape: (N, H, W, C) in RGB
    else:
        cap = cv2.VideoCapture(video_path)
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

        fps = original_fps if target_fps < 0 else target_fps
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
        frames = np.stack(frames, axis=0)

    return frames, fps


def extract_video_depth(video_path, model, device, dtype, is_metric, 
                         max_len=-1, target_fps=-1, max_res=1280, input_size=518):
    """Extract depth maps from a video using Video-Depth-Anything.
    
    Args:
        video_path: Path to input video file
        model: Video-Depth-Anything model
        device: Computation device
        dtype: Model data type
        is_metric: Whether using metric depth model
        max_len: Maximum frames to process (-1 for unlimited)
        target_fps: Target FPS (-1 for original)
        max_res: Maximum resolution dimension
        input_size: Input size for depth model (default 518)
        
    Returns:
        tuple: (depth_maps_array, fps) where depth_maps is np.ndarray of shape (N, 1, H, W)
    """
    print(f"Extracting depth from video: {video_path}")
    
    # Read frames
    frames, fps = read_video_frames(video_path, max_len, target_fps, max_res)
    num_frames, height, width, _ = frames.shape
    print(f"Video: {num_frames} frames, {width}x{height}, {fps:.1f} fps")
    
    # Use Video-Depth-Anything inference
    depth_maps, actual_fps = model.infer_video_depth(
        frames=frames,
        target_fps=fps if target_fps == -1 else target_fps,
        input_size=input_size,
        device=torch.device(device).type,
        fp32=dtype == torch.float32
    )
    
    # depth_maps shape: (N, H, W) - numpy array
    print(f"Depth extraction complete: {depth_maps.shape[0]} depth maps")
    
    return depth_maps, actual_fps


def extract_video_frames(video_path, max_len=-1, target_fps=-1, max_res=1280):
    """Extract video frames for SBS conversion.
    
    Args:
        video_path: Path to input video file
        max_len: Maximum frames (-1 for unlimited)
        target_fps: Target FPS (-1 for original)
        max_res: Maximum resolution dimension
        
    Returns:
        tuple: (frames_array, fps) where frames is np.ndarray of shape (N, H, W, C)
    """
    return read_video_frames(video_path, max_len, target_fps, max_res)


def convert_video_to_sbs(
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
    max_len=-1,
    target_fps=-1,
    max_res=1280,
    input_size=518,
    temporal_smoothing=0.2,
    batch_size=16,
    depth_only=False,
):
    """Convert a video to side-by-side (SBS) 3D video.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory for output files
        model: Depth estimation model (Video-Depth-Anything or DepthAnythingV2)
        device: Computation device
        dtype: Model data type
        is_metric: Whether using metric depth model
        sbs_method: "mesh_warping" or "grid_sampling"
        depth_scale: SBS 3D strength (try 30-50)
        sbs_mode: "parallel" or "cross-eyed"
        sbs_blur: Depth blur (odd number, 3-15)
        max_len: Maximum frames (-1 for unlimited)
        target_fps: Target FPS (-1 for original)
        max_res: Maximum resolution dimension
        input_size: Input size for depth model (518)
        temporal_smoothing: Temporal smoothing factor (0.0-0.5)
        batch_size: Frames per processing batch
        depth_only: Save only depth maps (no SBS)
        
    Returns:
        bool: True if successful
    """
    if hasattr(model, "infer_video_depth_one"):
        return convert_video_streaming(
            video_path, output_dir, model, device, dtype,
            sbs_method=sbs_method, depth_scale=depth_scale, sbs_mode=sbs_mode,
            sbs_blur=sbs_blur, max_len=max_len, target_fps=target_fps,
            max_res=max_res, input_size=input_size,
            temporal_smoothing=temporal_smoothing, depth_only=depth_only,
        )

    from video_depth_anything.video_depth import VideoDepthAnything
    
    name, ext = os.path.splitext(os.path.basename(video_path))
    
    # Step 1: Extract depth maps using Video-Depth-Anything
    is_vda_model = isinstance(model, VideoDepthAnything)
    
    if is_vda_model:
        # Use Video-Depth-Anything for video depth estimation
        depth_maps, fps = extract_video_depth(
            video_path, model, device, dtype, is_metric,
            max_len=max_len, target_fps=target_fps, 
            max_res=max_res, input_size=input_size
        )
    else:
        # Fall back to frame-by-frame DepthAnythingV2 (slower but works)
        print("Using frame-by-frame depth estimation (not optimized for video)...")
        frames, fps = extract_video_frames(video_path, max_len, target_fps, max_res)
        depth_maps_list = []
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for i in tqdm(range(len(frames)), desc="Per-frame depth"):
            pil_img = Image.fromarray(frames[i].astype(np.uint8))
            tensor = transform(pil_img).unsqueeze(0).to(device=device, dtype=dtype)
            
            orig_H, orig_W = tensor.shape[2:]
            new_H, new_W = orig_H, orig_W
            if new_W % 14 != 0:
                new_W -= new_W % 14
            if new_H % 14 != 0:
                new_H -= new_H % 14
            
            if new_H != orig_H or new_W != orig_W:
                tensor = F.interpolate(tensor, size=(new_H, new_W), mode="bilinear", align_corners=False)
            
            with torch.no_grad():
                depth = model(tensor)
            
            depth = depth.squeeze(0).squeeze(0)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2
            if depth.shape[0] != final_H or depth.shape[1] != final_W:
                depth = F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0), size=(final_H, final_W),
                    mode="bilinear", align_corners=False,
                ).squeeze()
            
            depth = torch.clamp(depth, 0, 1)
            if is_metric:
                depth = 1.0 - depth
            
            depth_maps_list.append(depth.cpu().numpy())
        
        depth_maps = np.stack(depth_maps_list)
    
    # Step 2: Convert to SBS 3D
    if depth_only:
        print("Saving depth maps only...")
        depth_output_dir = os.path.join(output_dir, f"{name}_depth_frames")
        os.makedirs(depth_output_dir, exist_ok=True)
        for i, depth in enumerate(depth_maps):
            depth_visual = (depth * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth_visual)
            out_path = os.path.join(depth_output_dir, f"frame_{i:05d}_depth.png")
            depth_image.save(out_path)
        print(f"Depth maps saved to: {depth_output_dir}")
        return True
    
    # Extract frames for SBS conversion
    frames, _ = extract_video_frames(video_path, max_len, target_fps, max_res)
    num_frames, height, width, channels = frames.shape
    
    # Convert depth maps to tensor format expected by SBS processor
    # depth_maps shape: (N, H, W) -> (N, H, W, 1)
    depth_tensor = torch.from_numpy(depth_maps).unsqueeze(-1).float().to(device)
    frames_tensor = torch.from_numpy(frames).float().div_(255.0).to(device)
    
    print(f"Converting {num_frames} frames to SBS 3D...")
    
    # Use existing SBS video processor
    sbs_result = process_video_sbs(
        frames=frames_tensor,  # Shape: (N, H, W, C)
        depth_maps=depth_tensor,  # Shape: (N, H, W, 1)
        method=sbs_method,
        depth_scale=depth_scale,
        mode=sbs_mode,
        depth_blur_strength=sbs_blur,
        temporal_smoothing=temporal_smoothing,
        batch_size=batch_size,
    )
    
    # sbs_result is a tuple with the SBS tensor
    sbs_tensor = sbs_result[0]  # Shape: (N, H, W*2, C)
    
    # Step 3: Encode as video
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output codec and extension
    out_video_path = os.path.join(output_dir, f"{name}_sbs{ext}")
    
    print(f"Encoding SBS video to: {out_video_path}")
    writer = imageio.get_writer(out_video_path, fps=fps, macro_block_size=1, 
                                 codec='libx264', ffmpeg_params=['-crf', '18'])
    
    for i in tqdm(range(num_frames), desc="Encoding"):
        sbs_frame = sbs_tensor[i].cpu().numpy()
        # Convert from [0, 1] float to [0, 255] uint8
        sbs_frame_uint8 = (sbs_frame * 255).astype(np.uint8)
        writer.append_data(sbs_frame_uint8)
    
    writer.close()
    print(f"SBS video saved to: {out_video_path}")
    
    return True


# ================================================================================
# CLI entry point
# ================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D videos to Quest-ready side-by-side (SBS) 3D video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Video file or folder of videos.")
    parser.add_argument("--output-dir", default="output/sbs_video", help="Where to save results.")
    parser.add_argument("--encoder", default="vitl", choices=["vits", "vitb", "vitl"],
                        help="Video-Depth-Anything model variant.")
    parser.add_argument("--depth-model", default="depth_anything_v2_vitl_fp16.safetensors", 
                        choices=AVAILABLE_MODELS,
                        help="Fallback depth model for per-frame processing.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--depth-only", action="store_true", help="Save depth maps only (no SBS).")
    parser.add_argument("--max-len", type=int, default=-1, help="Max frames to process (-1 = all).")
    parser.add_argument("--target-fps", type=int, default=-1, help="Target FPS (-1 = original).")
    parser.add_argument("--max-res", type=int, default=1280, help="Max resolution dimension.")
    parser.add_argument("--input-size", type=int, default=518, help="Input size for depth model.")
    parser.add_argument("--depth-scale", type=int, default=40, help="SBS 3D strength (try 30-50).")
    parser.add_argument("--sbs-method", choices=["mesh_warping", "grid_sampling"], 
                        default="mesh_warping")
    parser.add_argument("--sbs-mode", choices=["parallel", "cross-eyed"], default="parallel")
    parser.add_argument("--sbs-blur", type=int, default=7, help="Depth blur (odd number, 3-15).")
    parser.add_argument("--temporal-smoothing", type=float, default=0.2, 
                        help="Temporal smoothing (0.0-0.5).")
    parser.add_argument("--batch-size", type=int, default=16, help="Frames per batch.")
    parser.add_argument("--metric", action="store_true", help="Use metric depth model.")
    
    args = parser.parse_args()
    
    # Collect video files
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    
    if os.path.isfile(args.input):
        videos = [args.input]
    elif os.path.isdir(args.input):
        videos = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(glob.glob(os.path.join(args.input, f"*{ext}")))
            videos.extend(glob.glob(os.path.join(args.input, f"*{ext.upper()}")))
        videos = sorted(set(videos))
    else:
        print(f"Input not found: {args.input}")
        sys.exit(1)
    
    if not videos:
        print(f"No video files found in: {args.input}")
        sys.exit(1)
    
    print(f"Found {len(videos)} video(s)")
    mode_label = "depth maps only" if args.depth_only else "Quest-ready SBS 3D video"
    print(f"Mode: {mode_label}")
    print(f"Output: {os.path.abspath(args.output_dir)}")
    
    if not args.yes and sys.stdin.isatty():
        answer = input("Proceed? [Y/n]: ").strip().lower()
        if answer in ('n', 'no'):
            print("Cancelled.")
            sys.exit(0)
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA GPU detected. Using GPU.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon MPS detected. Using MPS.")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")
    
    # Load Video-Depth-Anything model for video depth estimation
    try:
        print(f"\nLoading Video-Depth-Anything model (encoder={args.encoder})...")
        vda_model, vda_dtype, vda_is_metric = load_video_depth_anything_model(
            encoder=args.encoder, metric=args.metric, device=device
        )
    except Exception as e:
        print(f"Failed to load Video-Depth-Anything model: {e}")
        print("Falling back to per-frame DepthAnythingV2 (slower)...")
        try:
            model, dtype, is_metric = load_depth_model(args.depth_model, device, args.models_dir)
            vda_model = model  # Use as fallback
            vda_dtype = dtype
            vda_is_metric = is_metric
        except Exception as e2:
            print(f"Failed to load fallback model: {e2}")
            sys.exit(1)
    
    # Process each video
    ok = 0
    for video_path in videos:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        try:
            convert_video_to_sbs(
                video_path=video_path,
                output_dir=args.output_dir,
                model=vda_model,
                device=device,
                dtype=vda_dtype if 'vda_dtype' in dir() else dtype,
                is_metric=vda_is_metric if 'vda_is_metric' in dir() else is_metric,
                sbs_method=args.sbs_method,
                depth_scale=args.depth_scale,
                sbs_mode=args.sbs_mode,
                sbs_blur=args.sbs_blur,
                max_len=args.max_len,
                target_fps=args.target_fps,
                max_res=args.max_res,
                input_size=args.input_size,
                temporal_smoothing=args.temporal_smoothing,
                batch_size=args.batch_size,
                depth_only=args.depth_only,
            )
            ok += 1
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nDone. {ok}/{len(videos)} succeeded.")
    if not args.depth_only:
        print("Load the output videos on your Quest (Skybox, Pigasus, etc.) in side-by-side 3D mode.")


if __name__ == "__main__":
    main()
