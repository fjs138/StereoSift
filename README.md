# StereoSift
StereoSift is a local desktop toolkit for 2D-to-3D conversion and image QC.

## Deployment

StereoSift is a local app, so there is no hosted deployment to point at.
You run it from the repo on your machine.

## The Goal

I wanted one place to turn flat images and videos into side-by-side 3D, and
another place to sort image batches into pass, warning, and fail without a lot
of manual clicking.

## Technology Stack

| Technology | Use | Description |
| :-- | :-- | :-- |
| Python | Core language | Main language for the app and scripts |
| PyTorch | Model runtime | Runs the depth and QC models locally |
| CustomTkinter | Desktop UI | GUI for the Convert and Judge tabs |
| Depth Anything V2 | Image depth | Converts 2D images into SBS 3D |
| Video Depth Anything | Video depth | Converts videos frame by frame with temporal consistency |
| YOLO11n | QC pose gate | Finds people and catches obvious duplicate structure in the QC flow |
| Moondream2 | QC fallback scan | Checks tricky structure cases when the pose rules are not enough |
| LM Studio / oMLX | Optional backend | Local vision backend for stronger QC when you want it |
| requests | HTTP client | Talks to the optional backend |
| Pillow | Image handling | Loads, resizes, and saves images |
| imageio / imageio-ffmpeg | Video I/O | Reads and writes video files |
| opencv-python | Vision helpers | Extra image and video utilities |
| ultralytics | YOLO loader | Loads the YOLO checkpoint |
| transformers | Model loading | Loads Moondream2 from Hugging Face |
| huggingface_hub | Model cache | Handles model downloads and local caching |

## Project Specifications

* Convert 2D images into SBS 3D.
* Convert 2D videos into SBS 3D.
* Keep original audio when converting video.
* Show a GUI for both conversion and QC.
* Sort QC results into `pass`, `warning`, and `fail`.
* Copy originals by default, and only move files when asked.
* Support a strict offline mode that stays local and skips model-backed checks.
* Optionally connect to a local OpenAI-compatible vision backend if you already have one running.

## How It Works

StereoSift has two main paths.

Convert uses Depth Anything V2 for images and Video Depth Anything for video.
Judge uses pixel checks first, then YOLO for person/object detection, and then
an structure scan when that path is enabled.

There is also an optional local backend path for Judge. If you point it at LM
Studio or oMLX, StereoSift sends the image to that server instead of using the
local moondream2 scan.

Strict offline mode turns off YOLO, deep scan, and backend calls entirely. That
is the safest choice if you want no network-capable behavior at all.

## Structure of Project

| File/Folder | Purpose |
| :-- | :-- |
| `gui.py` | CustomTkinter desktop GUI |
| `convert.py` | 2D to SBS 3D conversion CLI |
| `qc_pipeline.py` | QC pipeline for pixel checks, YOLO, moondream2, and optional backend QC |
| `depth_model.py` | Depth Anything V2 loading |
| `video_converter.py` | Video Depth Anything streaming and encoding |
| `sbs/sbs.py` | SBS warping and conversion helpers |
| `tests/` | Automated tests |
| `models/` | Downloaded checkpoints |
| `video_depth_anything_repo/` | Vendored Video Depth Anything code |
| `depth_anything_v2/` | Depth Anything V2 implementation |

## Current Status

The app is working, but a few edges are still being tuned:

* QC verdicts are being tightened so obvious structure problems do not slip by
* the optional backend can require an API key, so the GUI needs to make that obvious
* strict offline mode is in place for peace of mind when you do not want any outbound behavior

That’s basically where it is right now: useful, local-first, and still getting
cleaned up around QC behavior and model choices.

## Installation

Requires Python 3.10+ and FFmpeg (pulled in through `imageio-ffmpeg`).

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python install_torch.py
pip install -r requirements.txt
```

Models download automatically on first use and are stored in `models/`.

If you want a fully offline run, turn on strict offline mode in the Judge tab
or use `--strict-offline` on the CLI.

## GUI

```bash
python gui.py
```

Two tabs:

* Convert turns 2D images or videos into SBS 3D. Pick a model size, choose the
  output style, and run it.
* Judge sorts a folder of images into pass, warning, and fail. It shows scores,
  person counts, issues, and structure notes when the structure scan is on.
* Judge also has a strict offline toggle that keeps everything local and blocks
  backend/model-backed checks.

## SBS Conversion

```bash
# Images
python convert.py --input photo.jpg --output-dir output --yes
python convert.py --input ~/Pictures/batch --output-dir output --yes

# Red-cyan anaglyph, or both output formats
python convert.py --input photo.jpg --output-dir output --output-format anaglyph --convergence 0.5 --yes
python convert.py --input photo.jpg --output-dir output --output-format both --yes

# Video
python convert.py --input movie.mp4 --output-dir output --video-encoder vits --yes

# Interactive launcher
sh sbs.sh
```

Video encoder choices: `vits` is fast, `vitb` is balanced, and `vitl` is the
best quality. Image model selection follows the same idea through `--model`.
Anaglyph output currently applies to images, and `--convergence` controls where
the depth plane sits when you view red-cyan output.

## Image QC

```bash
# Basic: exposure and contrast checks + YOLO person/object detection
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc

# With structure scan
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc --deep-scan

# Strictness: relaxed | balanced (default) | strict
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc \
    --deep-scan --strictness strict

# Strict offline mode
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc --strict-offline

# Interactive launcher
sh qc.sh
```

Results go into `pass/`, `warning/`, `fail/`, plus a `report.json` summary.
Originals are copied by default. Use `--move` only if you really want destructive sorting.

Local judgment is intentionally conservative. Only an explicit, confident
structural `PASS` goes into `pass/`. Clear defects, uncertain verdicts,
malformed responses, and scan failures land in `fail/`, which acts as the
manual review queue. Exposure and other minor aesthetic issues do not affect
the verdict.

### QC Pipeline Layers

| Layer | Model | What it catches |
| :-- | :-- | :-- |
| Pixel metrics | None | Records exposure and contrast for reference |
| Pose structure gate | YOLO11n (~6 MB) | Person presence, object count, and obvious duplicate structure |
| Structure scan | moondream2 fallback (~2 GB) | Tricky fused or duplicated structure when the pose gate is not enough |

YOLO and moondream2 download automatically on first use. The pose gate runs
first on person images, and the moondream2 fallback only kicks in after that
unless you are in strict offline mode.

Strict offline mode disables those model-backed checks entirely, so the QC path
stays local to image decoding and pixel heuristics only.

### Optional Backend

If you already have a local vision model running in LM Studio or oMLX, you can
point the Judge tab at it.

* LM Studio: `http://127.0.0.1:1234/v1`
* oMLX: `http://127.0.0.1:8000/v1`

The full `/v1/chat/completions` URL is also accepted. If authentication is on,
paste the Bearer token into the API key field or pass it with `--api-key`.

```bash
python qc_pipeline.py \
  --input ~/Pictures/to-review \
  --output-dir output/qc \
  --backend-url http://127.0.0.1:8000/v1 \
  --model Qwen3.6-35B-A3B-MLX-4bit
```

That path replaces the local fallback scan with a standard multimodal
OpenAI-compatible request. Images are resized to at most 1536 px and sent
inline over localhost.

## License

This project incorporates code and model integrations derived from
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything).
Consult the upstream license files before commercial use, especially for the
Base and Large checkpoints.
