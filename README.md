# StereoSift

StereoSift is a local image and video toolkit with two jobs:

1. Convert 2D images and videos into side-by-side (SBS) stereoscopic 3D media.
2. Review image batches and sort copies into `pass`, `warning`, and `fail` folders.

Everything runs locally through PyTorch — no cloud, no API keys, no running servers
required. Models download automatically on first use.

## Features

- GUI and CLI interfaces
- Image SBS conversion using Depth Anything V2
- Video SBS conversion using Video Depth Anything (constant-memory streaming)
- Original audio preserved in converted videos
- CUDA, Apple Silicon MPS, and CPU support
- Image QC with local YOLO object detection and optional moondream2 structure scan
- Safe QC mode that copies originals by default; move only when explicitly requested
- Optional OpenAI-compatible vision backend (LM Studio, Ollama) for QC

## Installation

Requires Python 3.10+ and FFmpeg (supplied via `imageio-ffmpeg`).

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
python install_torch.py
pip install -r requirements.txt
```

All model checkpoints download automatically on first use and are stored in
`models/`. They are not committed to Git.

## GUI

```bash
python gui.py
```

Two tabs:

- **Convert** — pick a file or folder, choose model size (Small/Base/Large),
  adjust 3D strength and depth blur, hit Convert. Images and videos are detected
  automatically and routed to the correct model.
- **Judge** — pick an input folder, run QC. Results appear in a live table with
  status badges (pass/warning/fail), score, person count, detected objects, and
  structure notes when deep scan is enabled.

## SBS conversion (CLI)

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

Video encoder choices: `vits` (fast, default), `vitb` (balanced), `vitl` (best quality).
Image model is selected the same way via `--model`; see `--help` for all options.
Anaglyph output currently applies to images; `--convergence` controls which depth
plane appears at screen level when viewed through red-cyan glasses.

## Image QC (CLI)

```bash
# Basic: exposure and contrast checks + YOLO person/object detection
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc

# With moondream2 structure scan (catches fused figures, doubled heads, etc.)
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc --deep-scan

# Strictness: relaxed | balanced (default) | strict
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc \
    --deep-scan --strictness strict

# Interactive launcher
sh qc.sh
```

Results are sorted into `pass/`, `warning/`, `fail/`, and a `report.json` summary.
Originals are copied by default. Add `--move` only when destructive sorting is
explicitly desired.

### QC pipeline layers

| Layer | Model | What it catches |
|---|---|---|
| Pixel checks | None | Dark/bright exposure, low/high contrast |
| Object detection | YOLO11n (~6 MB) | Person presence, object count |
| Structure deep scan | moondream2 (~2 GB) | Fused structure, doubled heads, extra/missing limbs, malformed hands |

YOLO and moondream2 download automatically on first use. moondream2 only runs
on images containing people unless `--scan-all` is passed.

### Optional: vision backend

For teams with a local LLM server already running:

```bash
python qc_pipeline.py \
  --input ~/Pictures/to-review \
  --output-dir output/qc \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model llama-3.2-11b-vision-instruct
```

This replaces moondream2 with an API call to any OpenAI-compatible endpoint.

## Project layout

```
convert.py                  — unified image/video conversion CLI
gui.py                      — CustomTkinter desktop GUI
depth_model.py              — Depth Anything V2 loading
video_converter.py          — Video Depth Anything streaming and encoding
qc_pipeline.py              — QC pipeline: pixel checks, YOLO, moondream2
sbs/sbs.py                  — stereoscopic warping algorithms
depth_anything_v2/          — image depth model implementation
video_depth_anything_repo/  — vendored Video Depth Anything implementation
models/                     — downloaded checkpoints (gitignored)
tests/                      — automated tests
```

## License and upstream work

This project incorporates code and model integrations derived from
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and
[Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything).
Consult the upstream license files before commercial use, especially for the
Base and Large checkpoints.
