<p align="center">
  <img src="assets/stereosift-logo.png" alt="StereoSift logo" width="220">
</p>

<h1 align="center">StereoSift</h1>

<p align="center">
  Local-first desktop toolkit for 2D-to-3D conversion, AI upscaling, batch image
  QC, and vision-assisted folder organization. Runs entirely offline on local
  models, or against any OpenAI-compatible vision endpoint.
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Platform" src="https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey">
  <img alt="Offline capable" src="https://img.shields.io/badge/offline-capable-success">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  <img alt="Status: alpha" src="https://img.shields.io/badge/status-alpha-orange">
</p>

---

## Why

Media-heavy desktop work — converting stills to stereo 3D, upscaling a batch for
a headset, triaging a folder of generated images, sorting a mixed dump into
categories — normally means four separate scripts, four sets of arguments, and
no shared conventions between them.

StereoSift puts all four behind one window with consistent behavior: recursive
input preserves source structure, originals are copied rather than moved unless
you ask, and every path can run with no network access at all.

## What It Does

- **Convert** — 2D images and video to side-by-side 3D using monocular depth estimation
- **Upscale** — tiled Real-ESRGAN with aspect-safe target sizing, for images and video
- **Judge** — batch image QC routing to `pass`, `warning`, `fail`, and an unscored queue
- **Organize** — vision-model sorting into your own arbitrary category labels

## Screenshots

Four tabs, one window. Everything runs on your machine.

| Judge | Organize |
| :-- | :-- |
| ![Judge tab — batch image QC sorting into pass, warning, fail, and review](assets/judge.png) | ![Organize tab — vision-model sorting into user-defined category labels](assets/organize.png) |
| Batch QC routes each image to `pass`, `warning`, `fail`, or a review queue. | A vision model picks exactly one of your labels per image. |

| Upscale | Convert |
| :-- | :-- |
| ![Upscale tab — Real-ESRGAN upscaling with headset-ready targets](assets/upscale.png) | ![Convert tab — 2D to SBS 3D conversion with depth controls](assets/convert.png) |
| Real-ESRGAN x2plus with tiled inference and aspect-safe target sizes. | Depth-based 2D to SBS 3D, with strength, blur, and convergence controls. |

## Architecture

```
                        ┌──────────────────────────┐
                        │   gui.py  (CustomTkinter)│
                        │   Convert │ Upscale │    │
                        │   Judge   │ Organize│    │
                        └────┬──────┬──────┬───────┘
                             │      │      │
        ┌────────────────────┘      │      └──────────────────┐
        ▼                           ▼                         ▼
┌───────────────┐        ┌──────────────────┐      ┌────────────────────┐
│  convert.py   │        │   upscaler.py    │      │   qc_pipeline.py   │
│               │        │                  │      │                    │
│ depth_model   │        │  Real-ESRGAN     │      │  ┌──────────────┐  │
│  └ Depth      │        │   x2plus         │      │  │ pixel metrics│  │
│    Anything V2│        │  tiled inference │      │  └──────┬───────┘  │
│               │        │  aspect-safe     │      │         ▼          │
│ video_        │        │   resize         │      │  ┌──────────────┐  │
│  converter.py │        └──────────────────┘      │  │ YOLO11n gate │  │
│  └ Video Depth│                                  │  └──────┬───────┘  │
│    Anything   │                                  │         ▼          │
│    (streaming)│                                  │  ┌──────────────┐  │
│               │                                  │  │ moondream2   │  │
│ sbs/sbs.py    │                                  │  │ deep scan    │  │
│  └ L/R warp   │                                  │  └──────────────┘  │
└───────────────┘                                  └─────────┬──────────┘
                                                             │
                                              ┌──────────────┴──────────────┐
                                              │  OR (mutually exclusive)    │
                                              ▼                             ▼
                                    ┌───────────────────┐        ┌──────────────────┐
                                    │ OpenAI-compatible │        │ local model path │
                                    │ vision backend    │        │ (CLI default;    │
                                    │ LM Studio / oMLX  │        │  clear URL in GUI)│
                                    │ (GUI default)     │        └──────────────────┘
                                    └───────────────────┘
```

Every tab is a thin GUI layer over a module that also runs standalone from the
command line. `media_utils.py` holds the shared image/video detection and
collection helpers so all four paths agree on what counts as input and how
recursive structure is preserved.

### QC routing policy

Judge has two mutually exclusive paths:

**Vision backend** — GUI default (URL pre-filled); CLI with `--backend-url`:

| Outcome | Routed to |
| :-- | :-- |
| Model returns `pass` | `pass/` |
| Model returns `warning`, or malformed JSON (batch keeps moving) | `warning/` |
| Model returns `fail` | `fail/` |
| No recognizable status in the reply | `unscored/` |

**Local models** — CLI default; GUI when the backend URL field is cleared:

| Outcome | Routed to |
| :-- | :-- |
| No structural issues detected | `pass/` |
| Major defect (uncertain cases with 3+ people become `warning`) | `fail/` |
| Uncertain structure, or deep scan unavailable | `warning/` |

Clear the backend URL (or omit `--backend-url` on the CLI) to use local models:
pixel heuristics always, YOLO subject detection by default, and an optional
moondream2 deep scan for duplicated or incorrectly joined structures.

Separating `unscored/` from `fail/` matters on the backend path: an image nobody
rated is not the same as an image that failed, and collapsing the two would
silently inflate the failure rate.

Strict offline mode forces the local path and additionally disables YOLO and the
deep scan, leaving pixel-only QC with no network-capable behavior at all.

Exposure and other aesthetic issues are recorded but never affect the verdict.

### QC pipeline layers

| Layer | Model | What it catches |
| :-- | :-- | :-- |
| Pixel metrics | None | Exposure and contrast, recorded for reference |
| Structure gate | YOLO11n (~6 MB) | Subject presence, object count, obvious duplicate or fused features |
| Structural scan | moondream2 (~2 GB) | Tricky structural defects when the gate is not conclusive |

The structure gate runs first on images with subjects; the moondream2 fallback
only engages after it when `--deep-scan` is enabled (off by default).

## Technology Stack

| Technology | Role | Why it's here |
| :-- | :-- | :-- |
| Python | Core language | Application, pipelines, and CLI entry points |
| PyTorch | Model runtime | Executes depth and QC models locally, CPU or GPU |
| CustomTkinter | Desktop UI | Native-feeling GUI without a browser runtime |
| Depth Anything V2 | Image depth | Monocular depth maps driving SBS displacement |
| Video Depth Anything | Video depth | Temporally consistent depth, streamed frame by frame |
| Real-ESRGAN x2plus | Upscaling | Tiled x2 restoration before an exact target resize |
| YOLO11n | QC subject gate | Fast subject and object detection as the first structural check |
| moondream2 | QC fallback | Vision-language scan for defects the gate cannot resolve |
| LM Studio / oMLX | Optional backend | Any OpenAI-compatible vision endpoint can replace the local path |
| Pillow / opencv-python | Image handling | Decode, resize, and save |
| imageio / imageio-ffmpeg | Video I/O | Frame extraction and re-encoding, bundling FFmpeg |
| ultralytics | Model loader | Loads the YOLO checkpoint |
| transformers / huggingface_hub | Model loading & cache | Fetches and caches moondream2 and Qwen-VL checkpoints |

## Installation

Requires Python 3.10+ and FFmpeg (pulled in through `imageio-ffmpeg`).

```bash
git clone https://github.com/fjs138/StereoSift.git
cd StereoSift

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

python install_torch.py
pip install -r requirements.txt
```

Model checkpoints download automatically on first use and are cached in
`models/`. For a fully offline run, enable strict offline mode in the Judge tab
or pass `--strict-offline` on the CLI.

## Usage

### GUI

```bash
python gui.py
```

The Judge tab pre-fills `http://127.0.0.1:8001/v1` for a local oMLX or LM Studio
server. Clear that field (or leave `--backend-url` unset on the CLI) to run fully
local QC with YOLO instead.

Interactive shell wrappers are also available if you prefer prompts over flags:
`sh sbs.sh` for conversion and `sh qc.sh` for QC (both expect a project venv).

### SBS conversion

```bash
# Images
python convert.py --input photo.jpg --output-dir output --yes
python convert.py --input ~/Pictures/batch --output-dir output --yes

# Red-cyan anaglyph, or both output formats
python convert.py --input photo.jpg --output-dir output \
    --output-format anaglyph --convergence 0.5 --yes

# Video
python convert.py --input movie.mp4 --output-dir output --video-encoder vits --yes

# First five seconds only, as a preview before committing to a long clip
python convert.py --input movie.mp4 --output-dir output \
    --video-encoder vits --max-seconds 5 --yes
```

Encoder choices trade speed for quality: `vits` is fast, `vitb` balanced, `vitl`
best. Video uses the streaming Video Depth Anything path — each frame is
depth-estimated, converted to left/right, and written straight to the output
without loading the whole file into memory. Outputs carry a `_SBS_LR` suffix so
headset players detect left/right mode.

### Upscaling

```bash
python upscaler.py --input photo.jpg --quest-3-sbs --output-dir output/upscaled
python upscaler.py --input ~/Pictures/batch --long-edge 7680 --output-dir output/8k
python upscaler.py --input clip.mp4 --long-edge 3840 --max-seconds 5 --output-dir output/upscaled
```

Aspect ratio is preserved, media already at the target is not enlarged, and
tiled inference keeps memory bounded.

### Image QC

```bash
# Exposure and contrast checks plus YOLO subject detection
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc

# With the structural fallback scan
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc --deep-scan

# Strictness: relaxed | balanced (default) | strict
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc \
    --deep-scan --strictness strict

# Fully local, no model-backed checks
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc --strict-offline

# Against a local vision backend
python qc_pipeline.py --input ~/Pictures/to-review --output-dir output/qc \
    --backend-url http://127.0.0.1:8000/v1 --model Qwen3.6-35B-A3B-MLX-4bit
```

Originals are copied by default. `--move` is destructive and opt-in.

### Optional backend

Point Judge at a vision model you already have running:

- LM Studio — `http://127.0.0.1:1234/v1`
- oMLX — `http://127.0.0.1:8001/v1`

The full `/v1/chat/completions` URL is also accepted. If auth is enabled, supply
the token in the API key field or via `--api-key`. Images are resized to at most
1536 px and sent inline over localhost.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q .
```

Run these inside the project venv — the system Python will fail on `PIL` and
`cv2` imports.

### Structure benchmark

`tools/benchmark_structure.py` evaluates a vision model against a labeled folder,
either using a local Qwen-VL checkpoint or an OpenAI-compatible backend:

```bash
python tools/benchmark_structure.py \
  --input path/to/images \
  --labels path/to/labels.json \
  --output output/qwen-benchmark.json
```

Labels are a JSON object mapping filename to `pass` or `fail`. The report covers
latency, errors, accuracy, and **fail precision and recall separately** — a model
that misses defects cannot hide behind headline accuracy.

`qwen_structure.py` is reference-only benchmark code. It is inert unless invoked
directly and is not part of the GUI, converter, or production routing policy.

## Project Structure

| Path | Purpose |
| :-- | :-- |
| `gui.py` | CustomTkinter desktop GUI |
| `convert.py` | 2D to SBS 3D conversion CLI |
| `upscaler.py` | Real-ESRGAN upscaling CLI |
| `qc_pipeline.py` | QC and user-defined organization pipelines |
| `media_utils.py` | Shared image/video detection and collection helpers |
| `depth_model.py` | Depth Anything V2 loading |
| `video_converter.py` | Video Depth Anything streaming and encoding |
| `sbs/sbs.py` | SBS warping and conversion helpers |
| `install_torch.py` | Platform-specific PyTorch installer |
| `qc.sh` / `sbs.sh` | Interactive shell launchers for QC and conversion |
| `qwen_structure.py` | Reference-only Qwen-VL benchmark path |
| `tools/benchmark_structure.py` | Labeled-folder benchmark harness |
| `tests/` | Automated tests |
| `models/` | Downloaded checkpoints |
| `depth_anything_v2/` | Vendored upstream — Depth Anything V2 |
| `video_depth_anything_repo/` | Vendored upstream — Video Depth Anything |

## Design Notes

**Why local-first.** The images people run through QC and organization are often
ones they would not upload anywhere. The CLI defaults to local models; strict
offline mode disables every network-capable path so that guarantee is checkable
rather than promised.

**Why copy instead of move.** A misrouted verdict on a destructive sort loses
the original. Copying is the default and `--move` must be requested explicitly.

**Why a separate `unscored/` bucket.** Conflating "not rated" with "failed"
corrupts any measurement of how the pipeline is performing.

**Why vendored upstreams.** Depth Anything V2 and Video Depth Anything are
pinned in-tree so a model repository moving or changing its API does not break
existing installs. Licenses are tracked in `THIRD_PARTY_NOTICES.md`.

**Why the benchmark reports fail recall.** On a defect-detection task, accuracy
is dominated by the majority class. A model that passes everything can score
well and be useless.

## License

MIT © Frank Santaguida — see [LICENSE](LICENSE).

StereoSift incorporates third-party code, architectures, and model checkpoints.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution, upstream
links, and the role of each component. **Model licenses differ from the licenses
of their implementation code** — verify the specific checkpoint license before
redistribution or commercial use.
