<p align="center">
  <img src="assets/stereosift-logo.png" alt="StereoSift logo" width="220">
</p>

<h1 align="center">StereoSift</h1>

<p align="center"><strong>Alpha</strong></p>

<p align="center">
  Local desktop toolkit for 2D-to-3D conversion, AI upscaling, image QC, and vision-assisted folder organization.
</p>

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
| Real-ESRGAN x2plus | Image upscaling | Restores detail in tiled x2 passes before an exact target resize |
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
* Show a GUI for conversion, upscaling, QC, and image organization.
* Sort QC results into `pass`, `warning`, `fail`, and `violations`.
* Copy originals by default, and only move files when asked.
* Support a strict offline mode that stays local and skips model-backed checks.
* Optionally connect to a local OpenAI-compatible vision backend if you already have one running.
* Sort images into arbitrary user-defined categories such as `outdoors, indoors`.

## How It Works

StereoSift has four main paths.

Convert uses Depth Anything V2 for images and Video Depth Anything for video.
Upscale uses Real-ESRGAN x2plus with tiled inference and aspect-safe sizing.
Judge uses pixel checks first, then YOLO for person/object detection, and an
optional structure fallback when you want a second opinion.
Organize asks an OpenAI-compatible vision model to choose exactly one of your
category labels for each image, then copies or moves it into that subfolder.

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
| `qc_pipeline.py` | QC and user-defined image organization pipelines |
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
* the new upscaling tab is in place, using Real-ESRGAN x2plus with a Quest 3 preset
* image upscaling is now a first-class step before SBS conversion when you want cleaner stereo output

That’s basically where it is right now: useful, local-first, and still getting
cleaned up around QC behavior and model choices.

## Progress Journal

This is the short “what we tried, what stuck, and what did not” log so the
project history stays visible without digging through code.

| Try | Result | Decision |
| :-- | :-- | :-- |
| Depth Anything V2 for image conversion | Works well for stills and keeps the SBS path simple | Kept |
| Video Depth Anything for video conversion | Better fit for temporal consistency than frame-by-frame image depth | Kept |
| YOLO11n as the first QC gate | Catches cheap, obvious person-count and structure issues | Kept, but not treated as a full judge |
| Moondream2 as the fallback scan | Helps with trickier structure cases when YOLO is not enough | Kept as the second-pass check |
| Qwen3-VL as an structure judge | Useful for isolated benchmarking, but not stable enough to become the production router | Left experimental |
| Larger local vision backends for Judge | Can produce stronger reasoning, but latency depends heavily on image detail, resize policy, and prompt shape | Kept optional, not the default assumption |
| Smaller or faster vision model swaps | Often answer quickly, but some collapse into generic "looks fine" outputs unless the prompt is tuned for that model | Treated as model-specific, not plug-and-play |
| Strict structured JSON responses | Good for automation, but weaker models may minimize explanation and hide uncertainty behind short answers | Kept for routing, but watched carefully during evaluation |
| More permissive / less over-constrained QC prompts | In several experiments, this produced more useful image-specific reasoning than rigidly over-instructed prompts | Kept as a practical prompt-design lesson |
| High-detail image requests to local backends | Improves signal in some borderline cases, but often costs more latency than the QC task justifies | Now treated as a tradeoff, not an automatic default |
| Strict offline mode | Keeps the QC path fully local when you want zero outbound behavior | Kept as a guardrail |
| Real-ESRGAN x2plus for Quest prep | Gives cleaner pre-SBS upscaling for Quest-oriented images | Kept and exposed as the new Upscale tab |

The main pattern is pretty simple: YOLO is the cheap first pass, Moondream is
the harder fallback, Qwen stays a benchmark tool for now, and Real-ESRGAN is the
image-prep step when you want to feed SBS cleaner source material.

Some extra lessons from the QC experiments:

* Swapping only the model name was rarely enough. Different local vision models needed different prompt pressure, response constraints, and image detail settings.
* Faster models were not automatically worse, but they were more likely to default to generic reassurance unless the request clearly forced evidence-based judgments.
* Very large context or token ceilings were usually not the main bottleneck for Judge. Model size, image preprocessing, and per-image visual detail had a bigger effect on throughput.
* For this project, the best practical results usually came from combining cheap deterministic gates with a second opinion, instead of asking one model to be perfect at everything.
* Throughput matters: a local batch tool can be technically accurate and still feel wrong if each image takes too long to review, so latency is treated as part of quality.

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

Four tabs:

* Convert turns 2D images or videos into SBS 3D. Pick a model size, choose the
  output style, and run it.
* Upscale runs tiled Real-ESRGAN x2plus on one image or a folder. Its default
  Quest 3 preset fits each future eye within 2064×2208, producing SBS up to
  4128×2208; a true 7680 px source option is also available.
* Judge sorts a folder of images into pass, warning, and fail. It shows scores,
  person counts, issues, and structure notes when the optional structure scan is on.
* Judge also has a strict offline toggle that keeps everything local and blocks
  backend/model-backed checks.
* Organize accepts choices such as `outdoors, indoors` or `color, black-and-white`, asks
  your oMLX/LM Studio vision model for the best match, and creates one output
  subfolder per label. It copies by default and can optionally move originals.

The Organize tab writes every model response to `model_responses.log` as it is
received and updates `report.json` after every image. Both files can be opened
while a batch is still running. A failed request is shown as an error, leaves
that source image untouched, and does not stop the rest of the batch.

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

## Image Upscaling

The x2plus checkpoint downloads to `models/` on first use. Aspect ratio is
preserved, images already at the target are not enlarged, and tiled inference
keeps memory use manageable.

```bash
python upscaler.py --input photo.jpg --quest-3-sbs --output-dir output/upscaled
python upscaler.py --input ~/Pictures/batch --long-edge 7680 --output-dir output/8k
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

Results go into `pass/`, `warning/`, `fail/`, `unscored/`, plus a `report.json`
summary and a `model_responses.log` file with human-readable backend output.
Originals are copied by default. Use `--move` only if you really want destructive sorting.

Local judgment is intentionally conservative. Only an explicit, confident
structural `PASS` goes into `pass/`. Clear defects, uncertain verdicts,
malformed responses, and scan failures land in `fail/`, which acts as the
manual review queue. Backend responses that mention a safety/policy
violation are routed to `unscored/` instead so they stay separate from
ordinary structure failures. Exposure and other minor aesthetic issues do not
affect the verdict.

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

### Experimental Qwen Benchmark

`qwen_structure.py` is an isolated structure judge for model evaluation; it is not
part of the production routing policy. It can run either against a local
Qwen-VL checkpoint in the Hugging Face cache or against an OpenAI-compatible
vision backend such as oMLX. For the local offline path, once the default
`Qwen/Qwen3-VL-4B-Instruct` checkpoint is present in the Hugging Face cache,
benchmark a labeled folder fully offline with:

```bash
python tools/benchmark_structure.py \
  --input path/to/images \
  --labels path/to/labels.json \
  --output output/qwen-benchmark.json
```

The label file is a JSON object mapping each filename to `pass` or `fail`.
The report includes latency, errors, accuracy, and fail precision/recall so a
model that misses defects is not hidden behind headline accuracy.

### Optional Backend

If you already have a local vision model running in LM Studio or oMLX, you can
point the Judge tab at it.

* LM Studio: `http://127.0.0.1:1234/v1`
* oMLX: `http://127.0.0.1:8000/v1`

The full `/v1/chat/completions` URL is also accepted. If authentication is on,
paste the Bearer token into the API key field or pass it with `--api-key`.

The standalone Qwen structure judge also accepts the same backend URL pattern,
which makes it easy to benchmark the exact vision model you have loaded in
oMLX.

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
