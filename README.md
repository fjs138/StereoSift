# StereoSift

StereoSift is a local image and video toolkit with two jobs:

1. Convert ordinary 2D images and videos into side-by-side (SBS) stereoscopic media.
2. Review image batches and sort copies into `pass`, `warning`, and `fail` folders.

Inference runs locally with PyTorch. Images use Depth Anything V2; videos use the
official Video Depth Anything streaming model so long movies do not accumulate
every frame or depth map in memory.

## Features

- Image and full-resolution video SBS conversion
- Constant-memory streaming for long videos
- Original audio preserved in converted videos
- Apple Silicon MPS, CUDA, and CPU support
- Safe QC mode that copies originals by default
- Optional OpenAI-compatible local vision backend for structure/artifact review

## Installation

Requires Python 3.10+ and FFmpeg support supplied through `imageio-ffmpeg`.

```bash
python3 -m venv venv
source venv/bin/activate
python install_torch.py
pip install -r requirements.txt
```

Model checkpoints are downloaded on first use and stored locally. They are not
committed to Git.

## SBS conversion

The friendly launcher asks for input and output paths:

```bash
sh sbs.sh
```

Direct CLI examples:

```bash
python convert.py --input photo.jpg --output-dir output --yes
python convert.py --input movie.mp4 --output-dir output --video-encoder vits --max-res -1 --yes
```

Video encoder choices are `vits` (small/default), `vitb` (base), and `vitl`
(large). Larger models require substantially more memory.

## Visual judgment

```bash
sh qc.sh
```

Without a vision backend, QC checks exposure, contrast, and visual complexity.
Those basic checks cannot understand structure. For duplicated heads, torsos,
limbs, fused figures, and similar generative artifacts, supply an
OpenAI-compatible vision endpoint such as LM Studio:

```bash
python qc_pipeline.py \
  --input ~/Pictures/to-review \
  --output-dir ~/Pictures/reviewed \
  --backend-url http://127.0.0.1:1234/v1/chat/completions \
  --model llama-3.2-11b-vision-instruct
```

Results include `pass/`, `warning/`, `fail/`, and `report.json`. Originals are
copied. Add `--move` only when destructive sorting is explicitly desired.

## Project layout

- `convert.py` — unified image/video conversion CLI
- `depth_model.py` — Depth Anything V2 loading
- `video_converter.py` — Video Depth Anything streaming and encoding
- `qc_pipeline.py` — visual judgment and routing
- `sbs/sbs.py` — stereoscopic warping algorithms
- `depth_anything_v2/` — image depth model implementation
- `video_depth_anything_repo/` — vendored official video depth implementation
- `tests/` — automated QC tests

## License and upstream work

This project incorporates code and model integrations derived from Depth
Anything V2 and Video Depth Anything. Consult the upstream license files before
commercial use, especially for Base and Large video checkpoints.
