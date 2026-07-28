# Third-Party Notices

StereoSift uses third-party software and model checkpoints. This file records
the projects that are directly used by the application or included in the
repository. It is not a replacement for the full upstream license texts.

## Depth Anything V2

Used for image depth estimation in the Convert workflow.

- Project: <https://github.com/DepthAnything/Depth-Anything-V2>
- Local source: `depth_anything_v2/`
- Paper: <https://arxiv.org/abs/2406.09414>
- Code license: Apache-2.0
- Model licensing: the upstream project identifies the Small checkpoint as
  Apache-2.0 and the Base/Large/Giant checkpoints as CC BY-NC-4.0.
- StereoSift checkpoint source: `yushan777/DepthAnythingV2` on Hugging Face.
  The exact checkpoint selected by the user must be checked independently.

The local implementation contains source derived from the Depth Anything V2
project and includes upstream Apache-2.0/file-level attribution notices where
applicable.

## Video Depth Anything

Used for temporally consistent video depth estimation in the Convert workflow.

- Project: <https://github.com/DepthAnything/Video-Depth-Anything>
- Local source: `video_depth_anything_repo/`
- Paper: <https://arxiv.org/abs/2501.12375>
- Upstream code license: Apache-2.0
- Model licensing: Video Depth Anything Small is Apache-2.0; Base and Large
  checkpoints are CC BY-NC-4.0.
- The repository contains the upstream license at
  `video_depth_anything_repo/LICENSE`.

## Real-ESRGAN

Used for tiled image and video upscaling.

- Project: <https://github.com/xinntao/Real-ESRGAN>
- Model checkpoint: `RealESRGAN_x2plus.pth`
- Checkpoint source:
  <https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1>
- Upstream code license: BSD-3-Clause
- StereoSift contains a compatible RRDBNet implementation rather than
  importing the Real-ESRGAN Python package directly.

## Ultralytics YOLO

Used by the Judge workflow for object detection and human-pose checks.

- Project: <https://github.com/ultralytics/ultralytics>
- Python package: `ultralytics`
- Checkpoints: `yolo11n.pt` and `yolo11n-pose.pt`
- Review the applicable Ultralytics license for the installed package and
  checkpoint before redistribution or commercial deployment. Ultralytics
  provides AGPL-3.0 and separate enterprise licensing options.

## Moondream2

Optional local fallback model for the Judge workflow's structure scan.

- Model repository: <https://huggingface.co/vikhyatk/moondream2>
- Code path: `qc_pipeline.py`
- The model is downloaded only when the optional deep scan is enabled.
- Review the model repository's current license and terms for the pinned
  revision `2025-01-09`.

## Transformers

Used as a model-loading runtime by the optional Moondream2 fallback and the
experimental Qwen benchmark.

- Project: <https://github.com/huggingface/transformers>
- Package: `transformers`
- License: Apache-2.0
- Transformers is a runtime dependency; StereoSift does not redistribute the
  Transformers package itself.

## PyTorch

Used as the tensor and inference runtime throughout StereoSift.

- Project: <https://github.com/pytorch/pytorch>
- Packages: `torch`, `torchvision`, and optionally `torchaudio`
- License: BSD-style license; consult the installed distribution's license
  files for the exact version.

## Experimental Qwen benchmark

`qwen_structure.py` and `tools/benchmark_structure.py` are reference/experimental
code and are not part of the normal GUI routing.

- Model family: <https://huggingface.co/Qwen>
- Default local checkpoint: `Qwen/Qwen3-VL-4B-Instruct`
- The model's own license and acceptable-use terms apply separately.
- Optional OpenAI-compatible backends such as LM Studio or oMLX are external
  user-selected services and models; StereoSift does not ship those models.

## Depth Anything 3

Depth Anything 3 is not currently used by StereoSift. It should not be listed
as a dependency or attribution until its code or checkpoints are adopted.
