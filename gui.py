#!/usr/bin/env python3
"""StereoSift — CustomTkinter GUI.

Four main tabs:
  • Judge          — local QC: pass / warning / fail / unscored sorting
  • Organize       — vision-model sorting into user-defined folders
  • Upscale        — images → Quest-ready high resolution
  • Convert        — 2D images / videos → SBS 3D

All heavy work runs in a background thread so the UI stays responsive.
Progress and log output stream back to the main thread via a queue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import json
import os
import platform
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from urllib import error as urllib_error
from urllib import request as urllib_request

import customtkinter as ctk
from media_utils import (
    collect_images,
    collect_videos,
    detect_input_kind,
    relative_output_subdir,
)

# ── appearance ───────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

STATUS_COLORS = {"pass": "#2ecc71", "warning": "#f39c12", "fail": "#e74c3c"}

# Maps human-readable size label → (image model filename suffix, video encoder)
# fp16/fp32 is resolved at runtime based on device.
_SIZE_LABELS   = ["Small", "Base", "Large"]
_IMG_ENCODERS  = {"Small": "vits", "Base": "vitb", "Large": "vitl"}
_VID_ENCODERS  = {"Small": "vits", "Base": "vitb", "Large": "vitl"}
# Seed value only. The real list is discovered from the backend's /models
# endpoint via Refresh; this exists so the menu is never empty on first launch.
_BACKEND_MODEL_CHOICES = [
    "Qwen3.6-35B-A3B-MLX-4bit",
]
_MODEL_SELECTED_PREFIX = "[Selected] "
@dataclass(frozen=True)
class ConvertOptions:
    input_path: str
    output_dir: str
    input_kind: str
    size: str
    method: str
    sbs_mode: str
    depth_scale: int
    sbs_blur: int
    depth_only: bool
    output_format: str
    convergence: float
    video_max_res: int
    video_input_size: int
    video_target_fps: int
    video_preview_seconds: int
    recursive: bool

    @property
    def is_video(self) -> bool:
        return self.input_kind == "video"


def _resolve_img_model(size_label: str, device_type: str) -> str:
    """Return the full model filename for an image depth model."""
    enc      = _IMG_ENCODERS[size_label]
    precision = "fp16" if device_type in ("cuda", "mps") else "fp32"
    return f"depth_anything_v2_{enc}_{precision}.safetensors"


# ── shared helpers ────────────────────────────────────────────────────────────

class _RunCancelled(Exception):
    """Raised when the user cancels a background GUI task."""


def _looks_like_file_path(path: str) -> bool:
    trimmed = path.strip()
    if not trimmed:
        return False
    expanded = os.path.expanduser(trimmed)
    if os.path.isfile(expanded):
        return True
    if os.path.isdir(expanded):
        return False
    basename = os.path.basename(trimmed.rstrip(os.sep))
    return bool(os.path.splitext(basename)[1])


def _input_kind(path: str, *, recursive: bool = False) -> str:
    """Return image, video, mixed, folder, missing, or unknown for a file/folder."""
    kind = detect_input_kind(path, recursive=recursive)
    return "unknown" if kind == "empty" else kind


def _suggest_output_path(path: str, output_suffix: str, *, is_file: bool) -> str:
    if is_file:
        stem = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(os.path.dirname(path), f"{stem}-{output_suffix}")
    return os.path.join(
        os.path.dirname(path),
        f"{os.path.basename(path)}-{output_suffix}",
    )


def _update_output_suggestion(
    output_var: ctk.StringVar | None,
    previous_input: str,
    new_input: str,
    output_suffix: str,
    *,
    previous_is_file: bool | None = None,
    new_is_file: bool | None = None,
) -> None:
    if output_var is None or not new_input:
        return
    current_output = output_var.get().strip()
    previous_suggestion = (
        _suggest_output_path(
            previous_input,
            output_suffix,
            is_file=(
                _looks_like_file_path(previous_input)
                if previous_is_file is None
                else previous_is_file
            ),
        )
        if previous_input
        else ""
    )
    if not current_output or current_output == previous_suggestion:
        output_var.set(
            _suggest_output_path(
                new_input,
                output_suffix,
                is_file=(
                    _looks_like_file_path(new_input)
                    if new_is_file is None
                    else new_is_file
                ),
            )
        )


class _OutputAutofillController:
    """Keep the output path aligned with the input until the user overrides it."""

    def __init__(
        self,
        input_var: ctk.StringVar,
        output_var: ctk.StringVar,
        output_suffix: str,
    ) -> None:
        self._input_var = input_var
        self._output_var = output_var
        self._output_suffix = output_suffix
        self._last_input = input_var.get().strip()
        self._last_suggestion = output_var.get().strip()
        self._auto_enabled = True
        self._setting_output = False
        self._input_var.trace_add("write", self._on_input_change)
        self._output_var.trace_add("write", self._on_output_change)

    def _on_input_change(self, *_args) -> None:
        new_input = self._input_var.get().strip()
        if new_input == self._last_input:
            return
        current_output = self._output_var.get().strip()
        should_update = (
            self._auto_enabled
            or not current_output
            or current_output == self._last_suggestion
        )
        if should_update and new_input:
            self._setting_output = True
            try:
                suggestion = _suggest_output_path(
                    new_input,
                    self._output_suffix,
                    is_file=_looks_like_file_path(new_input),
                )
                self._output_var.set(suggestion)
                self._last_suggestion = suggestion
            finally:
                self._setting_output = False
            self._auto_enabled = True
        self._last_input = new_input

    def _on_output_change(self, *_args) -> None:
        if self._setting_output:
            return
        current_output = self._output_var.get().strip()
        self._auto_enabled = not current_output or current_output == self._last_suggestion


def _respect_worker_controls(
    stop_event: threading.Event,
    pause_event: threading.Event,
) -> None:
    while pause_event.is_set():
        if stop_event.is_set():
            raise _RunCancelled()
        time.sleep(0.1)
    if stop_event.is_set():
        raise _RunCancelled()


def _make_controlled_log(
    log_queue: queue.Queue,
    stop_event: threading.Event,
    pause_event: threading.Event,
):
    def _log(text: str) -> None:
        _respect_worker_controls(stop_event, pause_event)
        log_queue.put(("log", text))
        _respect_worker_controls(stop_event, pause_event)

    return _log


def _browse_file(
    var: ctk.StringVar,
    parent,
    output_var_to_update: ctk.StringVar | None = None,
    output_suffix: str = "judged",
) -> None:
    """Open one file chooser and optionally suggest a sibling output folder."""
    path = filedialog.askopenfilename(parent=parent)
    if path:
        previous_path = var.get().strip()
        var.set(path)
        _update_output_suggestion(
            output_var_to_update,
            previous_path,
            path,
            output_suffix,
            previous_is_file=True,
            new_is_file=True,
        )


def _browse_folder(
    var: ctk.StringVar,
    parent,
    output_var_to_update: ctk.StringVar | None = None,
    output_suffix: str = "judged",
) -> None:
    """Open exactly one folder chooser and optionally suggest a sibling output folder."""
    path = filedialog.askdirectory(parent=parent)
    if path:
        previous_path = var.get().strip()
        var.set(path)
        _update_output_suggestion(
            output_var_to_update,
            previous_path,
            path,
            output_suffix,
            previous_is_file=False,
            new_is_file=False,
        )


def _open_folder(path: str) -> None:
    if not path or not os.path.isdir(path):
        messagebox.showinfo("Not found", "Output folder does not exist yet.")
        return
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", path])
    elif system == "Windows":
        subprocess.Popen(["explorer", path])
    else:
        subprocess.Popen(["xdg-open", path])


def _video_option_int(value: str) -> int:
    value = value.strip().lower()
    if value in {"original", "source", "full"}:
        return -1
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else -1


def _write_json_report(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _split_labels(text: str) -> list[str]:
    parts = [part.strip() for part in text.replace("\n", ",").split(",")]
    return [part for part in parts if part]


def _label_chip_color(label: str) -> str:
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6"]
    idx = sum(ord(ch) for ch in label.lower()) % len(palette)
    return palette[idx]


def _normalize_model_choices(models: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for model in models:
        candidate = str(model).strip()
        if not candidate:
            continue
        folded = candidate.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        normalized.append(candidate)
    return normalized


def _models_url(backend_url: str) -> str:
    """Accept a server root, OpenAI base URL, or full models endpoint."""
    url = backend_url.strip().rstrip("/")
    if url.endswith("/models"):
        return url
    if url.endswith("/chat/completions"):
        return f"{url[:-len('/chat/completions')]}/models"
    if url.endswith("/v1"):
        return f"{url}/models"
    return f"{url}/v1/models"


def _merge_backend_model_choices(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        merged.extend(group)
    return _normalize_model_choices(merged)


def _fetch_backend_model_choices(
    backend_url: str,
    api_key: str | None = None,
    *,
    timeout: float = 2.5,
) -> list[str]:
    """Best-effort model discovery for OpenAI-compatible backends."""
    url = _models_url(backend_url)
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib_request.Request(url, headers=headers, method="GET")
    with urllib_request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    return _normalize_model_choices(
        item.get("id", "")
        for item in data
        if isinstance(item, dict)
    )


def _strip_model_menu_label(value: str) -> str:
    candidate = str(value).strip()
    if candidate.startswith(_MODEL_SELECTED_PREFIX):
        return candidate[len(_MODEL_SELECTED_PREFIX):]
    return candidate


def _decorate_model_menu_values(models: list[str], selected: str) -> list[str]:
    chosen = _strip_model_menu_label(selected)
    return [
        f"{_MODEL_SELECTED_PREFIX}{model}" if model == chosen else model
        for model in models
    ]


def _progress_display(done: int, total: int) -> tuple[float, int]:
    """Return determinate bar fraction and displayed percent.

    In-progress work should never display 100%, which is reserved for completion.
    """
    if total <= 0:
        return 0.0, 0
    frac = min(max(done / total, 0.0), 1.0)
    percent = round(frac * 100)
    if done < total:
        percent = min(percent, 99)
    return frac, percent


# ── shared log / progress panel ──────────────────────────────────────────────

class LogPanel(ctk.CTkFrame):
    """Scrollable log box + determinate/indeterminate progress bar."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._box = ctk.CTkTextbox(self, state="disabled", wrap="word", height=160)
        self._box.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 0))

        self._bar = ctk.CTkProgressBar(
            self,
            mode="indeterminate",
            progress_color="#3b82f6",
            fg_color="#1f2937",
        )
        self._bar.grid(row=1, column=0, sticky="ew", padx=4, pady=(4, 0))
        self._bar.set(0)

        self._lbl = ctk.CTkLabel(self, text="", text_color="#aaa")
        self._lbl.grid(row=2, column=0, sticky="w", padx=6, pady=(0, 4))
        self._timer_lbl = ctk.CTkLabel(self, text="", text_color="#aaa")
        self._timer_lbl.grid(row=2, column=0, sticky="e", padx=6, pady=(0, 4))

        self._started_at: float | None = None
        self._last_total: int = 0
        self._last_done: int = 0

    def log(self, text: str) -> None:
        self._box.configure(state="normal")
        self._box.insert("end", text + "\n")
        self._box.see("end")
        self._box.configure(state="disabled")

    def clear(self) -> None:
        self._box.configure(state="normal")
        self._box.delete("1.0", "end")
        self._box.configure(state="disabled")
        self._lbl.configure(text="")
        self._timer_lbl.configure(text="")
        self._bar.set(0)
        self._started_at = None
        self._last_total = 0
        self._last_done = 0

    def start_spin(self) -> None:
        self._started_at = time.perf_counter()
        self._last_total = 0
        self._last_done = 0
        self._timer_lbl.configure(text="Elapsed: 0:00")
        self._bar.configure(mode="indeterminate")
        self._bar.set(0)
        self._bar.start()

    def _format_seconds(self, seconds: float) -> str:
        seconds = max(0.0, seconds)
        minutes, sec = divmod(int(seconds + 0.5), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{sec:02d}"
        return f"{minutes}:{sec:02d}"

    def _update_timer(self) -> None:
        if self._started_at is None:
            return
        elapsed = time.perf_counter() - self._started_at
        text = f"Elapsed: {self._format_seconds(elapsed)}"
        if self._last_total > 0 and self._last_done > 0:
            rate = elapsed / self._last_done
            remaining = max(0, self._last_total - self._last_done)
            text += f"  |  ETA: {self._format_seconds(rate * remaining)}"
        self._timer_lbl.configure(text=text)

    def set_progress(self, done: int, total: int, label: str | None = None) -> None:
        self._last_done = done
        self._last_total = total
        if total > 0:
            frac, percent = _progress_display(done, total)
            self._bar.configure(mode="determinate")
            self._bar.stop()
            self._bar.set(frac)
            prefix = f"{label}: " if label else ""
            self._lbl.configure(text=f"{prefix}{done} / {total}  ({percent}%)")
        self._update_timer()

    def stop(self, *, completed: bool = True) -> None:
        self._bar.stop()
        self._bar.configure(mode="determinate")
        if completed:
            self._bar.set(1)
        if self._started_at is not None:
            elapsed = time.perf_counter() - self._started_at
            self._timer_lbl.configure(text=f"Elapsed: {self._format_seconds(elapsed)}")


# ── Convert tab ───────────────────────────────────────────────────────────────

class ConvertTab(ctk.CTkFrame):
    """2D → SBS 3D conversion for images and videos.

    Images use DepthAnythingV2 (frame-level model).
    Videos use Video Depth Anything (temporal streaming model).
    The selected input path determines which handler and options are shown.
    """

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._run_locked_widgets: list[tk.Widget] = []
        self._build()
        self._poll()

    def _build(self):
        # ── paths ─────────────────────────────────────────────────────────────
        paths = ctk.CTkFrame(self)
        paths.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        paths.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(paths, text="Input (file or folder)", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=5)
        self._input_var = ctk.StringVar()
        self._input_entry = ctk.CTkEntry(paths, textvariable=self._input_var)
        self._input_entry.grid(row=0, column=1, sticky="ew", pady=5)
        self._input_file_btn = ctk.CTkButton(paths, text="File…", width=65,
                                             command=lambda: _browse_file(self._input_var, self))
        self._input_file_btn.grid(
            row=0, column=2, padx=(6, 3), pady=5)
        self._input_folder_btn = ctk.CTkButton(paths, text="Folder…", width=70,
                                               command=lambda: _browse_folder(self._input_var, self))
        self._input_folder_btn.grid(
            row=0, column=3, padx=(3, 8), pady=5)

        ctk.CTkLabel(paths, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        self._output_var = ctk.StringVar(
            value=os.path.join(os.getcwd(), "output"))
        self._output_entry = ctk.CTkEntry(paths, textvariable=self._output_var)
        self._output_entry.grid(row=1, column=1, sticky="ew", pady=5)
        self._output_browse_btn = ctk.CTkButton(paths, text="Browse", width=80,
                                                command=lambda: _browse_folder(self._output_var, self))
        self._output_browse_btn.grid(
            row=1, column=2, columnspan=2, padx=(6, 8), pady=5)
        self._output_autofill = _OutputAutofillController(
            self._input_var,
            self._output_var,
            "converted",
        )
        self._recursive_var = ctk.BooleanVar(value=False)
        self._recursive_chk = ctk.CTkCheckBox(
            paths,
            text="Include files in subfolders (preserve structure)",
            variable=self._recursive_var,
            command=self._refresh_input_type,
        )
        self._recursive_chk.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        # ── options ───────────────────────────────────────────────────────────
        opts = ctk.CTkFrame(self)
        opts.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        for c in range(5):
            opts.grid_columnconfigure(c, weight=1)

        # Row 0 — labels
        for col, text in enumerate(["Input type", "Model size", "Output format",
                                     "Method", "Viewing mode"]):
            ctk.CTkLabel(opts, text=text).grid(
                row=0, column=col, padx=8, pady=(8, 2))

        # Input type is detected from the selected file/folder.
        self._input_type_var = ctk.StringVar(value="Pick an input")
        self._input_type_lbl = ctk.CTkLabel(
            opts,
            textvariable=self._input_type_var,
            text_color="#ddd",
            anchor="center",
        )
        self._input_type_lbl.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="ew")

        # Model size picker (Small / Base / Large)
        self._size_var = ctk.StringVar(value="Large")
        self._size_menu = ctk.CTkOptionMenu(opts, variable=self._size_var,
                                            values=_SIZE_LABELS)
        self._size_menu.grid(
            row=1, column=1, padx=8, pady=(0, 8), sticky="ew")

        # Model description label — updates with mode
        self._model_lbl = ctk.CTkLabel(opts, text="", text_color="#888",
                                        font=ctk.CTkFont(size=11))
        self._model_lbl.grid(row=2, column=0, columnspan=2,
                              padx=8, sticky="w", pady=(0, 4))

        # Output format — SBS, anaglyph, or both
        self._output_format_var = ctk.StringVar(value="sbs")
        self._fmt_menu = ctk.CTkOptionMenu(
            opts, variable=self._output_format_var,
            values=["sbs", "anaglyph", "both"],
            command=self._on_format_change)
        self._fmt_menu.grid(row=1, column=2, padx=8, pady=(0, 8), sticky="ew")

        # Method
        self._method_var = ctk.StringVar(value="mesh_warping")
        self._method_menu = ctk.CTkOptionMenu(opts, variable=self._method_var,
                                              values=["mesh_warping", "grid_sampling"])
        self._method_menu.grid(
            row=1, column=3, padx=8, pady=(0, 8), sticky="ew")

        # Viewing mode (hidden when anaglyph-only)
        self._sbs_mode_var = ctk.StringVar(value="parallel")
        self._sbs_mode_menu = ctk.CTkOptionMenu(
            opts, variable=self._sbs_mode_var,
            values=["parallel", "cross-eyed"])
        self._sbs_mode_menu.grid(row=1, column=4, padx=8, pady=(0, 8), sticky="ew")

        # Depth-only checkbox on its own row
        self._depth_only_var = ctk.BooleanVar(value=False)
        self._depth_only_chk = ctk.CTkCheckBox(opts, text="Debug: save depth map instead of SBS",
                                               variable=self._depth_only_var)
        self._depth_only_chk.grid(
            row=2, column=2, columnspan=3, padx=8, pady=(0, 4), sticky="w")

        # Row 3 — sliders
        ctk.CTkLabel(opts, text="3D strength", anchor="w").grid(
            row=3, column=0, padx=8, sticky="w")
        self._depth_scale_var = ctk.IntVar(value=40)
        self._depth_scale_slider = ctk.CTkSlider(opts, from_=10, to=100, number_of_steps=90,
                                                 variable=self._depth_scale_var)
        self._depth_scale_slider.grid(
            row=4, column=0, columnspan=2, padx=8, sticky="ew", pady=(0, 8))
        self._ds_lbl = ctk.CTkLabel(opts, text="40")
        self._ds_lbl.grid(row=4, column=2, padx=4, sticky="w")
        self._depth_scale_var.trace_add(
            "write", lambda *_: self._ds_lbl.configure(
                text=str(self._depth_scale_var.get())))
        self._auto_video_strength_active = False

        ctk.CTkLabel(opts, text="Depth blur", anchor="w").grid(
            row=3, column=3, padx=8, sticky="w")
        self._blur_var = ctk.IntVar(value=7)
        self._blur_slider = ctk.CTkSlider(opts, from_=3, to=15, number_of_steps=6,
                                          variable=self._blur_var)
        self._blur_slider.grid(
            row=4, column=3, padx=8, sticky="ew", pady=(0, 8))
        self._blur_lbl = ctk.CTkLabel(opts, text="7")
        self._blur_lbl.grid(row=4, column=4, padx=4, sticky="w")
        self._blur_var.trace_add(
            "write", lambda *_: self._blur_lbl.configure(
                text=str(self._blur_var.get())))

        # Convergence (anaglyph only — shown/hidden by format)
        self._conv_row_lbl = ctk.CTkLabel(
            opts, text="Convergence (anaglyph)", anchor="w")
        self._conv_row_lbl.grid(row=5, column=0, columnspan=2,
                                 padx=8, sticky="w", pady=(4, 0))
        self._conv_var = ctk.DoubleVar(value=0.5)
        self._conv_slider = ctk.CTkSlider(
            opts, from_=0.0, to=1.0, number_of_steps=20,  # type: ignore
            variable=self._conv_var)
        self._conv_slider.grid(row=6, column=0, columnspan=2,
                               padx=8, sticky="ew", pady=(0, 8))
        self._conv_lbl = ctk.CTkLabel(opts, text="0.5")
        self._conv_lbl.grid(row=6, column=2, padx=4, sticky="w")
        self._conv_var.trace_add(
            "write", lambda *_: self._conv_lbl.configure(
                text=f"{self._conv_var.get():.2f}"))

        # Video-specific controls. Hidden for image mode.
        self._video_opts = ctk.CTkFrame(self)
        self._video_opts.grid_columnconfigure((0, 1, 2, 3), weight=1)
        for col, text in enumerate(["Video max size", "Depth input", "Output FPS", "Preview"]):
            ctk.CTkLabel(self._video_opts, text=text).grid(
                row=0, column=col, padx=8, pady=(8, 2))
        self._video_max_res_var = ctk.StringVar(value="Original")
        self._video_max_res_menu = ctk.CTkOptionMenu(
            self._video_opts,
            variable=self._video_max_res_var,
            values=["Original", "720", "1080", "1280"],
        )
        self._video_max_res_menu.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="ew")
        self._video_input_size_var = ctk.StringVar(value="518")
        self._video_input_size_menu = ctk.CTkOptionMenu(
            self._video_opts,
            variable=self._video_input_size_var,
            values=["392", "518"],
        )
        self._video_input_size_menu.grid(row=1, column=1, padx=8, pady=(0, 8), sticky="ew")
        self._video_target_fps_var = ctk.StringVar(value="Original")
        self._video_target_fps_menu = ctk.CTkOptionMenu(
            self._video_opts,
            variable=self._video_target_fps_var,
            values=["Original", "24", "30"],
        )
        self._video_target_fps_menu.grid(row=1, column=2, padx=8, pady=(0, 8), sticky="ew")
        self._video_preview_var = ctk.StringVar(value="Full video")
        self._video_preview_menu = ctk.CTkOptionMenu(
            self._video_opts,
            variable=self._video_preview_var,
            values=["Full video", "First 5 seconds"],
        )
        self._video_preview_menu.grid(row=1, column=3, padx=8, pady=(0, 8), sticky="ew")
        self._quest_tip_lbl = ctk.CTkLabel(
            self._video_opts,
            text=(
                "Quest tip: videos are saved as *_SBS_LR.*. If the headset/player opens "
                "one flat, choose SBS / Left-Right 3D in the player."
            ),
            text_color="#aaa",
            wraplength=780,
            justify="left",
        )
        self._quest_tip_lbl.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))
        self._run_locked_widgets.extend([
            self._input_entry, self._input_file_btn, self._input_folder_btn,
            self._output_entry, self._output_browse_btn,
            self._recursive_chk,
            self._size_menu, self._fmt_menu, self._method_menu, self._sbs_mode_menu,
            self._depth_only_chk, self._depth_scale_slider, self._blur_slider,
            self._conv_slider, self._video_max_res_menu, self._video_input_size_menu,
            self._video_target_fps_menu, self._video_preview_menu,
        ])

        self._input_var.trace_add("write", lambda *_: self._refresh_input_type())
        self._refresh_input_type()
        self._on_format_change("sbs")    # hide convergence initially

        # ── log + button ──────────────────────────────────────────────────────
        self._log = LogPanel(self)
        self._log.grid(row=3, column=0, sticky="nsew", padx=12, pady=4)
        self.grid_rowconfigure(3, weight=1)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=4, column=0, padx=12, pady=(4, 12), sticky="ew")
        btn_row.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._run_btn = ctk.CTkButton(
            btn_row, text="Convert", height=38, command=self._run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self._pause_btn = ctk.CTkButton(
            btn_row, text="Pause", height=38, command=self._toggle_pause,
            state="disabled", fg_color="#444", hover_color="#555",
        )
        self._pause_btn.grid(row=0, column=1, padx=4, sticky="ew")
        self._cancel_btn = ctk.CTkButton(
            btn_row, text="Cancel", height=38, command=self._cancel,
            state="disabled", fg_color="#cc2222", hover_color="#dd3333",
        )
        self._cancel_btn.grid(row=0, column=2, padx=(4, 0), sticky="ew")
        ctk.CTkButton(
            btn_row, text="Open output folder", height=38,
            fg_color="#444", hover_color="#555",
            command=lambda: _open_folder(self._output_var.get().strip()),
        ).grid(row=0, column=3, padx=(4, 0), sticky="ew")

    def _refresh_input_type(self) -> None:
        kind = self._detected_input_kind()
        if kind == "video":
            self._input_type_var.set("Video detected")
            self._apply_input_kind("video")
        elif kind == "mixed":
            self._input_type_var.set("Mixed folder — images and videos selected")
            self._apply_input_kind("mixed")
        elif kind == "image":
            self._input_type_var.set("Image detected")
            self._apply_input_kind("image")
        elif kind == "folder":
            self._input_type_var.set("No images/videos found")
            self._apply_input_kind("image")
        elif kind == "missing":
            self._input_type_var.set("Input not found")
            self._apply_input_kind("image")
        else:
            self._input_type_var.set("Pick an input")
            self._apply_input_kind("image")

    def _detected_input_kind(self) -> str:
        return _input_kind(self._input_var.get(), recursive=self._recursive_var.get())

    def _apply_input_kind(self, kind: str) -> None:
        if kind == "image":
            self._model_lbl.configure(
                text="DepthAnythingV2 — static image depth model")
            self._fmt_menu.configure(state="normal")
            self._video_opts.grid_forget()
            if self._auto_video_strength_active and self._depth_scale_var.get() == 70:
                self._depth_scale_var.set(40)
            self._auto_video_strength_active = False
        else:
            self._model_lbl.configure(
                text="Video Depth Anything — temporal streaming model")
            self._video_opts.grid(row=2, column=0, sticky="ew", padx=12, pady=4)
            # Videos are rendered as Quest-ready left/right SBS.  The red-cyan
            # anaglyph formatter is image-only, so keep the video path honest.
            self._output_format_var.set("sbs")
            self._fmt_menu.configure(state="disabled")
            if self._depth_scale_var.get() == 40:
                self._depth_scale_var.set(70)
                self._auto_video_strength_active = True
        self._on_format_change(self._output_format_var.get())

    def _on_format_change(self, value: str) -> None:
        """Show convergence slider for anaglyph; hide viewing mode for anaglyph-only."""
        show_conv = value in ("anaglyph", "both")
        show_sbs_mode = value in ("sbs", "both")
        state_conv = "normal" if show_conv else "disabled"
        state_sbs  = "normal" if show_sbs_mode else "disabled"
        self._conv_slider.configure(state=state_conv)
        self._conv_row_lbl.configure(
            text_color="#ccc" if show_conv else "#555")
        self._conv_lbl.configure(
            text_color="#ccc" if show_conv else "#555")
        self._sbs_mode_menu.configure(state=state_sbs)

    # ── worker ────────────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return
        inp = self._input_var.get().strip()
        out = self._output_var.get().strip()
        if not inp:
            messagebox.showwarning("Missing input",
                                   "Please select an input file or folder.")
            return
        if not out:
            messagebox.showwarning("Missing output",
                                   "Please select an output folder.")
            return
        self._running = True
        self._stop_event.clear()
        self._pause_event.clear()
        self._set_run_controls_enabled(False)
        self._run_btn.configure(state="disabled", text="Running…")
        self._pause_btn.configure(state="normal", text="Pause")
        self._cancel_btn.configure(state="normal", text="Cancel")
        self._log.clear()
        self._log.start_spin()
        preview_seconds = 5 if self._video_preview_var.get() == "First 5 seconds" else 0
        detected_kind = self._detected_input_kind()
        input_kind = detected_kind if detected_kind in {"image", "video", "mixed"} else "image"
        opts = ConvertOptions(
            input_path=inp,
            output_dir=out,
            input_kind=input_kind,
            size=self._size_var.get(),
            method=self._method_var.get(),
            sbs_mode=self._sbs_mode_var.get(),
            depth_scale=self._depth_scale_var.get(),
            sbs_blur=self._blur_var.get(),
            depth_only=self._depth_only_var.get(),
            output_format=self._output_format_var.get(),
            convergence=round(self._conv_var.get(), 2),
            video_max_res=_video_option_int(self._video_max_res_var.get()),
            video_input_size=_video_option_int(self._video_input_size_var.get()),
            video_target_fps=_video_option_int(self._video_target_fps_var.get()),
            video_preview_seconds=preview_seconds,
            recursive=self._recursive_var.get(),
        )
        threading.Thread(
            target=self._worker,
            args=(opts, self._stop_event, self._pause_event),
            daemon=True,
        ).start()

    def _toggle_pause(self):
        if not self._running:
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._pause_btn.configure(text="Pause")
            self._log.log("Resuming conversion…")
        else:
            self._pause_event.set()
            self._pause_btn.configure(text="Resume")
            self._log.log("Pausing conversion after the current step…")

    def _cancel(self):
        if not self._running:
            return
        self._stop_event.set()
        self._cancel_btn.configure(state="disabled", text="Cancelling…")
        self._pause_btn.configure(state="disabled")
        self._log.log("Cancelling conversion…")

    def _worker(
        self,
        opts: ConvertOptions,
        stop_event: threading.Event,
        pause_event: threading.Event,
    ):
        q = self._q
        control = lambda: _respect_worker_controls(stop_event, pause_event)
        log = _make_controlled_log(q, stop_event, pause_event)
        try:
            from convert import get_device, convert_one

            device    = get_device()
            image_files = collect_images(opts.input_path, recursive=opts.recursive) if not opts.is_video else []
            video_files = collect_videos(opts.input_path, recursive=opts.recursive) if opts.input_kind in {"video", "mixed"} else []
            total_files = len(image_files) + len(video_files)
            if total_files == 0:
                raise ValueError(f"No supported images or videos found in {opts.input_path}")
            started_at = time.perf_counter()
            report: dict[str, object] = {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "options": asdict(opts),
                "device": str(device),
                "files": [],
            }
            processed_files = 0

            if image_files:
                from depth_model import load_depth_model
                model_name = _resolve_img_model(opts.size, device.type)
                q.put(("log",
                       f"Loading DepthAnythingV2 — {opts.size} ({model_name})…"))
                model, dtype, is_metric = load_depth_model(model_name, device)
                q.put(("log", f"Found {len(image_files)} image(s)"))
                for path in image_files:
                    control()
                    q.put(("progress", processed_files, total_files))
                    q.put(("log",
                           f"[{processed_files + 1}/{total_files}] {path}"))
                    file_output = os.path.join(
                        opts.output_dir, relative_output_subdir(opts.input_path, path)
                    )
                    ok = convert_one(
                        model, path, file_output,
                        device, dtype, is_metric,
                        depth_only=opts.depth_only,
                        depth_input_scale=0.5,
                        sbs_method=opts.method,
                        depth_scale=opts.depth_scale,
                        sbs_mode=opts.sbs_mode,
                        sbs_blur=opts.sbs_blur,
                        output_format=opts.output_format,
                        convergence=opts.convergence,
                        log=log,
                        control=control,
                    )
                    report["files"].append({
                        "input": path,
                        "type": "image",
                        "success": ok,
                    })
                    processed_files += 1
                    control()
                    q.put(("progress", processed_files, total_files))

            if video_files:
                from video_converter import (
                    load_video_depth_model, convert_video_to_sbs)
                encoder = _VID_ENCODERS[opts.size]
                q.put(("log",
                       f"Loading Video Depth Anything — {opts.size} ({encoder})…"))
                model, dtype, is_metric = load_video_depth_model(
                    encoder=encoder, device=device)
                q.put(("log", f"Found {len(video_files)} video(s)"))
                max_res = opts.video_max_res
                if device.type == "mps" and max_res == 1280:
                    max_res = 720
                    q.put(("log", "Apple Silicon safety limit: using 720 video max size."))
                target_fps = opts.video_target_fps
                if opts.video_preview_seconds > 0:
                    q.put(("log", f"Preview mode: processing about the first {opts.video_preview_seconds} seconds."))
                for video_index, path in enumerate(video_files):
                    control()
                    q.put(("progress", processed_files, total_files))
                    q.put(("log",
                           f"[{processed_files + 1}/{total_files}] {path}"))
                    file_output = os.path.join(
                        opts.output_dir, relative_output_subdir(opts.input_path, path)
                    )
                    ok = convert_video_to_sbs(
                        video_path=path,
                        output_dir=file_output,
                        model=model, device=device,
                        dtype=dtype, is_metric=is_metric,
                        sbs_method=opts.method,
                        depth_scale=opts.depth_scale,
                        sbs_mode=opts.sbs_mode,
                        sbs_blur=opts.sbs_blur,
                        max_res=max_res,
                        input_size=opts.video_input_size,
                        target_fps=target_fps,
                        max_seconds=opts.video_preview_seconds if opts.video_preview_seconds > 0 else -1,
                        depth_only=opts.depth_only,
                        log=log,
                        control=control,
                        progress=lambda done, total, file_index=video_index: q.put((
                            "progress_detail",
                            done,
                            total,
                            f"Video {file_index + 1}/{len(video_files)} frames",
                        )),
                    )
                    suffix = "depth" if opts.depth_only else "SBS_LR"
                    output_path = os.path.join(
                        file_output,
                        f"{os.path.splitext(os.path.basename(path))[0]}_{suffix}{os.path.splitext(path)[1]}",
                    )
                    report["files"].append({
                        "input": path,
                        "type": "video",
                        "output": output_path,
                        "success": ok,
                    })
                    processed_files += 1
                    control()
                    q.put(("progress", processed_files, total_files))

            report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            report["seconds"] = round(time.perf_counter() - started_at, 3)
            report_path = os.path.join(opts.output_dir, "conversion_report.json")
            _write_json_report(report_path, report)
            q.put(("log", f"Report saved: {report_path}"))
            q.put(("done", f"Finished — {total_files} file(s) converted."))
        except _RunCancelled:
            q.put(("stopped", "Conversion cancelled by user."))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            q.put(("log", tb))
            q.put(("error", tb.splitlines()[-1]))

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._run_locked_widgets:
            widget.configure(state=state)

    def _poll(self):
        try:
            while True:
                msg  = self._q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log.log(msg[1])
                elif kind == "progress":
                    self._log.set_progress(msg[1], msg[2])
                elif kind == "progress_detail":
                    self._log.set_progress(msg[1], msg[2], msg[3])
                elif kind in {"done", "stopped"}:
                    self._log.stop(completed=kind == "done")
                    self._log.log(msg[1])
                    self._set_run_controls_enabled(True)
                    self._refresh_input_type()
                    self._run_btn.configure(state="normal", text="Convert")
                    self._pause_btn.configure(state="disabled", text="Pause")
                    self._cancel_btn.configure(state="disabled", text="Cancel")
                    self._running = False
                elif kind == "error":
                    self._log.stop(completed=False)
                    messagebox.showerror("Error", msg[1])
                    self._set_run_controls_enabled(True)
                    self._refresh_input_type()
                    self._run_btn.configure(state="normal", text="Convert")
                    self._pause_btn.configure(state="disabled", text="Pause")
                    self._cancel_btn.configure(state="disabled", text="Cancel")
                    self._running = False
        except queue.Empty:
            pass
        self.after(100, self._poll)


# ── Upscale tab ───────────────────────────────────────────────────────────────

class UpscaleTab(ctk.CTkFrame):
    """Real-ESRGAN x2plus batch upscaling with a VR-safe target preset."""

    TARGETS = {
        "Quest 3 SBS (2064×2208 per eye)": (2064, 2208),
        "True 8K source (7680 px)": 7680,
    }

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._run_locked_widgets: list[tk.Widget] = []
        self._build()
        self._poll()

    def _build(self):
        paths = ctk.CTkFrame(self)
        paths.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        paths.grid_columnconfigure(1, weight=1)
        self._input_var = ctk.StringVar()
        self._output_var = ctk.StringVar(value=os.path.join(os.getcwd(), "output", "upscaled"))
        ctk.CTkLabel(paths, text="Input (file or folder)", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=5)
        self._input_entry = ctk.CTkEntry(paths, textvariable=self._input_var)
        self._input_entry.grid(row=0, column=1, sticky="ew", pady=5)
        self._input_file_btn = ctk.CTkButton(paths, text="File…", width=65,
                                             command=lambda: _browse_file(self._input_var, self))
        self._input_file_btn.grid(
            row=0, column=2, padx=(6, 3), pady=5)
        self._input_folder_btn = ctk.CTkButton(paths, text="Folder…", width=70,
                                               command=lambda: _browse_folder(self._input_var, self))
        self._input_folder_btn.grid(
            row=0, column=3, padx=(3, 8), pady=5)
        ctk.CTkLabel(paths, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        self._output_entry = ctk.CTkEntry(paths, textvariable=self._output_var)
        self._output_entry.grid(row=1, column=1, sticky="ew", pady=5)
        self._output_browse_btn = ctk.CTkButton(paths, text="Browse", width=80,
                                                command=lambda: _browse_folder(self._output_var, self))
        self._output_browse_btn.grid(
            row=1, column=2, columnspan=2, padx=(6, 8), pady=5)
        self._output_autofill = _OutputAutofillController(
            self._input_var,
            self._output_var,
            "upscaled",
        )
        self._input_type_var = ctk.StringVar(value="Pick an input")
        ctk.CTkLabel(
            paths,
            textvariable=self._input_type_var,
            text_color="#aaa",
            anchor="w",
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))
        self._recursive_var = ctk.BooleanVar(value=False)
        self._recursive_chk = ctk.CTkCheckBox(
            paths,
            text="Include files in subfolders (preserve structure)",
            variable=self._recursive_var,
            command=self._refresh_input_type,
        )
        self._recursive_chk.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))
        self._input_var.trace_add("write", lambda *_: self._refresh_input_type())

        opts = ctk.CTkFrame(self)
        opts.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        for col in range(3):
            opts.grid_columnconfigure(col, weight=1)
        ctk.CTkLabel(opts, text="Target").grid(row=0, column=0, padx=8, pady=(8, 2))
        ctk.CTkLabel(opts, text="Tile size").grid(row=0, column=1, padx=8, pady=(8, 2))
        ctk.CTkLabel(opts, text="Format").grid(row=0, column=2, padx=8, pady=(8, 2))
        self._target_var = ctk.StringVar(value="Quest 3 SBS (2064×2208 per eye)")
        self._target_menu = ctk.CTkOptionMenu(opts, variable=self._target_var,
                                              values=list(self.TARGETS))
        self._target_menu.grid(
            row=1, column=0, padx=8, pady=(0, 8), sticky="ew")
        self._tile_var = ctk.StringVar(value="256")
        self._tile_menu = ctk.CTkOptionMenu(opts, variable=self._tile_var,
                                            values=["128", "256", "384", "512"])
        self._tile_menu.grid(
            row=1, column=1, padx=8, pady=(0, 8), sticky="ew")
        self._format_var = ctk.StringVar(value="PNG")
        self._format_menu = ctk.CTkOptionMenu(opts, variable=self._format_var,
                                              values=["PNG", "JPEG"])
        self._format_menu.grid(
            row=1, column=2, padx=8, pady=(0, 8), sticky="ew")
        ctk.CTkLabel(
            opts,
            text=("Images save as PNG/JPEG. Videos stream frame-by-frame through "
                  "Real-ESRGAN and keep audio when possible."),
            text_color="#aaa", wraplength=760, justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 8))
        self._run_locked_widgets.extend([
            self._input_entry, self._input_file_btn, self._input_folder_btn,
            self._output_entry, self._output_browse_btn,
            self._recursive_chk,
            self._target_menu, self._tile_menu, self._format_menu,
        ])
        self._refresh_input_type()

        self._log = LogPanel(self)
        self._log.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="ew")
        buttons.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._run_btn = ctk.CTkButton(buttons, text="Upscale", height=38,
                                      command=self._run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self._pause_btn = ctk.CTkButton(
            buttons, text="Pause", height=38, command=self._toggle_pause,
            state="disabled", fg_color="#444", hover_color="#555",
        )
        self._pause_btn.grid(row=0, column=1, padx=4, sticky="ew")
        self._cancel_btn = ctk.CTkButton(
            buttons, text="Cancel", height=38, command=self._cancel,
            state="disabled", fg_color="#cc2222", hover_color="#dd3333",
        )
        self._cancel_btn.grid(row=0, column=2, padx=4, sticky="ew")
        ctk.CTkButton(buttons, text="Open output folder", height=38,
                      fg_color="#444", hover_color="#555",
                      command=lambda: _open_folder(self._output_var.get().strip())).grid(
            row=0, column=3, padx=(4, 0), sticky="ew")

    def _run(self):
        if self._running:
            return
        inp, out = self._input_var.get().strip(), self._output_var.get().strip()
        if not inp or not out:
            messagebox.showwarning("Missing path", "Please select input and output paths.")
            return
        self._running = True
        self._stop_event.clear()
        self._pause_event.clear()
        self._set_run_controls_enabled(False)
        self._run_btn.configure(state="disabled", text="Upscaling…")
        self._pause_btn.configure(state="normal", text="Pause")
        self._cancel_btn.configure(state="normal", text="Cancel")
        self._log.clear()
        self._log.start_spin()
        opts = (inp, out, self.TARGETS[self._target_var.get()],
                int(self._tile_var.get()), self._format_var.get(),
                self._recursive_var.get())
        threading.Thread(
            target=self._worker,
            args=(*opts, self._stop_event, self._pause_event),
            daemon=True,
        ).start()

    def _refresh_input_type(self) -> None:
        kind = _input_kind(self._input_var.get(), recursive=self._recursive_var.get())
        if kind == "video":
            self._input_type_var.set("Video detected — will upscale video frames and preserve audio.")
            self._format_menu.configure(state="disabled")
        elif kind == "mixed":
            self._input_type_var.set("Mixed folder — will upscale images and videos.")
            self._format_menu.configure(state="disabled")
        elif kind == "image":
            self._input_type_var.set("Image detected — will upscale image file(s).")
            self._format_menu.configure(state="normal")
        elif kind == "folder":
            self._input_type_var.set("No supported images/videos found.")
            self._format_menu.configure(state="normal")
        elif kind == "missing":
            self._input_type_var.set("Input not found.")
            self._format_menu.configure(state="normal")
        else:
            self._input_type_var.set("Pick an input")
            self._format_menu.configure(state="normal")

    def _toggle_pause(self):
        if not self._running:
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._pause_btn.configure(text="Pause")
            self._log.log("Resuming upscale…")
        else:
            self._pause_event.set()
            self._pause_btn.configure(text="Resume")
            self._log.log("Pausing upscale after the current step…")

    def _cancel(self):
        if not self._running:
            return
        self._stop_event.set()
        self._cancel_btn.configure(state="disabled", text="Cancelling…")
        self._pause_btn.configure(state="disabled")
        self._log.log("Cancelling upscale…")

    def _worker(
        self,
        inp,
        out,
        target,
        tile,
        output_format,
        recursive,
        stop_event: threading.Event,
        pause_event: threading.Event,
    ):
        control = lambda: _respect_worker_controls(stop_event, pause_event)
        log = _make_controlled_log(self._q, stop_event, pause_event)
        try:
            from upscaler import (
                ensure_model,
                RealESRGANx2,
                upscale_file,
                upscale_video,
            )
            kind = _input_kind(inp, recursive=recursive)
            image_files = collect_images(inp, recursive=recursive) if kind in {"image", "mixed"} else []
            video_files = collect_videos(inp, recursive=recursive) if kind in {"video", "mixed"} else []
            total_files = len(image_files) + len(video_files)
            if total_files == 0:
                raise ValueError(f"No supported images or videos found in {inp}")
            model_path = ensure_model(log=log, control=control)
            log(f"Loading Real-ESRGAN x2plus (tile {tile})…")
            engine = RealESRGANx2(model_path, tile=tile)
            log(
                f"Found {len(image_files)} image(s) and {len(video_files)} video(s); "
                f"device: {engine.device}"
            )
            processed_files = 0
            for path in image_files:
                control()
                self._q.put(("progress", processed_files, total_files))
                log(f"[{processed_files + 1}/{total_files}] {path}")
                file_output = os.path.join(out, relative_output_subdir(inp, path))
                target_box = target if isinstance(target, tuple) else None
                long_edge = target if isinstance(target, int) else max(target)
                upscale_file(path, file_output, engine, long_edge, output_format, log,
                             target_box=target_box, control=control)
                processed_files += 1
                control()
                self._q.put(("progress", processed_files, total_files))
            for video_index, path in enumerate(video_files):
                control()
                self._q.put(("progress", processed_files, total_files))
                log(f"[{processed_files + 1}/{total_files}] {path}")
                file_output = os.path.join(out, relative_output_subdir(inp, path))
                target_box = target if isinstance(target, tuple) else None
                long_edge = target if isinstance(target, int) else max(target)
                upscale_video(
                    path,
                    file_output,
                    engine,
                    long_edge,
                    target_box=target_box,
                    log=log,
                    control=control,
                    progress=lambda done, total, file_index=video_index: self._q.put((
                        "progress_detail",
                        done,
                        total,
                        f"Video {file_index + 1}/{len(video_files)} frames",
                    )),
                )
                processed_files += 1
                control()
                self._q.put(("progress", processed_files, total_files))
            self._q.put(("done", f"Finished — {total_files} file(s) upscaled."))
        except _RunCancelled:
            self._q.put(("stopped", "Upscaling cancelled by user."))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            self._q.put(("log", tb))
            self._q.put(("error", tb.splitlines()[-1]))

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._run_locked_widgets:
            widget.configure(state=state)

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "log":
                    self._log.log(msg[1])
                elif msg[0] == "progress":
                    self._log.set_progress(msg[1], msg[2])
                elif msg[0] == "progress_detail":
                    self._log.set_progress(msg[1], msg[2], msg[3])
                elif msg[0] in ("done", "stopped", "error"):
                    self._log.stop(completed=msg[0] == "done")
                    if msg[0] in ("done", "stopped"):
                        self._log.log(msg[1])
                    else:
                        messagebox.showerror("Upscale error", msg[1])
                    self._set_run_controls_enabled(True)
                    self._run_btn.configure(state="normal", text="Upscale")
                    self._pause_btn.configure(state="disabled", text="Pause")
                    self._cancel_btn.configure(state="disabled", text="Cancel")
                    self._refresh_input_type()
                    self._running = False
        except queue.Empty:
            pass
        self.after(100, self._poll)


# ── Judge tab ─────────────────────────────────────────────────────────────────

class JudgeTab(ctk.CTkFrame):
    """QC sorting: pixel checks + YOLO person detection + optional moondream2
    structure deep scan.  All local, all PyTorch, no server required.
    """

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._run_locked_widgets: list[tk.Widget] = []
        self._results: list[dict] = []
        self._result_row = 1
        self._backend_models = list(_BACKEND_MODEL_CHOICES)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._build()
        self._poll()

    def _build(self):
        # ── paths ─────────────────────────────────────────────────────────────
        paths = ctk.CTkFrame(self)
        paths.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        paths.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(paths, text="Input (file or folder)", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=5)
        self._input_var = ctk.StringVar()
        self._output_var = ctk.StringVar()
        self._input_entry = ctk.CTkEntry(paths, textvariable=self._input_var)
        self._input_entry.grid(row=0, column=1, sticky="ew", pady=5)
        self._input_file_btn = ctk.CTkButton(paths, text="File…", width=65,
                                             command=lambda: _browse_file(self._input_var, self))
        self._input_file_btn.grid(row=0, column=2, padx=(6, 3), pady=5)
        self._input_folder_btn = ctk.CTkButton(paths, text="Folder…", width=70,
                                               command=lambda: _browse_folder(self._input_var, self))
        self._input_folder_btn.grid(
            row=0, column=3, padx=(3, 8), pady=5)

        ctk.CTkLabel(paths, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        self._output_entry = ctk.CTkEntry(paths, textvariable=self._output_var)
        self._output_entry.grid(row=1, column=1, sticky="ew", pady=5)
        self._output_browse_btn = ctk.CTkButton(paths, text="Browse", width=80,
                                                command=lambda: _browse_folder(self._output_var, self))
        self._output_browse_btn.grid(
            row=1, column=2, columnspan=2, padx=(6, 8), pady=5)
        self._output_autofill = _OutputAutofillController(
            self._input_var,
            self._output_var,
            "judged",
        )
        self._recursive_var = ctk.BooleanVar(value=False)
        self._recursive_chk = ctk.CTkCheckBox(
            paths,
            text="Include files in subfolders (preserve structure)",
            variable=self._recursive_var,
        )
        self._recursive_chk.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        # ── options ───────────────────────────────────────────────────────────
        opts = ctk.CTkFrame(self)
        opts.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        opts.grid_columnconfigure(1, weight=1)
        opts.grid_columnconfigure(3, weight=1)

        # Info label
        ctk.CTkLabel(
            opts,
            text="YOLO pose catches obvious structural artifacts. The optional vision backend can use oMLX or LM Studio for a stronger second opinion.",
            text_color="#aaa", font=ctk.CTkFont(size=11), anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))

        self._summary_frame = ctk.CTkFrame(opts, fg_color="transparent")
        self._summary_frame.grid(row=1, column=0, columnspan=5,
                                 sticky="ew", padx=8, pady=(2, 4))
        for col in range(5):
            self._summary_frame.grid_columnconfigure(col, weight=1)

        self._processed_var = ctk.StringVar(value="Processed: 0")
        self._pass_var = ctk.StringVar(value="Pass: 0")
        self._warn_var = ctk.StringVar(value="Warning: 0")
        self._fail_var = ctk.StringVar(value="Fail: 0")
        self._unscored_var = ctk.StringVar(value="Unscored: 0")
        self._remaining_var = ctk.StringVar(value="Remaining: 0")

        self._summary_labels = []
        for col, var in enumerate([
            self._processed_var,
            self._pass_var,
            self._warn_var,
            self._fail_var,
            self._unscored_var,
            self._remaining_var,
        ]):
            lbl = ctk.CTkLabel(
                self._summary_frame,
                textvariable=var,
                corner_radius=8,
                fg_color="#232323",
                text_color="#ddd",
            )
            lbl.grid(row=0, column=col, padx=4, sticky="ew")
            self._summary_labels.append(lbl)

        # Move vs copy
        self._move_var = ctk.BooleanVar(value=False)
        self._move_chk = ctk.CTkCheckBox(
            opts,
            text="Move originals (destructive — default copies)",
            variable=self._move_var,
            text_color="#e74c3c",
        )
        self._move_chk.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 6))

        # Deep scan toggle
        self._deep_scan_var = ctk.BooleanVar(value=False)
        self._deep_scan_chk = ctk.CTkCheckBox(
            opts,
            text="Optional moondream2 fallback — warning on suspect structural artifacts, fail only on clear structural errors",
            variable=self._deep_scan_var,
            text_color="#7eb3ff",
        )
        self._deep_scan_chk.grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 2))

        self._strict_offline_var = ctk.BooleanVar(value=False)
        self._strict_offline_chk = ctk.CTkCheckBox(
            opts,
            text="Offline mode — brightness and contrast only (no models or downloads)",
            variable=self._strict_offline_var,
            text_color="#c7d2fe",
        )
        self._strict_offline_chk.grid(row=3, column=2, columnspan=3, sticky="w", padx=8, pady=(0, 2))

        # ── optional backend ──────────────────────────────────────────────────
        adv_toggle = ctk.CTkButton(
            opts, text="▶  Local/remote vision backend (replaces moondream2)",
            fg_color="transparent", hover_color="#333",
            anchor="w", font=ctk.CTkFont(size=11),
            command=self._toggle_advanced,
        )
        adv_toggle.grid(row=4, column=0, columnspan=2,
                        sticky="w", padx=6, pady=(4, 2))
        self._adv_toggle_btn = adv_toggle

        self._adv_frame = ctk.CTkFrame(opts, fg_color="transparent")
        self._adv_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self._adv_frame, text="Backend URL", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=4)
        self._backend_var = ctk.StringVar(value="http://127.0.0.1:8001/v1")
        self._backend_entry = ctk.CTkEntry(
            self._adv_frame, textvariable=self._backend_var,
            placeholder_text="oMLX: http://127.0.0.1:8001/v1",
        )
        self._backend_entry.grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=4
        )

        ctk.CTkLabel(self._adv_frame, text="Model name", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=4)
        self._backend_model_var = ctk.StringVar(value=self._backend_models[0])
        self._backend_model_menu = ctk.CTkOptionMenu(
            self._adv_frame,
            variable=self._backend_model_var,
            values=_decorate_model_menu_values(self._backend_models, self._backend_model_var.get()),
            width=340,
            command=self._on_backend_model_selected,
        )
        self._backend_model_menu.grid(
            row=1, column=1, sticky="ew", padx=(0, 4), pady=4)
        self._backend_model_refresh_btn = ctk.CTkButton(
            self._adv_frame,
            text="Refresh",
            width=80,
            command=lambda: self._refresh_backend_models(show_feedback=True),
        )
        self._backend_model_refresh_btn.grid(
            row=1, column=2, sticky="e", padx=(4, 8), pady=4)

        ctk.CTkLabel(self._adv_frame, text="API key", anchor="w").grid(
            row=2, column=0, sticky="w", padx=(8, 6), pady=4)
        self._backend_api_key_var = ctk.StringVar(value="")
        self._backend_api_key_entry = ctk.CTkEntry(
            self._adv_frame, textvariable=self._backend_api_key_var,
            placeholder_text="Optional oMLX/LM Studio Bearer token", show="•",
        )
        self._backend_api_key_entry.grid(
            row=2, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=(4, 8)
        )
        self._run_locked_widgets.extend([
            self._input_entry, self._input_file_btn, self._input_folder_btn,
            self._output_entry, self._output_browse_btn, self._recursive_chk, self._move_chk,
            self._deep_scan_chk, self._strict_offline_chk, self._adv_toggle_btn,
            self._backend_entry, self._backend_model_menu, self._backend_model_refresh_btn,
            self._backend_api_key_entry,
        ])

        self._adv_visible = False

        # ── results table ─────────────────────────────────────────────────────
        results_outer = ctk.CTkFrame(self)
        results_outer.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        results_outer.grid_columnconfigure(0, weight=1)
        results_outer.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(results_outer, text="Results",
                     font=ctk.CTkFont(weight="bold"), anchor="w").grid(
            row=0, column=0, sticky="w", padx=8, pady=(6, 2))

        canvas = tk.Canvas(results_outer, bg="#2b2b2b",
                           highlightthickness=0, height=180)
        scrollbar = ctk.CTkScrollbar(results_outer, command=canvas.yview)
        self._results_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        self._results_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._results_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew", padx=(4, 0))
        scrollbar.grid(row=1, column=1, sticky="ns")

        for col, (text, w) in enumerate([
            ("File", 200), ("Status", 80), ("Score", 55),
            ("Persons", 65), ("Issues", 200), ("Structure note", 200),
        ]):
            ctk.CTkLabel(self._results_frame, text=text,
                         font=ctk.CTkFont(weight="bold"),
                         width=w, anchor="w").grid(
                row=0, column=col, padx=3, pady=2, sticky="w")

        # ── log + buttons ─────────────────────────────────────────────────────
        self._log = LogPanel(self)
        self._log.grid(row=3, column=0, sticky="ew", padx=12, pady=4)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=4, column=0, padx=12, pady=(4, 12), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        btn_row.grid_columnconfigure(2, weight=1)
        btn_row.grid_columnconfigure(3, weight=1)

        self._run_btn = ctk.CTkButton(
            btn_row, text="Run QC", height=38, command=self._run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self._pause_btn = ctk.CTkButton(
            btn_row, text="Pause", height=38, command=self._toggle_pause, state="disabled",
            fg_color="#444", hover_color="#555")
        self._pause_btn.grid(row=0, column=1, padx=(4, 4), sticky="ew")

        self._stop_btn = ctk.CTkButton(
            btn_row, text="Cancel QC", height=38, command=self._cancel, state="disabled",
            fg_color="#cc2222", hover_color="#dd3333")
        self._stop_btn.grid(row=0, column=2, padx=(4, 4), sticky="ew")

        ctk.CTkButton(
            btn_row, text="Open output folder", height=38,
            fg_color="#444", hover_color="#555",
            command=lambda: _open_folder(self._output_var.get().strip()),
        ).grid(row=0, column=3, padx=(4, 0), sticky="ew")

    def _toggle_advanced(self):
        if self._adv_visible:
            self._adv_frame.grid_forget()
            self._adv_toggle_btn.configure(
                text="▶  Vision backend (optional)")
        else:
            self._adv_frame.grid(row=4, column=0, columnspan=2,
                                  sticky="ew", padx=4, pady=(0, 6))
            self._adv_toggle_btn.configure(
                text="▼  Vision backend (optional)")
            self._refresh_backend_models(show_feedback=False)
        self._adv_visible = not self._adv_visible

    def _set_backend_models(self, models: list[str]) -> None:
        current = _strip_model_menu_label(self._backend_model_var.get())
        self._backend_models = _merge_backend_model_choices(
            models, [current], _BACKEND_MODEL_CHOICES
        )
        self._backend_model_menu.configure(
            values=_decorate_model_menu_values(self._backend_models, current)
        )
        self._backend_model_var.set(
            current if current in self._backend_models else self._backend_models[0]
        )

    def _on_backend_model_selected(self, choice: str) -> None:
        selected = _strip_model_menu_label(choice)
        self._backend_model_var.set(selected)
        self._backend_model_menu.configure(
            values=_decorate_model_menu_values(self._backend_models, selected)
        )

    def _refresh_backend_models(self, *, show_feedback: bool) -> None:
        backend_url = self._backend_var.get().strip()
        if not backend_url:
            if show_feedback:
                messagebox.showwarning("Missing backend", "Please provide the backend URL first.")
            return
        try:
            discovered = _fetch_backend_model_choices(
                backend_url,
                self._backend_api_key_var.get().strip() or None,
            )
        except (OSError, ValueError, json.JSONDecodeError, urllib_error.URLError) as exc:
            if show_feedback:
                messagebox.showwarning("Model refresh failed", str(exc))
            return
        self._set_backend_models(_merge_backend_model_choices(discovered, _BACKEND_MODEL_CHOICES))
        if show_feedback:
            source = "backend + built-in list" if discovered else "built-in list"
            self._log.log(f"Model list refreshed from {source}.")

    def _add_result_row(self, result: dict, row_idx: int):
        status = result.get("status", "warning")
        color  = STATUS_COLORS.get(status, "#888")
        note   = result.get("structure_note") or "—"
        if len(note) > 80:
            note = note[:77] + "…"
        values = [
            (os.path.basename(result.get("filename", "")), 200, "w"),
            (status.upper(),                                80, "center"),
            (str(result.get("score", "—")),                55, "center"),
            (str(result.get("person_count", "—")),         65, "center"),
            (", ".join(result.get("issues") or []) or "—", 200, "w"),
            (note,                                         200, "w"),
        ]
        for col, (text, w, anchor) in enumerate(values):
            kw: dict = dict(width=w, anchor=anchor, wraplength=w - 6)
            if col == 1:
                kw.update(fg_color=color, corner_radius=6, text_color="white")
            ctk.CTkLabel(self._results_frame, text=text, **kw).grid(
                row=row_idx, column=col, padx=3, pady=1, sticky="w")

    def _update_summary(self):
        processed = len(self._results)
        counts = {s: sum(1 for r in self._results if r.get("status") == s)
                  for s in ("pass", "warning", "fail")}
        unscored = sum(
            1 for r in self._results
            if r.get("route_folder") == "unscored"
        )
        self._processed_var.set(f"Processed: {processed}")
        self._pass_var.set(f"Pass: {counts['pass']}")
        self._warn_var.set(f"Warning: {counts['warning']}")
        self._fail_var.set(f"Fail: {counts['fail']}")
        self._unscored_var.set(f"Unscored: {unscored}")
        remaining = max(0, getattr(self, "_total_items", 0) - processed)
        self._remaining_var.set(f"Remaining: {remaining}")

    # ── worker ────────────────────────────────────────────────────────────────

    def _run(self):
        if self._running:
            return
        inp = self._input_var.get().strip()
        if not inp:
            messagebox.showwarning("Missing input",
                                   "Please select an input file or folder.")
            return
        if not self._output_var.get().strip():
            messagebox.showwarning("Missing output", "Please select an output folder.")
            return
        if self._move_var.get():
            if not messagebox.askyesno(
                "Move files?",
                "This will MOVE originals into the output subfolders.\n"
                "This cannot be undone. Continue?",
            ):
                return

        self._running = True
        self._stop_event.clear()  # Clear any previous stop signal
        self._pause_event.clear()
        self._set_run_controls_enabled(False)
        self._run_btn.configure(state="disabled", text="Running…")
        self._pause_btn.configure(state="normal", text="Pause")
        self._stop_btn.configure(state="normal", text="Cancel QC")
        self._log.clear()
        self._log.start_spin()

        # Clear previous results (keep header row 0)
        for w in list(self._results_frame.winfo_children()):
            info = w.grid_info()
            if info and int(info.get("row", 0)) > 0:
                w.destroy()
        self._results.clear()
        self._result_row = 1
        self._total_items = 0

        opts = dict(
            input_path=inp,
            output_dir=self._output_var.get().strip(),
            backend_url=self._backend_var.get().strip() or None,
            model_name=self._backend_model_var.get().strip(),
            api_key=self._backend_api_key_var.get().strip() or None,
            move_files=self._move_var.get(),
            deep_scan=self._deep_scan_var.get(),
            strict_offline=self._strict_offline_var.get(),
            recursive=self._recursive_var.get(),
        )
        threading.Thread(
            target=self._worker,
            args=(opts, self._stop_event, self._pause_event),
            daemon=True,
        ).start()

    def _toggle_pause(self):
        if not self._running:
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._pause_btn.configure(text="Pause")
            self._log.log("Resuming QC…")
        else:
            self._pause_event.set()
            self._pause_btn.configure(text="Resume")
            self._log.log("Pausing QC after the current image…")

    def _cancel(self):
        if not self._running:
            return
        self._stop_event.set()
        self._stop_btn.configure(state="disabled", text="Cancelling…")
        self._pause_btn.configure(state="disabled")
        self._run_btn.configure(state="disabled")
        self._log.log("Cancelling QC pipeline…")

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._run_locked_widgets:
            widget.configure(state=state)

    def _worker(
        self,
        opts: dict,
        stop_event: threading.Event,
        pause_event: threading.Event,
    ):
        q = self._q
        control = lambda: _respect_worker_controls(stop_event, pause_event)
        try:
            import qc_pipeline as qc_module
            qc_module = importlib.reload(qc_module)
            collect_images = qc_module.collect_images
            classify_image = qc_module.classify_image
            classify_image_with_backend = qc_module.classify_image_with_backend
            QCSettings = qc_module.QCSettings

            images = collect_images(opts["input_path"], recursive=opts["recursive"])
            if not images:
                q.put(("error",
                       f"No images found in {opts['input_path']}"))
                return
            q.put(("total", len(images)))

            use_backend = bool(opts["backend_url"]) and not opts["strict_offline"]
            if opts["strict_offline"]:
                q.put(("log", "Offline mode — brightness and contrast checks only"))
            if use_backend:
                q.put(("log", "Using remote backend"))
            else:
                if opts["strict_offline"]:
                    label = "brightness and contrast checks only"
                else:
                    label = "structure scan enabled" if opts["deep_scan"] else "basic checks + YOLO"
                q.put(("log", f"Local inference — {label}"))

            settings = QCSettings(
                use_yolo=not opts["strict_offline"],
                use_deep_scan=opts["deep_scan"] and not opts["strict_offline"],
                strict_offline=opts["strict_offline"],
            )

            q.put(("log", f"Found {len(images)} image(s)"))
            results = []
            for i, path in enumerate(images):
                control()
                q.put(("progress", i, len(images)))
                q.put(("log",
                       f"[{i+1}/{len(images)}] {path}"))
                relative_dir = relative_output_subdir(opts["input_path"], path)
                try:
                    if use_backend:
                        r = classify_image_with_backend(
                            path, opts["backend_url"], opts["output_dir"],
                            model_name=opts["model_name"],
                            move_files=opts["move_files"],
                            api_key=opts["api_key"],
                            relative_dir=relative_dir,
                        )
                    else:
                        r = classify_image(
                            path, opts["output_dir"],
                            move_files=opts["move_files"],
                            settings=settings,
                            relative_dir=relative_dir,
                        )
                except Exception as exc:
                    r = {
                        "filename": os.path.basename(path),
                        "status": "warning",
                        "score": 50.0,
                        "issues": [f"QC skipped after error: {exc}"],
                        "structure_note": "",
                        "destination": path,
                    }
                    q.put(("log", f"Skipping {os.path.basename(path)} after error: {exc}"))
                results.append(r)
                q.put(("result", r))
                q.put(("progress", i + 1, len(images)))

            counts = {s: sum(1 for r in results if r["status"] == s)
                      for s in ("pass", "warning", "fail")}
            unscored = sum(
                1 for r in results if r.get("route_folder") == "unscored"
            )
            q.put(("done",
                   f"Done — {len(results)} images  |  "
                   f"✓ {counts['pass']} pass   "
                   f"⚠ {counts['warning']} warning   "
                   f"✗ {counts['fail']} fail   "
                   f"⛔ {unscored} unscored"))
        except _RunCancelled:
            q.put(("stopped", "QC cancelled by user. Partial results were kept."))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            q.put(("log", tb))
            q.put(("error", tb.splitlines()[-1]))

    def _poll(self):
        try:
            while True:
                msg  = self._q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log.log(msg[1])
                elif kind == "progress":
                    self._log.set_progress(msg[1], msg[2])
                elif kind == "result":
                    result = msg[1]
                    self._results.append(result)
                    self._add_result_row(result, self._result_row)
                    self._result_row += 1
                    self._update_summary()
                elif kind == "total":
                    self._total_items = msg[1]
                    self._update_summary()
                elif kind == "done":
                    self._log.stop()
                    self._log.log(msg[1])
                    self._update_summary()
                    self._set_run_controls_enabled(True)
                    self._run_btn.configure(state="normal", text="Run QC")
                    self._pause_btn.configure(state="disabled", text="Pause")
                    self._stop_btn.configure(state="disabled", text="Cancel QC")
                    self._running = False
                elif kind == "error":
                    self._log.stop(completed=False)
                    messagebox.showerror("Error", msg[1])
                    self._set_run_controls_enabled(True)
                    self._run_btn.configure(state="normal", text="Run QC")
                    self._pause_btn.configure(state="disabled", text="Pause")
                    self._stop_btn.configure(state="disabled", text="Cancel QC")
                    self._running = False
                elif kind == "stopped":
                    self._log.stop(completed=False)
                    self._log.log(msg[1])
                    self._set_run_controls_enabled(True)
                    self._run_btn.configure(state="normal", text="Run QC")
                    self._pause_btn.configure(state="disabled", text="Pause")
                    self._stop_btn.configure(state="disabled", text="Cancel QC")
                    self._running = False
        except queue.Empty:
            pass
        self.after(100, self._poll)


# ── Organize tab ──────────────────────────────────────────────────────────────

class OrganizeTab(ctk.CTkFrame):
    """Sort images into user-defined folders with a vision backend."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._run_locked_widgets: list[tk.Widget] = []
        self._results: list[dict] = []
        self._labels: list[str] = []
        self._total_items = 0
        self._result_row = 1
        self._backend_models = list(_BACKEND_MODEL_CHOICES)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._build()
        self._poll()

    def _build(self):
        paths = ctk.CTkFrame(self)
        paths.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        paths.grid_columnconfigure(1, weight=1)

        self._input_var = ctk.StringVar()
        self._output_var = ctk.StringVar()
        ctk.CTkLabel(paths, text="Input (file or folder)", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=5)
        self._input_entry = ctk.CTkEntry(paths, textvariable=self._input_var)
        self._input_entry.grid(row=0, column=1, sticky="ew", pady=5)
        self._input_file_btn = ctk.CTkButton(
            paths, text="File…", width=65,
            command=lambda: _browse_file(self._input_var, self),
        )
        self._input_file_btn.grid(row=0, column=2, padx=(6, 3), pady=5)
        self._input_folder_btn = ctk.CTkButton(
            paths, text="Folder…", width=70,
            command=lambda: _browse_folder(self._input_var, self),
        )
        self._input_folder_btn.grid(row=0, column=3, padx=(3, 8), pady=5)

        ctk.CTkLabel(paths, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        self._output_entry = ctk.CTkEntry(paths, textvariable=self._output_var)
        self._output_entry.grid(row=1, column=1, sticky="ew", pady=5)
        self._output_browse_btn = ctk.CTkButton(
            paths, text="Browse", width=80,
            command=lambda: _browse_folder(self._output_var, self),
        )
        self._output_browse_btn.grid(row=1, column=2, columnspan=2, padx=(6, 8), pady=5)
        self._output_autofill = _OutputAutofillController(
            self._input_var,
            self._output_var,
            "organized",
        )
        self._recursive_var = ctk.BooleanVar(value=False)
        self._recursive_chk = ctk.CTkCheckBox(
            paths,
            text="Include files in subfolders (preserve structure)",
            variable=self._recursive_var,
        )
        self._recursive_chk.grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 6))

        opts = ctk.CTkFrame(self)
        opts.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        opts.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(opts, text="Category labels", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=(8, 4))
        self._labels_var = ctk.StringVar()
        self._labels_entry = ctk.CTkEntry(
            opts,
            textvariable=self._labels_var,
            placeholder_text="outdoors, indoors  or  places, people",
        )
        self._labels_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=(8, 4))
        ctk.CTkLabel(
            opts,
            text="Enter two or more comma-separated choices. The model must choose exactly one per image.",
            text_color="#aaa", font=ctk.CTkFont(size=11), anchor="w",
        ).grid(row=1, column=1, columnspan=3, sticky="w", padx=(0, 8), pady=(0, 6))

        ctk.CTkLabel(opts, text="Backend URL", anchor="w").grid(
            row=2, column=0, sticky="w", padx=(8, 6), pady=4)
        self._backend_var = ctk.StringVar(value="http://127.0.0.1:8001/v1")
        self._backend_entry = ctk.CTkEntry(
            opts, textvariable=self._backend_var,
            placeholder_text="oMLX or LM Studio OpenAI-compatible URL",
        )
        self._backend_entry.grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=4
        )

        ctk.CTkLabel(opts, text="Model name", anchor="w").grid(
            row=3, column=0, sticky="w", padx=(8, 6), pady=4)
        self._model_var = ctk.StringVar(value=self._backend_models[0])
        self._model_menu = ctk.CTkOptionMenu(
            opts,
            variable=self._model_var,
            values=_decorate_model_menu_values(self._backend_models, self._model_var.get()),
            width=340,
            command=self._on_model_selected,
        )
        self._model_menu.grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=4)
        self._model_refresh_btn = ctk.CTkButton(
            opts,
            text="Refresh",
            width=80,
            command=lambda: self._refresh_backend_models(show_feedback=True),
        )
        self._model_refresh_btn.grid(row=3, column=3, sticky="e", padx=(0, 8), pady=4)

        ctk.CTkLabel(opts, text="API key", anchor="w").grid(
            row=4, column=0, sticky="w", padx=(8, 6), pady=4)
        self._api_key_var = ctk.StringVar(value="")
        self._api_key_entry = ctk.CTkEntry(
            opts, textvariable=self._api_key_var, show="•",
            placeholder_text="Optional Bearer token",
        )
        self._api_key_entry.grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=4
        )

        self._move_var = ctk.BooleanVar(value=False)
        self._move_chk = ctk.CTkCheckBox(
            opts,
            text="Move originals (destructive — default copies)",
            variable=self._move_var,
            text_color="#e74c3c",
        )
        self._move_chk.grid(row=5, column=0, columnspan=4, sticky="w", padx=8, pady=4)
        self._run_locked_widgets.extend([
            self._input_entry, self._input_file_btn, self._input_folder_btn,
            self._output_entry, self._output_browse_btn, self._recursive_chk, self._labels_entry,
            self._backend_entry, self._model_menu, self._model_refresh_btn, self._api_key_entry,
            self._move_chk,
        ])

        self._summary_var = ctk.StringVar(value="Processed: 0  ·  Remaining: 0")
        ctk.CTkLabel(
            opts, textvariable=self._summary_var, anchor="w",
            corner_radius=8, fg_color="#232323", text_color="#ddd",
            wraplength=800, justify="left",
        ).grid(row=6, column=0, columnspan=4, sticky="ew", padx=8, pady=(6, 8))

        results_outer = ctk.CTkFrame(self)
        results_outer.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        results_outer.grid_columnconfigure(0, weight=1)
        results_outer.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            results_outer, text="Organization results",
            font=ctk.CTkFont(weight="bold"), anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))

        canvas = tk.Canvas(results_outer, bg="#2b2b2b", highlightthickness=0, height=180)
        scrollbar = ctk.CTkScrollbar(results_outer, command=canvas.yview)
        self._results_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        self._results_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._results_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew", padx=(4, 0))
        scrollbar.grid(row=1, column=1, sticky="ns")

        for col, (text, width) in enumerate([
            ("File", 230), ("Label", 140), ("Confidence", 85), ("Reason", 360),
        ]):
            ctk.CTkLabel(
                self._results_frame, text=text,
                font=ctk.CTkFont(weight="bold"), width=width, anchor="w",
            ).grid(row=0, column=col, padx=3, pady=2, sticky="w")

        self._log = LogPanel(self)
        self._log.grid(row=3, column=0, sticky="ew", padx=12, pady=4)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=4, column=0, padx=12, pady=(4, 12), sticky="ew")
        btn_row.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._run_btn = ctk.CTkButton(
            btn_row, text="Organize images", height=38, command=self._run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self._pause_btn = ctk.CTkButton(
            btn_row, text="Pause", height=38, command=self._toggle_pause,
            state="disabled", fg_color="#444", hover_color="#555",
        )
        self._pause_btn.grid(row=0, column=1, padx=4, sticky="ew")
        self._stop_btn = ctk.CTkButton(
            btn_row, text="Cancel", height=38, command=self._cancel,
            state="disabled", fg_color="#cc2222", hover_color="#dd3333",
        )
        self._stop_btn.grid(row=0, column=2, padx=4, sticky="ew")
        ctk.CTkButton(
            btn_row, text="Open output folder", height=38,
            fg_color="#444", hover_color="#555",
            command=lambda: _open_folder(self._output_var.get().strip()),
        ).grid(row=0, column=3, padx=(4, 0), sticky="ew")
        self._refresh_backend_models(show_feedback=False)

    def _set_backend_models(self, models: list[str]) -> None:
        current = _strip_model_menu_label(self._model_var.get())
        self._backend_models = _merge_backend_model_choices(
            models, [current], _BACKEND_MODEL_CHOICES
        )
        self._model_menu.configure(
            values=_decorate_model_menu_values(self._backend_models, current)
        )
        self._model_var.set(current if current in self._backend_models else self._backend_models[0])

    def _on_model_selected(self, choice: str) -> None:
        selected = _strip_model_menu_label(choice)
        self._model_var.set(selected)
        self._model_menu.configure(
            values=_decorate_model_menu_values(self._backend_models, selected)
        )

    def _refresh_backend_models(self, *, show_feedback: bool) -> None:
        backend_url = self._backend_var.get().strip()
        if not backend_url:
            if show_feedback:
                messagebox.showwarning("Missing backend", "Please provide the backend URL first.")
            return
        try:
            discovered = _fetch_backend_model_choices(
                backend_url,
                self._api_key_var.get().strip() or None,
            )
        except (OSError, ValueError, json.JSONDecodeError, urllib_error.URLError) as exc:
            if show_feedback:
                messagebox.showwarning("Model refresh failed", str(exc))
            return
        self._set_backend_models(_merge_backend_model_choices(discovered, _BACKEND_MODEL_CHOICES))
        if show_feedback:
            source = "backend + built-in list" if discovered else "built-in list"
            self._log.log(f"Model list refreshed from {source}.")

    def _run(self):
        if self._running:
            return
        input_path = self._input_var.get().strip()
        output_dir = self._output_var.get().strip()
        backend_url = self._backend_var.get().strip()
        model_name = self._model_var.get().strip()
        if not input_path or not output_dir:
            messagebox.showwarning("Missing path", "Please select input and output paths.")
            return
        if not backend_url or not model_name:
            messagebox.showwarning(
                "Missing backend", "Please provide the vision backend URL and choose a model."
            )
            return

        try:
            from qc_pipeline import validate_organizer_labels
            labels = validate_organizer_labels(_split_labels(self._labels_var.get()))
        except ValueError as exc:
            messagebox.showwarning("Invalid categories", str(exc))
            return
        if len(labels) < 2:
            messagebox.showwarning(
                "More categories needed", "Please enter at least two category labels."
            )
            return
        if self._move_var.get() and not messagebox.askyesno(
            "Move files?",
            "This will MOVE originals into category subfolders.\n"
            "This cannot be undone. Continue?",
        ):
            return

        for widget in list(self._results_frame.winfo_children()):
            info = widget.grid_info()
            if info and int(info.get("row", 0)) > 0:
                widget.destroy()
        self._results.clear()
        self._labels = labels
        self._total_items = 0
        self._result_row = 1
        self._update_summary()
        self._stop_event.clear()
        self._pause_event.clear()
        self._running = True
        self._set_run_controls_enabled(False)
        self._run_btn.configure(state="disabled", text="Organizing…")
        self._pause_btn.configure(state="normal", text="Pause")
        self._stop_btn.configure(state="normal", text="Cancel")
        self._log.clear()
        self._log.start_spin()

        opts = {
            "input_path": input_path,
            "output_dir": output_dir,
            "labels": labels,
            "backend_url": backend_url,
            "model_name": model_name,
            "api_key": self._api_key_var.get().strip() or None,
            "move_files": self._move_var.get(),
            "recursive": self._recursive_var.get(),
        }
        threading.Thread(
            target=self._worker,
            args=(opts, self._stop_event, self._pause_event),
            daemon=True,
        ).start()

    def _toggle_pause(self):
        if not self._running:
            return
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._pause_btn.configure(text="Pause")
            self._log.log("Resuming organizer…")
        else:
            self._pause_event.set()
            self._pause_btn.configure(text="Resume")
            self._log.log("Pausing organizer after the current image…")

    def _cancel(self):
        if not self._running:
            return
        self._stop_event.set()
        self._stop_btn.configure(state="disabled", text="Cancelling…")
        self._pause_btn.configure(state="disabled")
        self._log.log("Cancelling organizer after the current image…")

    def _set_run_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._run_locked_widgets:
            widget.configure(state=state)

    def _worker(
        self,
        opts: dict,
        stop_event: threading.Event,
        pause_event: threading.Event,
    ):
        q = self._q
        control = lambda: _respect_worker_controls(stop_event, pause_event)
        try:
            import qc_pipeline as qc_module
            qc_module = importlib.reload(qc_module)
            images = qc_module.collect_images(
                opts["input_path"], recursive=opts["recursive"]
            )
            if not images:
                raise FileNotFoundError(f"No images found in {opts['input_path']}")

            os.makedirs(opts["output_dir"], exist_ok=True)
            for label in opts["labels"]:
                os.makedirs(os.path.join(opts["output_dir"], label), exist_ok=True)
            q.put(("total", len(images)))
            q.put(("log", f"Found {len(images)} image(s)"))
            q.put(("log", f"Categories: {', '.join(opts['labels'])}"))

            results = []
            for index, path in enumerate(images):
                control()
                q.put(("progress", index, len(images)))
                q.put(("log", f"[{index + 1}/{len(images)}] {path}"))
                relative_dir = relative_output_subdir(opts["input_path"], path)
                try:
                    result = qc_module.classify_image_with_labels(
                        path,
                        opts["backend_url"],
                        opts["labels"],
                        opts["output_dir"],
                        model_name=opts["model_name"],
                        move_files=opts["move_files"],
                        api_key=opts["api_key"],
                        relative_dir=relative_dir,
                    )
                except Exception as exc:
                    result = {
                        "filename": os.path.basename(path),
                        "label": "ERROR",
                        "status": "error",
                        "confidence": 0.0,
                        "reason": str(exc),
                        "destination": None,
                        "labels": opts["labels"],
                    }
                    q.put(("log", f"Could not organize {os.path.basename(path)}: {exc}"))
                results.append(result)
                q.put(("result", result))
                q.put(("progress", index + 1, len(images)))

            errors = sum(1 for result in results if result.get("status") == "error")
            q.put((
                "done",
                f"Done — {len(results)} image(s) processed, {errors} error(s).",
            ))
        except _RunCancelled:
            q.put(("stopped", "Organizer cancelled by user."))
        except Exception:
            import traceback
            traceback_text = traceback.format_exc()
            q.put(("log", traceback_text))
            q.put(("error", traceback_text.splitlines()[-1]))

    def _add_result_row(self, result: dict):
        label = str(result.get("label") or "ERROR")
        color = "#8b5cf6" if label == "ERROR" else _label_chip_color(label)
        reason = str(result.get("reason") or "No reason provided.")
        values = [
            (os.path.basename(result.get("filename", "")), 230, "w"),
            (label, 140, "center"),
            (f"{float(result.get('confidence') or 0):.0f}%", 85, "center"),
            (reason, 360, "w"),
        ]
        for col, (text, width, anchor) in enumerate(values):
            options = {"width": width, "anchor": anchor, "wraplength": width - 8}
            if col == 1:
                options.update(fg_color=color, corner_radius=6, text_color="white")
            ctk.CTkLabel(self._results_frame, text=text, **options).grid(
                row=self._result_row, column=col, padx=3, pady=1, sticky="w"
            )
        self._result_row += 1

    def _update_summary(self):
        processed = len(self._results)
        remaining = max(0, self._total_items - processed)
        counts = [
            f"{label}: {sum(1 for result in self._results if result.get('label') == label)}"
            for label in self._labels
        ]
        errors = sum(1 for result in self._results if result.get("status") == "error")
        parts = [f"Processed: {processed}", *counts]
        if errors:
            parts.append(f"Errors: {errors}")
        parts.append(f"Remaining: {remaining}")
        self._summary_var.set("  ·  ".join(parts))

    def _finish(self):
        self._run_btn.configure(state="normal", text="Organize images")
        self._pause_btn.configure(state="disabled", text="Pause")
        self._stop_btn.configure(state="disabled", text="Cancel")
        self._running = False

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log.log(msg[1])
                elif kind == "progress":
                    self._log.set_progress(msg[1], msg[2])
                elif kind == "total":
                    self._total_items = msg[1]
                    self._update_summary()
                elif kind == "result":
                    self._results.append(msg[1])
                    self._add_result_row(msg[1])
                    self._update_summary()
                elif kind in {"done", "stopped"}:
                    self._log.stop(completed=kind == "done")
                    self._log.log(msg[1])
                    self._update_summary()
                    self._set_run_controls_enabled(True)
                    self._finish()
                elif kind == "error":
                    self._log.stop(completed=False)
                    messagebox.showerror("Organizer error", msg[1])
                    self._set_run_controls_enabled(True)
                    self._finish()
        except queue.Empty:
            pass
        self.after(100, self._poll)


# ── Sanitize tab ──────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("StereoSift")
        self.geometry("880x760")
        self.minsize(720, 600)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header bar
        hdr = ctk.CTkFrame(self, corner_radius=0,
                           fg_color=("#1a1a2e", "#1a1a2e"))
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(
            hdr, text="StereoSift",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#7eb3ff",
        ).pack(side="left", padx=16, pady=10)
        subtitle = "Image QC  ·  AI Organize  ·  AI Upscale  ·  2D → SBS 3D"
        ctk.CTkLabel(hdr, text=subtitle, text_color="#666").pack(side="left", pady=10)

        # Tabs
        tabs = ctk.CTkTabview(self)
        tabs.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        tab_specs = [
            ("Judge", JudgeTab),
            ("Organize", OrganizeTab),
            ("Upscale", UpscaleTab),
            ("Convert", ConvertTab),
        ]
        for name, cls in tab_specs:
            tabs.add(name)
            tabs.tab(name).grid_columnconfigure(0, weight=1)
            tabs.tab(name).grid_rowconfigure(0, weight=1)
            cls(tabs.tab(name)).grid(row=0, column=0, sticky="nsew")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure relative paths (models/, output/) resolve from the project root.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = App()
    app.mainloop()
