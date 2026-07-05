#!/usr/bin/env python3
"""StereoSift — CustomTkinter GUI.

Three tabs:
  • Convert  — 2D images / videos → SBS 3D
  • Upscale  — images → Quest-ready high resolution
  • Judge    — local QC: pass / warning / fail / violations sorting

All heavy work runs in a background thread so the UI stays responsive.
Progress and log output stream back to the main thread via a queue.
"""

from __future__ import annotations

import importlib
import os
import platform
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

# ── appearance ───────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

STATUS_COLORS = {"pass": "#2ecc71", "warning": "#f39c12", "fail": "#e74c3c"}

# Maps human-readable size label → (image model filename suffix, video encoder)
# fp16/fp32 is resolved at runtime based on device.
_SIZE_LABELS   = ["Small", "Base", "Large"]
_IMG_ENCODERS  = {"Small": "vits", "Base": "vitb", "Large": "vitl"}
_VID_ENCODERS  = {"Small": "vits", "Base": "vitb", "Large": "vitl"}


def _resolve_img_model(size_label: str, device_type: str) -> str:
    """Return the full model filename for an image depth model."""
    enc      = _IMG_ENCODERS[size_label]
    precision = "fp16" if device_type in ("cuda", "mps") else "fp32"
    return f"depth_anything_v2_{enc}_{precision}.safetensors"


# ── shared helpers ────────────────────────────────────────────────────────────

def _browse_file(
    var: ctk.StringVar,
    parent,
    output_var_to_update: ctk.StringVar | None = None,
) -> None:
    """Open one file chooser and optionally suggest a sibling output folder."""
    path = filedialog.askopenfilename(parent=parent)
    if path:
        var.set(path)
        if output_var_to_update and not output_var_to_update.get():
            stem = os.path.splitext(os.path.basename(path))[0]
            output_var_to_update.set(
                os.path.join(os.path.dirname(path), f"{stem}-judged")
            )


def _browse_folder(var: ctk.StringVar, parent, output_var_to_update: ctk.StringVar | None = None) -> None:
    """Open exactly one folder chooser and optionally suggest a sibling output folder."""
    path = filedialog.askdirectory(parent=parent)
    if path:
        var.set(path)
        if output_var_to_update and not output_var_to_update.get():
            default_output = os.path.join(
                os.path.dirname(path), f"{os.path.basename(path)}-judged"
            )
            output_var_to_update.set(default_output)


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


def _split_labels(text: str) -> list[str]:
    parts = [part.strip() for part in text.replace("\n", ",").split(",")]
    return [part for part in parts if part]


def _label_chip_color(label: str) -> str:
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6"]
    idx = sum(ord(ch) for ch in label.lower()) % len(palette)
    return palette[idx]


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
        self._bar.set(0)

    def start_spin(self) -> None:
        self._bar.configure(mode="indeterminate")
        self._bar.set(0)
        self._bar.start()

    def set_progress(self, done: int, total: int) -> None:
        if total > 0:
            frac = done / total
            self._bar.configure(mode="determinate")
            self._bar.stop()
            self._bar.set(frac)
            self._lbl.configure(text=f"{done} / {total}  ({frac*100:.0f}%)")

    def stop(self) -> None:
        self._bar.stop()
        self._bar.configure(mode="determinate")
        self._bar.set(1)


# ── Convert tab ───────────────────────────────────────────────────────────────

class ConvertTab(ctk.CTkFrame):
    """2D → SBS 3D conversion for images and videos.

    Images use DepthAnythingV2 (frame-level model).
    Videos use Video Depth Anything (temporal streaming model).
    The two modes are mutually exclusive — the model picker updates to match.
    """

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)
        self._q: queue.Queue = queue.Queue()
        self._running = False
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
        ctk.CTkEntry(paths, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", pady=5)
        ctk.CTkButton(paths, text="File…", width=65,
                      command=lambda: _browse_file(self._input_var, self)).grid(
            row=0, column=2, padx=(6, 3), pady=5)
        ctk.CTkButton(paths, text="Folder…", width=70,
                      command=lambda: _browse_folder(self._input_var, self)).grid(
            row=0, column=3, padx=(3, 8), pady=5)

        ctk.CTkLabel(paths, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        self._output_var = ctk.StringVar(
            value=os.path.join(os.getcwd(), "output"))
        ctk.CTkEntry(paths, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", pady=5)
        ctk.CTkButton(paths, text="Browse", width=80,
                      command=lambda: _browse_folder(self._output_var, self)).grid(
            row=1, column=2, columnspan=2, padx=(6, 8), pady=5)

        # ── options ───────────────────────────────────────────────────────────
        opts = ctk.CTkFrame(self)
        opts.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        for c in range(5):
            opts.grid_columnconfigure(c, weight=1)

        # Row 0 — labels
        for col, text in enumerate(["Mode", "Model size", "Output format",
                                     "Method", "Viewing mode"]):
            ctk.CTkLabel(opts, text=text).grid(
                row=0, column=col, padx=8, pady=(8, 2))

        # Mode — Images or Video; drives which model label is shown
        self._mode_var = ctk.StringVar(value="Images")
        ctk.CTkOptionMenu(opts, variable=self._mode_var,
                          values=["Images", "Video"],
                          command=self._on_mode_change).grid(
            row=1, column=0, padx=8, pady=(0, 8), sticky="ew")

        # Model size picker (Small / Base / Large)
        self._size_var = ctk.StringVar(value="Large")
        ctk.CTkOptionMenu(opts, variable=self._size_var,
                          values=_SIZE_LABELS).grid(
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
        ctk.CTkOptionMenu(opts, variable=self._method_var,
                          values=["mesh_warping", "grid_sampling"]).grid(
            row=1, column=3, padx=8, pady=(0, 8), sticky="ew")

        # Viewing mode (hidden when anaglyph-only)
        self._sbs_mode_var = ctk.StringVar(value="parallel")
        self._sbs_mode_menu = ctk.CTkOptionMenu(
            opts, variable=self._sbs_mode_var,
            values=["parallel", "cross-eyed"])
        self._sbs_mode_menu.grid(row=1, column=4, padx=8, pady=(0, 8), sticky="ew")

        # Depth-only checkbox on its own row
        self._depth_only_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(opts, text="Depth map only",
                        variable=self._depth_only_var).grid(
            row=2, column=2, columnspan=3, padx=8, pady=(0, 4), sticky="w")

        # Row 3 — sliders
        ctk.CTkLabel(opts, text="3D strength", anchor="w").grid(
            row=3, column=0, padx=8, sticky="w")
        self._depth_scale_var = ctk.IntVar(value=40)
        ctk.CTkSlider(opts, from_=10, to=100, number_of_steps=90,
                      variable=self._depth_scale_var).grid(
            row=4, column=0, columnspan=2, padx=8, sticky="ew", pady=(0, 8))
        self._ds_lbl = ctk.CTkLabel(opts, text="40")
        self._ds_lbl.grid(row=4, column=2, padx=4, sticky="w")
        self._depth_scale_var.trace_add(
            "write", lambda *_: self._ds_lbl.configure(
                text=str(self._depth_scale_var.get())))

        ctk.CTkLabel(opts, text="Depth blur", anchor="w").grid(
            row=3, column=3, padx=8, sticky="w")
        self._blur_var = ctk.IntVar(value=7)
        ctk.CTkSlider(opts, from_=3, to=15, number_of_steps=6,
                      variable=self._blur_var).grid(
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

        self._on_mode_change("Images")   # set initial labels
        self._on_format_change("sbs")    # hide convergence initially

        # ── log + button ──────────────────────────────────────────────────────
        self._log = LogPanel(self)
        self._log.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        self.grid_rowconfigure(2, weight=1)

        self._run_btn = ctk.CTkButton(
            self, text="Convert", height=38, command=self._run)
        self._run_btn.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="ew")

    def _on_mode_change(self, value: str) -> None:
        if value == "Images":
            self._model_lbl.configure(
                text="DepthAnythingV2 — static image depth model")
        else:
            self._model_lbl.configure(
                text="Video Depth Anything — temporal streaming model")

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
        self._run_btn.configure(state="disabled", text="Running…")
        self._log.clear()
        self._log.start_spin()
        opts = dict(
            input_path=inp,
            output_dir=out,
            mode=self._mode_var.get(),
            size=self._size_var.get(),
            method=self._method_var.get(),
            sbs_mode=self._sbs_mode_var.get(),
            depth_scale=self._depth_scale_var.get(),
            sbs_blur=self._blur_var.get(),
            depth_only=self._depth_only_var.get(),
            output_format=self._output_format_var.get(),
            convergence=round(self._conv_var.get(), 2),
        )
        threading.Thread(target=self._worker, args=(opts,), daemon=True).start()

    def _worker(self, opts: dict):
        q = self._q
        try:
            import torch
            from convert import get_device, collect_images, collect_videos, convert_one

            device    = get_device()
            is_video  = opts["mode"] == "Video"

            if is_video:
                from video_converter import (
                    load_video_depth_model, convert_video_to_sbs)
                encoder = _VID_ENCODERS[opts["size"]]
                q.put(("log",
                       f"Loading Video Depth Anything — {opts['size']} ({encoder})…"))
                model, dtype, is_metric = load_video_depth_model(
                    encoder=encoder, device=device)
                files = collect_videos(opts["input_path"])
                q.put(("log", f"Found {len(files)} video(s)"))
                for i, path in enumerate(files):
                    q.put(("progress", i, len(files)))
                    q.put(("log",
                           f"[{i+1}/{len(files)}] {os.path.basename(path)}"))
                    convert_video_to_sbs(
                        video_path=path,
                        output_dir=opts["output_dir"],
                        model=model, device=device,
                        dtype=dtype, is_metric=is_metric,
                        sbs_method=opts["method"],
                        depth_scale=opts["depth_scale"],
                        sbs_mode=opts["sbs_mode"],
                        sbs_blur=opts["sbs_blur"],
                        depth_only=opts["depth_only"],
                    )
                    q.put(("progress", i + 1, len(files)))
            else:
                from depth_model import load_depth_model
                model_name = _resolve_img_model(opts["size"], device.type)
                q.put(("log",
                       f"Loading DepthAnythingV2 — {opts['size']} ({model_name})…"))
                model, dtype, is_metric = load_depth_model(model_name, device)
                files = collect_images(opts["input_path"])
                q.put(("log", f"Found {len(files)} image(s)"))
                for i, path in enumerate(files):
                    q.put(("progress", i, len(files)))
                    q.put(("log",
                           f"[{i+1}/{len(files)}] {os.path.basename(path)}"))
                    convert_one(
                        model, path, opts["output_dir"],
                        device, dtype, is_metric,
                        depth_only=opts["depth_only"],
                        depth_input_scale=0.5,
                        sbs_method=opts["method"],
                        depth_scale=opts["depth_scale"],
                        sbs_mode=opts["sbs_mode"],
                        sbs_blur=opts["sbs_blur"],
                        output_format=opts["output_format"],
                        convergence=opts["convergence"],
                        log=lambda msg: q.put(("log", msg)),
                    )
                    q.put(("progress", i + 1, len(files)))

            q.put(("done", f"Finished — {len(files)} file(s) converted."))
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
                elif kind == "done":
                    self._log.stop()
                    self._log.log(msg[1])
                    self._run_btn.configure(state="normal", text="Convert")
                    self._running = False
                elif kind == "error":
                    self._log.stop()
                    messagebox.showerror("Error", msg[1])
                    self._run_btn.configure(state="normal", text="Convert")
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
        self._build()
        self._poll()

    def _build(self):
        paths = ctk.CTkFrame(self)
        paths.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        paths.grid_columnconfigure(1, weight=1)
        self._input_var = ctk.StringVar()
        self._output_var = ctk.StringVar(value=os.path.join(os.getcwd(), "output", "upscaled"))
        ctk.CTkLabel(paths, text="Input (image or folder)", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=5)
        ctk.CTkEntry(paths, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", pady=5)
        ctk.CTkButton(paths, text="File…", width=65,
                      command=lambda: _browse_file(self._input_var, self)).grid(
            row=0, column=2, padx=(6, 3), pady=5)
        ctk.CTkButton(paths, text="Folder…", width=70,
                      command=lambda: _browse_folder(self._input_var, self)).grid(
            row=0, column=3, padx=(3, 8), pady=5)
        ctk.CTkLabel(paths, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        ctk.CTkEntry(paths, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", pady=5)
        ctk.CTkButton(paths, text="Browse", width=80,
                      command=lambda: _browse_folder(self._output_var, self)).grid(
            row=1, column=2, columnspan=2, padx=(6, 8), pady=5)

        opts = ctk.CTkFrame(self)
        opts.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        for col in range(3):
            opts.grid_columnconfigure(col, weight=1)
        ctk.CTkLabel(opts, text="Target").grid(row=0, column=0, padx=8, pady=(8, 2))
        ctk.CTkLabel(opts, text="Tile size").grid(row=0, column=1, padx=8, pady=(8, 2))
        ctk.CTkLabel(opts, text="Format").grid(row=0, column=2, padx=8, pady=(8, 2))
        self._target_var = ctk.StringVar(value="Quest 3 SBS (2064×2208 per eye)")
        ctk.CTkOptionMenu(opts, variable=self._target_var,
                          values=list(self.TARGETS)).grid(
            row=1, column=0, padx=8, pady=(0, 8), sticky="ew")
        self._tile_var = ctk.StringVar(value="256")
        ctk.CTkOptionMenu(opts, variable=self._tile_var,
                          values=["128", "256", "384", "512"]).grid(
            row=1, column=1, padx=8, pady=(0, 8), sticky="ew")
        self._format_var = ctk.StringVar(value="PNG")
        ctk.CTkOptionMenu(opts, variable=self._format_var,
                          values=["PNG", "JPEG"]).grid(
            row=1, column=2, padx=8, pady=(0, 8), sticky="ew")
        ctk.CTkLabel(
            opts,
            text=("Default fits each eye within the Quest 3 panel's 2064×2208 bounds, "
                  "producing SBS up to 4128×2208 without stretching or cropping."),
            text_color="#aaa", wraplength=760, justify="left",
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 8))

        self._log = LogPanel(self)
        self._log.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="ew")
        buttons.grid_columnconfigure((0, 1), weight=1)
        self._run_btn = ctk.CTkButton(buttons, text="Upscale", height=38,
                                      command=self._run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(buttons, text="Open output folder", height=38,
                      fg_color="#444", hover_color="#555",
                      command=lambda: _open_folder(self._output_var.get().strip())).grid(
            row=0, column=1, padx=(4, 0), sticky="ew")

    def _run(self):
        if self._running:
            return
        inp, out = self._input_var.get().strip(), self._output_var.get().strip()
        if not inp or not out:
            messagebox.showwarning("Missing path", "Please select input and output paths.")
            return
        self._running = True
        self._run_btn.configure(state="disabled", text="Upscaling…")
        self._log.clear()
        self._log.start_spin()
        opts = (inp, out, self.TARGETS[self._target_var.get()],
                int(self._tile_var.get()), self._format_var.get())
        threading.Thread(target=self._worker, args=opts, daemon=True).start()

    def _worker(self, inp, out, target, tile, output_format):
        try:
            from upscaler import collect_images, ensure_model, RealESRGANx2, upscale_file
            files = collect_images(inp)
            if not files:
                raise ValueError(f"No supported images found in {inp}")
            log = lambda text: self._q.put(("log", text))
            model_path = ensure_model(log=log)
            log(f"Loading Real-ESRGAN x2plus (tile {tile})…")
            engine = RealESRGANx2(model_path, tile=tile)
            log(f"Found {len(files)} image(s); device: {engine.device}")
            for index, path in enumerate(files):
                self._q.put(("progress", index, len(files)))
                log(f"[{index + 1}/{len(files)}] {os.path.basename(path)}")
                target_box = target if isinstance(target, tuple) else None
                long_edge = target if isinstance(target, int) else max(target)
                upscale_file(path, out, engine, long_edge, output_format, log,
                             target_box=target_box)
                self._q.put(("progress", index + 1, len(files)))
            self._q.put(("done", f"Finished — {len(files)} image(s) upscaled."))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            self._q.put(("log", tb))
            self._q.put(("error", tb.splitlines()[-1]))

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                if msg[0] == "log":
                    self._log.log(msg[1])
                elif msg[0] == "progress":
                    self._log.set_progress(msg[1], msg[2])
                elif msg[0] in ("done", "error"):
                    self._log.stop()
                    if msg[0] == "done":
                        self._log.log(msg[1])
                    else:
                        messagebox.showerror("Upscale error", msg[1])
                    self._run_btn.configure(state="normal", text="Upscale")
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
        self._results: list[dict] = []
        self._result_row = 1
        self._stop_event = threading.Event()
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
        ctk.CTkEntry(paths, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", pady=5)
        ctk.CTkButton(paths, text="File…", width=65,
                      command=lambda: _browse_file(
                          self._input_var, self, self._output_var
                      )).grid(row=0, column=2, padx=(6, 3), pady=5)
        ctk.CTkButton(paths, text="Folder…", width=70,
                      command=lambda: _browse_folder(self._input_var, self, self._output_var)).grid(
            row=0, column=3, padx=(3, 8), pady=5)

        ctk.CTkLabel(paths, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        ctk.CTkEntry(paths, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", pady=5)
        ctk.CTkButton(paths, text="Browse", width=80,
                      command=lambda: _browse_folder(self._output_var, self)).grid(
            row=1, column=2, columnspan=2, padx=(6, 8), pady=5)

        # ── options ───────────────────────────────────────────────────────────
        opts = ctk.CTkFrame(self)
        opts.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        opts.grid_columnconfigure(1, weight=1)

        # Info label
        ctk.CTkLabel(
            opts,
            text="YOLO pose catches obvious duplicate heads/torsos. The optional vision backend can use oMLX or LM Studio for a stronger second opinion.",
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
        ctk.CTkCheckBox(
            opts,
            text="Move originals (destructive — default copies)",
            variable=self._move_var,
            text_color="#e74c3c",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 6))

        # Deep scan toggle
        self._deep_scan_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opts,
            text="Optional moondream2 fallback — warning on suspect structure, fail only on clear duplicates",
            variable=self._deep_scan_var,
            text_color="#7eb3ff",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 2))

        self._strict_offline_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opts,
            text="Offline mode — brightness and contrast only (no models or downloads)",
            variable=self._strict_offline_var,
            text_color="#c7d2fe",
        ).grid(row=3, column=2, columnspan=3, sticky="w", padx=8, pady=(0, 2))

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

        ctk.CTkLabel(self._adv_frame, text="Backend URL", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=4)
        self._backend_var = ctk.StringVar(value="http://127.0.0.1:8001/v1")
        ctk.CTkEntry(
            self._adv_frame, textvariable=self._backend_var,
            placeholder_text="oMLX: http://127.0.0.1:8001/v1",
        ).grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=4)
        self._adv_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self._adv_frame, text="Model name", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=4)
        self._backend_model_var = ctk.StringVar(
            value="Qwen3.6-35B-A3B-MLX-4bit")
        ctk.CTkEntry(self._adv_frame,
                     textvariable=self._backend_model_var).grid(
            row=1, column=1, sticky="ew", padx=(0, 8), pady=4)

        ctk.CTkLabel(self._adv_frame, text="API key", anchor="w").grid(
            row=2, column=0, sticky="w", padx=(8, 6), pady=4)
        self._backend_api_key_var = ctk.StringVar()
        ctk.CTkEntry(
            self._adv_frame, textvariable=self._backend_api_key_var,
            placeholder_text="Optional oMLX/LM Studio Bearer token", show="•",
        ).grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(4, 8))

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

        self._run_btn = ctk.CTkButton(
            btn_row, text="Run QC", height=38, command=self._run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self._stop_btn = ctk.CTkButton(
            btn_row, text="Stop QC", height=38, command=self._stop, state="disabled",
            fg_color="#cc2222", hover_color="#dd3333")
        self._stop_btn.grid(row=0, column=1, padx=(4, 4), sticky="ew")

        ctk.CTkButton(
            btn_row, text="Open output folder", height=38,
            fg_color="#444", hover_color="#555",
            command=lambda: _open_folder(self._output_var.get().strip()),
        ).grid(row=0, column=2, padx=(4, 0), sticky="ew")

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
        self._adv_visible = not self._adv_visible

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
        violations = sum(
            1 for r in self._results
            if r.get("route_folder") == "unscored"
        )
        total = max(processed, 0)
        self._processed_var.set(f"Processed: {processed}")
        self._pass_var.set(f"Pass: {counts['pass']}")
        self._warn_var.set(f"Warning: {counts['warning']}")
        self._fail_var.set(f"Fail: {counts['fail']}")
        self._unscored_var.set(f"Unscored: {violations}")
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
        self._run_btn.configure(state="disabled", text="Running…")
        self._stop_btn.configure(state="normal") # Enable stop button
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
        )
        threading.Thread(target=self._worker, args=(opts, self._stop_event,), daemon=True).start()

    def _stop(self):
        self._stop_event.set()
        self._stop_btn.configure(state="disabled", text="Stopping…")
        self._run_btn.configure(state="disabled") # Also disable run button while stopping
        self._log.log("Stopping QC pipeline...")


    def _worker(self, opts: dict, stop_event: threading.Event):
        q = self._q
        try:
            import qc_pipeline as qc_module
            qc_module = importlib.reload(qc_module)
            collect_images = qc_module.collect_images
            classify_image = qc_module.classify_image
            classify_image_with_backend = qc_module.classify_image_with_backend
            QCSettings = qc_module.QCSettings
            qc_module._reset_human_readable_log(opts["output_dir"])

            images = collect_images(opts["input_path"])
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
                if stop_event.is_set():
                    q.put(("stopped", "QC pipeline stopped by user."))
                    return

                q.put(("progress", i, len(images)))
                q.put(("log",
                       f"[{i+1}/{len(images)}] {os.path.basename(path)}"))
                try:
                    if use_backend:
                        r = classify_image_with_backend(
                            path, opts["backend_url"], opts["output_dir"],
                            model_name=opts["model_name"],
                            move_files=opts["move_files"],
                            api_key=opts["api_key"],
                        )
                    else:
                        r = classify_image(
                            path, opts["output_dir"],
                            move_files=opts["move_files"],
                            settings=settings,
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

            import json
            os.makedirs(opts["output_dir"], exist_ok=True)
            report = os.path.join(opts["output_dir"], "report.json")
            with open(report, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2)

            counts = {s: sum(1 for r in results if r["status"] == s)
                      for s in ("pass", "warning", "fail")}
            violations = sum(
                1 for r in results if r.get("route_folder") == "unscored"
            )
            q.put(("done",
                   f"Done — {len(results)} images  |  "
                   f"✓ {counts['pass']} pass   "
                   f"⚠ {counts['warning']} warning   "
                   f"✗ {counts['fail']} fail   "
                   f"⛔ {violations} violations"))
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
                    self._run_btn.configure(state="normal", text="Run QC")
                    self._stop_btn.configure(state="disabled", text="Stop QC")
                    self._running = False
                elif kind == "error":
                    self._log.stop()
                    messagebox.showerror("Error", msg[1])
                    self._run_btn.configure(state="normal", text="Run QC")
                    self._stop_btn.configure(state="disabled", text="Stop QC")
                    self._running = False
                elif kind == "stopped":
                    self._log.stop()
                    self._log.log(msg[1])
                    self._run_btn.configure(state="normal", text="Run QC")
                    self._stop_btn.configure(state="disabled", text="Stop QC")
                    self._running = False
        except queue.Empty:
            pass
        self.after(100, self._poll)


# ── Main window ───────────────────────────────────────────────────────────────

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
        ctk.CTkLabel(
            hdr, text="2D → SBS 3D  ·  AI Upscale  ·  Image QC",
            text_color="#666",
        ).pack(side="left", pady=10)

        # Tabs
        tabs = ctk.CTkTabview(self)
        tabs.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        for name, cls in [("Convert", ConvertTab), ("Upscale", UpscaleTab),
                          ("Judge", JudgeTab)]:
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
