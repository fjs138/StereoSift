#!/usr/bin/env python3
"""StereoSift — CustomTkinter GUI.

Two tabs:
  • Convert  — 2D images / videos → SBS 3D
  • Judge    — QC pipeline: pass / warning / fail sorting

All heavy work runs in a background thread so the UI stays responsive.
Progress and log output stream back to the main thread via a queue.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

# ── appearance ───────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Status-badge colours
STATUS_COLORS = {"pass": "#2ecc71", "warning": "#f39c12", "fail": "#e74c3c"}

# ── tiny helpers ─────────────────────────────────────────────────────────────

def _browse_input(var: ctk.StringVar, parent) -> None:
    """Ask for a file OR folder; prefer folder if nothing is selected."""
    path = filedialog.askopenfilename(parent=parent)
    if not path:
        path = filedialog.askdirectory(parent=parent)
    if path:
        var.set(path)


def _browse_folder(var: ctk.StringVar, parent) -> None:
    path = filedialog.askdirectory(parent=parent)
    if path:
        var.set(path)


def _row(parent, label: str, row: int, pady: int = 4):
    """Return a label + entry + browse-button row, packed into a grid."""
    ctk.CTkLabel(parent, text=label, anchor="w").grid(
        row=row, column=0, sticky="w", padx=(0, 8), pady=pady
    )

# ── shared log / progress widget ─────────────────────────────────────────────

class LogPanel(ctk.CTkFrame):
    """Scrollable log + progress bar used by both tabs."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._box = ctk.CTkTextbox(self, state="disabled", wrap="word", height=180)
        self._box.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 0))

        self._bar = ctk.CTkProgressBar(self, mode="indeterminate")
        self._bar.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self._bar.set(0)

        self._pct_label = ctk.CTkLabel(self, text="")
        self._pct_label.grid(row=2, column=0, sticky="w", padx=4)

    def log(self, text: str) -> None:
        self._box.configure(state="normal")
        self._box.insert("end", text + "\n")
        self._box.see("end")
        self._box.configure(state="disabled")

    def clear(self) -> None:
        self._box.configure(state="normal")
        self._box.delete("1.0", "end")
        self._box.configure(state="disabled")
        self._pct_label.configure(text="")
        self._bar.set(0)

    def start_spin(self) -> None:
        self._bar.configure(mode="indeterminate")
        self._bar.start()

    def set_progress(self, done: int, total: int) -> None:
        if total > 0:
            frac = done / total
            self._bar.configure(mode="determinate")
            self._bar.set(frac)
            self._pct_label.configure(text=f"{done} / {total}  ({frac*100:.0f}%)")

    def stop(self) -> None:
        self._bar.stop()
        self._bar.configure(mode="determinate")
        self._bar.set(1)


# ── Convert tab ───────────────────────────────────────────────────────────────

class ConvertTab(ctk.CTkFrame):
    """Tab for 2D → SBS 3D conversion (images and videos)."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)

        self._q: queue.Queue = queue.Queue()
        self._running = False

        self._build_inputs()
        self._build_options()
        self._build_log()
        self._poll()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_inputs(self):
        frm = ctk.CTkFrame(self)
        frm.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        frm.grid_columnconfigure(1, weight=1)

        # Input
        ctk.CTkLabel(frm, text="Input (file or folder)", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=5)
        self._input_var = ctk.StringVar()
        ctk.CTkEntry(frm, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", pady=5)
        ctk.CTkButton(frm, text="Browse", width=80,
                      command=lambda: _browse_input(self._input_var, self)).grid(
            row=0, column=2, padx=(6, 8), pady=5)

        # Output
        ctk.CTkLabel(frm, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        self._output_var = ctk.StringVar(value=os.path.join(os.getcwd(), "output"))
        ctk.CTkEntry(frm, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", pady=5)
        ctk.CTkButton(frm, text="Browse", width=80,
                      command=lambda: _browse_folder(self._output_var, self)).grid(
            row=1, column=2, padx=(6, 8), pady=5)

    def _build_options(self):
        frm = ctk.CTkFrame(self)
        frm.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        for c in range(6):
            frm.grid_columnconfigure(c, weight=1)

        # ── row 0: mode + method + viewing mode ──────────────────────────────
        ctk.CTkLabel(frm, text="Mode").grid(row=0, column=0, padx=8, pady=(8, 2))
        self._mode_var = ctk.StringVar(value="Images")
        ctk.CTkOptionMenu(frm, variable=self._mode_var,
                          values=["Images", "Video"],
                          command=self._on_mode_change).grid(
            row=1, column=0, padx=8, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(frm, text="Method").grid(row=0, column=1, padx=8, pady=(8, 2))
        self._method_var = ctk.StringVar(value="mesh_warping")
        ctk.CTkOptionMenu(frm, variable=self._method_var,
                          values=["mesh_warping", "grid_sampling"]).grid(
            row=1, column=1, padx=8, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(frm, text="Viewing mode").grid(row=0, column=2, padx=8, pady=(8, 2))
        self._sbs_mode_var = ctk.StringVar(value="parallel")
        ctk.CTkOptionMenu(frm, variable=self._sbs_mode_var,
                          values=["parallel", "cross-eyed"]).grid(
            row=1, column=2, padx=8, pady=(0, 8), sticky="ew")

        ctk.CTkLabel(frm, text="Depth model").grid(row=0, column=3, padx=8, pady=(8, 2))
        self._img_model_var = ctk.StringVar(value="depth_anything_v2_vitl_fp16.safetensors")
        from depth_model import AVAILABLE_MODELS
        ctk.CTkOptionMenu(frm, variable=self._img_model_var,
                          values=AVAILABLE_MODELS).grid(
            row=1, column=3, padx=8, pady=(0, 8), sticky="ew")

        # Video encoder (shown/hidden by mode)
        ctk.CTkLabel(frm, text="Video encoder").grid(row=0, column=4, padx=8, pady=(8, 2))
        self._vid_encoder_var = ctk.StringVar(value="vits")
        self._vid_encoder_menu = ctk.CTkOptionMenu(
            frm, variable=self._vid_encoder_var, values=["vits", "vitb", "vitl"])
        self._vid_encoder_menu.grid(row=1, column=4, padx=8, pady=(0, 8), sticky="ew")

        # Depth-only toggle
        self._depth_only_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(frm, text="Depth map only", variable=self._depth_only_var).grid(
            row=1, column=5, padx=8, pady=(0, 8))

        # ── row 2: sliders ───────────────────────────────────────────────────
        ctk.CTkLabel(frm, text="3D strength (depth scale)").grid(
            row=2, column=0, columnspan=2, padx=8, sticky="w")
        self._depth_scale_var = ctk.IntVar(value=40)
        ctk.CTkSlider(frm, from_=10, to=100, number_of_steps=90,
                      variable=self._depth_scale_var).grid(
            row=3, column=0, columnspan=2, padx=8, sticky="ew", pady=(0, 8))
        self._depth_scale_lbl = ctk.CTkLabel(frm, text="40")
        self._depth_scale_lbl.grid(row=3, column=2, padx=4, sticky="w")
        self._depth_scale_var.trace_add("write",
            lambda *_: self._depth_scale_lbl.configure(
                text=str(self._depth_scale_var.get())))

        ctk.CTkLabel(frm, text="Depth blur").grid(
            row=2, column=3, padx=8, sticky="w")
        self._blur_var = ctk.IntVar(value=7)
        ctk.CTkSlider(frm, from_=3, to=15, number_of_steps=6,
                      variable=self._blur_var).grid(
            row=3, column=3, padx=8, sticky="ew", pady=(0, 8))
        self._blur_lbl = ctk.CTkLabel(frm, text="7")
        self._blur_lbl.grid(row=3, column=4, padx=4, sticky="w")
        self._blur_var.trace_add("write",
            lambda *_: self._blur_lbl.configure(text=str(self._blur_var.get())))

        self._on_mode_change("Images")

    def _build_log(self):
        self._log = LogPanel(self)
        self._log.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        self.grid_rowconfigure(2, weight=1)

        self._run_btn = ctk.CTkButton(self, text="Convert", height=38,
                                      command=self._run)
        self._run_btn.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="ew")

    # ── logic ─────────────────────────────────────────────────────────────────

    def _on_mode_change(self, value):
        state = "normal" if value == "Video" else "disabled"
        self._vid_encoder_menu.configure(state=state)

    def _run(self):
        if self._running:
            return
        inp = self._input_var.get().strip()
        out = self._output_var.get().strip()
        if not inp:
            messagebox.showwarning("Missing input", "Please select an input file or folder.")
            return
        if not out:
            messagebox.showwarning("Missing output", "Please select an output folder.")
            return

        self._running = True
        self._run_btn.configure(state="disabled", text="Running…")
        self._log.clear()
        self._log.start_spin()

        opts = dict(
            input_path=inp,
            output_dir=out,
            mode=self._mode_var.get(),
            img_model=self._img_model_var.get(),
            vid_encoder=self._vid_encoder_var.get(),
            method=self._method_var.get(),
            sbs_mode=self._sbs_mode_var.get(),
            depth_scale=self._depth_scale_var.get(),
            sbs_blur=self._blur_var.get(),
            depth_only=self._depth_only_var.get(),
        )
        threading.Thread(target=self._worker, args=(opts,), daemon=True).start()

    def _worker(self, opts: dict):
        q = self._q
        try:
            import torch
            from depth_model import load_depth_model
            from convert import collect_images, collect_videos, convert_one, get_device

            device = get_device()
            is_video = opts["mode"] == "Video"

            if is_video:
                from video_converter import load_video_depth_model, convert_video_to_sbs
                q.put(("log", f"Loading Video Depth Anything ({opts['vid_encoder']})…"))
                model, dtype, is_metric = load_video_depth_model(
                    encoder=opts["vid_encoder"], device=device)
                files = collect_videos(opts["input_path"])
                q.put(("log", f"Found {len(files)} video(s)"))
                for i, path in enumerate(files):
                    q.put(("log", f"[{i+1}/{len(files)}] {os.path.basename(path)}"))
                    q.put(("progress", i, len(files)))
                    convert_video_to_sbs(
                        video_path=path,
                        output_dir=opts["output_dir"],
                        model=model, device=device, dtype=dtype, is_metric=is_metric,
                        sbs_method=opts["method"],
                        depth_scale=opts["depth_scale"],
                        sbs_mode=opts["sbs_mode"],
                        sbs_blur=opts["sbs_blur"],
                        depth_only=opts["depth_only"],
                    )
                    q.put(("progress", i + 1, len(files)))
            else:
                q.put(("log", f"Loading image depth model…"))
                model, dtype, is_metric = load_depth_model(
                    opts["img_model"], device)
                files = collect_images(opts["input_path"])
                q.put(("log", f"Found {len(files)} image(s)"))
                for i, path in enumerate(files):
                    q.put(("log", f"[{i+1}/{len(files)}] {os.path.basename(path)}"))
                    q.put(("progress", i, len(files)))
                    convert_one(
                        model, path, opts["output_dir"], device, dtype, is_metric,
                        depth_only=opts["depth_only"],
                        depth_input_scale=0.5,
                        sbs_method=opts["method"],
                        depth_scale=opts["depth_scale"],
                        sbs_mode=opts["sbs_mode"],
                        sbs_blur=opts["sbs_blur"],
                        log=lambda msg: q.put(("log", msg)),
                    )
                    q.put(("progress", i + 1, len(files)))

            q.put(("done", f"Finished — {len(files)} file(s) converted."))
        except Exception as exc:
            import traceback
            q.put(("log", traceback.format_exc()))
            q.put(("error", str(exc)))

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
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


# ── Judge tab ─────────────────────────────────────────────────────────────────

class JudgeTab(ctk.CTkFrame):
    """Tab for QC image sorting: pass / warning / fail."""

    def __init__(self, master, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)

        self._q: queue.Queue = queue.Queue()
        self._running = False
        self._results: list[dict] = []

        self._build_inputs()
        self._build_options()
        self._build_results()
        self._build_log()
        self._poll()

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_inputs(self):
        frm = ctk.CTkFrame(self)
        frm.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        frm.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frm, text="Input folder", anchor="w").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=5)
        self._input_var = ctk.StringVar()
        ctk.CTkEntry(frm, textvariable=self._input_var).grid(
            row=0, column=1, sticky="ew", pady=5)
        ctk.CTkButton(frm, text="Browse", width=80,
                      command=lambda: _browse_folder(self._input_var, self)).grid(
            row=0, column=2, padx=(6, 8), pady=5)

        ctk.CTkLabel(frm, text="Output folder", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        self._output_var = ctk.StringVar(value=os.path.join(os.getcwd(), "output", "qc"))
        ctk.CTkEntry(frm, textvariable=self._output_var).grid(
            row=1, column=1, sticky="ew", pady=5)
        ctk.CTkButton(frm, text="Browse", width=80,
                      command=lambda: _browse_folder(self._output_var, self)).grid(
            row=1, column=2, padx=(6, 8), pady=5)

    def _build_options(self):
        frm = ctk.CTkFrame(self)
        frm.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        frm.grid_columnconfigure(1, weight=1)

        # Vision backend URL (optional)
        ctk.CTkLabel(frm, text="Vision backend URL\n(blank = basic checks only)",
                     anchor="w", justify="left").grid(
            row=0, column=0, sticky="w", padx=(8, 6), pady=5)
        self._backend_var = ctk.StringVar()
        ctk.CTkEntry(frm, textvariable=self._backend_var,
                     placeholder_text="http://127.0.0.1:1234/v1/chat/completions").grid(
            row=0, column=1, sticky="ew", padx=(0, 8), pady=5)

        # Model name
        ctk.CTkLabel(frm, text="Vision model name", anchor="w").grid(
            row=1, column=0, sticky="w", padx=(8, 6), pady=5)
        self._model_var = ctk.StringVar(value="llama-3.2-11b-vision-instruct")
        ctk.CTkEntry(frm, textvariable=self._model_var).grid(
            row=1, column=1, sticky="ew", padx=(0, 8), pady=5)

        # Move vs copy
        self._move_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(frm, text="Move originals (destructive — copies by default)",
                        variable=self._move_var,
                        text_color="#e74c3c").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 8))

    def _build_results(self):
        """Scrollable results table showing filename, status badge, score, issues."""
        outer = ctk.CTkFrame(self)
        outer.grid(row=2, column=0, sticky="nsew", padx=12, pady=4)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(outer, text="Results", anchor="w",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, sticky="w", padx=8, pady=(6, 2))

        # We embed a plain Tkinter canvas + frame for the scrollable rows
        canvas = tk.Canvas(outer, bg="#2b2b2b", highlightthickness=0, height=200)
        scrollbar = ctk.CTkScrollbar(outer, command=canvas.yview)
        self._results_frame = ctk.CTkFrame(canvas, fg_color="transparent")

        self._results_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._results_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=1, column=0, sticky="nsew", padx=(4, 0))
        scrollbar.grid(row=1, column=1, sticky="ns")
        outer.grid_columnconfigure(0, weight=1)

        self._results_canvas = canvas

        # Header row
        for col, (text, w) in enumerate([
            ("File", 260), ("Status", 80), ("Score", 60), ("Issues", 300)
        ]):
            ctk.CTkLabel(self._results_frame, text=text,
                         font=ctk.CTkFont(weight="bold"),
                         width=w, anchor="w").grid(
                row=0, column=col, padx=4, pady=2, sticky="w")

    def _add_result_row(self, result: dict, row_idx: int):
        status = result.get("status", "warning")
        color  = STATUS_COLORS.get(status, "#888")
        issues = ", ".join(result.get("issues") or []) or "—"
        score  = result.get("score", "—")
        fname  = os.path.basename(result.get("filename", ""))

        ctk.CTkLabel(self._results_frame, text=fname, anchor="w", width=260,
                     wraplength=255).grid(row=row_idx, column=0, padx=4, pady=1, sticky="w")
        ctk.CTkLabel(self._results_frame, text=status.upper(), anchor="center",
                     width=80, fg_color=color, corner_radius=6,
                     text_color="white").grid(row=row_idx, column=1, padx=4, pady=1)
        ctk.CTkLabel(self._results_frame, text=str(score), anchor="center",
                     width=60).grid(row=row_idx, column=2, padx=4, pady=1)
        ctk.CTkLabel(self._results_frame, text=issues, anchor="w", width=300,
                     wraplength=295).grid(row=row_idx, column=3, padx=4, pady=1, sticky="w")

    def _build_log(self):
        self._log = LogPanel(self)
        self._log.grid(row=3, column=0, sticky="ew", padx=12, pady=4)

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=4, column=0, padx=12, pady=(4, 12), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        self._run_btn = ctk.CTkButton(btn_row, text="Run QC", height=38,
                                      command=self._run)
        self._run_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self._open_btn = ctk.CTkButton(btn_row, text="Open output folder", height=38,
                                       fg_color="#555", hover_color="#666",
                                       command=self._open_output)
        self._open_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")

    # ── logic ─────────────────────────────────────────────────────────────────

    def _open_output(self):
        path = self._output_var.get().strip()
        if not path or not os.path.isdir(path):
            messagebox.showinfo("Not found", "Output folder does not exist yet.")
            return
        import subprocess, platform
        if platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        elif platform.system() == "Windows":
            subprocess.Popen(["explorer", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _run(self):
        if self._running:
            return
        inp = self._input_var.get().strip()
        out = self._output_var.get().strip()
        if not inp:
            messagebox.showwarning("Missing input", "Please select an input folder.")
            return

        # Warn before moving
        if self._move_var.get():
            if not messagebox.askyesno("Move files?",
                "This will MOVE originals into the output subfolders.\n"
                "This cannot be undone. Continue?"):
                return

        self._running = True
        self._run_btn.configure(state="disabled", text="Running…")
        self._log.clear()
        self._log.start_spin()

        # Clear old results
        for w in self._results_frame.winfo_children():
            if int(w.grid_info().get("row", 0)) > 0:
                w.destroy()
        self._results.clear()

        opts = dict(
            input_path=inp,
            output_dir=out,
            backend_url=self._backend_var.get().strip() or None,
            model_name=self._model_var.get().strip(),
            move_files=self._move_var.get(),
        )
        threading.Thread(target=self._worker, args=(opts,), daemon=True).start()

    def _worker(self, opts: dict):
        q = self._q
        try:
            from qc_pipeline import collect_images, classify_image, classify_image_with_backend
            images = collect_images(opts["input_path"])
            if not images:
                q.put(("error", f"No images found in {opts['input_path']}"))
                return

            q.put(("log", f"Found {len(images)} image(s)"))
            results = []
            for i, path in enumerate(images):
                q.put(("progress", i, len(images)))
                q.put(("log", f"[{i+1}/{len(images)}] {os.path.basename(path)}"))
                if opts["backend_url"]:
                    r = classify_image_with_backend(
                        path, opts["backend_url"], opts["output_dir"],
                        model_name=opts["model_name"],
                        move_files=opts["move_files"],
                    )
                else:
                    r = classify_image(path, opts["output_dir"],
                                       move_files=opts["move_files"])
                results.append(r)
                q.put(("result", r, i + 1))
                q.put(("progress", i + 1, len(images)))

            # Write report
            import json
            os.makedirs(opts["output_dir"], exist_ok=True)
            report = os.path.join(opts["output_dir"], "report.json")
            with open(report, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            counts = {s: sum(1 for r in results if r["status"] == s)
                      for s in ("pass", "warning", "fail")}
            q.put(("done",
                   f"Done — {len(results)} images  |  "
                   f"✓ {counts['pass']} pass  "
                   f"⚠ {counts['warning']} warning  "
                   f"✗ {counts['fail']} fail"))
        except Exception as exc:
            import traceback
            q.put(("log", traceback.format_exc()))
            q.put(("error", str(exc)))

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self._log.log(msg[1])
                elif kind == "progress":
                    self._log.set_progress(msg[1], msg[2])
                elif kind == "result":
                    result, row = msg[1], msg[2]
                    self._results.append(result)
                    self._add_result_row(result, row)
                elif kind == "done":
                    self._log.stop()
                    self._log.log(msg[1])
                    self._run_btn.configure(state="normal", text="Run QC")
                    self._running = False
                elif kind == "error":
                    self._log.stop()
                    messagebox.showerror("Error", msg[1])
                    self._run_btn.configure(state="normal", text="Run QC")
                    self._running = False
        except queue.Empty:
            pass
        self.after(100, self._poll)


# ── Main window ───────────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("StereoSift")
        self.geometry("860x740")
        self.minsize(700, 580)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, corner_radius=0, fg_color=("#1a1a2e", "#1a1a2e"))
        hdr.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(hdr, text="StereoSift",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color="#7eb3ff").pack(side="left", padx=16, pady=10)
        ctk.CTkLabel(hdr, text="2D → SBS 3D  ·  Image QC",
                     text_color="#888").pack(side="left", pady=10)

        # Tabs
        tabs = ctk.CTkTabview(self)
        tabs.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        tabs.add("Convert")
        tabs.add("Judge")

        tabs.tab("Convert").grid_columnconfigure(0, weight=1)
        tabs.tab("Convert").grid_rowconfigure(0, weight=1)
        ConvertTab(tabs.tab("Convert")).grid(row=0, column=0, sticky="nsew")

        tabs.tab("Judge").grid_columnconfigure(0, weight=1)
        tabs.tab("Judge").grid_rowconfigure(0, weight=1)
        JudgeTab(tabs.tab("Judge")).grid(row=0, column=0, sticky="nsew")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # On macOS the working directory needs to be the project root so relative
    # imports (models/, output/, etc.) resolve correctly.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    app = App()
    app.mainloop()
