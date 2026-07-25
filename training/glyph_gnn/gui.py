"""Interactive GUI for glyph character segmentation.

Renders font outlines, GNN edge predictions, and detected character groups.
Preprocessing and ONNX inference run in ``glyph-infer export``; threshold
changes re-run union-find locally for instant feedback.

Usage:
    python -m glyph_gnn.gui
    python -m glyph_gnn.gui --model model.onnx --font fonts/body/arial.ttf
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .postprocess import group_characters, sort_groups_by_x

GROUP_COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


@dataclass
class ExportData:
    text: str
    contours: list[list[list[float]]]
    node_x: list[float]
    node_y: list[float]
    node_contour_ids: list[int]
    edge_src: list[int]
    edge_dst: list[int]
    edge_prob: list[float]
    timing_preprocess_ms: float
    timing_inference_ms: float

    @classmethod
    def from_json(cls, raw: dict) -> ExportData:
        nodes = raw["nodes"]
        edges = raw["edges"]
        timing = raw["timing_ms"]
        return cls(
            text=raw["text"],
            contours=raw["contours"],
            node_x=nodes["x"],
            node_y=nodes["y"],
            node_contour_ids=nodes["contour_id"],
            edge_src=edges["src"],
            edge_dst=edges["dst"],
            edge_prob=edges["prob"],
            timing_preprocess_ms=timing["preprocess"],
            timing_inference_ms=timing["inference"],
        )

    @property
    def num_nodes(self) -> int:
        return len(self.node_x)


def find_infer_bin(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    root = Path(__file__).resolve().parents[2]
    for name in ("glyph-infer.exe", "glyph-infer"):
        for sub in ("release", "debug"):
            candidate = root / "target" / sub / name
            if candidate.is_file():
                return candidate
    found = shutil.which("glyph-infer")
    if found:
        return Path(found)
    return root / "target" / "release" / "glyph-infer.exe"


def run_export(
    infer_bin: Path,
    model: Path,
    font: Path,
    text: str,
    tracking: float,
    cpu: bool,
    knn: int,
    radius: float,
    contour_bridge: float,
) -> ExportData:
    cmd = [
        str(infer_bin),
        "--model",
        str(model),
        "--knn",
        str(knn),
        "--radius",
        str(radius),
        "--contour-bridge",
        str(contour_bridge),
        "export",
        "--font",
        str(font),
        "--text",
        text,
        f"--tracking={tracking}",
    ]
    if cpu:
        cmd.append("--cpu")
    # JSON from glyph-infer is UTF-8; Windows defaults to cp932 for text=True.
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        stdout = proc.stdout.decode("utf-8", errors="replace").strip()
        detail = stderr or stdout or f"exit {proc.returncode}"
        raise RuntimeError(detail)
    if not proc.stdout:
        raise RuntimeError("glyph-infer export produced no output")
    return ExportData.from_json(json.loads(proc.stdout.decode("utf-8")))


def contour_to_group(contour_groups: list[list[int]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for gid, contours in enumerate(contour_groups):
        for cid in contours:
            mapping[cid] = gid
    return mapping


class GlyphGuiApp(tk.Tk):
    def __init__(
        self,
        model: Path,
        font: Path,
        infer_bin: Path,
        text: str,
        tracking: float,
        threshold: float,
        cpu: bool,
        knn: int,
        radius: float,
        contour_bridge: float,
    ) -> None:
        super().__init__()
        self.title("Glyph Character Segmentation")
        self.geometry("1100x760")
        self.minsize(900, 600)

        self.model_path = tk.StringVar(value=str(model))
        self.font_path = tk.StringVar(value=str(font))
        self.infer_bin_path = tk.StringVar(value=str(infer_bin))
        self.text_var = tk.StringVar(value=text)
        self.tracking_var = tk.DoubleVar(value=tracking)
        self.threshold_var = tk.DoubleVar(value=threshold)
        self.cpu_var = tk.BooleanVar(value=cpu)
        self.knn_var = tk.IntVar(value=knn)
        self.radius_var = tk.DoubleVar(value=radius)
        self.contour_bridge_var = tk.DoubleVar(value=contour_bridge)
        self.show_graph_var = tk.BooleanVar(value=False)
        self.show_edges_var = tk.BooleanVar(value=False)

        self._data: ExportData | None = None
        self._busy = False
        self._threshold_job: str | None = None
        self._graph_job: str | None = None

        self._build_controls()
        self._build_canvas()
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, anchor="w").pack(
            fill="x", padx=8, pady=(0, 6)
        )

        self.after(200, self.run_inference)

    def _build_controls(self) -> None:
        panel = ttk.Frame(self, padding=8)
        panel.pack(fill="x")

        def row(label: str, var: tk.Variable, browse: str | None = None) -> None:
            frame = ttk.Frame(panel)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label, width=12).pack(side="left")
            entry = ttk.Entry(frame, textvariable=var)
            entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
            if browse:
                ttk.Button(frame, text="…", width=3, command=lambda: self._browse(var, browse)).pack(
                    side="left"
                )

        row("Model", self.model_path, "onnx")
        row("Font", self.font_path, "font")
        row("Infer bin", self.infer_bin_path, "infer")

        text_row = ttk.Frame(panel)
        text_row.pack(fill="x", pady=2)
        ttk.Label(text_row, text="Text", width=12).pack(side="left")
        text_entry = ttk.Entry(text_row, textvariable=self.text_var)
        text_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
        text_entry.bind("<Return>", lambda _e: self.run_inference())
        ttk.Button(text_row, text="Run", command=self.run_inference).pack(side="left")

        slider_row = ttk.Frame(panel)
        slider_row.pack(fill="x", pady=4)
        self._tracking_label = ttk.Label(slider_row, text="Tracking", width=12)
        self._tracking_label.pack(side="left")
        ttk.Scale(
            slider_row,
            from_=-0.3,
            to=0.1,
            variable=self.tracking_var,
            orient="horizontal",
            command=self._on_tracking_change,
        ).pack(side="left", fill="x", expand=True, padx=4)
        self._tracking_job: str | None = None

        threshold_row = ttk.Frame(panel)
        threshold_row.pack(fill="x", pady=4)
        self._threshold_label = ttk.Label(threshold_row, text="Threshold", width=12)
        self._threshold_label.pack(side="left")
        ttk.Scale(
            threshold_row,
            from_=0.0,
            to=1.0,
            variable=self.threshold_var,
            orient="horizontal",
            command=self._on_threshold_change,
        ).pack(side="left", fill="x", expand=True, padx=4)

        graph_row = ttk.Frame(panel)
        graph_row.pack(fill="x", pady=2)
        ttk.Label(graph_row, text="Graph", width=12).pack(side="left")
        ttk.Label(graph_row, text="kNN").pack(side="left", padx=(4, 2))
        knn_entry = ttk.Spinbox(graph_row, from_=1, to=64, width=5, textvariable=self.knn_var)
        knn_entry.pack(side="left")
        knn_entry.bind("<Return>", lambda _e: self._schedule_graph_rerun())
        knn_entry.bind("<FocusOut>", lambda _e: self._schedule_graph_rerun())
        ttk.Label(graph_row, text="radius").pack(side="left", padx=(8, 2))
        radius_entry = ttk.Entry(graph_row, width=6, textvariable=self.radius_var)
        radius_entry.pack(side="left")
        radius_entry.bind("<Return>", lambda _e: self._schedule_graph_rerun())
        radius_entry.bind("<FocusOut>", lambda _e: self._schedule_graph_rerun())
        ttk.Label(graph_row, text="bridge").pack(side="left", padx=(8, 2))
        bridge_entry = ttk.Entry(graph_row, width=6, textvariable=self.contour_bridge_var)
        bridge_entry.pack(side="left")
        bridge_entry.bind("<Return>", lambda _e: self._schedule_graph_rerun())
        bridge_entry.bind("<FocusOut>", lambda _e: self._schedule_graph_rerun())

        opts = ttk.Frame(panel)
        opts.pack(fill="x", pady=2)
        ttk.Checkbutton(opts, text="CPU only", variable=self.cpu_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            opts, text="Show graph nodes", variable=self.show_graph_var, command=self.redraw
        ).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            opts, text="Show edges (p≥t)", variable=self.show_edges_var, command=self.redraw
        ).pack(side="left")

        self._update_slider_labels()

    def _build_canvas(self) -> None:
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_title("Character groups")
        self.ax.set_xlabel("x (em)")
        self.ax.set_ylabel("y (em)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

    def _browse(self, var: tk.StringVar, kind: str) -> None:
        if kind == "onnx":
            path = filedialog.askopenfilename(filetypes=[("ONNX model", "*.onnx"), ("All", "*.*")])
        elif kind == "font":
            path = filedialog.askopenfilename(
                filetypes=[("Fonts", "*.ttf *.otf *.ttc"), ("All", "*.*")]
            )
        else:
            path = filedialog.askopenfilename()
        if path:
            var.set(path)

    def _update_slider_labels(self) -> None:
        self._tracking_label.config(text=f"Tracking {self.tracking_var.get():+.3f}")
        self._threshold_label.config(text=f"Threshold {self.threshold_var.get():.2f}")

    def _on_tracking_change(self, _value: str) -> None:
        self._update_slider_labels()
        if self._tracking_job is not None:
            self.after_cancel(self._tracking_job)
        self._tracking_job = self.after(300, self._tracking_rerun)

    def _tracking_rerun(self) -> None:
        self._tracking_job = None
        self.run_inference()

    def _schedule_graph_rerun(self) -> None:
        if self._graph_job is not None:
            self.after_cancel(self._graph_job)
        self._graph_job = self.after(300, self._graph_rerun)

    def _graph_rerun(self) -> None:
        self._graph_job = None
        self.run_inference()

    def _on_threshold_change(self, _value: str) -> None:
        self._update_slider_labels()
        if self._threshold_job is not None:
            self.after_cancel(self._threshold_job)
        self._threshold_job = self.after(60, self._threshold_redraw)

    def _threshold_redraw(self) -> None:
        self._threshold_job = None
        self.redraw()

    def run_inference(self) -> None:
        if self._busy:
            return
        model = Path(self.model_path.get())
        font = Path(self.font_path.get())
        infer_bin = Path(self.infer_bin_path.get())
        text = self.text_var.get()
        if not model.is_file():
            messagebox.showerror("Missing model", f"ONNX model not found:\n{model}")
            return
        if not font.is_file():
            messagebox.showerror("Missing font", f"Font file not found:\n{font}")
            return
        if not infer_bin.is_file():
            messagebox.showerror(
                "Missing glyph-infer",
                f"Inference binary not found:\n{infer_bin}\n\n"
                "Build it with: cargo build --release -p glyph-infer",
            )
            return
        if not text:
            messagebox.showwarning("Empty text", "Enter some text to segment.")
            return

        self._busy = True
        self.status_var.set("Running inference…")
        self.update_idletasks()
        try:
            self._data = run_export(
                infer_bin,
                model,
                font,
                text,
                self.tracking_var.get(),
                self.cpu_var.get(),
                int(self.knn_var.get()),
                float(self.radius_var.get()),
                float(self.contour_bridge_var.get()),
            )
        except (RuntimeError, json.JSONDecodeError, KeyError) as exc:
            messagebox.showerror("Inference failed", str(exc))
            self.status_var.set("Inference failed")
            self._busy = False
            return

        self._busy = False
        self.redraw()

    def redraw(self) -> None:
        data = self._data
        if data is None:
            return

        threshold = self.threshold_var.get()
        groups = group_characters(
            data.num_nodes,
            data.node_contour_ids,
            data.edge_src,
            data.edge_dst,
            data.edge_prob,
            threshold,
        )
        groups = sort_groups_by_x(groups, data.node_x, data.node_contour_ids)
        contour_group = contour_to_group(groups)

        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_title(f"“{data.text}” — {len(groups)} character group(s)")
        self.ax.set_xlabel("x (em)")
        self.ax.set_ylabel("y (em)")

        for cid, contour in enumerate(data.contours):
            if len(contour) < 2:
                continue
            gid = contour_group.get(cid, 0)
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            xs = [p[0] for p in contour] + [contour[0][0]]
            ys = [p[1] for p in contour] + [contour[0][1]]
            self.ax.plot(xs, ys, color=color, linewidth=1.4, solid_capstyle="round")

        if self.show_edges_var.get():
            segments = []
            colors = []
            for src, dst, prob in zip(
                data.edge_src, data.edge_dst, data.edge_prob, strict=True
            ):
                if prob < threshold:
                    continue
                segments.append(
                    [
                        (data.node_x[src], data.node_y[src]),
                        (data.node_x[dst], data.node_y[dst]),
                    ]
                )
                colors.append(plt.cm.RdYlGn(prob))
            if segments:
                self.ax.add_collection(
                    LineCollection(segments, colors=colors, linewidths=0.6, alpha=0.35)
                )

        if self.show_graph_var.get():
            for node, contour_id in enumerate(data.node_contour_ids):
                gid = contour_group.get(contour_id, 0)
                color = GROUP_COLORS[gid % len(GROUP_COLORS)]
                self.ax.plot(
                    data.node_x[node],
                    data.node_y[node],
                    "o",
                    color=color,
                    markersize=2.5,
                    alpha=0.85,
                )

        legend_handles = [
            Patch(facecolor=GROUP_COLORS[i % len(GROUP_COLORS)], label=f"char {i}")
            for i in range(len(groups))
        ]
        if legend_handles:
            self.ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        self.ax.autoscale()
        self.ax.margins(0.08)
        self.canvas.draw_idle()

        self.status_var.set(
            f"{data.num_nodes} nodes | {len(data.edge_prob)} edges | "
            f"kNN {int(self.knn_var.get())} r {float(self.radius_var.get()):.2f} "
            f"bridge {float(self.contour_bridge_var.get()):.2f} | "
            f"pre {data.timing_preprocess_ms:.1f}ms | "
            f"inf {data.timing_inference_ms:.1f}ms | threshold {threshold:.2f}"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Interactive glyph segmentation GUI")
    ap.add_argument("--model", type=Path, default=root / "model.onnx")
    ap.add_argument("--font", type=Path, default=root / "fonts" / "body" / "arial.ttf")
    ap.add_argument("--infer-bin", type=Path, default=None)
    ap.add_argument("--text", default="Overlap")
    ap.add_argument("--tracking", type=float, default=-0.12)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--knn", type=int, default=8, help="graph kNN (must match training)")
    ap.add_argument("--radius", type=float, default=0.25, help="graph radius in em (must match training)")
    ap.add_argument("--contour-bridge", type=float, default=0.35,
                    help="contour-pair bridge distance in em (must match training)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    try:
        app = GlyphGuiApp(
            model=args.model,
            font=args.font,
            infer_bin=find_infer_bin(args.infer_bin),
            text=args.text,
            tracking=args.tracking,
            threshold=args.threshold,
            cpu=args.cpu,
            knn=args.knn,
            radius=args.radius,
            contour_bridge=args.contour_bridge,
        )
    except tk.TclError as exc:
        print(f"GUI unavailable: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    app.mainloop()


if __name__ == "__main__":
    main()
