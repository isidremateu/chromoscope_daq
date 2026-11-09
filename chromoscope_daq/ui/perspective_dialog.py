"""Perspective correction dialog (README 'Perspective correction menu').

Load an image from disk (or use the last acquired frame), click 4 origin
points on it, enter each point's destination in mm plus the output X/Y
axis ranges, and save/load the whole definition as json. On Apply the
finished :class:`PerspectiveTransform` is handed to the main window.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk

from . import display
from ..perspective import (PerspectiveError, PerspectiveTransform,
                           load_image_rgb8)

#: Marker colors for the 4 selectable points.
_POINT_COLORS = ("#ff4040", "#40ff40", "#4080ff", "#ffff40")

CANVAS_SIZE = (640, 480)


class PerspectiveDialog(tk.Toplevel):
    """Modal-ish editor for one perspective correction definition."""

    def __init__(self, master, transform: Optional[PerspectiveTransform],
                 on_apply: Callable[[PerspectiveTransform], None],
                 get_last_frame: Optional[Callable[[], Optional[object]]]
                 = None) -> None:
        super().__init__(master)
        self.title("Perspective correction")
        self.transient(master)
        self.grab_set()
        self._on_apply_cb = on_apply
        self._get_last_frame = get_last_frame
        self._json_path: Optional[str] = None
        self._image_rgb8: Optional[np.ndarray] = None
        self._image_path: Optional[str] = None
        self._geom: Optional[display.ViewGeometry] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._source_points: List[Tuple[float, float]] = []
        self._marker_items: List[int] = []
        self._px_labels: List[ttk.Label] = []
        self._build_ui()
        if transform is not None:
            self._fill_from_transform(transform)

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)

        buttons = ttk.Frame(top)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Load image from disk...",
                   command=self._load_image).pack(side="left")
        ttk.Button(buttons, text="Use last frame",
                   command=self._use_last_frame).pack(side="left", padx=4)
        ttk.Button(buttons, text="Clear points",
                   command=self._clear_points).pack(side="left")
        self._image_info = tk.StringVar(value="No image loaded.")
        ttk.Label(buttons, textvariable=self._image_info).pack(
            side="left", padx=8)

        self._canvas = tk.Canvas(top, width=CANVAS_SIZE[0],
                                 height=CANVAS_SIZE[1], bg="black")
        self._canvas.pack(pady=6)
        self._canvas.bind("<Button-1>", self._on_click)
        self._hint = tk.StringVar(
            value="Load an image, then click 4 origin points on it "
                  "(top-left of the beam first — any order works).")
        ttk.Label(top, textvariable=self._hint).pack(anchor="w")

        body = ttk.Frame(top)
        body.pack(fill="x", pady=(4, 0))
        dest_box = ttk.LabelFrame(body, text="Destinations [mm]", padding=4)
        dest_box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._dest_vars: List[Tuple[tk.StringVar, tk.StringVar]] = []
        for index in range(4):
            ttk.Label(dest_box,
                      text=f"P{index + 1} (px)").grid(
                row=index, column=0, sticky="w")
            px_label = ttk.Label(dest_box, text="-")
            px_label.grid(row=index, column=1, sticky="w", padx=(0, 8))
            self._px_labels.append(px_label)
            ttk.Label(dest_box, text="X:").grid(row=index, column=2)
            x_var = tk.StringVar()
            ttk.Entry(dest_box, textvariable=x_var, width=8).grid(
                row=index, column=3, padx=2)
            ttk.Label(dest_box, text="Y:").grid(row=index, column=4)
            y_var = tk.StringVar()
            ttk.Entry(dest_box, textvariable=y_var, width=8).grid(
                row=index, column=5, padx=2)
            self._dest_vars.append((x_var, y_var))

        range_box = ttk.LabelFrame(body, text="Output axis ranges [mm]",
                                   padding=4)
        range_box.pack(side="left", fill="y")
        self._x_min_var = tk.StringVar()
        self._x_max_var = tk.StringVar()
        self._y_min_var = tk.StringVar()
        self._y_max_var = tk.StringVar()
        for row, (label, var) in enumerate((
                ("X min", self._x_min_var), ("X max", self._x_max_var),
                ("Y min", self._y_min_var), ("Y max", self._y_max_var))):
            ttk.Label(range_box, text=label).grid(row=row, column=0,
                                                  sticky="w")
            ttk.Entry(range_box, textvariable=var, width=8).grid(
                row=row, column=1, padx=2, pady=1)

        bottom = ttk.Frame(top)
        bottom.pack(fill="x", pady=(6, 0))
        ttk.Button(bottom, text="Save json...", command=self._save_json
                   ).pack(side="left")
        ttk.Button(bottom, text="Load json...", command=self._load_json
                   ).pack(side="left", padx=4)
        ttk.Button(bottom, text="Apply", command=self._apply).pack(
            side="right")
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(
            side="right", padx=4)

    # -- image handling ---------------------------------------------------

    def _load_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Load image for point selection",
            filetypes=(("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
                       ("All files", "*.*")))
        if not path:
            return
        try:
            image = load_image_rgb8(path)
        except PerspectiveError as error:
            messagebox.showerror("Perspective correction", str(error),
                                 parent=self)
            return
        self._set_image(image, path)

    def _use_last_frame(self) -> None:
        if self._get_last_frame is None:
            return
        bundle = self._get_last_frame()
        if bundle is None:
            messagebox.showinfo(
                "Perspective correction",
                "No frame available yet — acquire something first.",
                parent=self)
            return
        self._set_image(display.bundle_to_rgb8(bundle), "<last frame>")

    def _set_image(self, image: np.ndarray, name: str) -> None:
        self._image_rgb8 = image
        self._image_path = name
        self._clear_points()
        self._image_info.set(f"{name}  ({image.shape[1]} x {image.shape[0]})")
        self._draw_image()

    def _draw_image(self) -> None:
        image = self._image_rgb8
        geom = display.compute_view((image.shape[1], image.shape[0]),
                                    (self._canvas.winfo_width()
                                     or CANVAS_SIZE[0],
                                     self._canvas.winfo_height()
                                     or CANVAS_SIZE[1]))
        self._geom = geom
        crop = display.crop_for_view(image, geom)
        target = (max(1, int(geom.crop_w * geom.scale)),
                  max(1, int(geom.crop_h * geom.scale)))
        self._photo = ImageTk.PhotoImage(
            Image.fromarray(crop).resize(target, Image.NEAREST))
        self._canvas.delete("all")
        self._canvas.create_image(geom.offset_x, geom.offset_y,
                                  image=self._photo, anchor="nw")
        self._redraw_markers()

    # -- point selection ---------------------------------------------------

    def _on_click(self, event) -> None:
        if self._image_rgb8 is None or self._geom is None:
            return
        if len(self._source_points) >= 4:
            self._hint.set("4 points selected — use Clear points to redo.")
            return
        x, y = self._geom.canvas_to_image(event.x, event.y)
        self._source_points.append((x, y))
        self._hint.set(f"{4 - len(self._source_points)} point(s) left.")
        self._redraw_markers()

    def _redraw_markers(self) -> None:
        for item in self._marker_items:
            self._canvas.delete(item)
        self._marker_items.clear()
        if self._geom is None:
            return
        for index, (x, y) in enumerate(self._source_points):
            cx, cy = self._geom.image_to_canvas(x, y)
            color = _POINT_COLORS[index % len(_POINT_COLORS)]
            self._marker_items.append(self._canvas.create_oval(
                cx - 5, cy - 5, cx + 5, cy + 5, outline=color, width=2))
            self._marker_items.append(self._canvas.create_text(
                cx + 10, cy, text=f"P{index + 1}", fill=color,
                anchor="w"))
            if index < len(self._px_labels):
                self._px_labels[index].config(text=f"({x:.0f}, {y:.0f})")
        for index in range(len(self._source_points), 4):
            self._px_labels[index].config(text="-")

    def _clear_points(self) -> None:
        self._source_points.clear()
        self._redraw_markers()
        if self._image_rgb8 is not None:
            self._hint.set("Click 4 origin points on the image.")

    # -- json / apply -------------------------------------------------------

    def _build_transform(self) -> PerspectiveTransform:
        dest = tuple((float(x.get()), float(y.get()))
                     for x, y in self._dest_vars)
        ranges = (float(self._x_min_var.get()), float(self._x_max_var.get()),
                  float(self._y_min_var.get()), float(self._y_max_var.get()))
        transform = PerspectiveTransform(
            source_points=tuple(self._source_points),
            dest_points=dest,
            x_range=(ranges[0], ranges[1]),
            y_range=(ranges[2], ranges[3]),
        )
        transform.validate()
        return transform

    def _apply(self) -> None:
        try:
            transform = self._build_transform()
        except (PerspectiveError, ValueError) as error:
            messagebox.showerror("Perspective correction", str(error),
                                 parent=self)
            return
        # The source path is reported so the main window can show where
        # the correction came from (None when defined interactively).
        self._on_apply_cb(transform, self._json_path)
        self.destroy()

    def _save_json(self) -> None:
        try:
            transform = self._build_transform()
        except (PerspectiveError, ValueError) as error:
            messagebox.showerror("Perspective correction", str(error),
                                 parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=(("Perspective correction", "*.json"),))
        if not path:
            return
        try:
            transform.save_json(path)
        except OSError as error:
            messagebox.showerror("Perspective correction", str(error),
                                 parent=self)

    def _load_json(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=(("Perspective correction", "*.json"),))
        if not path:
            return
        try:
            transform = PerspectiveTransform.load_json(path)
        except PerspectiveError as error:
            messagebox.showerror("Perspective correction", str(error),
                                 parent=self)
            return
        self._json_path = path
        self._fill_from_transform(transform)

    def _fill_from_transform(self, transform: PerspectiveTransform) -> None:
        self._source_points = [tuple(point)
                               for point in transform.source_points]
        for index, (x_mm, y_mm) in enumerate(transform.dest_points):
            self._dest_vars[index][0].set(f"{x_mm:g}")
            self._dest_vars[index][1].set(f"{y_mm:g}")
        self._x_min_var.set(f"{transform.x_range[0]:g}")
        self._x_max_var.set(f"{transform.x_range[1]:g}")
        self._y_min_var.set(f"{transform.y_range[0]:g}")
        self._y_max_var.set(f"{transform.y_range[1]:g}")
        # Show markers only if an image with matching size is displayed.
        if self._image_rgb8 is not None:
            self._redraw_markers()
