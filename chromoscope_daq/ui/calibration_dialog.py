"""Intensity calibration dialog (README 'Intensity calibration').

Pick one of the four function types, enter the coefficients, and
save/load the definition as json. On Apply the calibration is handed to
the main window, enabling the particle-flux display toggle and the
calibrated variant saving.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

from ..calibration import (CALIBRATION_TYPES, COEFF_COUNTS,
                           CalibrationError, IntensityCalibration)


class CalibrationDialog(tk.Toplevel):
    """Editor for one intensity calibration definition."""

    def __init__(self, master, calibration: Optional[IntensityCalibration],
                 on_apply: Callable[[IntensityCalibration], None],
                 default_exposure_us: float = 1000.0) -> None:
        super().__init__(master)
        self.title("Intensity calibration")
        self.transient(master)
        self.grab_set()
        self._on_apply_cb = on_apply
        self._default_exposure_us = float(default_exposure_us)
        self._json_path: Optional[str] = None
        self._build_ui()
        if calibration is not None:
            self._kind_var.set(calibration.kind)
            self._on_kind_change()
            for var, value in zip(self._coeff_vars, calibration.coefficients):
                var.set(f"{value:g}")
            self._exposure_var.set(f"{calibration.exposure_norm_us:g}")
        else:
            self._exposure_var.set(f"{self._default_exposure_us:g}")

    def _build_ui(self) -> None:
        body = ttk.Frame(self, padding=10)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Function type:").grid(row=0, column=0,
                                                    sticky="w")
        self._kind_var = tk.StringVar(value=CALIBRATION_TYPES[0])
        kind_box = ttk.Combobox(body, textvariable=self._kind_var, width=12,
                                values=list(CALIBRATION_TYPES),
                                state="readonly")
        kind_box.grid(row=0, column=1, sticky="w", padx=4)
        kind_box.bind("<<ComboboxSelected>>", lambda _e: self._on_kind_change())

        self._formula_var = tk.StringVar()
        ttk.Label(body, textvariable=self._formula_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 8))

        self._coeff_vars = [tk.StringVar() for _ in range(3)]
        self._coeff_labels: list[ttk.Label] = []
        self._coeff_entries: list[ttk.Entry] = []
        for index, label in enumerate(("a", "b", "c")):
            coeff_label = ttk.Label(body, text=f"{label} =")
            coeff_label.grid(row=2 + index, column=0, sticky="w", pady=2)
            entry = ttk.Entry(body, textvariable=self._coeff_vars[index],
                              width=12)
            entry.grid(row=2 + index, column=1, sticky="w", padx=4, pady=2)
            self._coeff_labels.append(coeff_label)
            self._coeff_entries.append(entry)

        self._exposure_var = tk.StringVar()
        exposure_row = ttk.Frame(body)
        exposure_row.grid(row=5, column=0, columnspan=2, sticky="w",
                          pady=(6, 0))
        ttk.Label(exposure_row,
                  text="Exposure normalization [us]:").pack(side="left")
        ttk.Entry(exposure_row, textvariable=self._exposure_var,
                  width=12).pack(side="left", padx=4)
        ttk.Label(exposure_row,
                  text="flux = f(luminance) * t_norm / t_current",
                  foreground="#888888").pack(side="left", padx=8)

        buttons = ttk.Frame(body)
        buttons.grid(row=6, column=0, columnspan=2, sticky="we",
                     pady=(10, 0))
        ttk.Button(buttons, text="Save json...",
                   command=self._save_json).pack(side="left")
        ttk.Button(buttons, text="Load json...",
                   command=self._load_json).pack(side="left", padx=4)
        ttk.Button(buttons, text="Apply",
                   command=self._apply).pack(side="right")
        ttk.Button(buttons, text="Cancel",
                   command=self.destroy).pack(side="right", padx=4)
        self._on_kind_change()

    def _on_kind_change(self) -> None:
        kind = self._kind_var.get()
        count = COEFF_COUNTS[kind]
        formulas = {
            "linear": "flux = a * luminance + b",
            "quadratic": "flux = a * luminance^2 + b * luminance + c",
            "power law": "flux = a * luminance^b + c",
            "logarithmic": "flux = a * ln(luminance + b) + c",
        }
        self._formula_var.set(formulas[kind])
        for index in range(3):
            state = "normal" if index < count else "disabled"
            self._coeff_entries[index].config(state=state)
            self._coeff_labels[index].config(
                state="normal" if index < count else "disabled")

    def _build_calibration(self) -> IntensityCalibration:
        kind = self._kind_var.get()
        count = COEFF_COUNTS[kind]
        coefficients = tuple(float(var.get()) for var in self._coeff_vars
                             [:count])
        calibration = IntensityCalibration(
            kind=kind, coefficients=coefficients,
            exposure_norm_us=float(self._exposure_var.get()))
        calibration.validate()
        return calibration

    def _apply(self) -> None:
        try:
            calibration = self._build_calibration()
        except (CalibrationError, ValueError) as error:
            messagebox.showerror("Intensity calibration", str(error),
                                 parent=self)
            return
        # Source path so the main window can show where it came from.
        self._on_apply_cb(calibration, self._json_path)
        self.destroy()

    def _save_json(self) -> None:
        try:
            calibration = self._build_calibration()
        except (CalibrationError, ValueError) as error:
            messagebox.showerror("Intensity calibration", str(error),
                                 parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=(("Intensity calibration", "*.json"),))
        if not path:
            return
        try:
            calibration.save_json(path)
        except OSError as error:
            messagebox.showerror("Intensity calibration", str(error),
                                 parent=self)

    def _load_json(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=(("Intensity calibration", "*.json"),))
        if not path:
            return
        try:
            calibration = IntensityCalibration.load_json(path)
        except CalibrationError as error:
            messagebox.showerror("Intensity calibration", str(error),
                                 parent=self)
            return
        self._json_path = path
        self._kind_var.set(calibration.kind)
        self._on_kind_change()
        for var, value in zip(self._coeff_vars, calibration.coefficients):
            var.set(f"{value:g}")
        self._exposure_var.set(f"{calibration.exposure_norm_us:g}")
