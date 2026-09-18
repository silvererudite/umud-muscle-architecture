"""Qualitative panels: what the model saw and what it measured.

Rendered inside the inference kernel so the figures come from exactly the
weights that produced the submission.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .calibration import calibrate, crop_to_roi, to_gray
from .geometry import estimate_architecture, extract_aponeuroses
from .predict import _forward, _resample

INK, ACCENT, TEAL, RED = "#1d3557", "#e07a1f", "#2a9d8f", "#b3001b"


def annotate_frame(
    ax,
    path: Path,
    apo_model,
    fasc_model,
    device,
    size=(512, 512),
    tta: bool = True,
) -> dict:
    """Draw one frame with its predicted masks and reconstructed geometry."""
    cal = calibrate(path)
    gray = to_gray(np.asarray(Image.open(path)))
    roi = crop_to_roi(gray, cal)

    apo_probs, _ = _forward(apo_model, roi, size, device, tta)
    fasc_probs, field = _forward(fasc_model, roi, size, device, tta)
    apo_probs = _resample(apo_probs, roi.shape)
    fasc_probs = _resample(fasc_probs, roi.shape)

    orientation = None
    if field is not None:
        cos2t = _resample(field[0], roi.shape)
        sin2t = _resample(field[1], roi.shape)
        norm = np.hypot(cos2t, sin2t) + 1e-6
        orientation = (cos2t / norm, sin2t / norm)

    apo_mask = apo_probs > 0.5
    fasc_mask = fasc_probs > 0.5
    estimate = estimate_architecture(
        apo_mask, fasc_mask, cal.mm_per_px_x, cal.mm_per_px_y,
        orientation=orientation, fasc_prob=fasc_probs,
    )

    ax.imshow(roi, cmap="gray")
    if apo_mask.any():
        ax.contour(apo_mask, levels=[0.5], colors=[ACCENT], linewidths=1.1)
    if fasc_mask.any():
        ax.contour(fasc_mask, levels=[0.5], colors=[TEAL], linewidths=0.8)

    # Overlay the fitted aponeurosis curves and one reconstructed fascicle,
    # converted back from millimetres to the pixel grid for drawing.
    superficial, deep, _, _ = extract_aponeuroses(apo_mask, cal.mm_per_px_x, cal.mm_per_px_y)
    if superficial is not None and deep is not None:
        lo = max(superficial.x_min, deep.x_min)
        hi = min(superficial.x_max, deep.x_max)
        if hi > lo:
            xs_mm = np.linspace(lo, hi, 80)
            xs_px = xs_mm / cal.mm_per_px_x
            for curve in (superficial, deep):
                ax.plot(xs_px, np.asarray(curve.y_at(xs_mm)) / cal.mm_per_px_y,
                        color=RED, lw=1.6)
            x0_mm = 0.5 * (lo + hi)
            y0_mm = float(deep.y_at(x0_mm))
            theta = np.radians(estimate.fascicle_angle_deg)
            dx, dy = np.cos(theta), np.sin(theta)
            if dy > 0:
                dx, dy = -dx, -dy
            x1_mm = x0_mm + estimate.fl_mm * dx
            y1_mm = y0_mm + estimate.fl_mm * dy
            ax.plot([x0_mm / cal.mm_per_px_x, x1_mm / cal.mm_per_px_x],
                    [y0_mm / cal.mm_per_px_y, y1_mm / cal.mm_per_px_y],
                    color="#ffd166", lw=2.2)

    ax.set_title(
        f"{path.name}\nMT {estimate.mt_mm:.1f} mm · PA {estimate.pa_deg:.1f}° · "
        f"FL {estimate.fl_mm:.1f} mm · conf {estimate.confidence:.2f}",
        fontsize=8,
    )
    ax.axis("off")
    return {"name": path.name, **estimate.as_dict(), "device": cal.device}


def qualitative_panel(
    paths: Sequence[Path],
    apo_model,
    fasc_model,
    device,
    out_path: Path,
    size=(512, 512),
    columns: int = 3,
    title: str = "Predicted masks and reconstructed geometry",
) -> None:
    rows = int(np.ceil(len(paths) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, path in zip(axes, paths):
        annotate_frame(ax, path, apo_model, fasc_model, device, size)
    for ax in axes[len(paths):]:
        ax.axis("off")
    fig.suptitle(
        title + "  —  orange: aponeurosis mask · teal: fascicle mask · "
        "red: fitted aponeuroses · yellow: reconstructed fascicle",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
