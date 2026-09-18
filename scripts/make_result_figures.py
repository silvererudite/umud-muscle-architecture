"""Figures built from the Kaggle kernel outputs in outputs/.

Expects (from `kaggle kernels output`):
  outputs/training_summary.json   outputs/evaluation.json
  outputs/ablations.json          outputs/predictions.csv
  outputs/predictions_no_smoothing.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, ACCENT, TEAL, RED = "#1d3557", "#e07a1f", "#2a9d8f", "#b3001b"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.edgecolor": "#c9c9c9",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load(name: str):
    path = OUT / name
    if not path.exists():
        print(f"  (skipping: {name} not found)")
        return None
    return json.loads(path.read_text()) if name.endswith(".json") else pd.read_csv(path)


def fig_training_curves() -> None:
    summary = _load("training_summary.json")
    if summary is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for task, colour in (("apo", ACCENT), ("fasc", TEAL)):
        history = pd.DataFrame(summary[task]["history"])
        axes[0].plot(history.epoch, history.train_total, color=colour, ls="--", lw=1.2, label=f"{task} train")
        axes[0].plot(history.epoch, history.val_total, color=colour, lw=2, label=f"{task} val")
        axes[1].plot(history.epoch, history.val_dice, color=colour, lw=2, label=task)
        if "val_orientation_mae_deg" in history and history.val_orientation_mae_deg.notna().any():
            axes[2].plot(history.epoch, history.val_orientation_mae_deg, color=colour, lw=2, label=task)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss", fontsize=10); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Dice"); axes[1].set_title("Validation Dice", fontsize=10); axes[1].legend(fontsize=8)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("degrees")
    axes[2].set_title("Orientation-field error (fascicle head)", fontsize=10); axes[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", dpi=110); plt.close(fig)
    print("training_curves.png")


def fig_evaluation() -> None:
    evaluation = _load("evaluation.json")
    if evaluation is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    rows = []
    for task in ("apo", "fasc"):
        by_device = evaluation[f"{task}_segmentation"]["by_device"]
        for device, values in by_device.items():
            rows.append({"task": task, "device": device, "dice": values["dice"], "n": values["n"]})
    frame = pd.DataFrame(rows)
    frame = frame[frame.n >= 5].sort_values("dice")
    colours = [ACCENT if t == "apo" else TEAL for t in frame.task]
    ax.barh(range(len(frame)), frame.dice, color=colours)
    ax.set_yticks(range(len(frame)))
    ax.set_yticklabels([f"{d} (n={n})" for d, n in zip(frame.device, frame.n)], fontsize=7)
    ax.set_xlabel("Dice")
    ax.set_title("Segmentation by acquisition shape\n(orange = aponeurosis, teal = fascicle)", fontsize=9)

    ax = axes[1]
    ori = evaluation["fascicle_orientation_head"]
    errors = [r["angle_abs_err_deg"] for r in ori["rows"] if r.get("found") and np.isfinite(r.get("angle_abs_err_deg", np.nan))]
    ax.hist(errors, bins=40, color=TEAL, edgecolor="white")
    ax.axvline(np.median(errors), color=RED, lw=2, label=f"median {np.median(errors):.2f}°")
    ax.set_xlabel("fascicle-angle error (°)"); ax.set_ylabel("frames")
    ax.set_title("Orientation error vs. ground-truth masks", fontsize=9); ax.legend(fontsize=8)

    ax = axes[2]
    paired = evaluation["paired_end_to_end"]
    rows = pd.DataFrame(paired["rows"])
    rows = rows[rows["found"]]
    ax.scatter(rows.gt_pa, rows.pred_pa, s=22, alpha=0.7, color=INK, edgecolor="none")
    lim = [0, max(rows.gt_pa.max(), rows.pred_pa.max()) * 1.1]
    ax.plot(lim, lim, "--", color="#999", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("PA from ground-truth masks (°)"); ax.set_ylabel("PA from predicted masks (°)")
    ax.set_title(f"End-to-end on the {paired['n']} paired frames", fontsize=9)

    fig.tight_layout(); fig.savefig(FIG / "evaluation.png", dpi=110); plt.close(fig)
    print("evaluation.png")


def fig_predictions() -> None:
    frame = _load("predictions.csv")
    if frame is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, (field, label, colour) in zip(axes, [
        ("mt_mm", "muscle thickness (mm)", ACCENT),
        ("pa_deg", "pennation angle (°)", TEAL),
        ("fl_mm", "fascicle length (mm)", INK),
        ("confidence", "per-frame confidence", RED),
    ]):
        ax.hist(frame[field], bins=32, color=colour, edgecolor="white")
        ax.set_xlabel(label); ax.set_ylabel("frames")
        ax.set_title(f"median {frame[field].median():.1f}", fontsize=9)
    fig.suptitle("Test-set predictions (309 frames)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "predictions.png", dpi=110); plt.close(fig)
    print("predictions.png")


def fig_smoothing_effect() -> None:
    smoothed = _load("predictions.csv")
    raw = _load("predictions_no_smoothing.csv")
    if smoothed is None or raw is None:
        return
    runs = smoothed[smoothed.group_size > 1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, field, label in zip(axes, ["pa_deg", "fl_mm", "mt_mm"],
                                ["pennation angle (°)", "fascicle length (mm)", "thickness (mm)"]):
        before = raw.loc[runs.index, field]
        ax.scatter(before, runs[field], s=20, alpha=0.6, color=TEAL, edgecolor="none")
        lim = [min(before.min(), runs[field].min()), max(before.max(), runs[field].max())]
        ax.plot(lim, lim, "--", color="#999", lw=1)
        ax.set_xlabel(f"before smoothing"); ax.set_ylabel("after smoothing")
        ax.set_title(f"{label}\nmean |shift| {np.abs(before.to_numpy() - runs[field].to_numpy()).mean():.2f}", fontsize=9)
    fig.suptitle(f"Effect of sequence smoothing on the {len(runs)} frames inside acquisition runs", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "smoothing_effect.png", dpi=110); plt.close(fig)
    print("smoothing_effect.png")


if __name__ == "__main__":
    fig_training_curves()
    fig_evaluation()
    fig_predictions()
    fig_smoothing_effect()
    print(f"\nfigures in {FIG}")
