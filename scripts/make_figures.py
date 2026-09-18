"""Generate the figures used in docs/REPORT.md.

Reads the competition data locally (no training, no GPU) and writes PNGs into
docs/figures/.
"""

from __future__ import annotations

import glob
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
Image.MAX_IMAGE_PIXELS = None

from umud.calibration import calibrate, crop_to_roi, to_gray
from umud.config import COMPETITION_DIR, APO_IMAGES, APO_MASKS, FASC_IMAGES, FASC_MASKS, TEST_IMAGES
from umud.data import list_samples, list_test_images, paired_frames
from umud.sequence import frame_signature, group_frames

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK = "#1d3557"
ACCENT = "#e07a1f"
TEAL = "#2a9d8f"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#c9c9c9", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def fig_data_overview() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for row, (img_dir, mask_dir, tag, colour) in enumerate([
        (APO_IMAGES, APO_MASKS, "aponeurosis", ACCENT),
        (FASC_IMAGES, FASC_MASKS, "fascicle", TEAL),
    ]):
        names = sorted(p.name for p in (COMPETITION_DIR / img_dir).glob("*.tif"))
        picks = [names[5], names[len(names) // 3], names[2 * len(names) // 3], names[-20]]
        for col, name in enumerate(picks):
            with Image.open(COMPETITION_DIR / img_dir / name) as im:
                image = np.asarray(im.convert("L"))
                size = im.size
            with Image.open(COMPETITION_DIR / mask_dir / name) as mk:
                native = (mk.size[1], mk.size[0])
                mask = np.asarray(mk.convert("L").resize(size, Image.NEAREST)) > 127
            ax = axes[row][col]
            ax.imshow(image, cmap="gray")
            ax.contour(mask, levels=[0.5], colors=[colour], linewidths=1.0)
            flag = "" if native == image.shape else f"  mask {native[0]}x{native[1]}"
            ax.set_title(f"{tag} · {name}\n{image.shape[0]}x{image.shape[1]}{flag}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Training annotations: continuous aponeurosis bands (top) vs. short fascicle segments (bottom)",
                 fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "data_overview.png", dpi=110)
    plt.close(fig)
    print("data_overview.png")


def fig_calibration() -> None:
    paths = list_test_images()
    cals = [calibrate(p) for p in paths]
    families = Counter(c.device for c in cals)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, 0])
    labels, counts = zip(*sorted(families.items(), key=lambda kv: -kv[1]))
    ax.barh(range(len(labels)), counts, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("test frames")
    ax.set_title(f"Scanner families ({len(paths)} frames, 100 % calibrated)", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist([c.px_per_cm_y for c in cals], bins=30, color=ACCENT, edgecolor="white")
    ax.set_xlabel("pixels per cm (depth axis)")
    ax.set_ylabel("frames")
    ax.set_title("Recovered scale spans a 2.6x range", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    xs = [c.px_per_cm_x for c in cals]
    ys = [c.px_per_cm_y for c in cals]
    ax.scatter(xs, ys, s=14, alpha=0.55, color=TEAL, edgecolor="none")
    lim = [min(xs + ys) * 0.95, max(xs + ys) * 1.05]
    ax.plot(lim, lim, "--", color="#999", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("px/cm lateral"); ax.set_ylabel("px/cm depth")
    ratio = np.array([c.px_per_cm_y / c.px_per_cm_x for c in cals])
    ax.set_title("Test pixels are square to within "
                 f"{100 * np.abs(ratio - 1).max():.1f} %\n"
                 "(the aspect correction matters for training frames, not here)",
                 fontsize=9)

    # Ruler illustration on one frame of the largest family.
    example = next(p for p, c in zip(paths, cals) if c.device == "800x1200_right")
    cal = calibrate(example)
    gray = to_gray(np.asarray(Image.open(example)))
    ax = fig.add_subplot(gs[1, :2])
    ax.imshow(gray, cmap="gray", aspect="auto")
    left, top, right, bottom = cal.roi
    ax.add_patch(plt.Rectangle((left, top), right - left, bottom - top,
                               fill=False, edgecolor=ACCENT, lw=2))
    ax.axvline(1150, color=TEAL, lw=1.2, ls="--")
    for k in range(int((bottom - top) / (cal.px_per_cm_y / 2)) + 1):
        ax.plot([1140, 1162], [top + k * cal.px_per_cm_y / 2] * 2, color=TEAL, lw=0.8)
    ax.set_title(f"Ruler reading on {example.name}: ticks at x=1150, "
                 f"{cal.px_per_cm_y:.1f} px/cm, ROI in orange", fontsize=10)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    depth = [(c.roi[3] - c.roi[1]) / c.px_per_cm_y for c in cals]
    width = [(c.roi[2] - c.roi[0]) / c.px_per_cm_x for c in cals]
    ax.scatter(width, depth, s=16, alpha=0.6, color=INK, edgecolor="none")
    ax.set_xlabel("imaged width (cm)"); ax.set_ylabel("imaged depth (cm)")
    ax.set_title("Sanity check: implied field of view\nmatches documented 3-7 cm presets", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=110)
    plt.close(fig)
    print("calibration.png")


def fig_sequences() -> None:
    paths = list_test_images()
    cals = [calibrate(p) for p in paths]
    signatures = np.stack([frame_signature(p, c) for p, c in zip(paths, cals)])
    groups = group_frames(paths, cals)
    runs = [g for g in groups if g.is_sequence]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    lo, hi = 55, 100
    similarity = signatures[lo:hi] @ signatures[lo:hi].T
    im = axes[0].imshow(similarity, cmap="magma", vmin=0, vmax=1)
    for group in groups:
        start = group.indices[0]
        if lo <= start < hi and start != lo:
            axes[0].axhline(start - lo - 0.5, color="#4dd0c0", lw=0.7)
            axes[0].axvline(start - lo - 0.5, color="#4dd0c0", lw=0.7)
    axes[0].set_title("Frame-to-frame similarity, IMG_00056-00100\n(block structure = acquisition runs)", fontsize=10)
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("frame")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    consecutive = [float(signatures[i] @ signatures[i + 1]) for i in range(len(paths) - 1)]
    axes[1].hist(consecutive, bins=60, color=INK, edgecolor="white")
    axes[1].axvline(0.995, color=ACCENT, lw=2, label="grouping threshold 0.995")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("cosine similarity of consecutive frames")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title("The gap the threshold sits in", fontsize=10)
    axes[1].legend(fontsize=8)

    sizes = Counter(g.size for g in groups)
    keys = sorted(sizes)
    axes[2].bar([str(k) for k in keys], [sizes[k] for k in keys], color=TEAL)
    for i, k in enumerate(keys):
        axes[2].text(i, sizes[k], str(sizes[k]), ha="center", va="bottom", fontsize=9)
    axes[2].set_xlabel("frames per acquisition run"); axes[2].set_ylabel("runs")
    axes[2].set_title(f"{len(runs)} five-frame runs cover "
                      f"{sum(g.size for g in runs)}/{len(paths)} test frames", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG / "sequence_structure.png", dpi=110)
    plt.close(fig)
    print("sequence_structure.png")


def fig_geometry_diagram() -> None:
    """The reconstruction, drawn from the actual code path.

    The divergence in the right panel is the p90 measured on the 78
    ground-truth-annotated frames (3.1 deg), not an exaggeration chosen to make
    the point look bigger than it is.
    """
    from umud.geometry import Aponeurosis, fascicle_length_mm, muscle_thickness_mm

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    divergence_deg = 3.1          # measured p90; median is 1.7 deg

    for ax, (slope_s, title) in zip(axes, [
        (0.0, "Parallel aponeuroses\nintersection and MT/sin(PA) agree"),
        (np.tan(np.radians(divergence_deg)),
         f"Diverging by {divergence_deg:.1f}° (the measured p90)\n"
         "MT/sin(PA) overestimates"),
    ]):
        superficial = Aponeurosis(np.array([slope_s, 5.0]), 0.0, 95.0, 100, 0.1)
        deep = Aponeurosis(np.array([0.0, 25.0]), 0.0, 95.0, 100, 0.1)
        xs = np.linspace(0, 95, 100)
        ax.plot(xs, superficial.y_at(xs), color=INK, lw=2.5, label="superficial aponeurosis")
        ax.plot(xs, deep.y_at(xs), color="#5a7a9a", lw=2.5, label="deep aponeurosis")

        angle = -18.0
        thickness = muscle_thickness_mm(superficial, deep)
        fl, _ = fascicle_length_mm(superficial, deep, angle, thickness)
        x0, y0 = 12.0, float(deep.y_at(12.0))
        dx, dy = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        if dy > 0:
            dx, dy = -dx, -dy

        pa = abs(angle - deep.angle_deg_at(x0))
        parallel = thickness / np.sin(np.radians(pa))
        ax.plot([x0, x0 + parallel * dx], [y0, y0 + parallel * dy], color=TEAL,
                lw=3.2, ls="-", alpha=0.45, label=f"MT/sin(PA) = {parallel:.1f} mm")
        ax.plot([x0, x0 + fl * dx], [y0, y0 + fl * dy], color=ACCENT, lw=2.4,
                label=f"intersection = {fl:.1f} mm")

        ax.annotate("", xy=(x0, float(superficial.y_at(x0))), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<->", color="#b3001b", lw=1.6))
        ax.text(x0 - 4.5, 15, f"MT\n{thickness:.1f}", color="#b3001b", fontsize=9, ha="center")
        ax.text(x0 + 4, y0 - 1.4, f"PA {pa:.0f}°", color=ACCENT, fontsize=10)

        error = abs(parallel - fl) / fl
        ax.set_xlim(0, 96); ax.set_ylim(31, -1)
        ax.set_xlabel("lateral (mm)"); ax.set_ylabel("depth (mm)")
        ax.set_title(title + f"\n(textbook formula is {error:+.0%} off)", fontsize=10)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.95)

    fig.suptitle("Fascicle length: aponeurosis intersection vs. the textbook formula", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "geometry_reconstruction.png", dpi=110)
    plt.close(fig)
    print("geometry_reconstruction.png")


def fig_dataset_stats() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    apo = list_samples("apo")
    fasc = list_samples("fasc")
    for ax, samples, tag, colour in [(axes[0], apo, "aponeurosis", ACCENT),
                                     (axes[1], fasc, "fascicle", TEAL)]:
        counts = Counter(s.device_key for s in samples)
        top = counts.most_common(8)
        ax.barh([k for k, _ in top], [v for _, v in top], color=colour)
        ax.invert_yaxis()
        ax.set_xlabel("frames")
        resized = sum(1 for s in samples if s.was_resized)
        ax.set_title(f"{tag}: {len(samples)} frames, {len(counts)} acquisition shapes\n"
                     f"{resized} ({resized / len(samples):.0%}) were resized off their native grid",
                     fontsize=9)

    ax = axes[2]
    labels = ["apo\nfiles", "apo\ndistinct", "fasc\nfiles", "fasc\ndistinct", "paired\nframes"]
    values = [len(apo), 1028, len(fasc), 2046, len(paired_frames())]
    colours = [ACCENT, "#f2b880", TEAL, "#8ecfc7", INK]
    ax.bar(labels, values, color=colours)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Duplicates and the paired subset\n(715 fascicle files repeat another image)", fontsize=9)
    ax.set_ylabel("frames")

    fig.tight_layout()
    fig.savefig(FIG / "dataset_stats.png", dpi=110)
    plt.close(fig)
    print("dataset_stats.png")


if __name__ == "__main__":
    fig_data_overview()
    fig_dataset_stats()
    fig_calibration()
    fig_sequences()
    fig_geometry_diagram()
    print(f"\nfigures in {FIG}")
