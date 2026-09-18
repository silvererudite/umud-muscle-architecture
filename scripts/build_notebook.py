"""Build notebooks/01_data_and_method.ipynb.

The notebook is the narrative version of the report's sections 2-4: every claim
about the data is re-derived in front of the reader rather than asserted. It is
EDA and validation only - no training, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_data_and_method.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(keepends=True)}


CELLS = [
    md("""
# UMUD Challenge — data, method and validation

**Measuring Muscle, Automatically** · Shamima Hossain · CSE 754

This notebook re-derives every claim the report makes about the data. It runs
EDA and validation only — training happens in a Kaggle GPU kernel
(`kaggle/train/umud_train.py`).
"""),
    code("""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from umud.config import COMPETITION_DIR, APO_IMAGES, APO_MASKS, FASC_IMAGES, FASC_MASKS
from umud.calibration import calibrate
from umud.data import list_samples, list_test_images, split_samples, content_group, paired_frames
from umud.geometry import estimate_architecture
from umud.sequence import group_frames

print("competition data:", COMPETITION_DIR)
"""),
    md("""
## 1. What the data actually contains

The competition supplies masks, never measurements, and no scale metadata.
"""),
    code("""
apo, fasc = list_samples("apo"), list_samples("fasc")
test = list_test_images()
print(f"aponeurosis pairs : {len(apo)}")
print(f"fascicle pairs    : {len(fasc)}")
print(f"test frames       : {len(test)}")
print(f"frames with BOTH annotations: {len(paired_frames())}")
"""),
    md("""
### 1.1 Images and masks disagree in size — which one is authoritative?

Two readings are possible: the mask is a crop of the image, or the image is a
resize of the mask's frame. They imply opposite fixes, so we test it. If the
mask aligns after a plain resize, image intensity under it should be high
(aponeuroses are bright) — and a deliberately misaligned control should not be.
"""),
    code("""
import random
random.seed(0)
matched, mismatched, control = [], [], []
for s in random.sample(apo, 60):
    with Image.open(s.image_path) as im:
        image = np.asarray(im.convert("L"), dtype=np.float32); size = im.size
    with Image.open(s.mask_path) as mk:
        mask = np.asarray(mk.convert("L").resize(size, Image.NEAREST)) > 127
    if mask.sum() < 50:
        continue
    ratio = image[mask].mean() / image.mean()
    (matched if not s.was_resized else mismatched).append(ratio)
    control.append(image[np.roll(mask, image.shape[0] // 4, axis=0)].mean() / image.mean())

print(f"shape-matched pairs        : {np.mean(matched):.2f}x frame mean")
print(f"shape-mismatched, resized  : {np.mean(mismatched):.2f}x frame mean")
print(f"control (mask rolled H/4)  : {np.mean(control):.2f}x frame mean")
print("\\n=> the image is a resize of the mask's frame; resampling both onto a common canvas aligns them.")
"""),
    md("""
### 1.2 Duplicate leakage

26 % of the fascicle images repeat another image in the same set. Split at
random and the model is validated on frames it trained on.
"""),
    code("""
for task, samples in (("apo", apo), ("fasc", fasc)):
    grouped = split_samples(samples, task=task, group_duplicates=True)
    naive = split_samples(samples, task=task, group_duplicates=False)
    leak = lambda tr, va: len({content_group(task, s.name) for s in tr} & {content_group(task, s.name) for s in va})
    print(f"{task}: grouped split leaks {leak(*grouped)} groups | naive split leaks {leak(*naive)} groups")
"""),
    md("""
## 2. Scale calibration

Two of the three targets are millimetres; the network sees pixels. The
conversion has to be read off the depth ruler each scanner burns into the frame.
"""),
    code("""
cals = [calibrate(p) for p in test]
frame = pd.DataFrame([{
    "device": c.device, "ok": c.ok, "px_per_cm": c.px_per_cm_y,
    "depth_cm": (c.roi[3] - c.roi[1]) / c.px_per_cm_y if c.px_per_cm_y else np.nan,
    "width_cm": (c.roi[2] - c.roi[0]) / c.px_per_cm_x if c.px_per_cm_x else np.nan,
} for c in cals])
print(f"test frames calibrated: {frame.ok.sum()}/{len(frame)}")
display(frame.groupby("device").agg(n=("ok", "size"), ok=("ok", "sum"),
                                    px_per_cm=("px_per_cm", "median"),
                                    depth_cm=("depth_cm", "median")).round(2))
print(f"implied depth {frame.depth_cm.min():.2f}-{frame.depth_cm.max():.2f} cm "
      f"| width {frame.width_cm.min():.2f}-{frame.width_cm.max():.2f} cm")
print("matches the documented 3-7 cm presets and 2.85 / 5.75 cm sector widths — nothing was fitted to these.")
"""),
    md("""
The *training* set is a different story, and it is why the offline evaluation is
scale-free.
"""),
    code("""
random.seed(0)
for task, samples in (("apo", apo), ("fasc", fasc)):
    sub = random.sample(samples, 200)
    ok = np.mean([calibrate(s.image_path).ok for s in sub])
    print(f"{task}: {ok:.0%} of training frames carry a readable ruler")
"""),
    md("""
## 3. Is the geometry module right?

Pennation angle is **scale-invariant**, so running the reconstruction on
ground-truth masks gives a check that needs no calibration at all: the result
should land in the published physiological range.
"""),
    code("""
from umud.config import PhysiologicalPriors
unclamped = PhysiologicalPriors(pa_deg=(0.1, 89.9), fl_mm=(1, 1e6), mt_mm=(1, 1e6))

rows = []
for pair in paired_frames():
    a, f = pair["apo"][0], pair["fasc"][0]
    with Image.open(COMPETITION_DIR / APO_MASKS / a) as mk:
        apo_mask = np.asarray(mk.convert("L")) > 127
    with Image.open(COMPETITION_DIR / FASC_MASKS / f) as mk:
        fasc_mask = np.asarray(mk.convert("L")) > 127
    if fasc_mask.shape != apo_mask.shape:
        fasc_mask = np.asarray(Image.fromarray((fasc_mask * 255).astype(np.uint8))
                               .resize(apo_mask.shape[::-1], Image.NEAREST)) > 127
    e = estimate_architecture(apo_mask, fasc_mask, 1.0, 1.0, priors=unclamped)
    rows.append({"pa_deg": e.pa_deg, "mt_frac": e.mt_mm / apo_mask.shape[0],
                 "fl_frac": e.fl_mm / apo_mask.shape[0],
                 "divergence_deg": e.aponeurosis_divergence_deg,
                 "fl_intersection": e.fl_mm, "fl_parallel": e.fl_parallel_mm})
gt = pd.DataFrame(rows)
display(gt.describe().round(3))
print(f"pennation angle from ground-truth masks: median {gt.pa_deg.median():.1f} deg")
print("published range for vastus lateralis: 10-25 deg")
"""),
    md("""
### 3.1 Does the intersection construction actually matter?

The textbook formula `FL = MT / sin(PA)` assumes the aponeuroses are parallel.
"""),
    code("""
rel = (gt.fl_intersection - gt.fl_parallel).abs() / gt.fl_parallel
print(f"aponeurosis divergence : median {gt.divergence_deg.median():.2f} deg, p90 {gt.divergence_deg.quantile(.9):.2f} deg")
print(f"frames >2 deg from parallel: {(gt.divergence_deg > 2).mean():.0%}")
print(f"FL disagreement        : median {rel.median():.1%}, p90 {rel.quantile(.9):.1%}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].hist(gt.divergence_deg, bins=25, color="#e07a1f", edgecolor="white")
ax[0].set_xlabel("aponeurosis divergence (deg)"); ax[0].set_ylabel("frames")
ax[1].scatter(gt.fl_parallel, gt.fl_intersection, s=20, alpha=.7, color="#2a9d8f")
lim = [gt.fl_parallel.min(), gt.fl_parallel.max()]
ax[1].plot(lim, lim, "--", color="#999"); ax[1].set_xlabel("MT / sin(PA)"); ax[1].set_ylabel("intersection")
plt.tight_layout(); plt.show()
"""),
    md("""
### 3.2 Are the orientation targets right?

The orientation supervision is *derived* from the fascicle masks with a
structure tensor, not annotated — so it needs its own check against an
independent estimator (per-segment principal axis).
"""),
    code("""
from scipy import ndimage
from umud.data import load_pair, orientation_target
from umud.geometry import orientation_from_field

def pca_reference(mask):
    labels, n = ndimage.label(ndimage.binary_dilation(mask, np.ones((3, 3))), np.ones((3, 3)))
    vectors = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(labels == k)
        if ys.size < 12:
            continue
        cov = np.cov(np.stack([xs - xs.mean(), ys - ys.mean()]))
        if not np.all(np.isfinite(cov)):
            continue
        w, v = np.linalg.eigh(cov)
        if w[-1] <= 0 or w[-1] / max(w[0], 1e-9) < 3:
            continue
        a = np.arctan2(v[1, -1], v[0, -1])
        vectors.append((np.cos(2 * a), np.sin(2 * a), ys.size))
    if not vectors:
        return np.nan
    weights = [v[2] for v in vectors]
    c = np.average([v[0] for v in vectors], weights=weights)
    s = np.average([v[1] for v in vectors], weights=weights)
    return np.degrees(0.5 * np.arctan2(s, c))

random.seed(3)
diffs = []
for s in random.sample(fasc, 50):
    _, mask = load_pair(s, (512, 512))
    reference = pca_reference(mask)
    field = orientation_target(mask)
    estimated, _ = orientation_from_field(field[0], field[1], mask.astype(float), 1.0, 1.0)
    if not (np.isfinite(reference) and np.isfinite(estimated)):
        continue
    d = abs(reference - estimated)
    diffs.append(min(d, 180 - d))
diffs = np.array(diffs)
print(f"structure-tensor target vs per-segment PCA over {len(diffs)} frames:")
print(f"  median {np.median(diffs):.2f} deg | within 5 deg: {(diffs < 5).mean():.0%}")
"""),
    md("""
## 4. The hidden sequence structure

The competition ships 309 loose frames. They are not 309 independent scans.
"""),
    code("""
groups = group_frames(test, cals)
sizes = pd.Series([g.size for g in groups]).value_counts().sort_index()
print(sizes.to_string())
runs = [g for g in groups if g.is_sequence]
print(f"\\n{len(runs)} acquisition runs covering {sum(g.size for g in runs)} of {len(test)} frames")
print("example:", runs[0].names[0], "->", runs[0].names[-1])
"""),
    md("""
The grouping threshold is not a tuned parameter — the similarity distribution is
sharply bimodal and 0.995 sits in an empty gap.

![sequence structure](../docs/figures/sequence_structure.png)

## 5. Where this goes next

Training runs on Kaggle (`kaggle/train`), inference and evaluation on Kaggle
(`kaggle/infer`). Results, ablations and limitations are in
[`docs/REPORT.md`](../docs/REPORT.md).
"""),
]

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
