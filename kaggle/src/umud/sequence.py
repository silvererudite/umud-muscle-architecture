"""Frame grouping and temporal smoothing.

The proposal's third component is temporal coherence.  The competition ships 309
loose images with no sequence metadata, but they are not 309 independent scans:
28 runs of five consecutive video frames are interleaved with 169 single
images.  Recovering that structure is what makes a temporal module possible at
all, and it is worth doing because a run of five frames gives five nearly
independent looks at the same anatomy — averaging them cuts the variance of the
segmentation noise without touching its bias.

Grouping is deliberately conservative.  Frames join a group only when they are
adjacent in acquisition order, come from the same scanner family, share the same
depth setting, and are visually near-identical.  Two frames of the same muscle
taken minutes apart are *not* the same measurement, and smoothing across them
would inject bias to buy variance we do not need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from .calibration import Calibration, crop_to_roi, to_gray

Image.MAX_IMAGE_PIXELS = None

# Empirical: within a true five-frame run the downsampled correlation sits at
# 0.999+, while the next-closest pairs in the test set fall below 0.99.  0.995
# sits in that gap with room on both sides.
SEQUENCE_SIMILARITY = 0.995


@dataclass
class FrameGroup:
    indices: list[int]
    names: list[str]

    @property
    def size(self) -> int:
        return len(self.indices)

    @property
    def is_sequence(self) -> bool:
        return self.size > 1


def frame_signature(path: Path, cal: Calibration, size: int = 64) -> np.ndarray:
    """Contrast-normalised thumbnail of the imaged sector.

    Cropping to the calibrated region of interest first matters: the scanner
    chrome (patient banner, gain curve, depth ruler) is identical across every
    frame from a session and would make unrelated frames look correlated.
    """
    gray = to_gray(np.asarray(Image.open(path)))
    roi = crop_to_roi(gray, cal)
    small = np.asarray(
        Image.fromarray(roi.astype(np.uint8)).resize((size, size), Image.BILINEAR),
        dtype=np.float32,
    )
    small -= small.mean()
    norm = np.linalg.norm(small)
    return (small / norm).ravel() if norm > 1e-6 else small.ravel()


def group_frames(
    paths: Sequence[Path],
    calibrations: Sequence[Calibration],
    threshold: float = SEQUENCE_SIMILARITY,
) -> list[FrameGroup]:
    """Partition frames into acquisition runs.

    ``paths`` must be in acquisition order (the file names are zero-padded, so
    sorting them is acquisition order).
    """
    signatures = [frame_signature(p, c) for p, c in zip(paths, calibrations)]

    groups: list[FrameGroup] = []
    current = [0]
    for i in range(1, len(paths)):
        previous, current_cal = calibrations[i - 1], calibrations[i]
        same_device = previous.device == current_cal.device
        same_scale = abs(previous.px_per_cm_y - current_cal.px_per_cm_y) < 0.5
        similarity = float(signatures[i - 1] @ signatures[i])
        if same_device and same_scale and similarity >= threshold:
            current.append(i)
        else:
            groups.append(FrameGroup(current, [paths[j].name for j in current]))
            current = [i]
    groups.append(FrameGroup(current, [paths[j].name for j in current]))
    return groups


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    """Median of a weighted sample.

    A median rather than a mean because a single frame whose segmentation
    collapsed produces an outlier several times the true value, and a five-frame
    mean would carry a fifth of that error into every frame of the run.
    """
    v = np.asarray(list(values), dtype=np.float64)
    w = np.asarray(list(weights), dtype=np.float64)
    finite = np.isfinite(v) & np.isfinite(w)
    v, w = v[finite], w[finite]
    if v.size == 0:
        return float("nan")
    if w.sum() <= 0:
        w = np.ones_like(v)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= w.sum()
    return float(np.interp(0.5, cumulative, v))


def smooth_predictions(
    records: list[dict],
    groups: Sequence[FrameGroup],
    fields: Sequence[str] = ("pa_deg", "fl_mm", "mt_mm"),
    min_group_size: int = 2,
) -> list[dict]:
    """Replace each frame's estimate by the confidence-weighted group consensus.

    Returns a new list; ``records`` is not modified.  Every frame in a run
    receives the same value, which is the right answer when the frames really do
    show the same anatomy a few milliseconds apart.
    """
    smoothed = [dict(record) for record in records]
    for group in groups:
        # Record the run size on every frame, smoothed or not, so the ablation
        # can separate "was in a run" from "was changed by smoothing".
        for i in group.indices:
            smoothed[i]["group_size"] = group.size
        if group.size < min_group_size:
            continue
        members = [records[i] for i in group.indices]
        weights = [max(float(m.get("confidence", 0.0)), 1e-3) for m in members]
        consensus = {
            field: weighted_median([m[field] for m in members], weights)
            for field in fields
        }
        for i in group.indices:
            for field, value in consensus.items():
                if np.isfinite(value):
                    smoothed[i][field] = value
    return smoothed
