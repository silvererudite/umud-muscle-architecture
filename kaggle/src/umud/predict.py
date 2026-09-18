"""Inference: frames in, architecture measurements out.

The pipeline is the proposal's diagram, executed literally:

    frame
      -> ruler calibration (mm per pixel, region of interest)
      -> aponeurosis U-Net        ---.
      -> fascicle U-Net + orientation head
      -> geometric reconstruction (theta, thickness, Lf)
      -> sequence smoothing
      -> thickness, pennation angle, fascicle length (+ confidence)

Two details worth flagging.  Predictions are made on the *calibrated region of
interest*, not the raw frame: the scanner chrome is bright, high-contrast and
utterly unlike muscle, and leaving it in costs Dice for no reason.  And the
network runs at a fixed 512x512 while the geometry runs at ROI resolution, so
the masks are resampled back before anything is measured — measuring on the
network canvas would fold the resize's aspect distortion into every angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .calibration import Calibration, calibrate, crop_to_roi, to_gray
from .config import PRIORS, SUBMISSION_COLUMNS
from .geometry import ArchitectureEstimate, estimate_architecture
from .models import GeometryAwareUNet
from .sequence import group_frames, smooth_predictions

Image.MAX_IMAGE_PIXELS = None


def load_model(weights_path: str | Path, device: torch.device) -> tuple[GeometryAwareUNet, dict]:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    state_dict = checkpoint["state_dict"]
    model = GeometryAwareUNet(
        encoder=config["encoder"],
        pretrained=False,                      # weights come from the checkpoint
        orientation_head=has_orientation_head(state_dict),
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, config


def has_orientation_head(state_dict: dict) -> bool:
    """Whether a checkpoint carries the orientation branch.

    Read off the weights rather than the saved config: the aponeurosis model is
    built without the branch, and rebuilding it from a stale config would leave
    the loader with keys it cannot place.
    """
    return any(key.startswith("orientation_head.") for key in state_dict)


@dataclass
class FramePrediction:
    name: str
    estimate: ArchitectureEstimate
    calibration: Calibration

    def as_record(self) -> dict:
        record = {"image_id": self.name, **self.estimate.as_dict()}
        record["device"] = self.calibration.device
        record["px_per_cm_y"] = self.calibration.px_per_cm_y
        record["px_per_cm_x"] = self.calibration.px_per_cm_x
        return record


@torch.no_grad()
def _forward(model, roi_gray: np.ndarray, size: tuple[int, int], device, tta: bool):
    """Run the network on one ROI, optionally with a horizontal-flip average.

    Horizontal flip is the only test-time augmentation used, for the same reason
    it is the only training augmentation: it is the one transform whose effect on
    the orientation field we can undo exactly.  Flipping x negates ``sin 2t`` and
    leaves ``cos 2t`` alone, because the doubled angle reflects about the
    horizontal axis.
    """
    height, width = size
    tensor = torch.from_numpy(
        np.asarray(
            Image.fromarray(roi_gray.astype(np.uint8)).resize((width, height), Image.BILINEAR),
            dtype=np.float32,
        )[None, None] / 255.0
    ).to(device)

    outputs = model(tensor)
    probs = torch.sigmoid(outputs["mask"])
    orientation = outputs.get("orientation")

    if tta:
        flipped = model(torch.flip(tensor, dims=[3]))
        probs = 0.5 * (probs + torch.flip(torch.sigmoid(flipped["mask"]), dims=[3]))
        if orientation is not None and "orientation" in flipped:
            undone = torch.flip(flipped["orientation"], dims=[3])
            undone = torch.stack([undone[:, 0], -undone[:, 1]], dim=1)
            orientation = orientation + undone
            orientation = orientation / (orientation.norm(dim=1, keepdim=True) + 1e-6)

    probs = probs[0, 0].float().cpu().numpy()
    field = orientation[0].float().cpu().numpy() if orientation is not None else None
    return probs, field


def _resample(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return np.asarray(
        Image.fromarray(array.astype(np.float32), mode="F").resize((width, height), Image.BILINEAR),
        dtype=np.float32,
    )


def predict_frame(
    path: Path,
    apo_model,
    fasc_model,
    device: torch.device,
    size: tuple[int, int] = (512, 512),
    apo_threshold: float = 0.5,
    fasc_threshold: float = 0.5,
    tta: bool = True,
    use_orientation_head: bool = True,
) -> FramePrediction:
    cal = calibrate(path)
    gray = to_gray(np.asarray(Image.open(path)))
    roi = crop_to_roi(gray, cal)
    roi_shape = roi.shape

    apo_probs, _ = _forward(apo_model, roi, size, device, tta)
    fasc_probs, field = _forward(fasc_model, roi, size, device, tta)

    apo_probs = _resample(apo_probs, roi_shape)
    fasc_probs = _resample(fasc_probs, roi_shape)
    orientation = None
    if field is not None and use_orientation_head:
        cos2t = _resample(field[0], roi_shape)
        sin2t = _resample(field[1], roi_shape)
        norm = np.hypot(cos2t, sin2t) + 1e-6
        orientation = (cos2t / norm, sin2t / norm)

    # The ROI is a crop, not a resize, so the frame's mm-per-pixel carries over
    # unchanged; only the network canvas was rescaled, and we have already
    # resampled off it.
    estimate = estimate_architecture(
        apo_mask=(apo_probs > apo_threshold),
        fasc_mask=(fasc_probs > fasc_threshold),
        mm_per_px_x=cal.mm_per_px_x,
        mm_per_px_y=cal.mm_per_px_y,
        orientation=orientation,
        fasc_prob=fasc_probs,
    )
    return FramePrediction(path.name, estimate, cal)


def predict_all(
    paths: Sequence[Path],
    apo_weights: str | Path,
    fasc_weights: str | Path,
    device: torch.device | None = None,
    use_orientation_head: bool = True,
    use_sequence_smoothing: bool = True,
    tta: bool = True,
    progress: bool = True,
) -> list[dict]:
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    apo_model, _ = load_model(apo_weights, device)
    fasc_model, _ = load_model(fasc_weights, device)

    predictions: list[FramePrediction] = []
    for i, path in enumerate(paths):
        predictions.append(
            predict_frame(
                path, apo_model, fasc_model, device,
                tta=tta, use_orientation_head=use_orientation_head,
            )
        )
        if progress and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(paths)} frames", flush=True)

    records = [p.as_record() for p in predictions]
    _impute_failures(records)

    if use_sequence_smoothing:
        groups = group_frames(paths, [p.calibration for p in predictions])
        records = smooth_predictions(records, groups)
        n_runs = sum(1 for g in groups if g.is_sequence)
        if progress:
            print(f"  sequence smoothing: {n_runs} runs covering "
                  f"{sum(g.size for g in groups if g.is_sequence)} frames", flush=True)
    else:
        for record in records:
            record["group_size"] = 1
    return records


def _impute_failures(records: list[dict]) -> None:
    """Replace hard failures with the cohort median rather than a constant.

    A frame whose aponeuroses were never found carries no information, so the
    best available answer is the central value of the frames that did work — and
    taking it from this run rather than from a hard-coded prior keeps the
    fallback honest if the test distribution differs from the training one.
    """
    good = [r for r in records if r.get("confidence", 0.0) > 0.05 and not r.get("failure", "").startswith("aponeurosis")]
    if not good:
        return
    medians = {
        field: float(np.median([r[field] for r in good]))
        for field in ("pa_deg", "fl_mm", "mt_mm")
    }
    for record in records:
        if record.get("failure", "") == "aponeurosis_not_found":
            record.update(medians)
            record["failure"] = "imputed_cohort_median"


def to_submission(records: list[dict], path: str | Path, separator: str = ",") -> "object":
    """Write the competition CSV.

    The sample file is semicolon-separated; Kaggle accepts either, and we keep
    the column order it specifies.
    """
    import pandas as pd

    frame = pd.DataFrame(records)
    pa, fl, mt = PRIORS.pa_deg, PRIORS.fl_mm, PRIORS.mt_mm
    frame["pa_deg"] = frame["pa_deg"].clip(*pa)
    frame["fl_mm"] = frame["fl_mm"].clip(*fl)
    frame["mt_mm"] = frame["mt_mm"].clip(*mt)
    submission = frame[list(SUBMISSION_COLUMNS)].copy()
    submission.to_csv(path, index=False, sep=separator, float_format="%.5f")
    return submission
