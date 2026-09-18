"""Evaluation without expert measurements.

The competition gives no ground-truth thickness, pennation angle or fascicle
length for any training frame — only masks — and the leaderboard allows five
submissions a day.  Neither is a basis for choosing between design options, so
the project needs an offline protocol.

The one used here is a **pseudo-ground-truth** protocol: run the *same*
geometric reconstruction twice, once on the annotator's masks and once on the
network's, and compare.  What it measures is precisely the quantity the
segmentation contributes to the final error, with the geometry module held
fixed.  What it does not measure is the error of the geometry module itself
against a human measurement — that would need the expert-annotated UMUD subsets,
which are outside this competition's data, and the report says so rather than
pretending otherwise.

Two evaluation sets:

* **Per-task** — aponeurosis frames give thickness and aponeurosis inclination;
  fascicle frames give fascicle orientation.  Large (155 + 417 frames) but each
  covers only part of the geometry.
* **Paired** — the 78 frames carrying both annotations, where the full
  reconstruction (PA, FL, MT together) can be compared end to end.  Small, but
  it is the only set that exercises the whole pipeline.

Because the scale can only be recovered reliably for part of the training data
(the ruler survives the competition's resizing on some scanner families and not
others), the primary numbers are **scale-free**: thickness and fascicle length
as a fraction of frame height, angles in degrees on the frame's own pixel grid.
Millimetre errors are reported for the calibrated subset alongside.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image

from .calibration import calibrate
from .config import UNCLAMPED
from .data import Sample, list_samples, load_pair, paired_frames, split_samples
from .geometry import estimate_architecture, extract_aponeuroses, orientation_from_field, orientation_from_mask
from .losses import dice_score, iou_score

Image.MAX_IMAGE_PIXELS = None


# --------------------------------------------------------------------------
# Mask-level metrics
# --------------------------------------------------------------------------


@torch.no_grad()
def segmentation_metrics(
    model, samples: Sequence[Sample], size, device, batch_size: int = 8, threshold: float = 0.5
) -> dict:
    """Dice / IoU on a held-out split, broken down by scanner family."""
    model.eval()
    per_device: dict[str, list[tuple[float, float]]] = {}
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        images, masks = [], []
        for sample in chunk:
            image, mask = load_pair(sample, size)
            images.append(image.astype(np.float32) / 255.0)
            masks.append(mask.astype(np.float32))
        x = torch.from_numpy(np.stack(images)[:, None]).to(device)
        y = torch.from_numpy(np.stack(masks)[:, None]).to(device)
        logits = model(x)["mask"].float()
        for i, sample in enumerate(chunk):
            d = dice_score(logits[i:i + 1], y[i:i + 1], threshold)
            j = iou_score(logits[i:i + 1], y[i:i + 1], threshold)
            per_device.setdefault(sample.device_key, []).append((d, j))

    everything = [v for values in per_device.values() for v in values]
    return {
        "dice": float(np.mean([d for d, _ in everything])) if everything else float("nan"),
        "iou": float(np.mean([j for _, j in everything])) if everything else float("nan"),
        "n": len(everything),
        "by_device": {
            key: {
                "dice": float(np.mean([d for d, _ in values])),
                "iou": float(np.mean([j for _, j in values])),
                "n": len(values),
            }
            for key, values in sorted(per_device.items())
        },
    }


# --------------------------------------------------------------------------
# Geometry-level metrics
# --------------------------------------------------------------------------


def _predict_masks(model, image: np.ndarray, device, threshold: float = 0.5):
    x = torch.from_numpy(image.astype(np.float32)[None, None] / 255.0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.sigmoid(out["mask"])[0, 0].float().cpu().numpy()
        field = out["orientation"][0].float().cpu().numpy() if "orientation" in out else None
    return probs, (probs > threshold), field


def aponeurosis_geometry_error(
    model, samples: Sequence[Sample], size, device
) -> dict:
    """Thickness and aponeurosis-inclination error, ground-truth masks vs predicted.

    Thickness is reported as a fraction of frame height so that it is comparable
    across scanner families whose scales we cannot all recover; multiply by the
    frame's own mm-per-pixel to read it in millimetres.
    """
    rows = []
    for sample in samples:
        image, gt_mask = load_pair(sample, size)
        _, pred_mask, _ = _predict_masks(model, image, device)

        gt = extract_aponeuroses(gt_mask, 1.0, 1.0)
        pr = extract_aponeuroses(pred_mask, 1.0, 1.0)
        if gt[0] is None:
            continue    # no usable reference on this frame
        from .geometry import muscle_thickness_mm
        gt_thickness = muscle_thickness_mm(gt[0], gt[1])
        gt_deep = gt[1].angle_deg

        if pr[0] is None:
            rows.append({
                "name": sample.name, "device": sample.device_key, "found": False,
                "thickness_abs_err_frac": np.nan, "deep_angle_abs_err_deg": np.nan,
            })
            continue

        pred_thickness = muscle_thickness_mm(pr[0], pr[1])
        rows.append({
            "name": sample.name,
            "device": sample.device_key,
            "found": True,
            "gt_thickness_px": gt_thickness,
            "pred_thickness_px": pred_thickness,
            "thickness_abs_err_frac": abs(pred_thickness - gt_thickness) / size[0],
            "thickness_rel_err": abs(pred_thickness - gt_thickness) / max(gt_thickness, 1e-6),
            "deep_angle_abs_err_deg": abs(pr[1].angle_deg - gt_deep),
        })
    return _summarise(rows, ["thickness_abs_err_frac", "thickness_rel_err", "deep_angle_abs_err_deg"])


def fascicle_orientation_error(
    model, samples: Sequence[Sample], size, device, use_head: bool = True
) -> dict:
    """Fascicle-angle error, ground-truth masks vs predicted.

    ``use_head=False`` reproduces the classical post-processing route (structure
    tensor on the predicted mask) and is the ablation arm for the orientation
    head.
    """
    rows = []
    for sample in samples:
        image, gt_mask = load_pair(sample, size)
        probs, pred_mask, field = _predict_masks(model, image, device)

        gt_angle, gt_resultant = orientation_from_mask(gt_mask, 1.0, 1.0)
        if not np.isfinite(gt_angle):
            continue

        if use_head and field is not None:
            weight = np.where(pred_mask, probs, 0.0)
            pred_angle, resultant = orientation_from_field(field[0], field[1], weight, 1.0, 1.0)
        else:
            pred_angle, resultant = orientation_from_mask(pred_mask, 1.0, 1.0)

        if not np.isfinite(pred_angle):
            rows.append({"name": sample.name, "device": sample.device_key,
                         "found": False, "angle_abs_err_deg": np.nan})
            continue

        error = abs(pred_angle - gt_angle)
        if error > 90.0:
            error = 180.0 - error       # orientation is defined modulo 180 degrees
        rows.append({
            "name": sample.name, "device": sample.device_key, "found": True,
            "gt_angle_deg": gt_angle, "pred_angle_deg": pred_angle,
            "angle_abs_err_deg": error, "dispersion": 1.0 - resultant,
        })
    return _summarise(rows, ["angle_abs_err_deg", "dispersion"])


def paired_geometry_error(
    apo_model,
    fasc_model,
    size,
    device,
    data_root: Path | None = None,
    use_head: bool = True,
    limit: int | None = None,
) -> dict:
    """End-to-end error on the 78 frames that carry both annotations.

    This is the only set where all three targets can be reconstructed from
    ground truth, so it answers "how much does the segmentation cost us in PA,
    FL and MT together".

    Runs with clamping disabled.  The physiological bounds are in millimetres
    and this routine works on the pixel grid, so leaving them on pins both the
    prediction and the reference to the same ceiling and reports an error of
    exactly zero -- which is what the first run of this evaluation did.
    """
    from .config import APO_IMAGES, APO_MASKS, COMPETITION_DIR, FASC_MASKS

    root = Path(data_root or COMPETITION_DIR)
    pairs = paired_frames()
    if limit:
        pairs = pairs[:limit]

    rows = []
    for pair in pairs:
        apo_name, fasc_name = pair["apo"][0], pair["fasc"][0]
        image_path = root / APO_IMAGES / apo_name
        with Image.open(image_path) as im:
            image = np.asarray(im.convert("L").resize((size[1], size[0]), Image.BILINEAR))
        with Image.open(root / APO_MASKS / apo_name) as mk:
            gt_apo = (np.asarray(mk.convert("L").resize((size[1], size[0]), Image.NEAREST)) > 127).astype(np.uint8)
        with Image.open(root / FASC_MASKS / fasc_name) as mk:
            gt_fasc = (np.asarray(mk.convert("L").resize((size[1], size[0]), Image.NEAREST)) > 127).astype(np.uint8)

        cal = calibrate(image_path)
        # Work on the pixel grid; converting to millimetres here would import the
        # calibration's own error into a number meant to isolate segmentation.
        mm_x = mm_y = 1.0

        reference = estimate_architecture(gt_apo, gt_fasc, mm_x, mm_y, priors=UNCLAMPED)

        apo_probs, apo_pred, _ = _predict_masks(apo_model, image, device)
        fasc_probs, fasc_pred, field = _predict_masks(fasc_model, image, device)
        orientation = None
        if use_head and field is not None:
            orientation = (field[0], field[1])
        predicted = estimate_architecture(
            apo_pred, fasc_pred, mm_x, mm_y,
            orientation=orientation, fasc_prob=fasc_probs, priors=UNCLAMPED,
        )

        rows.append({
            "name": apo_name,
            "device": f"{gt_apo.shape[0]}x{gt_apo.shape[1]}",
            "found": predicted.confidence > 0,
            "calibrated": cal.ok,
            "gt_pa": reference.pa_deg, "pred_pa": predicted.pa_deg,
            "gt_fl_px": reference.fl_mm, "pred_fl_px": predicted.fl_mm,
            "gt_mt_px": reference.mt_mm, "pred_mt_px": predicted.mt_mm,
            "gt_divergence_deg": reference.aponeurosis_divergence_deg,
            "pred_divergence_deg": predicted.aponeurosis_divergence_deg,
            "gt_fl_parallel_px": reference.fl_parallel_mm,
            "pred_fl_parallel_px": predicted.fl_parallel_mm,
            "pa_abs_err_deg": abs(predicted.pa_deg - reference.pa_deg),
            "fl_rel_err": abs(predicted.fl_mm - reference.fl_mm) / max(reference.fl_mm, 1e-6),
            "mt_rel_err": abs(predicted.mt_mm - reference.mt_mm) / max(reference.mt_mm, 1e-6),
            # The textbook formula's error on the same frames, so the report can
            # say whether the intersection construction actually pays off once
            # segmentation noise is in the loop rather than only in principle.
            "fl_parallel_rel_err": abs(predicted.fl_parallel_mm - reference.fl_parallel_mm)
            / max(reference.fl_parallel_mm, 1e-6),
            "divergence_abs_err_deg": abs(
                predicted.aponeurosis_divergence_deg - reference.aponeurosis_divergence_deg
            ),
            "confidence": predicted.confidence,
        })
    return _summarise(rows, ["pa_abs_err_deg", "fl_rel_err", "mt_rel_err",
                             "fl_parallel_rel_err", "divergence_abs_err_deg"])


# --------------------------------------------------------------------------
# Summarising
# --------------------------------------------------------------------------


def stat(summary: dict, field: str, which: str = "median") -> float:
    """Read one statistic out of a summary, tolerating a missing field.

    ``_summarise`` omits a field entirely when nothing was measurable for it --
    a model that finds no aponeuroses produces no thickness error at all.
    Callers that index straight into the dict would then raise, typically after
    the expensive part of a run has already finished, so this degrades to NaN
    and lets the reported number say so.
    """
    value = summary.get(field)
    if isinstance(value, dict) and which in value:
        return float(value[which])
    return float("nan")


def _summarise(rows: list[dict], fields: Sequence[str]) -> dict:
    import pandas as pd

    if not rows:
        return {"n": 0, "rows": []}
    frame = pd.DataFrame(rows)
    summary: dict = {"n": len(frame), "found_rate": float(frame["found"].mean())}
    usable = frame[frame["found"]]
    for field in fields:
        if field not in frame.columns or usable.empty:
            continue
        values = usable[field].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        summary[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }
    if "device" in frame.columns and not usable.empty:
        summary["by_device"] = {
            str(device): {
                "n": int(len(group)),
                **{
                    field: float(np.nanmean(group[field].to_numpy(dtype=float)))
                    for field in fields if field in group.columns
                },
            }
            for device, group in usable.groupby("device")
        }
    summary["rows"] = rows
    return summary
