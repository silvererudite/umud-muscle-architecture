"""UMUD — inference, evaluation and submission (Kaggle kernel).

Consumes the weights produced by ``shamimahossain/umud-train-seg`` and writes

  * ``submission.csv``        — the competition entry
  * ``predictions.csv``       — every per-frame diagnostic, for the report
  * ``evaluation.json``       — held-out segmentation and geometry metrics
  * ``ablations.json``        — the two ablations the proposal promised

Everything here is deterministic; re-running it reproduces the submission byte
for byte.
"""

import json
import pathlib
import sys
import time


def _locate_source_package() -> str:
    for root in (pathlib.Path("/kaggle/input"), pathlib.Path("/kaggle/usr/lib")):
        if root.is_dir():
            for candidate in sorted(root.rglob("umud/config.py")):
                return str(candidate.parent.parent)
    raise SystemExit("umud-src not mounted")


def _locate_weights() -> pathlib.Path:
    for candidate in sorted(pathlib.Path("/kaggle/input").rglob("umud_apo.pt")):
        return candidate.parent
    raise SystemExit(
        "training output not mounted; /kaggle/input contains: "
        + str(sorted(p.name for p in pathlib.Path("/kaggle/input").glob("*")))
    )


SOURCE_DIR = _locate_source_package()
sys.path.insert(0, SOURCE_DIR)
print("umud package:", SOURCE_DIR, flush=True)

import numpy as np
import pandas as pd
import torch

from umud.config import find_competition_dir
from umud.data import list_samples, list_test_images, split_samples
from umud.evaluate import (
    aponeurosis_geometry_error,
    fascicle_orientation_error,
    paired_geometry_error,
    segmentation_metrics,
)
from umud.predict import load_model, predict_all, to_submission

COMPETITION = find_competition_dir()
WEIGHTS = _locate_weights()
OUT = pathlib.Path("/kaggle/working")
SIZE = (512, 512)

print("competition:", COMPETITION, flush=True)
print("weights:", WEIGHTS, sorted(p.name for p in WEIGHTS.glob("*.pt")), flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

apo_weights = WEIGHTS / "umud_apo.pt"
fasc_weights = WEIGHTS / "umud_fasc.pt"

# ---------------------------------------------------------------- submission
test_paths = list_test_images(COMPETITION)
print(f"\n=== inference on {len(test_paths)} test frames ===", flush=True)
started = time.time()
records = predict_all(
    test_paths, apo_weights, fasc_weights, device=device,
    use_orientation_head=True, use_sequence_smoothing=True, tta=True,
)
print(f"  done in {time.time() - started:.0f}s", flush=True)

frame = pd.DataFrame(records)
frame.to_csv(OUT / "predictions.csv", index=False)
submission = to_submission(records, OUT / "submission.csv", separator=",")

print("\n--- submission summary ---", flush=True)
print(submission.describe().to_string(), flush=True)
print("\nfailure modes:", flush=True)
print(frame["failure"].value_counts().to_string(), flush=True)
print("\nby scanner family:", flush=True)
print(frame.groupby("device")[["pa_deg", "fl_mm", "mt_mm", "confidence"]].mean().round(2).to_string(), flush=True)

assert len(submission) == 309, f"expected 309 rows, got {len(submission)}"
assert submission[["pa_deg", "fl_mm", "mt_mm"]].notna().all().all(), "NaNs in submission"

# ---------------------------------------------------------------- evaluation
print("\n=== held-out evaluation ===", flush=True)
apo_model, _ = load_model(apo_weights, device)
fasc_model, _ = load_model(fasc_weights, device)

evaluation = {}
for task, model in (("apo", apo_model), ("fasc", fasc_model)):
    samples = list_samples(task, COMPETITION)
    _, val = split_samples(samples, task=task)
    metrics = segmentation_metrics(model, val, SIZE, device)
    evaluation[f"{task}_segmentation"] = metrics
    print(f"  {task}: Dice {metrics['dice']:.4f}  IoU {metrics['iou']:.4f}  (n={metrics['n']})", flush=True)

apo_val = split_samples(list_samples("apo", COMPETITION), task="apo")[1]
fasc_val = split_samples(list_samples("fasc", COMPETITION), task="fasc")[1]

geo = aponeurosis_geometry_error(apo_model, apo_val, SIZE, device)
evaluation["aponeurosis_geometry"] = geo
print(f"  thickness: median rel. err {geo['thickness_rel_err']['median']:.3f}  "
      f"deep-angle median err {geo['deep_angle_abs_err_deg']['median']:.2f} deg", flush=True)

ori_head = fascicle_orientation_error(fasc_model, fasc_val, SIZE, device, use_head=True)
evaluation["fascicle_orientation_head"] = ori_head
print(f"  fascicle angle (orientation head): median {ori_head['angle_abs_err_deg']['median']:.2f} deg", flush=True)

paired = paired_geometry_error(apo_model, fasc_model, SIZE, device, COMPETITION, use_head=True)
evaluation["paired_end_to_end"] = paired
print(f"  paired end-to-end (n={paired['n']}): PA {paired['pa_abs_err_deg']['median']:.2f} deg, "
      f"FL {paired['fl_rel_err']['median']:.3f} rel, MT {paired['mt_rel_err']['median']:.3f} rel", flush=True)

(OUT / "evaluation.json").write_text(json.dumps(evaluation, indent=2, default=float))

# ---------------------------------------------------------------- ablations
print("\n=== ablations ===", flush=True)
ablations = {}

ori_tensor = fascicle_orientation_error(fasc_model, fasc_val, SIZE, device, use_head=False)
ablations["orientation_head"] = {
    "with_head_median_deg": ori_head["angle_abs_err_deg"]["median"],
    "structure_tensor_median_deg": ori_tensor["angle_abs_err_deg"]["median"],
    "with_head_mean_deg": ori_head["angle_abs_err_deg"]["mean"],
    "structure_tensor_mean_deg": ori_tensor["angle_abs_err_deg"]["mean"],
}
print(f"  orientation head  : {ori_head['angle_abs_err_deg']['median']:.2f} deg median", flush=True)
print(f"  structure tensor  : {ori_tensor['angle_abs_err_deg']['median']:.2f} deg median", flush=True)

paired_no_head = paired_geometry_error(apo_model, fasc_model, SIZE, device, COMPETITION, use_head=False)
ablations["paired_no_orientation_head"] = {
    "pa_median_deg": paired_no_head["pa_abs_err_deg"]["median"],
    "fl_median_rel": paired_no_head["fl_rel_err"]["median"],
}

# Sequence smoothing: how much does it move the test predictions, and how
# internally consistent were the runs before it?
raw = predict_all(
    test_paths, apo_weights, fasc_weights, device=device,
    use_orientation_head=True, use_sequence_smoothing=False, tta=True, progress=False,
)
raw_frame = pd.DataFrame(raw)
raw_frame.to_csv(OUT / "predictions_no_smoothing.csv", index=False)

from umud.calibration import calibrate
from umud.sequence import group_frames

groups = group_frames(test_paths, [calibrate(p) for p in test_paths])
runs = [g for g in groups if g.is_sequence]
within = {
    field: float(np.mean([raw_frame[field].iloc[g.indices].std() for g in runs]))
    for field in ("pa_deg", "fl_mm", "mt_mm")
}
shift = {
    field: float(np.abs(frame[field].to_numpy() - raw_frame[field].to_numpy()).mean())
    for field in ("pa_deg", "fl_mm", "mt_mm")
}
ablations["sequence_smoothing"] = {
    "n_runs": len(runs),
    "frames_in_runs": int(sum(g.size for g in runs)),
    "within_run_std_before": within,
    "mean_abs_shift": shift,
}
print(f"  {len(runs)} runs; within-run std before smoothing: "
      f"PA {within['pa_deg']:.2f} deg, FL {within['fl_mm']:.2f} mm, MT {within['mt_mm']:.2f} mm", flush=True)

# No-TTA arm, to show what the flip averaging buys.
no_tta = predict_all(
    test_paths, apo_weights, fasc_weights, device=device,
    use_orientation_head=True, use_sequence_smoothing=True, tta=False, progress=False,
)
no_tta_frame = pd.DataFrame(no_tta)
ablations["tta"] = {
    field: float(np.abs(frame[field].to_numpy() - no_tta_frame[field].to_numpy()).mean())
    for field in ("pa_deg", "fl_mm", "mt_mm")
}

(OUT / "ablations.json").write_text(json.dumps(ablations, indent=2, default=float))

# ---------------------------------------------------------------- qualitative
print("\n=== qualitative panels ===", flush=True)
from umud.visualise import qualitative_panel

by_device = frame.groupby("device")["image_id"].first().to_dict()
one_per_family = [p for p in test_paths if p.name in set(by_device.values())]
qualitative_panel(
    one_per_family, apo_model, fasc_model, device,
    OUT / "qualitative_by_device.png",
    title="One frame per scanner family",
)

ranked = frame.sort_values("confidence")
worst = [p for p in test_paths if p.name in set(ranked.image_id.head(3))]
best = [p for p in test_paths if p.name in set(ranked.image_id.tail(3))]
qualitative_panel(
    best + worst, apo_model, fasc_model, device,
    OUT / "qualitative_best_worst.png",
    title="Highest-confidence frames (top row) and lowest (bottom row)",
)
print("wrote qualitative panels", flush=True)

print("\nwrote submission.csv, predictions.csv, evaluation.json, ablations.json", flush=True)
