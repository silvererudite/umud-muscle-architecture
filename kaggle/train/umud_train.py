"""UMUD — train the geometry-aware segmentation models (Kaggle GPU kernel).

Trains two models:
  * ``apo``  — aponeurosis segmentation (1 048 annotated frames)
  * ``fasc`` — fascicle segmentation + dense orientation field (2 761 frames)

Both write ``umud_<task>.pt`` and ``history_<task>.json`` to /kaggle/working,
which the inference kernel mounts as a kernel source.

Source package: https://www.kaggle.com/datasets/shamimahossain/umud-src
"""

import json
import os
import pathlib
import sys
import time
from pathlib import Path


def _locate_source_package() -> str:
    """Find the mounted umud-src dataset.

    Kaggle normally mounts a dataset at /kaggle/input/<slug>, but the mount
    point can differ, so we search rather than assume and fail loudly with the
    actual directory listing if the package is missing.
    """
    roots = [pathlib.Path("/kaggle/input"), pathlib.Path("/kaggle/usr/lib")]
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.rglob("umud/config.py")):
            return str(candidate.parent.parent)
    listing = []
    for root in roots:
        if root.is_dir():
            listing += [str(p) for p in sorted(root.glob("*"))]
    raise SystemExit(f"umud-src not mounted. /kaggle/input contains: {listing}")


SOURCE_DIR = _locate_source_package()
print("umud package mounted at:", SOURCE_DIR, flush=True)
sys.path.insert(0, SOURCE_DIR)

import torch

from umud.config import TrainConfig, find_competition_dir
from umud.train import train

COMPETITION = find_competition_dir()
OUT = Path("/kaggle/working")

print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
print("competition dir:", COMPETITION, flush=True)
if not (COMPETITION / "test_images_v2").is_dir():
    raise SystemExit(
        "competition data not mounted; /kaggle/input contains: "
        + str(sorted(p.name for p in pathlib.Path("/kaggle/input").glob("*")))
    )
os.environ["UMUD_DATA_DIR"] = str(COMPETITION)

EPOCHS_APO = int(os.environ.get("UMUD_EPOCHS_APO", 45))
EPOCHS_FASC = int(os.environ.get("UMUD_EPOCHS_FASC", 35))

summaries = {}
started = time.time()

for task, epochs in (("apo", EPOCHS_APO), ("fasc", EPOCHS_FASC)):
    config = TrainConfig(
        task=task,
        encoder="resnet34",
        encoder_weights="imagenet",
        image_size=(512, 512),
        batch_size=8,
        epochs=epochs,
        lr=3e-4,
        num_workers=4,
        amp=True,
        orientation_head=True,      # only active for the fascicle task
        orientation_weight=0.5,
        out_dir=OUT,
    )
    print(f"\n{'=' * 70}\n{task.upper()}  ({epochs} epochs)\n{'=' * 70}", flush=True)
    summaries[task] = train(config, COMPETITION)

summaries["wall_clock_minutes"] = round((time.time() - started) / 60, 1)
(OUT / "training_summary.json").write_text(json.dumps(summaries, indent=2))

print("\n" + "=" * 70)
for task in ("apo", "fasc"):
    s = summaries[task]
    print(f"{task}: best val Dice {s['best_val_dice']:.4f}  "
          f"(pretrained encoder: {s['encoder_pretrained']})")
print(f"total {summaries['wall_clock_minutes']} min")
print("=" * 70)
