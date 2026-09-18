"""Central configuration: paths, physiological priors, model hyper-parameters."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths.  On Kaggle the competition is mounted read-only under /kaggle/input;
# locally it lives under data/.  Everything downstream goes through these.
# --------------------------------------------------------------------------

ON_KAGGLE = Path("/kaggle/input").exists()

# The marker is a directory that only the competition data has, so finding it
# identifies the competition root wherever Kaggle happens to have mounted it.
_MARKER = "test_images_v2"

_KAGGLE_COMP_CANDIDATES = (
    Path("/kaggle/input/umud-challenge-muscle-architecture-in-ultrasound-data"),
    Path("/kaggle/input/competitions/umud-challenge-muscle-architecture-in-ultrasound-data"),
)


def find_competition_dir() -> Path:
    """Locate the competition data.

    Kaggle mounts competitions under a path that varies between worker images
    (``/kaggle/input/<slug>`` on some, ``/kaggle/input/competitions/<slug>`` on
    others), so we check the known layouts and then fall back to searching for
    the marker directory instead of hard-coding one and failing silently with an
    empty dataset.
    """
    env = os.environ.get("UMUD_DATA_DIR")
    if env and (Path(env) / _MARKER).is_dir():
        return Path(env)

    for candidate in _KAGGLE_COMP_CANDIDATES:
        if (candidate / _MARKER).is_dir():
            return candidate

    root = Path("/kaggle/input")
    if root.is_dir():
        for depth in ("*", "*/*", "*/*/*"):
            for candidate in sorted(root.glob(depth)):
                if (candidate / _MARKER).is_dir():
                    return candidate

    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "umud"


# Backwards-compatible private alias.
_find_competition_dir = find_competition_dir


COMPETITION_DIR = find_competition_dir()

APO_IMAGES = "apo_imgs_v1/apo_images_new_model_v1"
APO_MASKS = "apo_masks_v1/apo_masks_new_model_v1"
FASC_IMAGES = "fasc_imgs_v1/fasc_images_new_model_v1"
FASC_MASKS = "fasc_masks_v1/fasc_masks_new_model_v1"
TEST_IMAGES = "test_images_v2/test_set_v2"

SUBMISSION_COLUMNS = ("image_id", "pa_deg", "fl_mm", "mt_mm")
N_TEST_IMAGES = 309


# --------------------------------------------------------------------------
# Physiological priors.
#
# Ranges are deliberately wide — they are guard-rails against catastrophic
# geometry failures (a mask that collapses, an aponeurosis pair that is
# mis-paired), not a way to inject the answer.  Sources: the ranges reported
# for human lower-limb muscle in Ritsche et al. 2024 (DL_Track_US) and the
# UMUD benchmark sets.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysiologicalPriors:
    pa_deg: tuple[float, float] = (3.0, 40.0)
    fl_mm: tuple[float, float] = (20.0, 140.0)
    mt_mm: tuple[float, float] = (5.0, 60.0)

    # Fallbacks used only when geometry fails outright.  These are the medians
    # of the values the pipeline produces on images where it *does* succeed,
    # recomputed at inference time; the constants here are the cold-start seed.
    pa_fallback: float = 18.0
    fl_fallback: float = 70.0
    mt_fallback: float = 21.0

    def clamp(self, pa: float, fl: float, mt: float) -> tuple[float, float, float]:
        return (
            min(max(pa, self.pa_deg[0]), self.pa_deg[1]),
            min(max(fl, self.fl_mm[0]), self.fl_mm[1]),
            min(max(mt, self.mt_mm[0]), self.mt_mm[1]),
        )


PRIORS = PhysiologicalPriors()

# Evaluation on the pixel grid must NOT clamp: the bounds are millimetres, and a
# thickness of 174 *pixels* would be pinned to the 60 mm ceiling on both the
# prediction and the reference, making the error identically zero and the
# measurement meaningless.  Offline evaluation therefore uses these.
UNCLAMPED = PhysiologicalPriors(
    pa_deg=(0.0, 90.0),
    fl_mm=(0.0, 1e9),
    mt_mm=(0.0, 1e9),
)


# --------------------------------------------------------------------------
# Training configuration.
# --------------------------------------------------------------------------


@dataclass
class TrainConfig:
    task: str = "apo"                      # "apo" | "fasc"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    image_size: tuple[int, int] = (512, 512)
    batch_size: int = 8
    epochs: int = 40
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    seed: int = 754
    num_workers: int = 2
    amp: bool = True

    # The geometry-aware extra head (the contribution of this project).
    orientation_head: bool = True
    orientation_weight: float = 0.5
    # Only the fascicle task carries a meaningful orientation field.
    orientation_tasks: tuple[str, ...] = ("fasc",)

    # Loss mixing for the mask head.
    dice_weight: float = 0.5
    bce_weight: float = 0.5

    out_dir: Path = field(default_factory=lambda: Path("/kaggle/working"))

    @property
    def uses_orientation(self) -> bool:
        return self.orientation_head and self.task in self.orientation_tasks
