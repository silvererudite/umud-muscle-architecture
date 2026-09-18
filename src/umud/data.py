"""Dataset, splits and the orientation-target construction.

Two facts about this data drive the whole module:

1.  **Images and masks frequently disagree in size.**  Whenever they do, the
    image is 800x1200 and the mask carries some other shape — the images were
    bulk-resized to a fixed canvas without preserving aspect ratio while the
    masks kept their native resolution.  Resizing both onto a common canvas
    therefore restores the correspondence.  (Verified empirically: mean image
    intensity under a resized aponeurosis mask is 3.8x the frame mean, against
    1.2x for the same mask rolled by a quarter frame.)

2.  **The fascicle masks are short segments, not whole fascicles.**  Annotators
    traced the locally visible piece of each fascicle.  That makes them a poor
    length target but an excellent *orientation* target, which is precisely what
    the orientation head consumes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from .config import (
    APO_IMAGES, APO_MASKS, COMPETITION_DIR, FASC_IMAGES, FASC_MASKS, TEST_IMAGES,
)

TASK_DIRS = {
    "apo": (APO_IMAGES, APO_MASKS),
    "fasc": (FASC_IMAGES, FASC_MASKS),
}


@dataclass(frozen=True)
class Sample:
    name: str
    image_path: Path
    mask_path: Path
    native_shape: tuple[int, int]      # the mask's shape == the acquisition shape
    image_shape: tuple[int, int]

    @property
    def was_resized(self) -> bool:
        return self.native_shape != self.image_shape

    @property
    def device_key(self) -> str:
        """Group key for the split and for the per-device error breakdown.

        The acquisition shape is the most reliable proxy for scanner family we
        have: the competition ships no device metadata, but each scanner writes
        a characteristic frame size.
        """
        return f"{self.native_shape[0]}x{self.native_shape[1]}"


def list_samples(task: str, root: Path | None = None) -> list[Sample]:
    root = Path(root or COMPETITION_DIR)
    img_dir, mask_dir = (root / d for d in TASK_DIRS[task])
    samples = []
    for image_path in sorted(img_dir.glob("*.tif")):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            continue
        with Image.open(image_path) as im:
            image_shape = (im.size[1], im.size[0])
        with Image.open(mask_path) as mk:
            native_shape = (mk.size[1], mk.size[0])
        samples.append(Sample(image_path.name, image_path, mask_path, native_shape, image_shape))
    return samples


def list_test_images(root: Path | None = None) -> list[Path]:
    root = Path(root or COMPETITION_DIR)
    return sorted(
        p for p in (root / TEST_IMAGES).iterdir()
        if p.suffix.lower() in (".tif", ".png") and not p.name.startswith(".")
    )


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def _stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def split_samples(
    samples: Sequence[Sample], val_fraction: float = 0.15, seed: int = 754
) -> tuple[list[Sample], list[Sample]]:
    """Deterministic split, stratified by device family.

    Stratifying matters here: two scanner families supply most of the frames, so
    a plain random split can leave a rare device entirely out of validation and
    make the generalisation study impossible to run.  Hashing rather than
    shuffling keeps the split identical between the local session and the Kaggle
    kernel without shipping an index file.
    """
    by_device: dict[str, list[Sample]] = {}
    for sample in samples:
        by_device.setdefault(sample.device_key, []).append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    for device, group in sorted(by_device.items()):
        ordered = sorted(group, key=lambda s: _stable_hash(f"{seed}:{s.name}"))
        n_val = max(1, int(round(val_fraction * len(ordered)))) if len(ordered) > 3 else 0
        val.extend(ordered[:n_val])
        train.extend(ordered[n_val:])
    return train, val


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_pair(sample: Sample, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image, mask)`` resampled onto a common ``size`` canvas.

    Both are resized from their own native resolution onto the target canvas —
    never the mask onto the image and then onto the canvas, which would compound
    two rounds of nearest-neighbour damage on structures only a few pixels wide.
    """
    height, width = size
    with Image.open(sample.image_path) as im:
        image = np.asarray(im.convert("L").resize((width, height), Image.BILINEAR))
    with Image.open(sample.mask_path) as mk:
        mask = np.asarray(mk.convert("L").resize((width, height), Image.NEAREST))
    return image, (mask > 127).astype(np.uint8)


def load_test_image(path: Path, size: tuple[int, int], roi=None) -> np.ndarray:
    height, width = size
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    if roi is not None:
        left, top, right, bottom = roi
        if right - left > 16 and bottom - top > 16:
            arr = arr[max(0, top):bottom, max(0, left):right]
    return np.asarray(Image.fromarray(arr).resize((width, height), Image.BILINEAR))


# --------------------------------------------------------------------------
# Orientation targets
# --------------------------------------------------------------------------


def orientation_target(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Build the dense ``(cos 2t, sin 2t)`` target from a fascicle mask.

    The annotation gives us line segments, not angles, so the target has to be
    derived.  We take the structure tensor of the blurred mask: its minor
    eigenvector is the direction along which the segment varies least, i.e. the
    fascicle direction.  Encoding as a doubled angle removes the 180-degree
    ambiguity that would otherwise make the regression target discontinuous
    along every fascicle.

    Returns a ``(2, H, W)`` float32 array; it is only meaningful where the mask
    is set, and the training loss masks it accordingly.
    """
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(mask.astype(np.float32), sigma)
    gy, gx = np.gradient(smooth)
    jxx = ndimage.gaussian_filter(gx * gx, sigma * 2)
    jyy = ndimage.gaussian_filter(gy * gy, sigma * 2)
    jxy = ndimage.gaussian_filter(gx * gy, sigma * 2)

    cos2t = -(jxx - jyy)
    sin2t = -(2.0 * jxy)
    norm = np.hypot(cos2t, sin2t) + 1e-8
    return np.stack([cos2t / norm, sin2t / norm]).astype(np.float32)
