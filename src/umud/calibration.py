"""Recover the millimetre-per-pixel scale from the ultrasound ruler.

Every target in this competition is a physical quantity — millimetres and
degrees — but the network only ever sees pixels.  Nothing downstream is
meaningful without the scale, and the scale is not supplied as metadata: it has
to be read back off the depth ruler the scanner burns into the image.

Each scanner family draws that ruler differently (left edge, right edge, bottom
edge; major ticks every 5 mm or every 10 mm; fixed or operator-selected depth),
so the recogniser is a small table of device signatures with a generic
autocorrelation fallback for anything unrecognised.

The device-signature table follows the analysis published by AmbrosM in the
public notebook "UMUD Quick and Dirty" (CC-BY-SA, linked in the README); the
tick geometry of each scanner family is an empirical fact about the data rather
than something we could derive, and we cite it rather than re-deriving it.  The
generic fallback, the pixel-aspect handling and the confidence reporting are
ours.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Calibration:
    """Scale and region of interest for one frame."""

    px_per_cm_y: float          # depth axis
    px_per_cm_x: float          # lateral axis
    roi: tuple[int, int, int, int]   # (left, top, right, bottom) in pixels
    device: str
    ok: bool
    note: str = ""

    @property
    def mm_per_px_y(self) -> float:
        return 10.0 / self.px_per_cm_y

    @property
    def mm_per_px_x(self) -> float:
        return 10.0 / self.px_per_cm_x

    @property
    def square_pixels(self) -> bool:
        return abs(self.px_per_cm_x - self.px_per_cm_y) / self.px_per_cm_y < 0.02

    def as_dict(self) -> dict:
        d = asdict(self)
        d["roi"] = list(self.roi)
        d["mm_per_px_x"] = self.mm_per_px_x
        d["mm_per_px_y"] = self.mm_per_px_y
        return d


def load_image(path: str | Path) -> np.ndarray:
    """Read a frame as an ``(H, W)`` or ``(H, W, 3)`` uint8 array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB") if im.mode not in ("L", "RGB") else im)


def to_gray(a: np.ndarray) -> np.ndarray:
    """Collapse a frame to a 2-D float array.

    Deliberately a whole-image operation: slicing a colour image first and
    converting afterwards silently leaves a trailing colour axis on 1-D
    profiles, which makes every threshold comparison below return an array.
    """
    a = np.asarray(a)
    return a.mean(axis=-1).astype(np.float64) if a.ndim == 3 else a.astype(np.float64)


# --------------------------------------------------------------------------
# Generic fallback: find a ruler column and read its tick spacing
# --------------------------------------------------------------------------


def _tick_positions(profile: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """Centres of the bright runs along a 1-D ruler profile."""
    bright = profile > threshold
    if not bright.any():
        return np.empty(0)
    edges = np.diff(bright.astype(np.int8))
    starts = list(np.nonzero(edges == 1)[0] + 1)
    ends = list(np.nonzero(edges == -1)[0] + 1)
    if bright[0]:
        starts.insert(0, 0)
    if bright[-1]:
        ends.append(len(bright))
    return np.array([0.5 * (s + e - 1) for s, e in zip(starts, ends) if e - s <= 12])


def _spacing_from_ticks(ticks: np.ndarray, max_irregularity: float = 0.12) -> Optional[float]:
    """Robust tick pitch: the median gap, provided the gaps are actually regular.

    The regularity test is what separates a ruler from speckle.  A drawn ruler
    has gaps whose spread is a few percent of their mean; bright specks in an
    ultrasound border produce "ticks" whose gaps are all over the place.  Without
    this check the detector happily invents a scale for an image that has no
    ruler at all, which is worse than admitting defeat: a wrong scale silently
    rescales every millimetre the pipeline reports.
    """
    if ticks.size < 5:
        return None
    gaps = np.diff(ticks)
    gaps = gaps[gaps > 2]
    if gaps.size < 4:
        return None
    pitch = float(np.median(gaps))
    inliers = gaps[np.abs(gaps - pitch) < 0.25 * pitch]
    if inliers.size < 4 or inliers.size < 0.7 * gaps.size:
        return None
    if float(np.std(inliers) / max(np.mean(inliers), 1e-6)) > max_irregularity:
        return None
    return float(np.median(inliers))


def _looks_like_chrome(profile: np.ndarray, threshold: float = 50.0) -> bool:
    """Whether a border profile looks like scanner chrome rather than tissue.

    Rulers are drawn as bright marks on a black margin, so the profile should be
    mostly dark with a small bright minority.  A strip that runs through actual
    tissue has a high, noisy baseline and fails both halves of this test.
    """
    dark_fraction = float((profile < threshold).mean())
    if dark_fraction < 0.6:
        return False
    background = float(np.median(profile))
    peak = float(profile.max())
    return background < threshold and peak > background + 80.0


def _generic_calibration(g: np.ndarray) -> Optional[tuple[float, str]]:
    """Scan the image borders for a ruler and return ``(px_per_cm, note)``.

    We try the four border strips, score each candidate column/row by how many
    evenly spaced bright runs it carries, and take the best.  The 5 mm/10 mm
    ambiguity is resolved by preferring the interpretation that puts the total
    imaged depth in the 2–10 cm range that clinical lower-limb scanning uses.
    """
    height, width = g.shape
    best: Optional[tuple[int, float, str]] = None

    strips = [
        ("left", [g[:, c] for c in range(2, min(24, width))]),
        ("right", [g[:, width - 1 - c] for c in range(2, min(24, width))]),
        ("bottom", [g[height - 1 - r, :] for r in range(2, min(24, height))]),
        ("top", [g[r, :] for r in range(2, min(24, height))]),
    ]
    for side, profiles in strips:
        for profile in profiles:
            if not _looks_like_chrome(profile):
                continue
            ticks = _tick_positions(profile)
            pitch = _spacing_from_ticks(ticks)
            if pitch is None or pitch < 4:
                continue
            span = ticks.max() - ticks.min()
            n = int(round(span / pitch))
            if n < 3:
                continue
            axis_len = height if side in ("left", "right") else width
            for mm_per_tick in (5.0, 10.0):
                px_per_cm = pitch * 10.0 / mm_per_tick
                extent_cm = axis_len / px_per_cm
                if not (2.0 <= extent_cm <= 12.0):
                    continue
                score = n
                if best is None or score > best[0]:
                    best = (score, px_per_cm, f"generic:{side}:{mm_per_tick:.0f}mm:{n}ticks")
    if best is None:
        return None
    return best[1], best[2]


# --------------------------------------------------------------------------
# Device signature table
# --------------------------------------------------------------------------


def _calibrate_png_family(a: np.ndarray, g: np.ndarray) -> Calibration:
    """Depth ruler in the left margin, minor ticks in column 6, major in 9.

    Operator-selected depth (3-6 cm); the imaged sector is 2.85 cm wide with
    square pixels, which is what lets us pin the lateral scale too.
    """
    col6 = g[:, 6]
    col9 = g[:, 9]
    first_tick = int(np.argmax(col6 > 50))
    second_minor = 150 + int(np.argmax(col6[150:] > 50))
    second_major = 150 + int(np.argmax(col9[150:] > 50))
    last_tick = len(a) - 1 - int(np.argmax(col6[::-1] > 50))

    if (second_major - first_tick) < 3 * (second_minor - first_tick):
        px_per_cm = float(second_major - first_tick)   # major = 10 mm
    else:
        px_per_cm = float(second_minor - first_tick)   # major = 50 mm, minor = 10 mm

    half = a.shape[1] // 2
    half_width = int(np.argmin(g[:, half:].sum(axis=0)))
    left, right = half - half_width, half + half_width
    width_cm = (right - left) / px_per_cm if px_per_cm > 0 else 0.0
    ok = px_per_cm > 0 and 2.7 < width_cm < 3.0
    return Calibration(
        px_per_cm_y=px_per_cm,
        px_per_cm_x=px_per_cm,          # square pixels
        roi=(left, first_tick, right, last_tick),
        device="png_left_ruler",
        ok=ok,
        note=f"width={width_cm:.2f}cm",
    )


_TICKS_TO_ROI = {
    7: (142, 1058), 8: (163, 1037), 9: (211, 989), 10: (249, 951),
    11: (282, 918), 12: (308, 892), 13: (331, 869), 14: (349, 851),
}


def _calibrate_800x1200_right(a: np.ndarray, g: np.ndarray) -> Calibration:
    """Ruler in the right margin (column 1150), operator-selected depth.

    Two ticks per centimetre up to 14 ticks; a 15-tick ruler is the 3 cm preset
    drawn with five ticks per centimetre instead.  The sector is 5.75 cm wide,
    which fixes the lateral scale independently of the depth scale — these
    images have genuinely non-square pixels.
    """
    col = g[:, 1150]
    first_tick = int(np.argmax(col > 50))
    second_major = first_tick + 20 + int(np.argmax(col[first_tick + 20:] > 50))
    last_tick = len(a) - 1 - int(np.argmax(col[::-1] > 50))
    n_ticks = int(round((last_tick - first_tick) / max(second_major - first_tick, 1)))

    if n_ticks in _TICKS_TO_ROI:
        px_per_cm_y = (last_tick - first_tick) / n_ticks * 2.0
        left, right = _TICKS_TO_ROI[n_ticks]
    elif n_ticks == 15:
        px_per_cm_y = (last_tick - first_tick) / 3.0
        left, right = 142, 1058
    else:
        return Calibration(0.0, 0.0, (0, 0, a.shape[1], a.shape[0]),
                           "800x1200_right", False, f"n_ticks={n_ticks}")

    px_per_cm_x = (right - left) / 5.75
    return Calibration(
        px_per_cm_y=px_per_cm_y,
        px_per_cm_x=px_per_cm_x,
        roi=(left, 91, right, last_tick),
        device="800x1200_right",
        ok=px_per_cm_y > 0,
        note=f"n_ticks={n_ticks}",
    )


def _calibrate_800x1200_left(a: np.ndarray, g: np.ndarray) -> Calibration:
    """Fixed 5 cm depth preset with the ruler in the left margin."""
    px_per_cm = (783 - 42) / 5.0
    return Calibration(px_per_cm, px_per_cm, (171, 42, 1029, 798),
                       "800x1200_left", True)


def _calibrate_644x1088(a: np.ndarray, g: np.ndarray) -> Calibration:
    px_per_cm = 630.5 / 5.0
    return Calibration(px_per_cm, px_per_cm, (140, 0, 947, 643), "644x1088", True)


def _calibrate_512_bottom(a: np.ndarray, g: np.ndarray) -> Calibration:
    """Lateral ruler along the bottom edge, fixed 5 cm span."""
    row = g[-5]
    px_per_cm = (442 - 53) / 5.0
    known = [(49, 438), (52, 441), (53, 442)]
    ok = any(row[i] > 150 and row[j] > 150 for i, j in known)
    if not ok:
        ticks = _tick_positions(row, threshold=120)
        if ticks.size >= 2:
            px_per_cm = float(ticks.max() - ticks.min()) / 5.0
            ok = px_per_cm > 0
    return Calibration(px_per_cm, px_per_cm, (0, 0, a.shape[1], a.shape[0] - 10),
                       "512_bottom", ok)


def _calibrate_853_bottom(a: np.ndarray, g: np.ndarray) -> Calibration:
    row = g[-5]
    for i, j in ((100, 934), (44, 879)):
        if j < len(row) and row[i] > 150 and row[j] > 150:
            px_per_cm = (j - i) / 5.0
            return Calibration(px_per_cm, px_per_cm, (0, 0, a.shape[1], a.shape[0]),
                               "853_bottom", True)
    ticks = _tick_positions(row, threshold=120)
    if ticks.size >= 2:
        px_per_cm = float(ticks.max() - ticks.min()) / 5.0
        return Calibration(px_per_cm, px_per_cm, (0, 0, a.shape[1], a.shape[0]),
                           "853_bottom", px_per_cm > 0, "edge-ticks")
    return Calibration(0.0, 0.0, (0, 0, a.shape[1], a.shape[0]), "853_bottom", False)


def calibrate(path_or_array, filename: str = "") -> Calibration:
    """Recover ``(px_per_cm, roi)`` for one frame.

    Dispatch is on image shape plus a signature pixel, because the scanner
    family is what determines the ruler layout and the shape is a reliable
    proxy for it.  Anything unrecognised falls through to the generic ruler
    detector, and anything that fails that is reported with ``ok=False`` rather
    than guessed at — the caller decides what to do with an uncalibrated frame.
    """
    if isinstance(path_or_array, (str, Path)):
        filename = filename or Path(path_or_array).name
        a = load_image(path_or_array)
    else:
        a = np.asarray(path_or_array)

    g = to_gray(a)
    height, width = a.shape[:2]
    suffix = Path(filename).suffix.lower()

    try:
        if suffix == ".png":
            cal = _calibrate_png_family(a, g)
        elif (height, width) == (800, 1200) and a.ndim == 3:
            if bool((g[87, 1147:1157] == 175).all()):
                cal = _calibrate_800x1200_right(a, g)
            elif bool((g[42, 67:74] > 115).all()):
                cal = _calibrate_800x1200_left(a, g)
            else:
                cal = Calibration(0.0, 0.0, (0, 0, width, height), "800x1200_unknown", False)
        elif (height, width) == (644, 1088):
            cal = _calibrate_644x1088(a, g)
        elif height in (512, 513):
            cal = _calibrate_512_bottom(a, g)
        elif height == 853:
            cal = _calibrate_853_bottom(a, g)
        else:
            cal = Calibration(0.0, 0.0, (0, 0, width, height), f"unknown_{height}x{width}", False)
    except Exception as exc:  # a malformed ruler must not kill a whole run
        cal = Calibration(0.0, 0.0, (0, 0, width, height), "error", False, str(exc)[:80])

    if cal.ok and cal.px_per_cm_y > 0:
        return cal

    generic = _generic_calibration(g)
    if generic is not None:
        px_per_cm, note = generic
        return Calibration(px_per_cm, px_per_cm, (0, 0, width, height),
                           cal.device + "+fallback", True, note)
    return cal


def crop_to_roi(a: np.ndarray, cal: Calibration) -> np.ndarray:
    left, top, right, bottom = cal.roi
    left, top = max(0, left), max(0, top)
    right = min(a.shape[1], right) if right > 0 else a.shape[1]
    bottom = min(a.shape[0], bottom) if bottom > 0 else a.shape[0]
    if right - left < 16 or bottom - top < 16:
        return a
    return a[top:bottom, left:right]
