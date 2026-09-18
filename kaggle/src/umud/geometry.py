"""Masks + orientation field  ->  pennation angle, fascicle length, thickness.

This is the "geometric reconstruction" box of the proposal's pipeline.  Three
design choices distinguish it from the usual post-processing step:

1.  Everything is solved in **millimetre space**, not pixel space.  An angle
    measured on the pixel grid is only the anatomical angle when the pixels are
    square; we rescale first, then measure.  On the *test* set this correction
    turns out to be nearly a no-op (depth and lateral scales agree to 0.35 %),
    but 70 % of the training frames were resized off their native aspect ratio
    by up to 12.5 %, so the geometry cannot assume square pixels.

2.  Pennation angle is referenced to the **deep aponeurosis**, not to the image
    horizontal, which is what the clinical definition actually says.

3.  Fascicle length is obtained by **intersecting the fascicle direction with
    the two fitted aponeurosis lines**, rather than by the textbook
    ``MT / sin(PA)``.  The textbook formula silently assumes the aponeuroses are
    parallel; in real images they diverge by a few degrees and the error that
    introduces grows as ``1/sin`` for shallow pennation angles.

Every routine returns a confidence in [0, 1] alongside its estimate so that the
sequence module can weight frames and the report can separate "the model was
wrong" from "the model knew it had nothing to work with".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .config import PRIORS

# --------------------------------------------------------------------------
# Small containers
# --------------------------------------------------------------------------


@dataclass
class Aponeurosis:
    """A fitted aponeurosis curve in millimetre space.

    Aponeuroses are visibly curved in a good fraction of the data (see
    ``docs/figures/data_overview.png``), so a straight-line fit biases both the
    thickness and — through the fascicle/aponeurosis intersection — the fascicle
    length.  We therefore fit a low-order polynomial and linearise it locally
    wherever a slope is needed.
    """

    coeffs: np.ndarray          # numpy.polyfit order: highest power first
    x_min: float
    x_max: float
    n_points: int
    residual: float

    def y_at(self, x):
        return np.polyval(self.coeffs, x)

    def slope_at(self, x: float) -> float:
        return float(np.polyval(np.polyder(self.coeffs), x))

    def angle_deg_at(self, x: float) -> float:
        """Local inclination w.r.t. the horizontal, in degrees."""
        return math.degrees(math.atan(self.slope_at(x)))

    @property
    def x_mid(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def angle_deg(self) -> float:
        return self.angle_deg_at(self.x_mid)

    def tangent_at(self, x: float) -> "Aponeurosis":
        """First-order expansion about ``x`` — used for closed-form intersection."""
        slope = self.slope_at(x)
        intercept = float(self.y_at(x)) - slope * x
        return Aponeurosis(
            coeffs=np.array([slope, intercept]),
            x_min=self.x_min,
            x_max=self.x_max,
            n_points=self.n_points,
            residual=self.residual,
        )



@dataclass
class ArchitectureEstimate:
    pa_deg: float
    fl_mm: float
    mt_mm: float
    confidence: float
    # Diagnostics — kept for the error analysis, never used by the metric.
    apo_confidence: float = 0.0
    fasc_confidence: float = 0.0
    fascicle_angle_deg: float = float("nan")
    deep_angle_deg: float = float("nan")
    superficial_angle_deg: float = float("nan")
    aponeurosis_divergence_deg: float = float("nan")
    fl_parallel_mm: float = float("nan")
    orientation_dispersion: float = float("nan")
    n_apo_components: int = 0
    failure: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Aponeurosis extraction
# --------------------------------------------------------------------------


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected components.  Uses cv2 when available, else scipy."""
    try:
        import cv2

        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels, n - 1
    except Exception:  # pragma: no cover - cv2 is present on Kaggle
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=bool)
        labels, n = ndimage.label(mask, structure=structure)
        return labels, n


def _fit_curve_mm(
    ys: np.ndarray,
    xs: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
    degree: int = 2,
) -> Optional[Aponeurosis]:
    """Least-squares polynomial through an aponeurosis band, in millimetres.

    One point per column (the column centroid) rather than one per pixel: the
    band is several pixels thick and hundreds long, so per-pixel fitting lets
    whichever part of the band happens to be thickest dominate the fit.

    The degree is reduced automatically when the band is short — fitting a
    parabola through a 40-pixel fragment produces wild extrapolation.
    """
    if xs.size < 2:
        return None

    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    unique_x, start = np.unique(xs, return_index=True)
    if unique_x.size < 5:
        return None
    sums = np.add.reduceat(ys, start)
    counts = np.diff(np.append(start, ys.size))
    centroid_y = sums / counts

    X = unique_x.astype(np.float64) * mm_per_px_x
    Y = centroid_y * mm_per_px_y

    degree = min(degree, 1 if unique_x.size < 40 else degree)
    coeffs = np.polyfit(X, Y, degree)
    residual = float(np.sqrt(np.mean((Y - np.polyval(coeffs, X)) ** 2)))

    return Aponeurosis(
        coeffs=np.asarray(coeffs, dtype=np.float64),
        x_min=float(X.min()),
        x_max=float(X.max()),
        n_points=int(unique_x.size),
        residual=residual,
    )


def extract_aponeuroses(
    apo_mask: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
    min_area_frac: float = 0.0004,
    min_width_frac: float = 0.10,
    degree: int = 2,
) -> tuple[Optional[Aponeurosis], Optional[Aponeurosis], float, int]:
    """Fit the superficial and deep aponeurosis curves.

    Returns ``(superficial, deep, confidence, n_candidates)``.  Candidates are
    ranked by horizontal extent — an aponeurosis is the longest near-horizontal
    structure in the mask — and the retained pair must be separated vertically
    by at least 4 mm, which prevents pairing a band with a fragment of itself.
    """
    mask = np.asarray(apo_mask) > 0
    height, width = mask.shape
    if not mask.any():
        return None, None, 0.0, 0

    labels, n = _connected_components(mask)
    if n == 0:
        return None, None, 0.0, 0

    min_area = max(30, int(min_area_frac * height * width))
    min_width = max(10, int(min_width_frac * width))

    candidates = []
    for label in range(1, n + 1):
        ys, xs = np.nonzero(labels == label)
        if ys.size < min_area:
            continue
        if (xs.max() - xs.min()) < min_width:
            continue
        curve = _fit_curve_mm(ys, xs, mm_per_px_x, mm_per_px_y, degree=degree)
        if curve is None:
            continue
        # Aponeuroses are near-horizontal; steeper structures are acoustic
        # shadows or on-screen annotation, not anatomy.
        if abs(curve.angle_deg) > 35.0:
            continue
        candidates.append((xs.max() - xs.min(), float(ys.mean()), curve))

    if len(candidates) < 2:
        return None, None, 0.0, len(candidates)

    candidates.sort(key=lambda c: -c[0])
    first = candidates[0]

    # The two retained bands must be genuinely far apart, or we risk pairing a
    # band with a detached fragment of itself.  Two criteria, whichever is
    # stricter: 4 mm of real anatomy, and 3 % of the frame height.  The second
    # is what keeps the rule meaningful when the caller works on the pixel grid
    # (mm_per_px = 1), as the scale-free evaluation does -- there, "4 mm" would
    # degenerate to four pixels.
    min_separation_px = max(0.03 * height, 4.0 / max(mm_per_px_y, 1e-9))
    partner = None
    for cand in candidates[1:]:
        if abs(cand[1] - first[1]) >= min_separation_px:
            partner = cand
            break
    if partner is None:
        return None, None, 0.0, len(candidates)

    upper, lower = sorted([first, partner], key=lambda c: c[1])
    superficial, deep = upper[2], lower[2]

    width_mm = width * mm_per_px_x
    coverage = min(
        (superficial.x_max - superficial.x_min) / width_mm,
        (deep.x_max - deep.x_min) / width_mm,
    )
    fit_quality = 1.0 / (1.0 + max(superficial.residual, deep.residual))
    confidence = float(np.clip(coverage, 0.0, 1.0) * fit_quality)
    return superficial, deep, confidence, len(candidates)


# --------------------------------------------------------------------------
# Fascicle orientation
# --------------------------------------------------------------------------


def orientation_from_field(
    cos2t: np.ndarray,
    sin2t: np.ndarray,
    weight: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
) -> tuple[float, float]:
    """Aggregate the network's dense orientation field into a single angle.

    Orientation is a director, not a vector: ``t`` and ``t + 180`` describe the
    same fascicle.  Averaging angles directly is therefore wrong; we average the
    doubled-angle unit vectors and halve the result.  The resultant length is a
    free dispersion measure — near 1 the field agrees everywhere, near 0 it is
    noise — and feeds the per-frame confidence.

    Returns ``(angle_deg_in_millimetre_space, resultant_length)``.
    """
    w = np.asarray(weight, dtype=np.float64)
    total = w.sum()
    if total <= 1e-8:
        return float("nan"), 0.0

    c = float((w * np.asarray(cos2t, dtype=np.float64)).sum() / total)
    s = float((w * np.asarray(sin2t, dtype=np.float64)).sum() / total)
    resultant = math.hypot(c, s)
    if resultant < 1e-8:
        return float("nan"), 0.0

    angle_px = 0.5 * math.atan2(s, c)
    return _pixel_angle_to_mm(angle_px, mm_per_px_x, mm_per_px_y), resultant


def _pixel_angle_to_mm(angle_px_rad: float, mm_per_px_x: float, mm_per_px_y: float) -> float:
    """Convert an angle measured on the pixel grid into physical degrees.

    A direction ``(dx, dy)`` in pixels becomes ``(dx * sx, dy * sy)`` in
    millimetres, so the tangent picks up a factor ``sy / sx``.  Ultrasound
    pixels are routinely non-square and several of the training images have been
    resized without preserving aspect ratio, so this correction is not cosmetic.
    """
    dx = math.cos(angle_px_rad) * mm_per_px_x
    dy = math.sin(angle_px_rad) * mm_per_px_y
    return math.degrees(math.atan2(dy, dx))


def orientation_from_mask(
    fasc_mask: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
    sigma: float = 2.0,
) -> tuple[float, float]:
    """Structure-tensor fallback used when no orientation head is available.

    This is the classical post-processing route the proposal argues against, so
    it doubles as the "no orientation head" ablation arm.
    """
    from scipy import ndimage

    mask = np.asarray(fasc_mask, dtype=np.float64)
    if mask.max() <= 0:
        return float("nan"), 0.0
    smooth = ndimage.gaussian_filter(mask, sigma)
    gy, gx = np.gradient(smooth)

    jxx = ndimage.gaussian_filter(gx * gx, sigma * 2)
    jyy = ndimage.gaussian_filter(gy * gy, sigma * 2)
    jxy = ndimage.gaussian_filter(gx * gy, sigma * 2)

    # The dominant structure direction is perpendicular to the dominant gradient
    # direction, which is the sign flip below.
    cos2t = -(jxx - jyy)
    sin2t = -(2.0 * jxy)
    weight = np.asarray(mask > 0.5, dtype=np.float64)
    if weight.sum() < 20:
        return float("nan"), 0.0

    norm = np.hypot(cos2t, sin2t) + 1e-12
    return orientation_from_field(
        cos2t / norm, sin2t / norm, weight, mm_per_px_x, mm_per_px_y
    )


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------


def _overlap(superficial: Aponeurosis, deep: Aponeurosis) -> tuple[float, float]:
    lo = max(superficial.x_min, deep.x_min)
    hi = min(superficial.x_max, deep.x_max)
    if hi <= lo:  # disjoint fits — fall back to the union
        lo = min(superficial.x_min, deep.x_min)
        hi = max(superficial.x_max, deep.x_max)
    return lo, hi


def muscle_thickness_mm(superficial: Aponeurosis, deep: Aponeurosis) -> float:
    """Median separation of the aponeuroses, perpendicular to the deep one.

    The median over the overlapping span, rather than the value at a single
    abscissa, keeps a locally bad fit at one edge of the image from moving the
    answer.
    """
    lo, hi = _overlap(superficial, deep)
    xs = np.linspace(lo, hi, 64)
    vertical = np.asarray(deep.y_at(xs)) - np.asarray(superficial.y_at(xs))
    slopes = np.array([deep.slope_at(float(x)) for x in xs])
    perpendicular = vertical * np.cos(np.arctan(slopes))
    return float(np.median(np.abs(perpendicular)))


def fascicle_length_mm(
    superficial: Aponeurosis,
    deep: Aponeurosis,
    fascicle_angle_deg: float,
    fallback_mt_mm: float,
) -> tuple[float, str]:
    """Length of the fascicle segment bounded by the two aponeuroses.

    Start at the middle of the deep aponeurosis, travel along the fascicle
    direction, and solve for where that ray meets the (locally linearised)
    superficial aponeurosis.  This is the step the proposal singles out: the
    textbook ``MT / sin(PA)`` silently assumes the two aponeuroses are parallel,
    and the error that assumption introduces grows like ``1/sin`` as the
    pennation angle gets shallow.

    Returns ``(length_mm, mode)`` where mode records which branch was taken, so
    the report can quantify how often the two disagree.
    """
    theta = math.radians(fascicle_angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)

    lo, hi = _overlap(superficial, deep)
    x0 = 0.5 * (lo + hi)
    y0 = float(deep.y_at(x0))

    # Fascicles run from the deep aponeurosis up toward the superficial one, and
    # image y grows downward, so orient the ray upward.
    if dy > 0:
        dx, dy = -dx, -dy

    tangent = superficial.tangent_at(x0)
    slope_s, intercept_s = float(tangent.coeffs[0]), float(tangent.coeffs[1])

    denominator = dy - slope_s * dx
    if abs(denominator) > 1e-9:
        t = (slope_s * x0 + intercept_s - y0) / denominator
        if t > 0 and math.isfinite(t):
            return float(t), "intersection"

    pa = abs(fascicle_angle_deg - deep.angle_deg_at(x0))
    sin_pa = math.sin(math.radians(max(pa, 1e-3)))
    if sin_pa < 1e-6:
        return float("nan"), "degenerate"
    return float(fallback_mt_mm / sin_pa), "parallel"


def estimate_architecture(
    apo_mask: np.ndarray,
    fasc_mask: np.ndarray,
    mm_per_px_x: float,
    mm_per_px_y: float,
    orientation: Optional[tuple[np.ndarray, np.ndarray]] = None,
    fasc_prob: Optional[np.ndarray] = None,
    priors=PRIORS,
) -> ArchitectureEstimate:
    """Full geometric reconstruction for one frame.

    ``orientation`` is the network's ``(cos 2t, sin 2t)`` field.  Passing
    ``None`` switches to the structure-tensor route, which is exactly the
    "no orientation head" ablation reported in ``docs/REPORT.md``.
    """
    superficial, deep, apo_conf, n_components = extract_aponeuroses(
        apo_mask, mm_per_px_x, mm_per_px_y
    )

    if superficial is None or deep is None:
        return ArchitectureEstimate(
            pa_deg=priors.pa_fallback,
            fl_mm=priors.fl_fallback,
            mt_mm=priors.mt_fallback,
            confidence=0.0,
            n_apo_components=n_components,
            failure="aponeurosis_not_found",
        )

    mt = muscle_thickness_mm(superficial, deep)
    lo, hi = _overlap(superficial, deep)
    x_mid = 0.5 * (lo + hi)
    deep_angle = deep.angle_deg_at(x_mid)
    superficial_angle = superficial.angle_deg_at(x_mid)

    mask = np.asarray(fasc_mask) > 0
    if orientation is not None:
        cos2t, sin2t = orientation
        weight = np.asarray(fasc_prob if fasc_prob is not None else mask, dtype=np.float64)
        weight = np.where(mask, weight, 0.0)
        fasc_angle, resultant = orientation_from_field(
            cos2t, sin2t, weight, mm_per_px_x, mm_per_px_y
        )
    else:
        fasc_angle, resultant = orientation_from_mask(mask, mm_per_px_x, mm_per_px_y)

    fasc_conf = float(np.clip(resultant, 0.0, 1.0)) if mask.mean() > 1e-4 else 0.0

    if not math.isfinite(fasc_angle):
        # A muscle still has a measurable thickness when its fascicles are not
        # readable, so we keep MT and flag the rest.
        pa = priors.pa_fallback
        fl = mt / math.sin(math.radians(pa))
        pa, fl, mt = priors.clamp(pa, fl, mt)
        return ArchitectureEstimate(
            pa_deg=pa,
            fl_mm=fl,
            mt_mm=mt,
            confidence=0.25 * apo_conf,
            apo_confidence=apo_conf,
            fasc_confidence=0.0,
            deep_angle_deg=deep_angle,
            superficial_angle_deg=superficial_angle,
            n_apo_components=n_components,
            failure="orientation_not_found",
        )

    pa = abs(fasc_angle - deep_angle)
    if pa > 90.0:
        pa = 180.0 - pa

    fl, mode = fascicle_length_mm(superficial, deep, fasc_angle, mt)
    fl_parallel = mt / math.sin(math.radians(max(pa, 1e-3)))
    if not math.isfinite(fl) or fl <= 0:
        fl, mode = fl_parallel, "parallel"

    pa_c, fl_c, mt_c = priors.clamp(pa, fl, mt)
    clamped = (pa_c != pa) or (fl_c != fl) or (mt_c != mt)
    confidence = float(np.clip(0.5 * apo_conf + 0.5 * fasc_conf, 0.0, 1.0))
    if clamped:
        confidence *= 0.6

    return ArchitectureEstimate(
        pa_deg=pa_c,
        fl_mm=fl_c,
        mt_mm=mt_c,
        confidence=confidence,
        apo_confidence=apo_conf,
        fasc_confidence=fasc_conf,
        fascicle_angle_deg=fasc_angle,
        deep_angle_deg=deep_angle,
        superficial_angle_deg=superficial_angle,
        aponeurosis_divergence_deg=abs(superficial_angle - deep_angle),
        fl_parallel_mm=fl_parallel,
        orientation_dispersion=1.0 - resultant,
        n_apo_components=n_components,
        failure=("clamped_" + mode) if clamped else mode,
    )
