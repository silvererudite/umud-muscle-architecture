"""Geometry tests on synthetic phantoms, where the answer is known exactly."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from umud.geometry import (
    estimate_architecture,
    extract_aponeuroses,
    fascicle_length_mm,
    muscle_thickness_mm,
    orientation_from_field,
)


def phantom(thickness_px=200, angle_deg=20.0, height=400, width=600, deep_slope=0.0):
    """Two aponeuroses and a set of fascicles at a known angle."""
    apo = np.zeros((height, width), np.uint8)
    top = (height - thickness_px) // 2
    for x in range(40, width - 40):
        apo[top:top + 6, x] = 1
        y = int(top + thickness_px + deep_slope * (x - width / 2))
        if 0 <= y < height - 6:
            apo[y:y + 6, x] = 1

    fasc = np.zeros((height, width), np.uint8)
    for x0 in range(0, width, 40):
        for t in range(600):
            x = int(x0 + t * np.cos(np.radians(-angle_deg)))
            y = int(height // 2 + t * np.sin(np.radians(-angle_deg)))
            if 0 <= x < width and top + 10 < y < top + thickness_px - 10:
                fasc[y, x] = 1
    return apo, fasc


def test_thickness_is_exact_on_a_flat_phantom():
    apo, _ = phantom(thickness_px=200)
    sup, deep, conf, _ = extract_aponeuroses(apo, 0.1, 0.1)
    assert sup is not None and deep is not None
    assert muscle_thickness_mm(sup, deep) == pytest.approx(20.0, abs=0.3)
    assert conf > 0.5


def test_pennation_angle_recovered_within_a_degree():
    for angle in (10.0, 20.0, 30.0):
        apo, fasc = phantom(angle_deg=angle)
        est = estimate_architecture(apo, fasc, 0.1, 0.1)
        assert est.pa_deg == pytest.approx(angle, abs=1.5), f"angle {angle}"


def test_non_square_pixels_change_the_measured_angle():
    """A pixel grid twice as tall as it is wide must report a steeper angle."""
    apo, fasc = phantom(angle_deg=20.0)
    isotropic = estimate_architecture(apo, fasc, 0.1, 0.1)
    anisotropic = estimate_architecture(apo, fasc, 0.05, 0.1)
    expected = np.degrees(np.arctan(np.tan(np.radians(isotropic.pa_deg)) * 2))
    assert anisotropic.pa_deg == pytest.approx(expected, abs=1.5)
    assert anisotropic.pa_deg > isotropic.pa_deg


def test_parallel_aponeuroses_match_the_textbook_formula():
    apo, fasc = phantom(thickness_px=200, angle_deg=20.0, deep_slope=0.0)
    est = estimate_architecture(apo, fasc, 0.1, 0.1)
    textbook = est.mt_mm / np.sin(np.radians(est.pa_deg))
    assert est.fl_mm == pytest.approx(textbook, rel=0.02)


def test_the_two_fascicle_length_constructions_genuinely_differ():
    """When the aponeuroses are not parallel, MT / sin(PA) and the intersection
    give materially different answers — so the choice between them is a real
    one, not a formality. (Which one is *better* is an empirical question, and
    the measured answer is the textbook formula; see docs/REPORT.md.)"""
    apo, fasc = phantom(thickness_px=200, angle_deg=12.0, deep_slope=0.25)
    est = estimate_architecture(apo, fasc, 0.1, 0.1)
    assert est.aponeurosis_divergence_deg > 5.0
    spread = abs(est.fl_intersection_mm - est.fl_parallel_mm) / est.fl_parallel_mm
    assert spread > 0.05


def test_fl_method_selects_between_the_two_constructions():
    apo, fasc = phantom(thickness_px=200, angle_deg=12.0, deep_slope=0.25)
    default = estimate_architecture(apo, fasc, 0.1, 0.1)
    explicit = estimate_architecture(apo, fasc, 0.1, 0.1, fl_method="intersection")

    # The default leans on thickness, which the pipeline recovers far more
    # reliably than any angle.
    assert default.fl_mm == pytest.approx(default.fl_parallel_mm)
    assert explicit.fl_mm == pytest.approx(explicit.fl_intersection_mm)
    assert default.fl_mm != pytest.approx(explicit.fl_mm)
    # Both values travel with every estimate regardless of the choice.
    for est in (default, explicit):
        assert np.isfinite(est.fl_parallel_mm) and np.isfinite(est.fl_intersection_mm)


def test_orientation_aggregation_is_modulo_180_degrees():
    """Averaging +85 and -85 degrees must give ~90, not ~0."""
    cos2t = np.array([[np.cos(np.radians(170.0)), np.cos(np.radians(-170.0))]])
    sin2t = np.array([[np.sin(np.radians(170.0)), np.sin(np.radians(-170.0))]])
    angle, resultant = orientation_from_field(cos2t, sin2t, np.ones((1, 2)), 1.0, 1.0)
    assert abs(angle) == pytest.approx(90.0, abs=1.0)
    assert resultant > 0.9


def test_empty_mask_fails_loudly_rather_than_silently():
    est = estimate_architecture(np.zeros((200, 300), np.uint8), np.zeros((200, 300), np.uint8), 0.1, 0.1)
    assert est.failure == "aponeurosis_not_found"
    assert est.confidence == 0.0


def test_orientation_target_matches_independent_pca_reference():
    """The orientation supervision is derived, not annotated, so it needs its
    own check: a structure tensor and a per-segment principal-axis fit are
    independent estimators of the same quantity and must agree."""
    from scipy import ndimage

    from umud.data import orientation_target

    for angle in (-30.0, -15.0, 0.0, 25.0):
        mask = np.zeros((256, 256), np.uint8)
        for x0 in range(20, 240, 30):
            for t in range(60):
                x = int(x0 + t * np.cos(np.radians(angle)))
                y = int(128 + t * np.sin(np.radians(angle)))
                if 0 <= x < 256 and 0 <= y < 256:
                    mask[y, x] = 1
        mask = ndimage.binary_dilation(mask, np.ones((2, 2))).astype(np.uint8)

        field = orientation_target(mask)
        estimated, resultant = orientation_from_field(
            field[0], field[1], mask.astype(float), 1.0, 1.0
        )
        error = abs(estimated - angle)
        error = min(error, 180.0 - error)
        assert error < 4.0, f"angle {angle}: got {estimated}"
        assert resultant > 0.7


def test_a_band_is_not_paired_with_a_fragment_of_itself():
    """Predicted masks fragment. Two pieces of the *same* aponeurosis, a few
    pixels apart, must not be mistaken for the superficial/deep pair — that
    would report a near-zero thickness with high confidence."""
    height, width = 400, 600
    mask = np.zeros((height, width), np.uint8)
    mask[100:106, 40:560] = 1          # one real band (centroid y = 102.5)
    mask[107:113, 60:540] = 1          # a detached sliver 7 px below it
    sup, deep, conf, n = extract_aponeuroses(mask, 1.0, 1.0)
    assert sup is None and deep is None, "paired a band with its own fragment"
    assert conf == 0.0
    assert n >= 2                       # both were seen, both were rejected as a pair


def test_a_genuine_pair_is_still_found_on_the_pixel_grid():
    mask = np.zeros((400, 600), np.uint8)
    mask[100:106, 40:560] = 1
    mask[300:306, 40:560] = 1
    sup, deep, conf, _ = extract_aponeuroses(mask, 1.0, 1.0)
    assert sup is not None and deep is not None
    assert muscle_thickness_mm(sup, deep) == pytest.approx(200.0, abs=3.0)
    assert conf > 0.5


def test_pixel_grid_evaluation_must_not_clamp():
    """The physiological bounds are millimetres. Applied to a pixel-grid
    measurement they pin prediction and reference to the same ceiling and report
    an error of exactly zero — which is what the first evaluation run did."""
    from umud.config import PRIORS, UNCLAMPED

    thickness_px, length_px = 174.0, 600.0
    _, clamped_fl, clamped_mt = PRIORS.clamp(20.0, length_px, thickness_px)
    assert clamped_fl == 140.0 and clamped_mt == 60.0, "the trap this test guards"

    _, free_fl, free_mt = UNCLAMPED.clamp(20.0, length_px, thickness_px)
    assert free_fl == length_px and free_mt == thickness_px


def test_confidence_does_not_peek_at_the_clamp():
    """Confidence must come only from evidence available before the estimate was
    produced. Docking it when a value hits a clamp makes it look predictive of
    failures it was simply told about, and any correlation with error is then
    circular rather than earned."""
    from umud.config import PhysiologicalPriors

    apo, fasc = phantom(thickness_px=200, angle_deg=6.0)

    generous = PhysiologicalPriors(pa_deg=(0.1, 89.9), fl_mm=(1.0, 1e6), mt_mm=(1.0, 1e6))
    tight = PhysiologicalPriors(pa_deg=(3.0, 40.0), fl_mm=(20.0, 60.0), mt_mm=(5.0, 60.0))

    free = estimate_architecture(apo, fasc, 0.1, 0.1, priors=generous)
    pinched = estimate_architecture(apo, fasc, 0.1, 0.1, priors=tight)

    assert pinched.fl_mm < free.fl_mm, "the tight prior must actually bite here"
    assert pinched.confidence == pytest.approx(free.confidence, abs=1e-9)


def test_both_fascicle_length_values_are_always_recorded():
    """The ablation must score the two constructions explicitly. Reading it off
    fl_mm makes it compare the current default against itself the moment the
    default changes — which is what happened when the default was switched."""
    apo, fasc = phantom(thickness_px=200, angle_deg=12.0, deep_slope=0.25)
    for method in ("parallel", "intersection"):
        est = estimate_architecture(apo, fasc, 0.1, 0.1, fl_method=method)
        assert np.isfinite(est.fl_parallel_mm)
        assert np.isfinite(est.fl_intersection_mm)
        assert est.fl_parallel_mm != pytest.approx(est.fl_intersection_mm)
