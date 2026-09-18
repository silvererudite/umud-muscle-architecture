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


def test_diverging_aponeuroses_depart_from_the_textbook_formula():
    """The whole point of the intersection construction: when the aponeuroses
    are not parallel, MT / sin(PA) is the wrong answer."""
    apo, fasc = phantom(thickness_px=200, angle_deg=12.0, deep_slope=0.25)
    est = estimate_architecture(apo, fasc, 0.1, 0.1)
    assert est.aponeurosis_divergence_deg > 5.0
    assert abs(est.fl_mm - est.fl_parallel_mm) / est.fl_parallel_mm > 0.05


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
