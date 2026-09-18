"""Calibration and sequence-grouping tests that do not need the competition data."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from umud.calibration import Calibration, _spacing_from_ticks, _tick_positions, to_gray
from umud.sequence import weighted_median


def test_to_gray_collapses_colour_before_slicing():
    """A 1-D profile taken from a colour frame must be 1-D."""
    colour = np.zeros((10, 20, 3), np.uint8)
    assert to_gray(colour).shape == (10, 20)
    assert to_gray(colour)[:, 3].shape == (10,)


def test_tick_detection_finds_evenly_spaced_marks():
    profile = np.zeros(400)
    for centre in range(20, 380, 37):
        profile[centre:centre + 3] = 200
    ticks = _tick_positions(profile)
    assert ticks.size == 10
    assert _spacing_from_ticks(ticks) == pytest.approx(37.0, abs=0.5)


def test_tick_detection_survives_a_missing_tick():
    profile = np.zeros(400)
    for i, centre in enumerate(range(20, 380, 37)):
        if i == 4:
            continue
        profile[centre:centre + 3] = 200
    assert _spacing_from_ticks(_tick_positions(profile)) == pytest.approx(37.0, abs=1.0)


def test_calibration_reports_pixel_squareness():
    square = Calibration(150.0, 150.0, (0, 0, 100, 100), "x", True)
    oblong = Calibration(150.0, 120.0, (0, 0, 100, 100), "x", True)
    assert square.square_pixels and not oblong.square_pixels
    assert square.mm_per_px_y == pytest.approx(10.0 / 150.0)


def test_weighted_median_ignores_a_low_confidence_outlier():
    values = [20.0, 21.0, 20.5, 95.0]      # the last frame's segmentation collapsed
    weights = [0.9, 0.9, 0.9, 0.01]
    assert weighted_median(values, weights) == pytest.approx(20.5, abs=0.6)


def test_weighted_median_handles_nans():
    assert np.isfinite(weighted_median([1.0, np.nan, 3.0], [1.0, 1.0, 1.0]))


def test_generic_detector_refuses_irregular_speckle():
    """A wrong scale is worse than no scale: it silently rescales every
    millimetre the pipeline reports.  Random bright specks must not pass."""
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, size=400).astype(float)
    assert _spacing_from_ticks(_tick_positions(noisy)) is None


def test_generic_detector_refuses_a_strip_running_through_tissue():
    from umud.calibration import _looks_like_chrome

    rng = np.random.default_rng(1)
    tissue = np.clip(rng.normal(90, 35, size=400), 0, 255)
    assert not _looks_like_chrome(tissue)

    chrome = np.zeros(400)
    for centre in range(20, 380, 37):
        chrome[centre:centre + 3] = 200
    assert _looks_like_chrome(chrome)


def test_generic_detector_accepts_a_real_ruler_on_dark_chrome():
    from umud.calibration import _generic_calibration

    frame = np.zeros((400, 300))
    frame[:, 30:270] = 90                       # tissue in the middle
    for centre in range(20, 380, 30):           # ruler in the left margin
        frame[centre:centre + 3, 4:10] = 220
    result = _generic_calibration(frame)
    assert result is not None
    px_per_cm, note = result
    assert px_per_cm == pytest.approx(60.0, abs=1.0)   # 30 px per 5 mm tick
    assert "left" in note
