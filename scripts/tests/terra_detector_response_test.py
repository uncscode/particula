"""Regression for missing samples at the edges of Terra MODIS detectors."""

from pathlib import Path
from runpy import run_path

import numpy as np


def test_fill_is_removed_before_interpolation_and_normalization():
    """A missing edge sample cannot create a false peak in a band average."""
    source = (
        Path(__file__).resolve().parents[2]
        / "studies/spectral-radiation"
        / "observations/responses.py"
    )
    detector_curve = run_path(str(source))["detector_curve"]
    samples = np.array(
        [
            [1, 1, 615, -99],
            [1, 1, 617, 0.25],
            [1, 1, 620, 1.0],
            [1, 1, 623, 0.25],
            [1, 1, 625, -99],
        ]
    )
    wave = np.arange(614, 627)
    expected = np.zeros(wave.shape)
    expected[3:10] = [0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25]
    np.testing.assert_allclose(
        detector_curve(samples, wave), expected / np.trapezoid(expected, wave)
    )
