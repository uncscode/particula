"""Verify the sectional log plot without altering the signed BDF output."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scripts.mam4xx_coagulation.regime_plots import sectional_curve, spectra


def test_unresolved_solver_tail_in_log_plot():
    """Render positive bins, preserve source values and reject large errors."""
    bins = np.array([-2e-15, 0.0, 3.0])
    original = bins.copy()
    plotted = sectional_curve(bins, np.array([1.0, 2.0, 4.0, 8.0]))
    np.testing.assert_allclose(plotted, [np.nan, np.nan, 3 / np.log10(2)])
    np.testing.assert_array_equal(bins, original)
    with pytest.raises(ValueError, match="exceeds BDF tolerance"):
        sectional_curve([-1e-8], np.array([1.0, 2.0]))


def test_render_all_three_regime_spectra():
    """Exercise every panel with the retained atmospheric comparison."""
    root = Path(__file__).resolve().parents[2]
    report = json.loads(
        (root / "website/results/regimes/data.json").read_text()
    )
    try:
        spectra(report)
        plt.gcf().canvas.draw()
        assert len(plt.gcf().axes) == 12
    finally:
        plt.close(plt.gcf())
