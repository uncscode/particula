"""Check the chamber inversion and time-zero number normalization on CPU."""

import json
from pathlib import Path

import numpy as np
from scripts.pi_chamber.retrieval import fit_humidity

ROOT = Path(__file__).resolve().parents[1]


def test_haze_fit_reproduces_chamber_adjoint_estimates():
    """Compare independent analytic and original CUDA interval fits."""
    fit = fit_humidity()
    np.testing.assert_allclose(
        [fit["FW"], fit["SW"], fit["shared"]],
        [-0.3466808813, -0.95, -0.4495549582],
        atol=1e-7,
    )


def test_starts_share_interior_number_and_only_wet_start_has_cloud():
    """Recover Table 1 inputs from the saved instantaneous spectra."""
    result = json.loads(
        (ROOT / "website/results/pi-chamber-spectra/data.json").read_text()
    )
    for run in [result["main"], result["refined"]]:
        bins = np.array(run["bin_cm3"])[0]
        borders = np.array(run["edges_um"])
        cloud = (borders[:-1] >= 1.8) & (borders[1:] <= 40)
        np.testing.assert_allclose(bins.sum(axis=1), [1844, 1844, 2365, 2365])
        np.testing.assert_allclose(
            bins[:, cloud].sum(axis=1), [0, 771, 0, 1038], atol=2
        )
