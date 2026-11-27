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


def test_saved_time_zero_bins_match_restart_cohorts():
    """Rebuild every time-zero bin by explicit masks on the study's µm grid."""
    from scripts.pi_chamber.temporal import FLOOR, _population

    saved = json.loads(
        (ROOT / "website/results/pi-chamber-spectra/data.json").read_text()
    )
    restart = json.loads(
        (ROOT / "docs/data/pi-chamber/restart.json").read_text()
    )
    for run in (saved["main"], saved["refined"]):
        sizes, weights, zones, _, n0 = _population(
            np.random.default_rng(run["seed"]), run["dt"], restart
        )
        diam = sizes[..., :n0]
        weights = weights[..., :n0] / (1 - FLOOR)
        inside = zones[:, :n0] == 0
        edges = run["edges_um"]
        for i, (low, high) in enumerate(
            zip(edges[:-1], edges[1:], strict=True)
        ):
            upper = diam <= high if high == edges[-1] else diam < high
            selected = inside & (low <= diam) & upper
            density = np.where(selected, weights, 0).sum(axis=-1).mean(axis=-1)
            np.testing.assert_allclose(
                np.array(run["bin_cm3"])[0, :, i],
                density,
                rtol=1e-12,
                atol=1e-9,
            )
