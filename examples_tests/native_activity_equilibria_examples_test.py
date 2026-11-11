"""Scientific checks for the functional activity/equilibrium example family."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "docs/Examples"


@pytest.mark.parametrize(
    "relative_path",
    [
        "Activity/activity_tutorial.py",
        "Activity/bat_activity_example.py",
        "Activity/equilibria_example.py",
        "Activity/ideal_activity_example.py",
        "Activity/kappa_activity_example.py",
        "Equilibria/Notebooks/activity_part1.py",
        "Equilibria/Notebooks/equilibria_part1.py",
    ],
)
def test_functional_activity_and_equilibrium_examples(
    monkeypatch, relative_path
):
    """Run the published examples and check analytical or bounded outcomes."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    values = runpy.run_path(str(ROOT / relative_path))
    if relative_path.endswith("ideal_activity_example.py"):
        mass, molar = values["mass"], values["strategy"].molar_mass
        expected = (mass / molar) / np.sum(mass / molar)
        np.testing.assert_allclose(values["activity"], expected, rtol=1e-12)
    elif relative_path.endswith("kappa_activity_example.py"):
        # Independent kappa water-volume relation, including kappa=0.
        water_volume, dry_volume = 0.5 / 1000.0, 0.5 / 1500.0
        for kappa in values["kappa_test_values"]:
            strategy = values["par"].particles.ActivityKappaParameter(
                kappa=np.array([0.0, kappa]),
                density=np.array([1000.0, 1500.0]),
                molar_mass=np.array([0.018015, 0.1]),
                water_index=0,
            )
            actual = strategy.activity(np.array([0.5, 0.5]))[0]
            np.testing.assert_allclose(
                actual,
                water_volume / (water_volume + kappa * dry_volume),
                rtol=1e-12,
            )
    elif relative_path.endswith("bat_activity_example.py"):
        assert np.isfinite(values["bat_organic_activity"])
        assert values["bat_organic_activity"] > 0
    elif "result" in values:
        result = values["result"]
        assert np.all(np.isfinite(result.partition_coefficients))
        assert np.all(
            (result.partition_coefficients >= 0)
            & (result.partition_coefficients <= 1)
        )
        assert np.isfinite(result.error) and result.error < 1e-8
        phases = [
            p for p in (result.alpha_phase, result.beta_phase) if p is not None
        ]
        np.testing.assert_allclose(
            sum(p.total_concentration - p.water_concentration for p in phases),
            np.sum(
                result.partition_coefficients * values["concentration_organic"]
            ),
            rtol=1e-12,
            atol=1e-12,
        )
    elif "total_oa_concentration" in values:
        organic = values["total_oa_concentration"]
        assert np.all(np.isfinite(organic)) and np.all(organic >= 0)
        assert np.all(organic <= values["concentration_organic_matter"].sum())
        assert np.all(np.isfinite(values["water_concentration"]))
    else:
        for name in ("alpha", "q_alpha", "activity_water", "activity_organic"):
            assert np.all(np.isfinite(values[name]))
        assert np.all((values["q_alpha"] >= 0) & (values["q_alpha"] <= 1))
    plt.close("all")
