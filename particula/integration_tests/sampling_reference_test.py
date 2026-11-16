"""Independent sampling uncertainty for represented-particle removal."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.dynamics.wall_loss.wall_loss_strategies import (
    SphericalWallLossStrategy,
)
from particula.particles import create_resolved_particle_data


@pytest.mark.parametrize("slots", [128, 512])
@pytest.mark.parametrize("samples", [16, 64])
def test_survival_ensemble_separates_slots_and_sample_count(
    slots, samples, monkeypatch
):
    """Hold physical population fixed while independently varying both counts.

    A constant hazard 0.7 s^-1 gives Bernoulli survival p=exp(-0.7).
    Equal multiplicity weights yield Var(fraction)=p*(1-p)/slots;
    the ensemble-mean variance is further divided by samples. This qualifies
    the sampling operator, while separate wall-loss tests qualify coefficients.
    """
    strategy = SphericalWallLossStrategy(1e-3, 0.5, "particle_resolved")
    monkeypatch.setattr(
        strategy,
        "loss_coefficient_for_particles",
        lambda radius, *args: np.full(radius.shape, 0.7),
    )
    fractions = []
    material_mass = np.array([1e-18, 3e-18])
    for seed in range(samples):
        particle = create_resolved_particle_data(
            masses=np.tile(material_mass, (slots, 1)),
            density=[1000.0, 2000.0],
            concentration=1e6 / slots,
            volume=0.25,
        )
        strategy.random_generator = np.random.default_rng(81700 + seed)
        strategy.step(particle, 298.15, 101325.0, 1.0)
        survivors = particle.concentration[0].sum()
        fractions.append(survivors / 1e6)
        remaining = np.sum(
            particle.masses[0] * particle.concentration[0, :, None], axis=0
        )
        removed = (1e6 - survivors) * material_mass
        npt.assert_allclose(
            remaining + removed, 1e6 * material_mass, rtol=2e-14, atol=0
        )
    probability = np.exp(-0.7)
    variance = probability * (1 - probability) / slots
    assert abs(np.mean(fractions) - probability) < 5 * np.sqrt(
        variance / samples
    )
    # Broad predeclared finite-ensemble interval, separate from mass closure.
    assert 0.3 * variance < np.var(fractions, ddof=1) < 2.5 * variance
