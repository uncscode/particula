"""Preserved fixed-composition stepping and its species-closure limitation."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.dynamics import BrownianCoagulationStrategy
from particula.dynamics.coagulation.coagulation_strategy.tests.native_rates_test import (
    MECHANISMS,
    _particle,
    _strategy,
)
from particula.particles import ParticleData


def _integral(values, radius, kind):
    """Use raw bin sums or explicit radius trapezoids, not container helpers."""
    if kind == "discrete":
        return values.sum(axis=0)
    width = np.diff(radius)
    if values.ndim == 2:
        width = width[:, None]
    return np.sum((values[1:] + values[:-1]) * width / 2, axis=0)


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize("name", MECHANISMS)
def test_all_mechanisms_preserve_fixed_material_and_number_update(kind, name):
    """The accepted method writes number density, leaving material fixed."""
    particle = _particle(kind)
    fields = ("masses", "concentration", "charge", "density", "volume")
    before = {
        key: (getattr(particle, key), getattr(particle, key).copy())
        for key in fields
    }
    strategy = _strategy(name, kind)
    expected = (
        particle.concentration
        + strategy.net_rate(particle, 298.15, 101325.0)[None] * 1e-3
    )
    assert strategy.step(particle, 298.15, 101325.0, 1e-3) is particle
    npt.assert_array_equal(particle.concentration, expected)
    for key, (identity, values) in before.items():
        assert getattr(particle, key) is identity
        if key != "concentration":
            npt.assert_array_equal(identity, values)


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize("field", ["masses", "charge"])
def test_fixed_material_can_be_read_only(kind, field):
    """Fixed material is protected input, not a density-step destination."""
    particle = _particle(kind)
    getattr(particle, field).flags.writeable = False
    before = getattr(particle, field).copy()
    _strategy("brownian", kind).step(particle, 298.15, 101325.0, 1e-3)
    npt.assert_array_equal(getattr(particle, field), before)


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_size_dependent_composition_defect_persists_under_refinement(
    kind, monkeypatch
):
    """Total-mass convergence does not establish individual-species closure.

    Ratios are signed inventory changes divided by duration times gross loss;
    small timesteps cannot hide the retained method's composition defect.
    Bounds characterize this fixture, not a general permitted error tolerance.
    """
    errors, total_errors = [], []
    for size in (128, 256, 512, 1024):
        radius = np.geomspace(1e-9, 1e-6, size)
        fraction = np.clip(0.5 + 0.3 * np.log(radius / 1e-7), 0.05, 0.95)
        total_mass = 4 * np.pi * 1000 * radius**3 / 3
        masses = np.column_stack(
            [total_mass * fraction, total_mass * (1 - fraction)]
        )
        pdf = 1e10 * np.exp(-0.5 * (np.log(radius / 1e-7) / 0.35) ** 2)
        pdf /= radius * 0.35 * np.sqrt(2 * np.pi)
        counts = (
            pdf
            if kind == "continuous_pdf"
            else pdf * np.diff(radius, append=2 * radius[-1] - radius[-2])
        )
        particle = ParticleData(
            masses=masses[None].copy(),
            concentration=counts[None].copy(),
            charge=np.zeros((1, size)),
            density=np.array([1000.0, 1000.0]),
            volume=np.ones(1),
            distribution_type=kind,
            radius_grid=radius.copy() if kind == "continuous_pdf" else None,
        )
        strategy = BrownianCoagulationStrategy(kind)
        monkeypatch.setattr(
            strategy,
            "kernel",
            lambda *args, size=size, **kwargs: np.full((size, size), 1e-15),
        )
        before = _integral(counts[:, None] * masses, radius, kind)
        gross_loss = 1e-15 * _integral(counts, radius, kind) * before
        strategy.step(particle, 298.15, 101325.0, 100.0)
        after = _integral(
            particle.concentration[0, :, None] * particle.masses[0],
            radius,
            kind,
        )
        errors.append((after - before) / (100 * gross_loss))
        total_errors.append(
            abs((after.sum() - before.sum()) / (100 * gross_loss.sum()))
        )
        npt.assert_array_equal(particle.masses[0], masses)
    errors, total_errors = np.array(errors), np.array(total_errors)
    assert np.all(total_errors[1:] < 0.6 * total_errors[:-1])
    assert total_errors[-1] < (0.004 if kind == "discrete" else 1e-5)
    if kind == "continuous_pdf":
        assert 0.077 < errors[-1, 0] < 0.079
        assert -0.123 < errors[-1, 1] < -0.121
    else:
        assert 0.073 < errors[-1, 0] < 0.076
        assert -0.127 < errors[-1, 1] < -0.123
