"""Independent native finite-step species and signed-charge transport checks."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.dynamics import BrownianCoagulationStrategy
from particula.dynamics.coagulation.coagulation_strategy.tests.native_coagulation_step_test import (
    _state,
    _unchanged,
)
from particula.dynamics.coagulation.coagulation_strategy.tests.native_rates_test import (
    MECHANISMS,
    _particle,
    _strategy,
)
from particula.particles import ParticleData
from scipy.integrate import quad

KINDS = ["discrete", "continuous_pdf"]


def _widths(radius):
    return np.diff(radius, append=2 * radius[-1] - radius[-2])


def _integral(values, radius, kind):
    """Integrate independently of the production population helpers."""
    if kind == "discrete":
        return values.sum(axis=0)
    width = np.diff(radius)
    if values.ndim == 2:
        width = width[:, None]
    return np.sum((values[1:] + values[:-1]) * width / 2, axis=0)


@pytest.mark.parametrize("kind", KINDS)
def test_constant_kernel_step_matches_independent_collision_integrals(
    kind, monkeypatch
):
    """Integrate analytic donor moments, without production splines or gains.

    A constant radius PDF and kernel make the pair law analytic. Species
    composition varies with radius and charge changes sign. Match the retained
    spline's constant continuation below the smallest tabulated donor radius;
    this finite-boundary fixture checks local transport, not global closure.
    """
    scale, coefficient, pdf, duration = 1e-7, 1e-15, 1e15, 1e5
    x = np.linspace(0.05, 3.0, 257)
    radius = scale * x
    mass_scale = 4 * np.pi * 1000 * scale**3 / 3

    def moments(z):
        return np.array(
            [
                z**3 * (0.4 + 0.1 * z),
                z**3 * (0.6 - 0.1 * z),
                z**3 * (1 - 0.5 * z),
            ]
        )

    original = moments(x).T
    original[:, :2] *= mass_scale
    width = _widths(radius) if kind == "discrete" else np.ones(x.size)
    counts = pdf * width
    particle = ParticleData(
        masses=original[None, :, :2].copy(),
        concentration=counts[None].copy(),
        charge=original[None, :, 2].copy(),
        density=np.array([1000.0, 1000.0]),
        volume=np.ones(1),
        distribution_type=kind,
        radius_grid=radius.copy() if kind == "continuous_pdf" else None,
    )
    before = _state(particle)
    strategy = BrownianCoagulationStrategy(kind)
    monkeypatch.setattr(
        strategy,
        "kernel",
        lambda *args, **kwargs: np.full((x.size, x.size), coefficient),
    )
    strategy.step(particle, 298.15, 101325.0, duration)
    population = pdf * (radius[-1] - radius[0])
    if kind == "discrete":
        population = counts.sum()
    loss = coefficient * counts * population
    actual = np.column_stack([particle.masses[0], particle.charge[0]])
    recovered_gain = (
        particle.concentration[0, :, None] * actual - counts[:, None] * original
    ) / duration + loss[:, None] * original

    # Evaluate selected destinations on either side of the charge sign change.
    for index in (12, 40, 80, 128, 192, 256):
        destination = x[index]
        limit = 2 ** (-1 / 3)
        for lane in range(3):

            def integrand(t, destination=destination, lane=lane):
                other = (1 - t**3) ** (1 / 3)
                donors = moments(max(x[0], destination * t))
                donors += moments(max(x[0], destination * other))
                return donors[lane] / other**2

            integral = quad(
                integrand,
                0,
                limit,
                epsabs=1e-12,
                points=[x[0] / destination]
                if x[0] / destination < limit
                else None,
            )[0]
            expected = coefficient * pdf**2 * radius[index] * integral
            expected *= width[index] * (mass_scale if lane < 2 else 1)
            # Existing trapezoid integration has O(n^-2) error; the signed
            # result stays well away from zero at these selected nodes.
            npt.assert_allclose(
                recovered_gain[index, lane], expected, rtol=8e-5, atol=0
            )
    for name, (identity, values) in before.items():
        assert getattr(particle, name) is identity
        if name in ("density", "volume"):
            npt.assert_array_equal(identity, values)
    if kind == "continuous_pdf":
        npt.assert_array_equal(particle.radius_grid, radius)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("second_density", [1000.0, 1600.0])
def test_varying_composition_finite_step_inventory_converges(
    kind, second_density, monkeypatch
):
    """Per-species and signed-charge defects decrease relative to gross loss."""
    errors = []
    for size in (128, 256, 512):
        radius = np.geomspace(1e-9, 1e-6, size)
        fraction = np.clip(0.5 + 0.3 * np.log(radius / 1e-7), 0.05, 0.95)
        mass = 4 * np.pi * radius**3 / 3
        mass /= fraction / 1000 + (1 - fraction) / second_density
        pdf = 1e10 * np.exp(-0.5 * (np.log(radius / 1e-7) / 0.35) ** 2)
        pdf /= radius * 0.35 * np.sqrt(2 * np.pi)
        counts = pdf if kind == "continuous_pdf" else pdf * _widths(radius)
        # Charge uses the difference of two additive species moments, with
        # both signs represented, rather than a constant per-bin charge.
        moments = np.column_stack(
            [
                mass * fraction,
                mass * (1 - fraction),
                mass * (2 * fraction - 1) / 1e-18,
            ]
        )
        particle = ParticleData(
            masses=moments[None, :, :2].copy(),
            concentration=counts[None].copy(),
            charge=moments[None, :, 2].copy(),
            density=np.array([1000.0, second_density]),
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
        before = _integral(counts[:, None] * moments, radius, kind)
        population = _integral(counts, radius, kind)
        gross = (
            1e-15
            * population
            * _integral(counts[:, None] * np.abs(moments), radius, kind)
        )
        duration = 100.0
        strategy.step(particle, 298.15, 101325.0, duration)
        after_moments = np.column_stack(
            [particle.masses[0], particle.charge[0]]
        )
        after = _integral(
            particle.concentration[0, :, None] * after_moments, radius, kind
        )
        errors.append(np.abs((after - before) / (duration * gross)))
    errors = np.array(errors)
    reduction = 0.55 if kind == "discrete" else 0.3
    assert np.all(errors[1:] < reduction * errors[:-1]), errors
    assert np.all(errors[-1] < (0.007 if kind == "discrete" else 3e-5)), errors


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", MECHANISMS)
def test_all_mechanisms_retain_number_law_and_array_ownership(kind, name):
    """Moment transport retains each supported mechanism's number update."""
    particle = _particle(kind)
    strategy = _strategy(name, kind)
    state = _state(particle)
    expected = (
        particle.concentration
        + strategy.net_rate(particle, 298.15, 101325.0)[None] * 1e-3
    )
    assert strategy.step(particle, 298.15, 101325.0, 1e-3) is particle
    npt.assert_array_equal(particle.concentration, expected)
    for field, (identity, values) in state.items():
        assert getattr(particle, field) is identity
        if field in ("density", "volume"):
            npt.assert_array_equal(identity, values)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("field", ["masses", "charge"])
def test_new_moment_destinations_must_be_writable(kind, field):
    """Reject before changing concentration when a moment cannot be written."""
    particle = _particle(kind)
    getattr(particle, field).flags.writeable = False
    before = _state(particle)
    with pytest.raises(ValueError, match="writable"):
        _strategy("brownian", kind).step(particle, 298.15, 101325.0, 1)
    _unchanged(particle, before)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("noop", ["time", "kernel", "population"])
def test_noop_preserves_moments_exactly(kind, noop, monkeypatch):
    """Zero work must not round means through an inventory division."""
    particle = _particle(kind)
    if noop == "population":
        particle.concentration[:] = 0
    strategy = _strategy("brownian", kind)
    if noop == "kernel":
        monkeypatch.setattr(
            strategy, "kernel", lambda *args, **kwargs: np.zeros((4, 4))
        )
    before = _state(particle)
    strategy.step(particle, 298.15, 101325.0, 0 if noop == "time" else 1)
    _unchanged(particle, before)


@pytest.mark.parametrize("kind", KINDS)
def test_absent_material_and_zero_charge_stay_zero(kind):
    """A collision cannot invent a species or charge absent from all donors."""
    particle = _particle(kind)
    particle.masses[0, :, 1] = 0
    particle.charge[:] = 0
    _strategy("brownian", kind).step(particle, 298.15, 101325.0, 1)
    npt.assert_array_equal(particle.masses[0, :, 1], 0)
    npt.assert_array_equal(particle.charge, 0)


def test_newly_occupied_pdf_node_receives_both_donors(monkeypatch):
    """Birth into an empty node carries the sum of donor mass and charge."""
    size = 64
    coordinate = np.linspace(1.0, 4.0, size)
    radius = coordinate * 1e-7
    pdf = 1e15 * (coordinate - 2) ** 2
    pdf[21] = 0
    particle = ParticleData(
        masses=np.tile([1e-18, 3e-18], (1, size, 1)),
        concentration=pdf[None],
        charge=np.full((1, size), -2.0),
        density=np.array([1000.0, 1600.0]),
        volume=np.ones(1),
        distribution_type="continuous_pdf",
        radius_grid=radius,
    )
    strategy = BrownianCoagulationStrategy("continuous_pdf")
    monkeypatch.setattr(
        strategy, "kernel", lambda *args, **kwargs: np.full((size, size), 1e-15)
    )
    strategy.step(particle, 298.15, 101325.0, 1000)
    assert particle.concentration[0, 21] > 0
    npt.assert_allclose(
        particle.masses[0, 21], [2e-18, 6e-18], rtol=3e-15, atol=0
    )
    npt.assert_allclose(particle.charge[0, 21], -4.0, rtol=3e-15, atol=0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("failure", ["negative", "overflow", "orphan"])
def test_late_moment_rejection_preserves_every_caller_array(
    kind, failure, monkeypatch
):
    """A later species failure must not publish earlier staged moments."""
    from particula.dynamics.coagulation.coagulation_strategy import _native_step

    particle = _particle(kind)
    strategy = _strategy("brownian", kind)
    before = _state(particle)
    original = _native_step._density_moment_rate
    calls = []

    def moment_rate(particle, kernel, loss, moment):
        calls.append(moment)
        if len(calls) == 2:
            if failure == "overflow":
                return np.full_like(moment, np.finfo(np.float64).max)
            if failure == "negative":
                return -2 * particle.concentration[0] * moment
            return np.ones_like(moment)
        return original(particle, kernel, loss, moment)

    monkeypatch.setattr(_native_step, "_density_moment_rate", moment_rate)
    if failure == "orphan":
        monkeypatch.setattr(
            strategy,
            "net_rate",
            lambda *args: -particle.concentration[0] / 2,
        )

        # Preserve zero inventory at the first material so the second lane
        # establishes the orphan error after concentration reaches zero.
        def orphan_rate(particle, kernel, loss, moment):
            calls.append(moment)
            return -particle.concentration[0] * moment / 2 + (
                np.ones_like(moment) if len(calls) == 2 else 0
            )

        monkeypatch.setattr(_native_step, "_density_moment_rate", orphan_rate)
    with pytest.raises(ValueError, match="moment"):
        strategy.step(particle, 298.15, 101325.0, 2)
    assert len(calls) == 2
    _unchanged(particle, before)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("field", ["masses", "charge"])
def test_moment_writers_cannot_alias_protected_storage(kind, field):
    """The new species and charge writers must preserve density and volume."""
    particle = _particle(kind)
    if field == "masses":
        particle.volume = particle.masses.reshape(-1)[:1]
        particle.volume[:] = 1
    else:
        particle.density = particle.charge[0, :2]
        particle.density[:] = [1000, 1600]
    before = _state(particle)
    with pytest.raises(ValueError, match="alias"):
        _strategy("brownian", kind).step(particle, 298.15, 101325.0, 1)
    _unchanged(particle, before)
