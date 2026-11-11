"""Native coagulation stepping, weighted inventories and mutation boundaries."""

from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest
from particula.dynamics.coagulation.coagulation_strategy import _native_step
from particula.dynamics.coagulation.coagulation_strategy.coagulation_strategy_abc import (
    _get_binned_kernel_data_from_particle_data,
    _get_particle_resolved_kernel_radius_data,
)
from particula.dynamics.coagulation.coagulation_strategy.tests.native_rates_test import (
    CHARGED,
    MECHANISMS,
    _particle,
    _strategy,
)
from particula.dynamics.coagulation.particle_resolved_step import (
    particle_resolved_method,
)
from particula.particles import ParticleData


def _state(particle):
    return {
        name: (getattr(particle, name), getattr(particle, name).copy())
        for name in ("masses", "concentration", "charge", "density", "volume")
    }


def _unchanged(particle, state):
    for name, (original, values) in state.items():
        assert getattr(particle, name) is original
        npt.assert_array_equal(original, values)


def _inventory(particle):
    # Deliberately independent of ParticleData's inventory/density helpers.
    count = particle.concentration[0]
    species = np.sum(particle.masses[0] * count[:, None], axis=0)
    charge = np.dot(particle.charge[0], count)
    return species, charge


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("weight", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("name", MECHANISMS)
def test_resolved_merge_weights_volume_charge_and_identity(
    monkeypatch, volume, weight, name
):
    """Resolved merge weights volume charge and identity."""
    particle = _particle("particle_resolved", volume)
    particle.concentration[:] = weight
    particle.concentration[0, 1] = 0  # Positive material but inactive slot.
    particle.masses[0, 0, 1] = 0  # Valid zero-material composition.
    state = _state(particle)
    mass_before, charge_before = _inventory(particle)
    strategy = _strategy(name, "particle_resolved")
    rng_before = deepcopy(strategy.random_generator.bit_generator.state)

    def select(**kwargs):
        assert kwargs["volume"] == volume
        assert kwargs["time_step"] == 3.0 * weight
        assert kwargs["random_generator"] is strategy.random_generator
        npt.assert_allclose(
            kwargs["particle_radius"], particle.radii[0, [0, 2, 3]]
        )
        return np.array([[0, 2], [1, 2]], dtype=np.int64)

    monkeypatch.setattr(
        particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        select,
    )
    assert strategy.step(particle, 298.15, 101325.0, 3.0) is particle
    mass_after, charge_after = _inventory(particle)
    npt.assert_allclose(mass_after, mass_before, rtol=2e-15, atol=0)
    assert charge_after == charge_before
    npt.assert_array_equal(particle.concentration, [[0, 0, 0, weight]])
    npt.assert_allclose(
        particle.masses[0, 3],
        np.sum(state["masses"][1][0, [0, 2, 3]], axis=0),
        rtol=2e-15,
        atol=0,
    )
    npt.assert_array_equal(particle.masses[0, 1], state["masses"][1][0, 1])
    for key, (original, values) in state.items():
        assert getattr(particle, key) is original
        if key in ("density", "volume"):
            npt.assert_array_equal(original, values)
    assert strategy.random_generator.bit_generator.state == rng_before


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_binning_preserves_species_counts_and_zero_material(volume):
    """Binning preserves species counts and zero material."""
    particle = _particle("particle_resolved", volume)
    particle.masses[0, :, 1] = 0
    particle.concentration[0, 1] = 0
    before = _state(particle)
    edges = np.array([1e-10, 1e-5])
    result = _get_binned_kernel_data_from_particle_data(particle, edges)
    assert result.distribution_type == "particle_resolved"
    assert result.masses.shape == (1, 1, 2)
    npt.assert_array_equal(result.volume, [volume])
    npt.assert_allclose(
        _inventory(result)[0], _inventory(particle)[0], rtol=2e-15
    )
    assert result.concentration.sum() == particle.concentration.sum()
    assert result.charge[0, 0] == np.median(particle.charge[0, [0, 2, 3]])
    npt.assert_array_equal(result.masses[0, :, 1], 0)
    _unchanged(particle, before)
    for name in before:
        assert not np.shares_memory(
            getattr(result, name), getattr(particle, name)
        )


@pytest.mark.parametrize("name", MECHANISMS)
@pytest.mark.parametrize("duration", [0.0, 1.0])
def test_unequal_weights_reject_before_selection(monkeypatch, name, duration):
    """Unequal weights reject before selection."""
    particle = _particle("particle_resolved", 0.25)
    strategy = _strategy(name, "particle_resolved")
    state, rng = (
        _state(particle),
        deepcopy(strategy.random_generator.bit_generator.state),
    )

    def forbidden(**kwargs):
        pytest.fail("unsupported weights must reject before drawing")

    monkeypatch.setattr(
        particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        forbidden,
    )
    with pytest.raises(ValueError, match="equal positive"):
        strategy.step(particle, 298.15, 101325.0, duration)
    _unchanged(particle, state)
    assert strategy.random_generator.bit_generator.state == rng


@pytest.mark.parametrize(
    "kind", ["discrete", "continuous_pdf", "particle_resolved"]
)
@pytest.mark.parametrize(
    "field", ["masses", "concentration", "charge", "density", "volume"]
)
def test_mutated_physical_state_rejects_before_writes(kind, field):
    """Mutated physical state rejects before writes."""
    particle = _particle(kind)
    particle.concentration[:] = 1
    getattr(particle, field).flat[0] = np.nan
    state = _state(particle)
    with pytest.raises(ValueError):
        _strategy("brownian", kind).step(particle, 298.15, 101325.0, 0)
    _unchanged(particle, state)


@pytest.mark.parametrize(
    "kind", ["discrete", "continuous_pdf", "particle_resolved"]
)
@pytest.mark.parametrize("duration", [-1, np.nan, np.inf, True, np.array(1.0)])
def test_bad_time_rejects_even_zero_population(kind, duration):
    """Bad time rejects even zero population."""
    particle = _particle(kind)
    particle.concentration[:] = 0
    state = _state(particle)
    with pytest.raises(ValueError, match="time_step"):
        _strategy("brownian", kind).step(particle, 298.15, 101325.0, duration)
    _unchanged(particle, state)


@pytest.mark.parametrize("size", [0, 1, 4])
def test_empty_or_single_active_is_rng_preserving(monkeypatch, size):
    """Empty or single active is rng preserving."""
    particle = _particle("particle_resolved", 4.0)
    particle.masses = particle.masses[:, :size].copy()
    particle.concentration = particle.concentration[:, :size].copy()
    particle.charge = particle.charge[:, :size].copy()
    particle.concentration[:] = 0
    if size == 1:
        particle.concentration[:] = 4
    state = _state(particle)
    strategy = _strategy("brownian", "particle_resolved")
    rng = deepcopy(strategy.random_generator.bit_generator.state)
    assert strategy.step(particle, 298.15, 101325.0, 1.0) is particle
    _unchanged(particle, state)
    assert strategy.random_generator.bit_generator.state == rng


@pytest.mark.parametrize("bad", [True, 1, 0, -1, 2.5, np.nan])
def test_invalid_bin_count_preflight(bad):
    """Invalid bin count preflight."""
    particle = _particle("particle_resolved")
    particle.concentration[:] = 0
    strategy = _strategy("brownian", "particle_resolved")
    strategy.particle_resolved_bins_number = bad
    with pytest.raises(ValueError, match="total_bins"):
        strategy.step(particle, 298.15, 101325.0, 0)


@pytest.mark.parametrize(
    "edges", [[1e-7], [2e-7, 1e-7], [1e-7, np.inf], [0, 1], [[1, 2]]]
)
def test_invalid_edges_reject(edges):
    """Invalid edges reject."""
    particle = _particle("particle_resolved")
    with pytest.raises(ValueError, match="edges"):
        _get_particle_resolved_kernel_radius_data(particle, np.asarray(edges))


def test_edges_do_not_silently_drop_active_particles():
    """Edges do not silently drop active particles."""
    particle = _particle("particle_resolved")
    with pytest.raises(ValueError, match="cover every active"):
        _get_binned_kernel_data_from_particle_data(
            particle, np.array([1e-12, 1e-11])
        )


@pytest.mark.parametrize(
    "kind", ["discrete", "continuous_pdf", "particle_resolved"]
)
def test_readonly_writer_and_protected_alias_reject(kind):
    """Readonly writer and protected alias reject."""
    particle = _particle(kind)
    particle.concentration[:] = 1
    strategy = _strategy("brownian", kind)
    particle.concentration.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        strategy.step(particle, 298.15, 101325.0, 0)
    particle.concentration.flags.writeable = True
    particle.charge = particle.concentration.view()
    state = _state(particle)
    with pytest.raises(ValueError, match="alias"):
        strategy.step(particle, 298.15, 101325.0, 0)
    _unchanged(particle, state)


@pytest.mark.parametrize(
    "pairs",
    [
        np.array([[0, 9]]),
        np.array([[-1, 2]]),
        np.array([[0.0, 2.0]]),
        np.array([0, 2]),
        np.array([[0, 2], [0, 3]]),
        np.array([[0, 1], [1, 2]]),
    ],
)
def test_invalid_selection_rejects_without_particle_writes(monkeypatch, pairs):
    """Invalid selection rejects without particle writes."""
    particle = _particle("particle_resolved")
    particle.concentration[:] = 1
    state = _state(particle)
    monkeypatch.setattr(
        particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        lambda **kwargs: pairs,
    )
    with pytest.raises(ValueError, match="selector"):
        _strategy("brownian", "particle_resolved").step(
            particle, 298.15, 101325.0, 1
        )
    _unchanged(particle, state)


def test_self_root_is_noop_and_duplicate_recipients_conserve():
    """Self root is noop and duplicate recipients conserve."""
    particle = _particle("particle_resolved")
    particle.concentration[:] = 4
    inventory = _inventory(particle)
    _native_step._merge(
        particle, np.arange(4), np.array([[0, 2], [1, 2], [2, 2]])
    )
    npt.assert_allclose(_inventory(particle)[0], inventory[0], rtol=2e-15)
    assert _inventory(particle)[1] == inventory[1]
    assert particle.concentration[0, 2] == 4


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_density_step_equals_existing_rate_times_duration(kind):
    """Density step equals existing rate times duration."""
    particle = _particle(kind)
    strategy = _strategy("brownian", kind)
    before = _state(particle)
    rate = strategy.net_rate(particle, 298.15, 101325.0)
    # Material rates differ between the PMF and radius-PDF measures.
    duration = 1e-3
    expected = particle.concentration + rate[None, :] * duration
    assert strategy.step(particle, 298.15, 101325.0, duration) is particle
    npt.assert_array_equal(particle.concentration, expected)
    for name in ("masses", "charge", "density", "volume"):
        npt.assert_array_equal(getattr(particle, name), before[name][1])


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_negative_density_proposal_rejects_without_clipping(kind, monkeypatch):
    """Negative density proposal rejects without clipping."""
    particle = _particle(kind)
    state = _state(particle)
    strategy = _strategy("brownian", kind)
    monkeypatch.setattr(
        strategy, "net_rate", lambda *args: -2 * particle.concentration[0]
    )
    with pytest.raises(ValueError, match="smaller finite"):
        strategy.step(particle, 298.15, 101325.0, 1)
    _unchanged(particle, state)


@pytest.mark.parametrize("name", list(CHARGED))
@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_direct_callback_uses_original_active_material(
    monkeypatch, name, volume
):
    """Direct callback uses original active material."""
    particle = _particle("particle_resolved", volume)
    particle.concentration[:] = 2
    particle.concentration[0, 1] = 0
    strategy = _strategy(name, "particle_resolved")
    strategy.use_direct_kernel = True
    expected = strategy.kernel(particle, 298.15, 101325.0)[0, 3]

    def select(**kwargs):
        result = kwargs["kernel_index_func"](np.array([0]), np.array([2]))
        npt.assert_allclose(result, [expected], rtol=2e-14, atol=0)
        return np.empty((0, 2), dtype=np.int64)

    monkeypatch.setattr(
        particle_resolved_method,
        "get_particle_resolved_coagulation_step",
        select,
    )
    strategy.step(particle, 298.15, 101325.0, 1)


@pytest.mark.parametrize("name", MECHANISMS)
@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_real_selector_conserves_species_and_charge(name, volume):
    """Real selector conserves species and charge."""
    particle = _particle("particle_resolved", volume)
    particle.concentration[:] = 2
    strategy = _strategy(name, "particle_resolved")
    strategy.use_direct_kernel = name in CHARGED
    strategy.random_generator = np.random.default_rng(123)
    inventory = _inventory(particle)
    strategy.step(particle, 298.15, 101325.0, 1e16)
    npt.assert_allclose(_inventory(particle)[0], inventory[0], rtol=2e-15)
    npt.assert_allclose(_inventory(particle)[1], inventory[1], rtol=2e-15)
    assert np.all(particle.concentration >= 0)


@pytest.mark.parametrize("size", [4, 7, 10])
@pytest.mark.parametrize("weight", [0.1, 0.25, 1.0, 4.0])
def test_monodisperse_single_kernel_point_is_finite(size, weight):
    """Monodisperse single kernel point is finite."""
    particle = _particle("particle_resolved")
    particle.masses = np.repeat(particle.masses[:, :1], size, axis=1)
    particle.charge = np.zeros((1, size))
    particle.concentration = np.full((1, size), weight)
    strategy = _strategy("brownian", "particle_resolved")
    strategy.random_generator = np.random.default_rng(2)
    inventory = _inventory(particle)[0]
    strategy.step(particle, 298.15, 101325.0, 1e20)
    npt.assert_allclose(_inventory(particle)[0], inventory, rtol=2e-15)
    assert np.count_nonzero(particle.concentration) < size


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("weight", [0.25, 1.0, 4.0])
def test_normalized_selector_replay_preserves_existing_draws(volume, weight):
    """Same physical hazard yields identical draws on this NumPy backend."""
    first = _particle("particle_resolved", 1.0)
    second = _particle("particle_resolved", volume)
    first.concentration[:] = 1
    second.concentration[:] = weight
    left = _strategy("hard", "particle_resolved")
    right = _strategy("hard", "particle_resolved")
    left.use_direct_kernel = right.use_direct_kernel = True
    left.random_generator = np.random.default_rng(45)
    right.random_generator = np.random.default_rng(45)
    left.step(first, 298.15, 101325.0, 1e15)
    right.step(second, 298.15, 101325.0, 1e15 * volume / weight)
    npt.assert_array_equal(second.masses, first.masses)
    npt.assert_array_equal(second.charge, first.charge)
    npt.assert_array_equal(second.concentration, first.concentration * weight)
    assert (
        right.random_generator.bit_generator.state
        == left.random_generator.bit_generator.state
    )


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize(
    "rate",
    [np.zeros(3), np.full(4, np.nan), np.full(4, 1j), np.ones(4, dtype=bool)],
)
def test_invalid_rate_proposals_are_atomic(monkeypatch, kind, rate):
    """Reject invalid custom rate outputs before any density publication."""
    particle = _particle(kind)
    state = _state(particle)
    strategy = _strategy("brownian", kind)
    monkeypatch.setattr(strategy, "net_rate", lambda *args: rate)
    with pytest.raises(ValueError, match="finite slot shape"):
        strategy.step(particle, 298.15, 101325.0, 1.0)
    _unchanged(particle, state)


@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_uniform_composition_inventory_error_converges_with_grid(kind):
    """Bound retained quadrature error against gross transfer, not tiny dt.

    This fixture establishes uniform-composition convergence only. It does not
    establish conservation for size-dependent composition on the fixed grid.
    """
    errors = []
    for count in (128, 256, 512):
        radius = np.geomspace(1e-9, 1e-6, count)
        mass = 4 * np.pi / 3 * 1000 * radius**3
        pdf = (
            1e10
            * np.exp(-0.5 * (np.log(radius / 1e-7) / 0.35) ** 2)
            / (radius * 0.35 * np.sqrt(2 * np.pi))
        )
        density = (
            pdf
            if kind == "continuous_pdf"
            else pdf * np.diff(radius, append=2 * radius[-1] - radius[-2])
        )
        particle = ParticleData(
            masses=np.stack([0.4 * mass, 0.6 * mass], axis=1)[None],
            concentration=density[None],
            charge=np.zeros((1, count)),
            density=np.array([1000.0, 1000.0]),
            volume=np.ones(1),
            distribution_type=kind,
            radius_grid=radius if kind == "continuous_pdf" else None,
        )
        strategy = _strategy("brownian", kind)
        kernel = np.full((count, count), 1e-15)
        loss = strategy.loss_rate(particle, kernel)
        gain = strategy.gain_rate(particle, kernel)
        gross = particle.masses[0] * loss[:, None]
        defect = particle.masses[0] * (gain - loss)[:, None]
        if kind == "continuous_pdf":
            # Independent trapezoid inventory, without container helpers.
            widths = np.diff(radius)[:, None]
            gross_total = np.sum((gross[1:] + gross[:-1]) * widths / 2, axis=0)
            defect_total = np.sum(
                (defect[1:] + defect[:-1]) * widths / 2, axis=0
            )
        else:
            gross_total, defect_total = gross.sum(axis=0), defect.sum(axis=0)
        errors.append(np.abs(defect_total / gross_total))
    errors = np.array(errors)
    # Observed first-order PMF conversion and second-order PDF quadrature.
    reduction = 0.55 if kind == "discrete" else 0.3
    assert np.all(errors[1:] < reduction * errors[:-1])
    assert np.all(errors[-1] < (0.007 if kind == "discrete" else 3e-5))
