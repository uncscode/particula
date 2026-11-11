"""Native CPU wall-loss units, independent sinks, stochastic laws and admission."""

from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest
from particula.dynamics import (
    ChargedWallLossStrategy,
    RectangularWallLossStrategy,
    SphericalWallLossStrategy,
)
from particula.particles import ParticleData

FAMILIES = [
    "spherical",
    "rectangular",
    "charged_spherical",
    "charged_rectangular",
]
KINDS = ["discrete", "continuous_pdf", "particle_resolved"]


def _strategy(family, kind):
    parameters = {"wall_eddy_diffusivity": 1e-3, "distribution_type": kind}
    if "charged" in family:
        return ChargedWallLossStrategy(
            **parameters,
            chamber_geometry=family.removeprefix("charged_"),
            chamber_radius=0.5,
            chamber_dimensions=(1.0, 0.6, 0.3),
            wall_potential=0.05,
            wall_electric_field=10.0,
        )
    if family == "spherical":
        return SphericalWallLossStrategy(**parameters, chamber_radius=0.5)
    return RectangularWallLossStrategy(
        **parameters, chamber_dimensions=(1.0, 0.6, 0.3)
    )


def _particle(kind="particle_resolved", volume=1.0):
    return ParticleData(
        masses=np.array(
            [[[1, 2], [4, 3], [10, 9], [25, 30], [50, 12], [0, 0]]],
            dtype=np.float64,
        )
        * 1e-18,
        concentration=np.array([[0.25, 2.5, 0, 4, 1, 0]])
        * (1e17 if kind == "continuous_pdf" else 1),
        charge=np.array([[-1.0, 2, 17, 0, -3, 11]]),
        density=np.array([1000.0, 1600.0]),
        volume=np.array([volume], dtype=np.float64),
        distribution_type=kind,
        radius_grid=np.array([1, 2, 4, 8, 10, 20]) * 1e-7
        if kind == "continuous_pdf"
        else None,
    )


def _snapshot(particle):
    fields = ["masses", "concentration", "charge", "density", "volume"]
    if particle.radius_grid is not None:
        fields.append("radius_grid")
    return {
        name: (getattr(particle, name), getattr(particle, name).copy())
        for name in fields
    }


def _unchanged(particle, before):
    for name, (identity, values) in before.items():
        assert getattr(particle, name) is identity
        npt.assert_array_equal(identity, values)


def _constant(strategy, monkeypatch, value=0.2):
    name = (
        "compute_coefficient_from_arrays"
        if isinstance(strategy, ChargedWallLossStrategy)
        else "loss_coefficient_for_particles"
    )
    monkeypatch.setattr(
        strategy, name, lambda radius, *args: np.full(radius.shape, value)
    )


def _inventory(masses, concentration, radius=None):
    weighted = masses[0] * concentration[0, :, None]
    if radius is None:
        return weighted.sum(axis=0)
    return np.sum(
        (weighted[1:] + weighted[:-1]) * np.diff(radius)[:, None] / 2, axis=0
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_density_loss_and_removed_species_follow_forward_euler(
    family, kind, monkeypatch
):
    """Independent density-measure accounting preserves the CPU Euler law."""
    particle = _particle(kind)
    strategy = _strategy(family, kind)
    _constant(strategy, monkeypatch)
    before = _snapshot(particle)
    expected = particle.concentration * 0.9
    npt.assert_allclose(
        strategy.rate(particle, 298.15, 101325),
        -0.2 * particle.concentration[0],
        rtol=2e-15,
    )
    assert strategy.step(particle, 298.15, 101325, 0.5) is particle
    npt.assert_allclose(particle.concentration, expected, rtol=2e-15)
    initial = _inventory(
        before["masses"][1], before["concentration"][1], particle.radius_grid
    )
    final = _inventory(
        particle.masses, particle.concentration, particle.radius_grid
    )
    removed = _inventory(
        before["masses"][1],
        0.1 * before["concentration"][1],
        particle.radius_grid,
    )
    npt.assert_allclose(final + removed, initial, rtol=2e-15, atol=0)
    for name, (identity, values) in before.items():
        assert getattr(particle, name) is identity
        if name != "concentration":
            npt.assert_array_equal(identity, values)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_resolved_sink_uses_raw_unequal_counts_and_preserves_survivors(
    family, volume, monkeypatch
):
    """Removal consumes whole weighted representatives without N/V roundoff."""
    particle = _particle(volume=volume)
    strategy = _strategy(family, "particle_resolved")
    _constant(strategy, monkeypatch)
    before = _snapshot(particle)
    expected_active = (
        np.array([0, 1, 3, 4]) if "charged" in family else np.arange(5)
    )
    draws = (
        np.array([0, 1, 0, 1])
        if "charged" in family
        else np.array([0, 1, 1, 0, 1])
    )

    class Draws:
        def binomial(self, n, p):
            assert n == 1
            npt.assert_allclose(
                p, np.full(expected_active.size, np.exp(-0.1)), rtol=2e-15
            )
            return draws

    strategy.random_generator = Draws()
    expected_rate = -0.2 * before["concentration"][1][0] / volume
    npt.assert_allclose(
        strategy.rate(particle, 298.15, 101325), expected_rate, rtol=2e-15
    )
    strategy.step(particle, 298.15, 101325, 0.5)
    lost = np.array([0, 3])
    expected_counts = before["concentration"][1].copy()
    expected_counts[0, lost] = 0
    npt.assert_array_equal(particle.concentration, expected_counts)
    npt.assert_array_equal(particle.masses[0, lost], 0)
    survivors = np.array([1, 2, 4, 5])
    npt.assert_array_equal(
        particle.masses[0, survivors], before["masses"][1][0, survivors]
    )
    removed_counts = before["concentration"][1] - expected_counts
    initial = _inventory(before["masses"][1], before["concentration"][1])
    removed = _inventory(before["masses"][1], removed_counts)
    npt.assert_allclose(
        _inventory(particle.masses, particle.concentration) + removed,
        initial,
        rtol=2e-15,
        atol=0,
    )
    npt.assert_allclose(
        (initial - removed) / volume,
        _inventory(particle.masses, particle.concentration) / volume,
        rtol=2e-15,
        atol=0,
    )
    charge_before = np.dot(
        before["charge"][1][0], before["concentration"][1][0]
    )
    charge_after = np.dot(particle.charge[0], particle.concentration[0])
    charge_removed = np.dot(before["charge"][1][0], removed_counts[0])
    npt.assert_allclose(
        charge_after + charge_removed, charge_before, rtol=2e-15, atol=0
    )
    for name, (identity, values) in before.items():
        assert getattr(particle, name) is identity
        if name not in ("masses", "concentration"):
            npt.assert_array_equal(identity, values)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_weighted_bernoulli_survival_matches_independent_statistics(
    family, volume, monkeypatch
):
    """Weighted extensive inventory has mean p*sum(w) and variance p(1-p)sum(w²)."""
    size = 12000
    particle = _particle(volume=volume)
    particle.masses = np.tile([1e-18, 3e-18], (1, size, 1))
    particle.concentration = np.tile([0.25, 1.0, 4.0], (1, size // 3))
    particle.charge = np.full((1, size), 2.0)
    counts = particle.concentration[0].copy()
    strategy = _strategy(family, "particle_resolved")
    strategy.random_generator = np.random.default_rng(7321)
    _constant(strategy, monkeypatch, value=0.7)
    strategy.step(particle, 298.15, 101325, 1.0)
    probability = np.exp(-0.7)
    expected = probability * counts.sum()
    sigma = np.sqrt(probability * (1 - probability) * np.dot(counts, counts))
    assert abs(particle.concentration.sum() - expected) < 6 * sigma
    npt.assert_allclose(
        _inventory(particle.masses, particle.concentration),
        particle.concentration.sum() * np.array([1e-18, 3e-18]),
        rtol=5e-14,
        atol=0,
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "field", ["masses", "concentration", "charge", "density", "volume"]
)
def test_mutated_storage_rejects_before_writes_or_draws(family, kind, field):
    """Every current physical field is rechecked even for zero-time calls."""
    particle = _particle(kind)
    getattr(particle, field).flat[0] = np.nan
    before = _snapshot(particle)
    strategy = _strategy(family, kind)
    rng = deepcopy(strategy.random_generator.bit_generator.state)
    with pytest.raises(ValueError):
        strategy.step(particle, 298.15, 101325, 0)
    _unchanged(particle, before)
    assert strategy.random_generator.bit_generator.state == rng


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", KINDS)
def test_multibox_and_mismatched_representation_reject(family, kind):
    """Shared containers do not expand CPU execution or infer a distribution."""
    particle = _particle(kind)
    strategy = _strategy(family, kind)
    strategy.distribution_type = "invalid"
    before = _snapshot(particle)
    with pytest.raises(ValueError, match="distribution_type"):
        strategy.step(particle, 298.15, 101325, 0)
    _unchanged(particle, before)
    strategy.distribution_type = kind
    for name in ("masses", "concentration", "charge", "volume"):
        setattr(particle, name, np.repeat(getattr(particle, name), 2, axis=0))
    before = _snapshot(particle)
    with pytest.raises(ValueError, match="n_boxes=1"):
        strategy.step(particle, 298.15, 101325, 0)
    _unchanged(particle, before)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("duration", [-1, np.inf, np.nan, True, np.array(1.0)])
def test_invalid_time_rejects_without_mutation(family, kind, duration):
    """A malformed step cannot be hidden by zero-work shortcuts."""
    particle = _particle(kind)
    before = _snapshot(particle)
    with pytest.raises(ValueError, match="time_step"):
        _strategy(family, kind).step(particle, 298.15, 101325, duration)
    _unchanged(particle, before)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
def test_unstable_euler_step_rejects_atomically(family, kind, monkeypatch):
    """Retain Euler; reject a negative proposal without clipping or a new solver."""
    particle = _particle(kind)
    strategy = _strategy(family, kind)
    _constant(strategy, monkeypatch, 2)
    before = _snapshot(particle)
    with pytest.raises(ValueError, match="smaller finite"):
        strategy.step(particle, 298.15, 101325, 1)
    _unchanged(particle, before)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", KINDS)
def test_native_material_inputs_match_existing_physical_coefficient_laws(
    family, kind
):
    """Independently derive material inputs to the unchanged lower-level laws."""
    particle = _particle(kind, 0.25 if kind == "particle_resolved" else 1)
    strategy = _strategy(family, kind)
    selected = (
        np.array([0, 1, 3, 4]) if kind == "particle_resolved" else np.arange(5)
    )
    masses = particle.masses[0, selected]
    radius = (3 * np.sum(masses / particle.density, axis=1) / (4 * np.pi)) ** (
        1 / 3
    )
    density = np.sum(masses * particle.density, axis=1) / masses.sum(axis=1)
    if "charged" in family:
        # A stale legacy cache must not override current native charges.
        strategy._particle_charge_cache = np.full(selected.size, 1000.0)
        expected = strategy.compute_coefficient_from_arrays(
            radius, density, particle.charge[0, selected], 298.15, 101325
        )
    else:
        expected = strategy.loss_coefficient_for_particles(
            radius, density, 298.15, 101325
        )
    result = strategy.loss_coefficient(particle, 298.15, 101325)
    npt.assert_allclose(result[selected], expected, rtol=2e-14, atol=0)
    absent = np.ones(6, dtype=bool)
    absent[selected] = False
    npt.assert_array_equal(result[absent], 0)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "entry", ["loss_coefficient", "loss_rate", "rate", "step"]
)
def test_mutated_controls_reject_at_every_native_entry(family, kind, entry):
    """Runtime geometry validation is independent of constructor validation."""
    particle = _particle(kind)
    strategy = _strategy(family, kind)
    strategy.wall_eddy_diffusivity = np.nan
    before = _snapshot(particle)
    arguments = (
        (particle, 298.15, 101325, 0)
        if entry == "step"
        else (particle, 298.15, 101325)
    )
    with pytest.raises(ValueError, match="wall_eddy_diffusivity"):
        getattr(strategy, entry)(*arguments)
    _unchanged(particle, before)


@pytest.mark.parametrize(
    "field,value",
    [
        ("chamber_geometry", "other"),
        ("chamber_radius", 0),
        ("wall_potential", np.nan),
        ("wall_electric_field", [1, 2]),
        ("wall_electric_field", [1, np.inf, 3]),
        ("wall_electric_field", np.array(1)),
    ],
)
def test_mutated_charged_configuration_rejects_before_drawing(field, value):
    """Field and geometry errors cannot hide behind a zero time step."""
    particle = _particle()
    strategy = _strategy("charged_spherical", "particle_resolved")
    setattr(strategy, field, value)
    rng = deepcopy(strategy.random_generator.bit_generator.state)
    with pytest.raises(ValueError):
        strategy.step(particle, 298.15, 101325, 0)
    assert strategy.random_generator.bit_generator.state == rng


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", ["readonly", "alias", "zero_time", "empty"])
def test_storage_and_zero_work_contract(family, kind, case):
    """Reject overlapping/writable storage errors; valid no-ops preserve bytes."""
    particle = _particle(kind)
    strategy = _strategy(family, kind)
    if case == "readonly":
        particle.concentration.flags.writeable = False
    if case == "alias":
        particle.charge = particle.concentration.view()
    if case == "empty":
        particle.concentration[:] = 0
    before = _snapshot(particle)
    rng = deepcopy(strategy.random_generator.bit_generator.state)
    if case in ("readonly", "alias"):
        with pytest.raises(ValueError, match="writable|aliases"):
            strategy.step(particle, 298.15, 101325, 0)
    else:
        strategy.step(particle, 298.15, 101325, 0 if case == "zero_time" else 1)
    _unchanged(particle, before)
    assert strategy.random_generator.bit_generator.state == rng


@pytest.mark.parametrize("family", FAMILIES)
def test_empty_resolved_storage_preserves_identity(family):
    """Empty capacity is a valid write-free no-op."""
    particle = _particle()
    for name in ("masses", "concentration", "charge"):
        setattr(particle, name, getattr(particle, name)[:, :0].copy())
    before = _snapshot(particle)
    strategy = _strategy(family, "particle_resolved")
    npt.assert_array_equal(strategy.rate(particle, 298.15, 101325), np.zeros(0))
    strategy.step(particle, 298.15, 101325, 1)
    _unchanged(particle, before)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("value", [np.nan, -1, np.inf, 1j, True])
def test_invalid_coefficient_rejects_before_any_draw(
    family, value, monkeypatch
):
    """A helper's invalid physical output cannot be clamped into valid state."""
    particle = _particle()
    strategy = _strategy(family, "particle_resolved")
    _constant(strategy, monkeypatch, value)
    before = _snapshot(particle)
    rng = deepcopy(strategy.random_generator.bit_generator.state)
    with pytest.raises(ValueError, match="coefficient"):
        strategy.step(particle, 298.15, 101325, 1)
    _unchanged(particle, before)
    assert strategy.random_generator.bit_generator.state == rng


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["discrete", "continuous_pdf"])
@pytest.mark.parametrize("volume", [0.25, 4.0])
def test_density_representations_reject_nonunit_volume(family, kind, volume):
    """PMF/PDF density storage must never silently receive N/V scaling."""
    particle = _particle(kind)
    particle.volume[0] = volume
    before = _snapshot(particle)
    with pytest.raises(ValueError):
        _strategy(family, kind).step(particle, 298.15, 101325, 0)
    _unchanged(particle, before)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("field", ["temperature", "pressure"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, np.array(1.0)])
def test_invalid_environment_rejects_even_for_empty_work(family, field, value):
    """Environmental state is preflighted before empty/no-time returns."""
    particle = _particle()
    particle.concentration[:] = 0
    before = _snapshot(particle)
    arguments = {"temperature": 298.15, "pressure": 101325}
    arguments[field] = value
    with pytest.raises(ValueError, match=field):
        _strategy(family, "particle_resolved").step(
            particle, time_step=0, **arguments
        )
    _unchanged(particle, before)


@pytest.mark.parametrize("family", FAMILIES)
def test_occupied_zero_mass_is_not_silently_removed(family):
    """Reject inconsistent material before running a removal experiment."""
    particle = _particle()
    particle.masses[0, 0] = 0
    before = _snapshot(particle)
    with pytest.raises(ValueError, match="positive mass"):
        _strategy(family, "particle_resolved").step(particle, 298.15, 101325, 1)
    _unchanged(particle, before)


@pytest.mark.parametrize("family", FAMILIES)
def test_resolved_mass_must_be_writable(family):
    """Read-only material cannot cause partial publication of new counts."""
    particle = _particle()
    particle.masses.flags.writeable = False
    before = _snapshot(particle)
    with pytest.raises(ValueError, match="writable"):
        _strategy(family, "particle_resolved").step(particle, 298.15, 101325, 1)
    _unchanged(particle, before)


@pytest.mark.parametrize("field", ["chamber_dimensions", "wall_electric_field"])
def test_writer_cannot_alias_process_configuration(field):
    """Native in-place updates must not rewrite geometry or field arrays."""
    particle = _particle()
    particle.concentration[0, :5] = 1
    strategy = _strategy("charged_rectangular", "particle_resolved")
    setattr(strategy, field, particle.concentration[0, :3])
    before = _snapshot(particle)
    with pytest.raises(ValueError, match="aliases"):
        strategy.step(particle, 298.15, 101325, 0)
    _unchanged(particle, before)


def test_array_geometry_and_field_retain_coefficients_and_ownership():
    """Valid mutable array configuration has the same physical coefficient."""
    particle = _particle()
    strategy = _strategy("charged_rectangular", "particle_resolved")
    strategy.wall_electric_field = (10.0, 0.0, 0.0)
    expected = strategy.loss_coefficient(particle, 298.15, 101325)
    dimensions = np.array(strategy.chamber_dimensions)
    field = np.array(strategy.wall_electric_field)
    strategy.chamber_dimensions = dimensions
    strategy.wall_electric_field = field
    npt.assert_array_equal(
        strategy.loss_coefficient(particle, 298.15, 101325), expected
    )
    strategy.step(particle, 298.15, 101325, 0.1)
    assert strategy.chamber_dimensions is dimensions
    assert strategy.wall_electric_field is field
    npt.assert_array_equal(dimensions, [1.0, 0.6, 0.3])
    npt.assert_array_equal(field, [10.0, 0.0, 0.0])
