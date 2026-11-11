"""Independent native condensation limiter and ownership fixtures."""

import numpy as np
import pytest
from particula.dynamics.condensation._native_step import (
    finalize_native_transfer,
    validate_native_step_storage,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationLatentHeat,
)
from particula.dynamics.condensation.tests.species_mapping_test import (
    make_state,
    native_strategy,
)
from particula.gas.latent_heat_strategies import (
    ConstantLatentHeat,
    LatentHeatStrategy,
)
from particula.util.constants import GAS_CONSTANT


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_finalizer_credits_only_actual_evaporation_before_uptake(volume):
    """Unequal weights, two opposed transfers, inert gas and material lanes."""
    masses = (
        np.array([[3, 4, 9], [5, 6, 8], [7, 8, 7]], dtype=np.float64) * 1e-18
    )
    weights = np.array([2.0, 0.5, 0.0]) / volume
    gas = np.array([17.0, 0.0, 1.0]) * 1e-18 / volume
    proposal = (
        np.array(
            [[100, -100, -1000], [-100, 100, 1000], [50, 50, -1]],
            dtype=np.float64,
        )
        * 1e-18
    )
    before = masses.copy(), weights.copy(), gas.copy(), proposal.copy()
    updated, remaining = finalize_native_transfer(
        masses,
        weights,
        gas,
        proposal,
        np.array([2, 1], dtype=np.intp),
        np.array([0, 1], dtype=np.intp),
    )
    # A: 5*.5/V released plus 1/V gas feeds 2/V recipients -> +1.75 kg/particle
    # in e-18 units. B: 4*2/V released feeds .5/V recipients -> +16.
    expected = np.array([[4.75, 0, 9], [0, 22, 8], [7, 8, 7]]) * 1e-18
    np.testing.assert_allclose(updated, expected, rtol=1e-12, atol=1e-30)
    np.testing.assert_allclose(remaining, [17e-18 / volume, 0, 0], atol=1e-30)
    # Compute each extensive species inventory explicitly, including gas map.
    np.testing.assert_allclose(
        [
            (2 * updated[0, 0] + 0.5 * updated[1, 0]) + remaining[2] * volume,
            (2 * updated[0, 1] + 0.5 * updated[1, 1]) + remaining[1] * volume,
        ],
        [9.5e-18, 11e-18],
        rtol=1e-12,
        atol=1e-30,
    )
    assert np.all(updated >= 0) and np.all(remaining >= 0)
    for actual, original in zip(
        (masses, weights, gas, proposal), before, strict=True
    ):
        np.testing.assert_array_equal(actual, original)


def test_finalizer_unlimited_small_transfer_matches_per_particle_arithmetic():
    """Independent gas/particle comparisons retain unrelated gas storage."""
    masses = np.array([[2.0, 3.0], [5.0, 7.0]]) * 1e-18
    weights = np.array([8.0, 2.0])
    gas = np.array([0.0, 1e-15, 2e-15])
    changes = np.array([[0.125, -0.25], [0.5, 0.75]]) * 1e-18
    updated, remaining = finalize_native_transfer(
        masses,
        weights,
        gas,
        changes,
        np.array([2, 1]),
        np.array([0, 1]),
    )
    np.testing.assert_allclose(
        updated, masses + changes, rtol=1e-14, atol=1e-32
    )
    np.testing.assert_allclose(
        remaining,
        [0, 1e-15 + 0.5e-18, 2e-15 - 2e-18],
        rtol=1e-14,
        atol=1e-32,
    )


def test_finalizer_handles_empty_map_and_empty_particle_axis():
    """No work preserves full gas widths and returns detached staged storage."""
    masses = np.empty((0, 2))
    gas = np.array([1.0, 2.0, 3.0])
    updated, remaining = finalize_native_transfer(
        masses,
        np.empty(0),
        gas,
        masses.copy(),
        np.array([], dtype=np.intp),
        np.array([], dtype=np.intp),
    )
    assert updated.shape == masses.shape and updated is not masses
    assert remaining is not gas
    np.testing.assert_array_equal(remaining, gas)


@pytest.mark.parametrize("fault", ["shape", "nan", "infinite", "overflow"])
def test_finalizer_invalid_proposal_has_no_partial_state(fault):
    """Failure in a later mapped species cannot publish earlier staged work."""
    masses = np.array([[2.0, 3.0]])
    weights = np.array([2.0])
    gas = np.array([10.0, 20.0, 30.0])
    proposal = np.array([[1.0, 1.0]])
    if fault == "shape":
        proposal = np.ones((1, 3))
    else:
        proposal[0, 1] = {"nan": np.nan, "infinite": np.inf, "overflow": 1e308}[
            fault
        ]
    before = masses.copy(), weights.copy(), gas.copy(), proposal.copy()
    with pytest.raises(ValueError):
        finalize_native_transfer(
            masses, weights, gas, proposal, np.array([2, 1]), np.array([0, 1])
        )
    for actual, original in zip(
        (masses, weights, gas, proposal), before, strict=True
    ):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize(
    "field", ["masses", "gas", "density", "volume", "charge"]
)
def test_native_writer_ownership_rejects_before_mutation(field):
    """Even protected-field aliases must reject before a particle write."""
    particles, gas, _ = make_state()
    if field == "masses":
        particles.masses.flags.writeable = False
    elif field == "gas":
        gas.concentration.flags.writeable = False
    elif field == "density":
        particles.density = particles.masses[0, 0]
    elif field == "volume":
        particles.volume = particles.masses.ravel()[:1]
    else:
        particles.charge = particles.masses[:, :, 0]
    old_masses, old_gas = particles.masses.copy(), gas.concentration.copy()
    with pytest.raises(ValueError):
        validate_native_step_storage(particles, gas, True)
    np.testing.assert_array_equal(particles.masses, old_masses)
    np.testing.assert_array_equal(gas.concentration, old_gas)


def test_read_only_external_gas_is_supported_when_updates_disabled():
    """A reservoir is protected input, so it need not be writable."""
    particles, gas, _ = make_state()
    gas.concentration.flags.writeable = False
    validate_native_step_storage(particles, gas, False)


def _independent_rates(masses, density, gas, molar, pure, heat):
    """Evaluate scalar transport/thermal equations without production helpers."""
    temperature, pressure, viscosity = 298.0, 101325.0, 1.8e-5
    radius = (3 * (masses / density).sum(axis=1) / (4 * np.pi)) ** (1 / 3)
    fractions = masses / masses.sum(axis=1)[:, None]
    result = np.zeros_like(masses)
    for source, target, diffusion, alpha in (
        (2, 0, 2e-5, 0.9),
        (1, 1, 1e-5, 0.7),
    ):
        local_heat = np.broadcast_to(np.asarray(heat), gas.shape)[source]
        path = (2 * viscosity / pressure) / np.sqrt(
            8 * molar[source] / (np.pi * GAS_CONSTANT * temperature)
        )
        kn = path / radius
        correction = (
            0.75
            * alpha
            * (1 + kn)
            / (kn**2 + kn + 0.283 * alpha * kn + 0.75 * alpha)
        )
        equilibrium = (
            pure[source]
            * fractions[:, target]
            * np.exp(
                2
                * 0.072
                * molar[source]
                / (GAS_CONSTANT * temperature * 1000 * radius)
            )
        )
        delta = (
            gas[source] * GAS_CONSTANT * temperature / molar[source]
            - equilibrium
        )
        rate = (
            4 * np.pi * radius * diffusion * correction * delta * molar[source]
        )
        rate /= GAS_CONSTANT * temperature
        conductivity = 1e-3 * (4.39 + 0.071 * temperature)
        thermal = 1 + (
            diffusion
            * local_heat
            * equilibrium
            / (temperature * conductivity)
            * (local_heat * molar[source] / (temperature * GAS_CONSTANT) - 1)
            / (GAS_CONSTANT * temperature / molar[source])
        )
        result[:, target] = rate / thermal
    return result


@pytest.mark.parametrize(
    "kind,volume",
    [
        ("particle_resolved", 0.25),
        ("particle_resolved", 1.0),
        ("particle_resolved", 4.0),
        ("discrete", 1.0),
        ("continuous_pdf", 1.0),
    ],
)
@pytest.mark.parametrize(
    "strategy_type,heat",
    [
        (CondensationIsothermal, 0.0),
        (CondensationLatentHeat, 2e6),
    ],
)
@pytest.mark.parametrize("update_gases", [True, False])
@pytest.mark.parametrize("direction", ["uptake", "evaporation"])
def test_native_single_step_against_independent_physics(
    kind, volume, strategy_type, heat, update_gases, direction
):
    """Separately prove particle changes, mapped gas changes and extensive heat."""
    particles, gas, mapping = make_state(kind, volume)
    if kind == "continuous_pdf":
        particles.concentration *= 1e9
    extra = {"latent_heat_strategy": ConstantLatentHeat(heat)} if heat else {}
    strategy = native_strategy(
        strategy_type, gas, mapping, update_gases=update_gases, **extra
    )
    pure = np.array([0.0, 0.2, 0.1])
    if direction == "uptake":
        pure[:] = 0
        strategy.vapor_pressure_strategy.vapor_pressure = pure.copy()
        gas.concentration[0, 1:] = 1e-12
        dt = 1.0
    else:
        gas.concentration[0, 1:] = 1e-18
        dt = 1e-3
    old_masses, old_gas = particles.masses.copy(), gas.concentration.copy()
    protected = [
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.partitioning,
    ]
    if particles.radius_grid is not None:
        protected.append(particles.radius_grid)
    snapshots = [(array, array.copy()) for array in protected]
    identities = particles.masses, gas.concentration
    weights = particles.concentration[0].copy()
    if kind == "particle_resolved":
        weights /= volume
    elif kind == "continuous_pdf":
        weights *= np.array([0.5e-9, 1.5e-9, 1e-9])
    expected_delta = dt * _independent_rates(
        old_masses[0], particles.density, old_gas[0], gas.molar_mass, pure, heat
    )
    expected_gas_delta = np.array(
        [
            0,
            -np.sum(expected_delta[:, 1] * weights),
            -np.sum(expected_delta[:, 0] * weights),
        ]
    )
    assert np.all(old_masses[0] + expected_delta > 0)
    assert np.all(old_gas[0] + expected_gas_delta >= 0)
    result = strategy.step(particles, gas, 298.0, 101325.0, dt, 1.8e-5)
    assert result[0] is particles and result[1] is gas
    assert (
        particles.masses is identities[0] and gas.concentration is identities[1]
    )
    np.testing.assert_allclose(
        particles.masses[0] - old_masses[0],
        expected_delta,
        rtol=1e-9,
        atol=1e-33,
    )
    if update_gases:
        actual_gas_delta = gas.concentration[0] - old_gas[0]
        # Only account for subtraction rounding at the original gas magnitude.
        # Participating changes remain far larger than this machine-error bound.
        tolerance = 1e-9 * abs(expected_gas_delta) + 4 * np.spacing(
            abs(old_gas[0])
        )
        assert np.all(abs(actual_gas_delta - expected_gas_delta) <= tolerance)
        actual_particle_density = np.sum(
            (particles.masses[0] - old_masses[0]) * weights[:, None], axis=0
        )
        assert np.all(
            abs(actual_particle_density + actual_gas_delta[[2, 1]])
            <= tolerance[[2, 1]]
        )
    else:
        np.testing.assert_array_equal(gas.concentration, old_gas)
    if heat:
        expected_energy = (
            np.sum(expected_delta * weights[:, None]) * volume * heat
        )
        np.testing.assert_allclose(
            strategy.last_latent_heat_energy,
            expected_energy,
            rtol=1e-9,
            atol=1e-27,
        )
    for array, before in snapshots:
        np.testing.assert_array_equal(array, before)
    current_protected = [
        particles.concentration,
        particles.charge,
        particles.density,
        particles.volume,
        gas.molar_mass,
        gas.partitioning,
    ]
    if particles.radius_grid is not None:
        current_protected.append(particles.radius_grid)
    assert all(
        current is original
        for current, original in zip(current_protected, protected, strict=True)
    )


class _FullGasHeat(LatentHeatStrategy):
    """Supply deliberately different latent heat in the complete gas order."""

    def latent_heat(self, temperature):
        return np.array([1e8, 2e6, 1e6])


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_latent_energy_uses_mapped_full_gas_order_and_extensive_mass(volume):
    """Gas width differs from particle width and inert heat cannot contribute."""
    particles, gas, mapping = make_state(volume=volume)
    gas.concentration[0, 1:] = 1e-18
    strategy = native_strategy(
        CondensationLatentHeat,
        gas,
        mapping,
        latent_heat_strategy=_FullGasHeat(),
    )
    old = particles.masses.copy()
    heat = np.array([1e8, 2e6, 1e6])
    expected = 1e-3 * _independent_rates(
        old[0],
        particles.density,
        gas.concentration[0],
        gas.molar_mass,
        np.array([0, 0.2, 0.1]),
        heat,
    )
    strategy.step(particles, gas, 298, 101325, 1e-3, 1.8e-5)
    np.testing.assert_allclose(
        particles.masses[0] - old[0], expected, rtol=1e-12, atol=1e-33
    )
    # Counts, not density: this is J, not J/m3.
    expected_energy = np.sum(
        expected * particles.concentration[0, :, None] * [1e6, 2e6]
    )
    np.testing.assert_allclose(
        strategy.last_latent_heat_energy,
        expected_energy,
        rtol=1e-12,
        atol=1e-27,
    )


@pytest.mark.parametrize("field", ["concentration", "charge"])
@pytest.mark.parametrize("alias", [False, True])
def test_resolved_retirement_storage_rejected_before_any_update(field, alias):
    """The added retirement writers have read-only and overlap preflight."""
    from particula.dynamics.condensation.tests.species_mapping_test import (
        snapshot,
    )

    particles, gas, mapping = make_state()
    if alias:
        setattr(particles, field, particles.masses[:, :, 0])
    else:
        getattr(particles, field).flags.writeable = False
    before = snapshot(particles, gas)
    strategy = native_strategy(CondensationIsothermal, gas, mapping)
    with pytest.raises(ValueError, match="writable|alias"):
        strategy.step(particles, gas, 298, 101325, 1.0)
    assert snapshot(particles, gas) == before
