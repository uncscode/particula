"""Native two-pass staggered condensation with independent scalar oracles."""

import copy

import numpy as np
import pytest
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermalStaggered,
)
from particula.dynamics.condensation.tests.native_step_test import (
    _independent_rates,
)
from particula.dynamics.condensation.tests.species_mapping_test import (
    make_state,
    native_strategy,
    snapshot,
)


def _oracle(
    masses,
    density,
    concentration,
    weights,
    gas,
    molar,
    theta,
    batches,
    dt,
    update_gases,
    pure=None,
):
    """Run scalar sweeps with fixed materials, fresh gas and actual inventories."""
    current = masses.copy()
    current_gas = gas.copy()
    pure = np.array([0, 0.2, 0.1]) if pure is None else pure
    for fraction in (theta, 1 - theta):
        for batch in batches:
            rates = _independent_rates(
                masses[batch],
                density,
                current_gas,
                molar,
                pure,
                0,
            )
            rates[concentration[batch] == 0] = 0
            proposal = rates * (dt * fraction[batch, None])
            for source, target in ((2, 0), (1, 1)):
                deltas = [
                    max(
                        float(proposal[k, target]),
                        -float(current[index, target]),
                    )
                    if weights[index]
                    else 0.0
                    for k, index in enumerate(batch)
                ]
                released = sum(
                    -min(delta, 0) * weights[index]
                    for delta, index in zip(deltas, batch, strict=True)
                )
                requested = sum(
                    max(delta, 0) * weights[index]
                    for delta, index in zip(deltas, batch, strict=True)
                )
                available = current_gas[source] + released
                factor = min(1, available / requested) if requested else 1
                transferred = 0.0
                for delta, index in zip(deltas, batch, strict=True):
                    applied = delta * factor if delta > 0 else delta
                    previous = current[index, target]
                    current[index, target] += applied
                    transferred += (
                        current[index, target] - previous
                    ) * weights[index]
                if update_gases:
                    current_gas[source] -= transferred
    return current, current_gas


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
@pytest.mark.parametrize("mode", ["half", "random", "batch"])
@pytest.mark.parametrize("batches", [1, 2, 9])
@pytest.mark.parametrize("update_gases", [True, False])
def test_native_staggered_scalar_oracle(
    kind, volume, mode, batches, update_gases
):
    """Prove per-node theta durations, whole-grid PDF weights and gas coupling."""
    particles, gas, mapping = make_state(kind, volume)
    gas.concentration[0, 1:] = 1e-18
    if kind == "continuous_pdf":
        particles.concentration *= 1e9
    weights = particles.concentration[0].copy()
    if kind == "particle_resolved":
        weights /= volume
    elif kind == "continuous_pdf":
        weights *= [0.5e-9, 1.5e-9, 1e-9]
    seed, dt = 37, 0.01
    theta = (
        np.full(3, 0.5)
        if mode == "half"
        else np.ones(3)
        if mode == "batch"
        else np.random.default_rng(seed).uniform(0, 1, 3)
    )
    permutation = np.arange(3)
    np.random.default_rng(seed).shuffle(permutation)
    oracle_batches = np.array_split(permutation, min(batches, 3))
    strategy = native_strategy(
        CondensationIsothermalStaggered,
        gas,
        mapping,
        theta_mode=mode,
        num_batches=batches,
        random_state=seed,
        update_gases=update_gases,
    )
    old_mass, old_gas = particles.masses.copy(), gas.concentration.copy()
    mass_id, gas_id = particles.masses, gas.concentration
    expected_mass, expected_gas = _oracle(
        old_mass[0],
        particles.density,
        particles.concentration[0],
        weights,
        old_gas[0],
        gas.molar_mass,
        theta,
        oracle_batches,
        dt,
        update_gases,
    )
    result = strategy.step(particles, gas, 298, 101325, dt, 1.8e-5)
    assert result[0] is particles and result[1] is gas
    assert particles.masses is mass_id and gas.concentration is gas_id
    np.testing.assert_allclose(
        particles.masses[0], expected_mass, rtol=1e-12, atol=1e-30
    )
    np.testing.assert_allclose(
        gas.concentration[0], expected_gas, rtol=1e-12, atol=1e-30
    )
    if update_gases:
        before = (
            np.sum(old_mass[0] * weights[:, None], axis=0) + old_gas[0, [2, 1]]
        )
        after = (
            np.sum(particles.masses[0] * weights[:, None], axis=0)
            + gas.concentration[0, [2, 1]]
        )
        np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-30)
    else:
        np.testing.assert_array_equal(gas.concentration, old_gas)


@pytest.mark.parametrize(
    "random_type", [np.random.default_rng, np.random.RandomState]
)
def test_staggered_reuses_caller_random_state_without_reseeding(random_type):
    """Theta draws followed by shuffling advance the exact supplied generator."""
    particles, gas, mapping = make_state()
    rng, reference = random_type(19), random_type(19)
    strategy = native_strategy(
        CondensationIsothermalStaggered,
        gas,
        mapping,
        theta_mode="random",
        num_batches=2,
        random_state=rng,
    )
    for _ in range(2):
        if isinstance(reference, np.random.Generator):
            reference.uniform(0, 1, 3)
        else:
            reference.random(3)
        order = np.arange(3)
        reference.shuffle(order)
        strategy.step(particles, gas, 298, 101325, 1e-4, 1.8e-5)
    assert strategy.random_state is rng
    np.testing.assert_array_equal(rng.random(10), reference.random(10))


@pytest.mark.parametrize(
    "field,value",
    [
        ("theta_mode", "unknown"),
        ("num_batches", 0),
        ("num_batches", 1.5),
        ("num_batches", True),
        ("shuffle_each_step", "yes"),
    ],
)
def test_mutated_controls_reject_before_random_draw_or_state_write(
    field, value
):
    """Invalid mutable process configuration preserves caller state and RNG."""
    particles, gas, mapping = make_state()
    rng = np.random.default_rng(19)
    strategy = native_strategy(
        CondensationIsothermalStaggered,
        gas,
        mapping,
        theta_mode="random",
        random_state=rng,
    )
    setattr(strategy, field, value)
    before = snapshot(particles, gas)
    rng_before = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        strategy.step(particles, gas, 298, 101325, 0.01, 1.8e-5)
    assert snapshot(particles, gas) == before
    assert rng.bit_generator.state == rng_before


@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_repeated_evaporation_cannot_spend_original_mass_twice(volume):
    """A stiff second sweep sees zero remaining mass despite frozen rates."""
    particles, gas, mapping = make_state(volume=volume)
    gas.concentration[0, 1:] = 0
    original = particles.masses.copy()
    weights = particles.concentration[0] / volume
    strategy = native_strategy(
        CondensationIsothermalStaggered,
        gas,
        mapping,
        theta_mode="half",
        num_batches=2,
        shuffle_each_step=False,
    )
    strategy.step(particles, gas, 298, 101325, 1e6, 1.8e-5)
    np.testing.assert_array_equal(particles.masses, 0)
    expected = np.sum(original[0] * weights[:, None], axis=0)
    np.testing.assert_allclose(
        gas.concentration[0, [2, 1]], expected, rtol=1e-12, atol=1e-30
    )


def test_staggered_uptake_reads_gas_updated_by_preceding_batches():
    """Use appreciable gas depletion so frozen-gas proposals cannot pass."""
    particles, gas, mapping = make_state()
    gas.concentration[0, 1:] = [2e-25, 1e-25]
    strategy = native_strategy(
        CondensationIsothermalStaggered,
        gas,
        mapping,
        theta_mode="half",
        num_batches=3,
        shuffle_each_step=False,
    )
    strategy.vapor_pressure_strategy.vapor_pressure = np.zeros(3)
    old_masses, old_gas = particles.masses.copy(), gas.concentration.copy()
    weights = particles.concentration[0] / 0.25
    batches = [np.array([0]), np.array([1]), np.array([2])]
    expected_mass, expected_gas = _oracle(
        old_masses[0],
        particles.density,
        particles.concentration[0],
        weights,
        old_gas[0],
        gas.molar_mass,
        np.full(3, 0.5),
        batches,
        1e9,
        True,
        pure=np.zeros(3),
    )
    frozen_mass, _ = _oracle(
        old_masses[0],
        particles.density,
        particles.concentration[0],
        weights,
        old_gas[0],
        gas.molar_mass,
        np.full(3, 0.5),
        batches,
        1e9,
        False,
        pure=np.zeros(3),
    )
    assert np.max(abs(frozen_mass - expected_mass)) > 1e-30
    strategy.step(particles, gas, 298, 101325, 1e9, 1.8e-5)
    np.testing.assert_allclose(
        particles.masses[0] - old_masses[0],
        expected_mass - old_masses[0],
        rtol=1e-6,
        atol=1e-33,
    )
    np.testing.assert_allclose(
        gas.concentration[0, 1:], expected_gas[1:], rtol=1e-12, atol=1e-40
    )


def test_late_staggered_failure_does_not_publish_earlier_batches(monkeypatch):
    """Private working state protects both carriers through the final batch."""
    particles, gas, mapping = make_state()
    strategy = native_strategy(
        CondensationIsothermalStaggered,
        gas,
        mapping,
        num_batches=3,
        shuffle_each_step=False,
    )
    before = snapshot(particles, gas)
    original = strategy._native_rate_from_parameters
    calls = 0

    def fail_later(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("later batch model failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(strategy, "_native_rate_from_parameters", fail_later)
    with pytest.raises(ValueError, match="later batch"):
        strategy.step(particles, gas, 298, 101325, 0.01, 1.8e-5)
    assert calls == 3
    assert snapshot(particles, gas) == before
