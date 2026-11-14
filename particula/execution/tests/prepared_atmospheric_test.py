"""Physical pair routing, rejection and restart in the real collision sampler."""

import numpy as np
import pytest

from particula import Aerosol
from particula.dynamics.ions.attachment import attachment_coefficient
from particula.dynamics.ions.coagulation import (
    NEGATIVE_ION_MASS,
    POSITIVE_ION_MASS,
    AtmosphericCoagulation,
)
from particula.dynamics.tests.native_runnable_test import state
from particula.particles import ParticleData

wp = pytest.importorskip("warp")


@pytest.fixture(params=["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)])
def device(request):
    """Require real CUDA for the device variant."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def pair_case(kinds=(1, -1), radius=10e-9, aerosol_charge=0):
    """Build two unit particles with explicit 0.5 nm free-ion diameters."""
    masses = np.zeros((1, 2, 3))
    charge = np.zeros((1, 2))
    ion_volume = np.pi / 6 * (0.5e-9) ** 3
    for slot, kind in enumerate(kinds):
        if kind == 0:
            masses[0, slot, 2] = 4 * np.pi / 3 * radius**3 * 1000
            charge[0, slot] = aerosol_charge
        else:
            if kind in (1, 2):
                masses[0, slot, 0] = POSITIVE_ION_MASS
            if kind in (-1, 2):
                masses[0, slot, 1] = NEGATIVE_ION_MASS
            charge[0, slot] = kind if kind != 2 else 0
    base = state()
    return Aerosol(
        particles=ParticleData(
            masses,
            np.ones((1, 2)),
            charge,
            np.array(
                [
                    POSITIVE_ION_MASS / ion_volume,
                    NEGATIVE_ION_MASS / ion_volume,
                    1000,
                ]
            ),
            np.array([1e-12]),
            distribution_type="particle_resolved",
        ),
        gas=base.gas,
        environment=base.environment,
    )


@pytest.mark.parametrize("kinds", [(1, -1), (1, 1), (-1, -1), (2, 1)])
def test_recombination_routing_and_inert_products(device, kinds):
    """Opposite ions merge once; equal signs and neutral products never merge."""
    initial = pair_case(kinds)
    process = AtmosphericCoagulation(0, 1)
    with process.prepare(initial, device=device, time_step=1.0) as simulation:
        simulation.step()
        result = simulation.snapshot()
        majorant = simulation._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        if kinds == (1, -1):
            assert majorant == process.recombination
            assert result.particles.concentration.sum() == 1
            assert np.all(result.particles.charge == 0)
        else:
            assert majorant == 0
            np.testing.assert_array_equal(
                result.particles.masses, initial.particles.masses
            )
        simulation.run(3)
        np.testing.assert_array_equal(
            simulation.snapshot().particles.masses, result.particles.masses
        )
    np.testing.assert_array_equal(
        result.particles.masses.sum(axis=1),
        initial.particles.masses.sum(axis=1),
    )


@pytest.mark.parametrize("polarity", [-1, 1])
def test_attachment_uses_lf_rate_and_conserves_charge(device, polarity):
    """A forced single attachment carries the exact ion mass and signed charge."""
    initial = pair_case((0, polarity))
    rate = attachment_coefficient(10e-9, 0, polarity)
    process = AtmosphericCoagulation(0, 1)
    with process.prepare(
        initial, device=device, time_step=2e-12 / rate
    ) as simulation:
        before = simulation.checkpoint()
        if device != "cpu":
            simulation.capture()
        simulation.step()
        result = simulation.snapshot()
        majorant = simulation._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        np.testing.assert_allclose(majorant, rate, rtol=1e-12)
    assert result.particles.concentration.sum() == 1
    assert result.particles.charge.sum() == polarity
    np.testing.assert_array_equal(
        result.particles.masses.sum(axis=1),
        initial.particles.masses.sum(axis=1),
    )
    with process.resume(before) as restarted:
        restarted.step()
        np.testing.assert_array_equal(
            restarted.snapshot().particles.masses, result.particles.masses
        )


def test_unsupported_attachment_faults_before_rng_or_merge(device):
    """An omitted repulsive fit is never silently set to zero."""
    initial = pair_case((0, 1), radius=1.5e-9, aerosol_charge=1)
    with AtmosphericCoagulation(0, 1).prepare(
        initial, device=device, time_step=1.0
    ) as simulation:
        binding = simulation._bindings["coagulation"]
        rng = binding.brownian_parallel.selector_inputs[27]
        before = rng.numpy().copy()
        simulation.step()
        with pytest.raises(ValueError, match="attachment rate is unsupported"):
            simulation.synchronize()
        np.testing.assert_array_equal(rng.numpy(), before)
        np.testing.assert_array_equal(
            simulation._session.particles.masses.numpy(),
            initial.particles.masses,
        )


@pytest.mark.parametrize("omitted, expected", [("lower", 0), ("upper", 1e-15)])
def test_explicit_omitted_rate_bounds(device, omitted, expected):
    """Optional sensitivity bounds are propagated to the sampler majorant."""
    initial = pair_case((0, 1), radius=1.5e-9, aerosol_charge=1)
    with AtmosphericCoagulation(0, 1, omitted=omitted).prepare(
        initial, device=device, time_step=1e-6
    ) as simulation:
        simulation.step()
        simulation.synchronize()
        majorant = simulation._bindings[
            "coagulation"
        ].brownian_parallel.majorants.numpy()[0]
        assert majorant == expected


def test_recombination_ensemble_matches_finite_population_oracle(device):
    """Compare stochastic decay to an exact finite-state death process.

    With n positive and n negative ions the event rate is alpha*n^2/V.
    Matrix exponentiation supplies an independent distribution, including
    finite-population fluctuations rather than a mean-field approximation.
    """
    from scipy.linalg import expm

    from particula.gas import EnvironmentData, GasData

    boxes, pairs = 512, 8
    base = pair_case()
    masses = np.tile(base.particles.masses, (boxes, pairs, 1))
    volume = 1.7e-12 * pairs
    initial = Aerosol(
        particles=ParticleData(
            masses,
            np.ones((boxes, 2 * pairs)),
            np.tile(base.particles.charge, (boxes, pairs)),
            base.particles.density,
            np.full(boxes, volume),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            base.gas.name,
            base.gas.molar_mass,
            np.tile(base.gas.concentration, (boxes, 1)),
            base.gas.partitioning,
        ),
        environment=EnvironmentData(
            np.full(boxes, 298.15),
            np.full(boxes, 101325.0),
            np.tile(base.environment.saturation_ratio, (boxes, 1)),
        ),
    )
    generator = np.zeros((pairs + 1, pairs + 1))
    for count in range(1, pairs + 1):
        rate = count**2 / pairs
        generator[count, count] = -rate
        generator[count - 1, count] = rate
    probability = expm(generator)[:, pairs]
    states = np.arange(pairs + 1)
    mean = states @ probability
    variance = states**2 @ probability - mean**2
    with AtmosphericCoagulation(0, 1).prepare(
        initial, device=device, time_step=0.01, root_seed=381
    ) as simulation:
        simulation.run(100)
        final = simulation.snapshot().particles
    positive = (final.charge == 1).sum(axis=1)
    negative = (final.charge == -1).sum(axis=1)
    np.testing.assert_array_equal(positive, negative)
    # Six standard errors is a fixed stochastic acceptance interval, not
    # a pathwise CPU/GPU requirement. dt=0.01 resolves the reaction timescale.
    assert abs(positive.mean() - mean) < 6 * np.sqrt(variance / boxes)
    np.testing.assert_allclose(
        final.masses.sum(axis=1),
        initial.particles.masses.sum(axis=1),
        rtol=2e-15,
        atol=0,
    )


@pytest.mark.parametrize("invalid", ["cluster", "temperature", "counts"])
def test_evolving_unsupported_state_faults_before_sampling(device, invalid):
    """Device validation also protects state changed after preparation."""
    with AtmosphericCoagulation(0, 1).prepare(
        pair_case(), device=device, time_step=1.0
    ) as simulation:
        session = simulation._session
        if invalid == "cluster":
            array = session.particles.masses
            value = array.numpy()
            value[0, 0, 0] *= 2
        elif invalid == "temperature":
            array = session.environment.temperature
            value = np.array([300.0])
        else:
            array = session.particles.concentration
            value = np.array([[2.0, 2.0]])
        wp.copy(array, wp.array(value, dtype=wp.float64, device=device))
        rng = simulation._bindings[
            "coagulation"
        ].brownian_parallel.selector_inputs[27]
        before = rng.numpy().copy()
        simulation.step()
        with pytest.raises(ValueError, match="valid unit ions"):
            simulation.synchronize()
        np.testing.assert_array_equal(rng.numpy(), before)


def test_atmospheric_configuration_rejects_incompatible_source():
    """Physical LF reference masses and reserved material identity agree."""
    from particula.dynamics.ions.source import IonPairSource

    initial = pair_case()
    source = IonPairSource(1e6, 0, 1, 2e-25, 1e-25)
    with pytest.raises(ValueError, match="150 u and 90 u"):
        (source | AtmosphericCoagulation(0, 1)).prepare(
            initial, device="cpu", time_step=0.1
        )
    source = IonPairSource(1e6, 1, 0, POSITIVE_ION_MASS, NEGATIVE_ION_MASS)
    with pytest.raises(ValueError, match="matching material maps"):
        (source | AtmosphericCoagulation(0, 1)).prepare(
            initial, device="cpu", time_step=0.1
        )


def test_composed_source_cleanup_cohort_capture_restart(device):
    """Independent RNG streams, exports and original seed lineage continue."""
    from particula.dynamics.ions.cleanup import IonNeutralCleanup
    from particula.dynamics.ions.source import IonPairSource
    from particula.execution.cohort import SeedCohort

    base = pair_case((0, 1))
    masses = np.zeros((1, 128, 4))
    masses[:, :2, :3] = base.particles.masses
    counts = np.zeros((1, 128))
    counts[:, :2] = 1
    charges = np.zeros((1, 128))
    charges[:, :2] = base.particles.charge
    initial = Aerosol(
        particles=ParticleData(
            masses,
            counts,
            charges,
            np.append(base.particles.density, 1000),
            base.particles.volume,
            distribution_type="particle_resolved",
        ),
        gas=base.gas,
        environment=base.environment,
    )
    process = (
        IonPairSource(1e14, 0, 1, POSITIVE_ION_MASS, NEGATIVE_ION_MASS)
        | AtmosphericCoagulation(0, 1)
        | IonNeutralCleanup(0, 1)
        | SeedCohort(2, 3)
    )
    with process.prepare(
        initial, device=device, time_step=0.01, root_seed=913
    ) as simulation:
        simulation.run(20)
        checkpoint = simulation.checkpoint()
        if device != "cpu":
            simulation.capture()
        simulation.run(20)
        result = simulation.snapshot()
        cohort = simulation.cohort_diagnostics()
        source = simulation.ion_source_diagnostics()
        cleanup = simulation.ion_cleanup_diagnostics()
        rng = (
            simulation._bindings["coagulation"]
            .brownian_parallel.selector_inputs[27]
            .numpy()
        )
    with process.resume(checkpoint) as simulation:
        simulation.run(20)
        np.testing.assert_array_equal(
            simulation.snapshot().particles.masses, result.particles.masses
        )
        for method, expected in (
            (simulation.cohort_diagnostics, cohort),
            (simulation.ion_source_diagnostics, source),
            (simulation.ion_cleanup_diagnostics, cleanup),
        ):
            actual = method()
            for name in expected:
                np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_array_equal(
            simulation._bindings["coagulation"]
            .brownian_parallel.selector_inputs[27]
            .numpy(),
            rng,
        )
    np.testing.assert_array_equal(cohort["unresolved"], [1])
    np.testing.assert_array_equal(cohort["reached"], [0])
    np.testing.assert_allclose(
        result.particles.masses.sum(axis=1) + cleanup["exported_mass"],
        initial.particles.masses.sum(axis=1) + source["injected_mass"],
        rtol=2e-15,
        atol=0,
    )
    assert result.particles.charge.sum() == initial.particles.charge.sum()


def test_atmospheric_materials_cannot_label_seed_cohort():
    """Even without a source, reserved ion lanes cannot label backgrounds."""
    from particula.execution.cohort import SeedCohort

    process = AtmosphericCoagulation(0, 1) | SeedCohort(2, 1)
    with pytest.raises(ValueError, match="distinct from cohort markers"):
        process.prepare(pair_case((0, 1)), device="cpu", time_step=0.1)
