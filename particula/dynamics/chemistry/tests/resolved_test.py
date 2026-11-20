"""Resolved particle chemistry, evolving geometry and shared gas references."""

from contextlib import nullcontext

import numpy as np
import pytest
from particula import Aerosol, ProcessRegistry
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.process import ChemistrySolver
from particula.dynamics.chemistry.resolved import ResolvedChemistry
from particula.dynamics.chemistry.resolved_adapter import (
    ResolvedChemistryAdapter,
)
from particula.execution.cpu_simulation import CPUSimulation
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def material(name, mass, elements):
    """Return the versioned material entry for a named product or precursor."""
    return (
        f"particle:{name}",
        {"molar_mass_kg_per_mol": mass, "elements": elements},
    )


def experiment(case, equal=False, solver=None):
    """Two closed boxes with unequal volumes, weights, and a spare slot."""
    uptake = case == "uptake"
    n = np.array([[1e-17, 2e-17, 0], [3e-17, 1e-17, 0]])
    counts = np.array([[3e12, 8e12, 0], [1.2e13, 4e12, 0]])
    masses = np.zeros((2, 3, 4))
    masses[:, :, 0] = n * 0.05
    catalog = {}
    if uptake:
        catalog = dict(
            [material("P", 0.04, {"X": 1}), material("Q", 0.04, {"X": 1})]
        )
        catalog["gas:X"] = {"elements": {"X": 1}}
        lanes = {"particle:P": 3, "particle:Q": 1}
        reactions = [
            {
                "id": s,
                "reactants": {"gas:X": 1},
                "products": {f"particle:{s}": 1},
                "rate": {"type": "reactive_uptake", "gamma": g},
            }
            for s, g in [("P", 0.006), ("Q", 0.002)]
        ]
    else:
        catalog = dict(
            [
                material("A", 0.01, {"A": 1}),
                material("B", 0.02, {"B": 1}),
                material("C", 0.03, {"A": 1, "B": 1}),
            ]
        )
        lanes = {"particle:A": 3, "particle:B": 2, "particle:C": 1}
        masses[:, :, 3] = n * 0.01
        masses[:, :, 2] = n * 0.02
        reactions = [
            {
                "id": "combine",
                "reactants": {"particle:A": 1, "particle:B": 1},
                "products": {"particle:C": 1},
                "rate": {
                    "type": "arrhenius",
                    "A": 0.0006,
                    "b": 1,
                    "T_ref_K": 298.0,
                    "Ea_J_per_mol": 200.0,
                    "units": "m^3/mol/s",
                },
            }
        ]
    record = {
        "version": 2,
        "basis": "mol/m^3-phase",
        "species": catalog,
        "reactions": reactions,
    }
    density = (
        np.array([1000.0, 1600.0, 800.0, 1200.0])
        if not equal
        else np.full(4, 1000.0)
    )
    state = Aerosol(
        particles=ParticleData(
            masses,
            counts,
            np.zeros_like(n),
            density,
            np.array([1.0, 2.0]),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["spare", "X"],
            np.array([0.02, 0.04]),
            np.array([[1.0e-7, 1e-6], [3.0e-6, 2e-6]]),
            np.zeros(2, bool),
        ),
        environment=EnvironmentData(
            temperature=np.array([298.0, 318.0]),
            pressure=np.full(2, 1000.0),
            saturation_ratio=np.zeros((2, 2)),
        ),
    )
    mechanism = ReactionMechanism.from_dict(record)
    bound = mechanism.bind_resolved(state, lanes)
    return (
        state,
        ResolvedChemistry(
            bound, solver or ChemistrySolver(rtol=1e-9, atol=1e-17)
        ),
        lanes,
    )


def _bulk_reference(mass, density, temp, v, duration):
    out = mass.copy()
    n = mass[:, 3] / 0.01
    dv = 0.01 / density[3] + 0.02 / density[2] - 0.03 / density[1]
    k = 0.0006 * (temp / 298) * np.exp(-200 / (8.31446261815324 * temp))
    for slot, n0 in enumerate(n):

        def elapsed(f, v0=v[slot], n0=n0):
            return ((v0 / n0 - dv) * (1 / f - 1) + dv * np.log(1 / f)) / k

        f = brentq(lambda x: elapsed(x) - duration, 1e-10, 1.0, xtol=1e-14)
        out[slot, 1:] = [
            0.03 * n0 * (1 - f),
            0.02 * n0 * f,
            0.01 * n0 * f,
        ]
    return out


def _uptake_reference(mass, density, temp, v, duration, concentration, weights):
    out = mass.copy()
    # Constant branching probabilities give both products a common
    # deposited volume per incoming mole. All radii gain the same dr.
    fraction = np.array([0.25, 0.75])
    vm = 0.04 * (fraction @ (1 / density[[1, 3]]))
    speed = np.sqrt(8 * 8.31446261815324 * temp / (np.pi * 0.04))
    r0 = np.cbrt(v * 3 / (4 * np.pi))

    c0 = concentration / 0.04

    def gained(z):
        return 4 * np.pi * z * (r0 * r0 + r0 * z + z * z / 3) / vm

    def growth(t, z):
        return 0.008 * speed / 4 * vm * (c0 - weights @ gained(z))

    sol = solve_ivp(
        growth,
        (0, duration),
        [0.0],
        method="DOP853",
        rtol=3e-13,
        atol=1e-24,
    )
    assert sol.success
    deposited = gained(sol.y[0, -1])
    remaining = 0.04 * (c0 - weights @ deposited)
    out[:, 1] = 0.04 * deposited * fraction[0]
    out[:, 3] = 0.04 * deposited * fraction[1]
    return remaining, out


def reference(state, case, duration):
    """Solve the scalar volume integral or common radial growth independently."""
    p = state.particles
    out = p.masses.copy()
    gas = state.gas.concentration.copy()
    for b, temp in enumerate(state.environment.temperature):
        active = p.concentration[b] > 0
        mass = p.masses[b, active]
        v = (mass / p.density).sum(axis=1)
        if case == "bulk":
            out[b, active] = _bulk_reference(mass, p.density, temp, v, duration)
        else:
            gas[b, 1], out[b, active] = _uptake_reference(
                mass,
                p.density,
                temp,
                v,
                duration,
                gas[b, 1],
                p.concentration[b, active] / p.volume[b],
            )
    return gas, out


def owner_for(state, process, device, dt):
    """Create the requested backend without a device fallback."""
    import warp as wp

    if device == "numpy":
        return nullcontext(CPUSimulation(state, process, time_step=dt)), None
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    registry = ProcessRegistry()
    registry.register(ResolvedChemistry, ResolvedChemistryAdapter())
    return process.prepare(
        state, device=device, time_step=dt, process_registry=registry
    ), registry


@pytest.mark.parametrize(
    "case,equal", [("bulk", True), ("bulk", False), ("uptake", False)]
)
@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
def test_resolved_reference_and_restart(case, equal, device):
    """Check each material lane, extensive mass, inactive slots and restart."""
    state, process, _ = experiment(case, equal)
    duration = 1.5
    expected_gas, expected_particle = reference(state, case, duration)
    counts = state.particles.concentration.copy()
    initial_mass = state.particles.masses.copy()
    volume = state.particles.volume.copy()
    initial_gas = state.gas.concentration.copy()
    owner, registry = owner_for(state, process, device, duration / 2)
    with owner as sim:
        if device == "numpy":
            sim.step(duration / 2)
        else:
            sim.step()
        cp = sim.checkpoint()
        if device == "numpy":
            sim.step(duration)
            resumed = nullcontext(CPUSimulation.resume(cp, process))
        else:
            sim.step()
            resumed = process.resume(cp, process_registry=registry)
        result = sim.snapshot()
    with resumed as sim:
        if device == "numpy":
            sim.step(duration)
        else:
            sim.step()
        repeated = sim.snapshot()
    np.testing.assert_array_equal(
        result.gas.concentration, repeated.gas.concentration
    )
    np.testing.assert_array_equal(
        result.particles.masses, repeated.particles.masses
    )
    np.testing.assert_allclose(
        result.gas.concentration, expected_gas, rtol=5e-8, atol=1e-16
    )
    np.testing.assert_allclose(
        result.particles.masses, expected_particle, rtol=4e-8, atol=1e-32
    )
    np.testing.assert_array_equal(
        result.particles.masses[:, :, 0], initial_mass[:, :, 0]
    )
    np.testing.assert_array_equal(result.particles.masses[:, 2], 0.0)
    np.testing.assert_array_equal(result.particles.concentration, counts)
    np.testing.assert_array_equal(
        result.gas.concentration[:, 0], initial_gas[:, 0]
    )
    np.testing.assert_allclose(
        result.gas.concentration.sum(axis=1) * volume
        + (result.particles.masses.sum(axis=2) * counts).sum(axis=1),
        initial_gas.sum(axis=1) * volume
        + (initial_mass.sum(axis=2) * counts).sum(axis=1),
        rtol=1e-12,
        atol=1e-22,
    )


def test_binding_and_rejections(tmp_path):
    """Round-trip physical identity and reject ambiguous or stale layouts."""
    state, process, lanes = experiment("uptake")
    mechanism = process.bound.mechanism
    for ext in ("json", "yaml"):
        path = tmp_path / f"mechanism.{ext}"
        mechanism.save(path)
        assert (
            ReactionMechanism.load(path).bind_resolved(state, lanes)
            == process.bound
        )
    with pytest.raises(ValueError, match="explicitly map"):
        mechanism.bind_resolved(state, {})
    with pytest.raises(ValueError, match="distinct columns"):
        mechanism.bind_resolved(state, dict.fromkeys(lanes, 1))
    for field, change, msg in [
        ("rate", {"type": "reactive_uptake", "gamma": 1.0}, "probabilities"),
        ("products", {"particle:P": 2}, "do not balance"),
        (
            "rate",
            {"type": "constant", "k": 1.0, "units": "1/s"},
            "single concentration phase",
        ),
    ]:
        record = mechanism.to_dict()
        record["reactions"][0][field] = change
        with pytest.raises(ValueError, match=msg):
            ReactionMechanism(record).bind_resolved(state, lanes)
    record = mechanism.to_dict()
    record["species"]["particle:P"]["elements"] = {"Y": 1}
    with pytest.raises(ValueError, match="elemental imbalance"):
        ReactionMechanism(record).bind_resolved(state, lanes)
    state.particles.density[1] *= 2
    before = state.gas.concentration.copy()
    with pytest.raises(ValueError, match="rebind"):
        process.execute(state, 1.0)
    np.testing.assert_array_equal(state.gas.concentration, before)


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
def test_depletion_and_refinement(device):
    """Resolve shrinking gas concentration and verify refinement independently."""
    errors = []
    for tol in (1e-4, 1e-8):
        state, process, _ = experiment(
            "uptake", solver=ChemistrySolver(rtol=tol, atol=1e-18)
        )
        g, p = reference(state, "uptake", 3.0)
        initial = state.gas.concentration.copy()
        owner, _ = owner_for(state, process, device, 3.0)
        with owner as sim:
            if device == "numpy":
                sim.step(3.0)
            else:
                sim.step()
            result = sim.snapshot()
        errors.append(
            (
                np.max(np.abs(result.gas.concentration - g)),
                np.max(np.abs(result.particles.masses - p)),
            )
        )
        owner, _ = owner_for(state, process, device, 45.0)
        with owner as sim:
            if device == "numpy":
                sim.step(45.0)
            else:
                sim.step()
            depleted = sim.snapshot()
        left = depleted.gas.concentration[:, 1]
        assert np.all((left >= 0) & (left < 1e-6 * initial[:, 1]))
        added = (depleted.particles.masses - state.particles.masses).sum(axis=2)
        np.testing.assert_allclose(
            (added * state.particles.concentration).sum(axis=1),
            (initial[:, 1] - left) * state.particles.volume,
            rtol=1e-12,
            atol=1e-22,
        )
    assert np.all(np.array(errors[1]) < np.array(errors[0]) / 10)


def test_capture_and_configuration_identity():
    """Compare actual CUDA graph replay with ordinary launches and restore."""
    state, process, lanes = experiment("uptake")
    eager, registry = owner_for(state, process, "cuda:0", 0.25)
    captured, _ = owner_for(state, process, "cuda:0", 0.25)
    with eager as baseline, captured as graph:
        graph.capture()
        assert graph.current_time == 0.0
        for _ in range(2):
            graph.step()
            baseline.step()
        a, b = graph.snapshot(), baseline.snapshot()
        np.testing.assert_array_equal(a.gas.concentration, b.gas.concentration)
        np.testing.assert_array_equal(a.particles.masses, b.particles.masses)
        cp = graph.checkpoint()
        graph.step()
        with process.resume(cp, process_registry=registry) as restored:
            restored.capture()
            restored.step()
            np.testing.assert_array_equal(
                graph.snapshot().particles.masses,
                restored.snapshot().particles.masses,
            )
        changed = ResolvedChemistry(
            process.bound.mechanism.bind_resolved(
                state, {k: 4 - v for k, v in lanes.items()}
            )
        )
        with pytest.raises(ValueError):
            changed.resume(cp, process_registry=registry)


def test_failed_trials_and_preflight_budget():
    """A failure in the second box leaves the input intact."""
    state, process, _ = experiment(
        "uptake", solver=ChemistrySolver(first_step=1.0, max_trials=1)
    )
    state.gas.concentration[0, 1] = 1e-30
    original_gas = state.gas.concentration.copy()
    original = state.particles.masses.copy()
    with pytest.raises(RuntimeError, match="maximum trial count"):
        process.execute(state, 1.0)
    np.testing.assert_array_equal(state.particles.masses, original)
    np.testing.assert_array_equal(state.gas.concentration, original_gas)
    with pytest.raises(ValueError, match="scratch bytes"):
        ResolvedChemistryAdapter(workspace_bytes=1).plan(process, state)
    state.particles.concentration[1, 2] = 1
    with pytest.raises(ValueError, match="slot"):
        process.validate(state)


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
def test_condensation_precedes_reactive_uptake(device):
    """Condensation grows surface area before a distinct gas reacts on it."""
    import particula as par
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )

    state, chemistry, _ = experiment("uptake")
    state.gas.partitioning[0] = True
    duration = 0.05
    cond = par.dynamics.MassCondensation(
        par.dynamics.CondensationIsothermal(
            molar_mass=state.gas.molar_mass.copy(),
            activity_strategy=par.particles.ActivityIdealMolar(
                np.array([0.05, 0.04, 0.02, 0.04])
            ),
            surface_strategy=par.particles.SurfaceStrategyVolume(
                density=1000.0
            ),
            vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(
                1e-20
            ),
            species_map=CondensationSpeciesMap(
                tuple(state.gas.name), ((0, 2),)
            ),
        )
    )
    solo, _ = owner_for(state, cond, device, duration)
    combined, _ = owner_for(state, cond | chemistry, device, duration)
    with solo as first, combined as run:
        if device == "numpy":
            first.step(duration)
            run.step(duration)
        else:
            first.step()
            run.step()
        mid = first.snapshot()
        actual = run.snapshot()
    assert np.all(mid.gas.concentration[:, 0] < state.gas.concentration[:, 0])
    np.testing.assert_array_equal(
        mid.gas.concentration[:, 1], state.gas.concentration[:, 1]
    )
    expected_gas, expected_mass = reference(mid, "uptake", duration)
    np.testing.assert_allclose(
        actual.gas.concentration, expected_gas, rtol=5e-8, atol=1e-16
    )
    np.testing.assert_allclose(
        actual.particles.masses, expected_mass, rtol=5e-8, atol=1e-32
    )
    np.testing.assert_allclose(
        (actual.gas.concentration - state.gas.concentration).sum(axis=1)
        * state.particles.volume,
        -(
            (actual.particles.masses - state.particles.masses).sum(axis=2)
            * state.particles.concentration
        ).sum(axis=1),
        rtol=1e-11,
        atol=1e-22,
    )


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
def test_combined_network_and_empty_population(device):
    """Same mechanism includes gas conversion, uptake and bulk aging."""
    state, original, lanes = experiment("uptake")
    state.particles.masses[1] = 0
    state.particles.concentration[1] = 0
    # Use equal mass isomers for an independent two-species gas invariant.
    state.gas.molar_mass[0] = 0.04
    record = original.bound.mechanism.to_dict()
    record["species"]["gas:spare"] = {"elements": {"X": 1}}
    for name, left, right, k in [
        ("gas conversion", "gas:X", "gas:spare", 0.25),
        ("aging", "particle:P", "particle:Q", 0.5),
    ]:
        record["reactions"].append(
            {
                "id": name,
                "reactants": {left: 1},
                "products": {right: 1},
                "rate": {"type": "constant", "k": k, "units": "1/s"},
            }
        )
    process = ResolvedChemistry(
        ReactionMechanism(record).bind_resolved(state, lanes),
        ChemistrySolver(rtol=1e-10, atol=1e-18),
    )
    duration = 1.0
    expected = state.gas.concentration[1, 1] * np.exp(-0.25 * duration)
    owner, _ = owner_for(state, process, device, duration)
    with owner as sim:
        if device == "numpy":
            sim.step(duration)
        else:
            sim.step()
        result = sim.snapshot()
    np.testing.assert_allclose(
        result.gas.concentration[1, 1], expected, rtol=1e-8, atol=1e-16
    )
    np.testing.assert_allclose(
        result.gas.concentration[1].sum(),
        state.gas.concentration[1].sum(),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(result.particles.masses[1], 0)
    gained = result.particles.masses[0] - state.particles.masses[0]
    assert np.all(gained[:2, 1] > gained[:2, 3] / 3)
    np.testing.assert_allclose(
        gained.sum(axis=1) @ state.particles.concentration[0],
        (state.gas.concentration[0] - result.gas.concentration[0]).sum()
        * state.particles.volume[0],
        rtol=1e-12,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_resident_trial_failure(device):
    """A failed multi-box device interval marks the owner unusable."""
    state, process, _ = experiment(
        "uptake", solver=ChemistrySolver(first_step=1.0, max_trials=1)
    )
    owner, _ = owner_for(state, process, device, 1.0)
    with owner as sim:
        with pytest.raises((ValueError, RuntimeError)):
            sim.step()
            sim.synchronize()
        with pytest.raises((ValueError, RuntimeError)):
            sim.snapshot()


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
def test_rebind_reordered_materials(device):
    """Physical species follow explicit mappings across a storage permutation."""
    original, process, lanes = experiment("bulk")
    state, _, _ = experiment("bulk")
    order = [2, 0, 3, 1]
    state.particles.masses = state.particles.masses[:, :, order].copy()
    state.particles.density = state.particles.density[order].copy()
    with pytest.raises(ValueError, match="rebind"):
        process.validate(state)
    # Reorder the two gas columns as well as changing every particle lane.
    state.gas.name.reverse()
    state.gas.molar_mass = state.gas.molar_mass[::-1].copy()
    state.gas.concentration = state.gas.concentration[:, ::-1].copy()
    rebound = ResolvedChemistry(
        process.bound.mechanism.bind_resolved(
            state, {n: order.index(i) for n, i in lanes.items()}
        ),
        process.solver,
    )
    expected_gas, expected_mass = reference(original, "bulk", 0.5)
    owner, _ = owner_for(state, rebound, device, 0.5)
    with owner as sim:
        if device == "numpy":
            sim.step(0.5)
        else:
            sim.step()
        actual = sim.snapshot()
    np.testing.assert_array_equal(
        actual.gas.concentration, expected_gas[:, ::-1]
    )
    np.testing.assert_allclose(
        actual.particles.masses,
        expected_mass[:, :, order],
        rtol=4e-8,
        atol=1e-32,
    )
