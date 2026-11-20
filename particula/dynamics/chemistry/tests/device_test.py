"""Direct resident qualification on available explicit Warp devices."""

import numpy as np
import pytest
import warp as wp
from particula.dynamics.chemistry.adapter import GasChemistryAdapter
from particula.dynamics.chemistry.process import GasChemistry
from particula.dynamics.chemistry.tests.process_test import (
    robertson_reference,
    state_and_process,
)
from particula.execution.extension_registry import ProcessRegistry
from particula.util.constants import GAS_CONSTANT


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    ("case", "method", "step", "count"),
    [
        ("arrhenius", "rk23", 0.25, 4),
        ("arrhenius", "ros3", 0.25, 4),
        ("stiff", "ros3", 25.0, 4),
    ],
)
def test_resident_chemistry(device, case, method, step, count):
    """Match analytic or independent stiff solves and continue bit-for-bit."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    aerosol, process = state_and_process(case, method)
    registry = ProcessRegistry()
    registry.register(GasChemistry, GasChemistryAdapter())
    results = []
    with process.prepare(
        aerosol, device=device, time_step=step, process_registry=registry
    ) as owner:
        for i in range(count):
            owner.step()
            results.append(owner.snapshot().gas.concentration)
            if i == 1:
                saved = owner.checkpoint()
    with process.resume(saved, process_registry=registry) as restored:
        restored.run(count - 2)
        np.testing.assert_array_equal(
            restored.snapshot().gas.concentration, results[-1]
        )
    times = step * np.arange(1, count + 1)
    out = np.asarray(results)
    if case == "stiff":
        expected = robertson_reference(times, "Radau")
        for box in range(2):
            np.testing.assert_allclose(
                out[:, box, :3] / 0.1, expected, rtol=2e-6, atol=1e-12
            )
    else:
        temp = aerosol.environment.temperature
        k = 2 * temp / 298 * np.exp(-1000 / (GAS_CONSTANT * temp))
        initial = np.array([1.0, 2.0])
        reactant = initial / (1 + k * initial * times[:, None])
        expected = np.stack((reactant, reactant, initial - reactant), axis=-1)
        np.testing.assert_allclose(
            out[:, :, :3] / aerosol.gas.molar_mass[:3],
            expected,
            rtol=1e-7,
            atol=1e-13,
        )
    for result in results:
        np.testing.assert_array_equal(
            result[:, 3], aerosol.gas.concentration[:, 3]
        )
        np.testing.assert_allclose(
            result.sum(axis=1),
            aerosol.gas.concentration.sum(axis=1),
            rtol=1e-12,
        )


def test_workspace_and_failed_trials():
    """Reject excess workspace before upload and latch failed device trials."""
    from dataclasses import replace

    state, base = state_and_process("stiff", "ros3")
    with pytest.raises(ValueError, match="scratch bytes"):
        GasChemistryAdapter(workspace_bytes=1).plan(base, state)
    limited = GasChemistry(base.bound, replace(base.solver, max_trials=1))
    initial = state.gas.concentration.copy()
    with pytest.raises(RuntimeError, match="trial"):
        limited.execute(state, 1.0)
    np.testing.assert_array_equal(state.gas.concentration, initial)
    registry = ProcessRegistry()
    registry.register(GasChemistry, GasChemistryAdapter())
    with limited.prepare(
        state, device="cpu", time_step=1.0, process_registry=registry
    ) as owner:
        owner.step()
        with pytest.raises(ValueError, match="failure 100"):
            owner.snapshot()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_resume_configuration_guard(device):
    """Reject a changed chemistry network, numerical method or gas mapping."""
    from dataclasses import replace

    from particula.dynamics.chemistry import ReactionMechanism

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    state, process = state_and_process("stiff", "ros3")
    registry = ProcessRegistry()
    registry.register(GasChemistry, GasChemistryAdapter())
    with process.prepare(
        state, device=device, time_step=0.01, process_registry=registry
    ) as sim:
        sim.step()
        saved = sim.checkpoint()
    record = process.bound.mechanism.to_dict()
    record["reactions"][0]["rate"]["k"] *= 2
    alternatives = (
        GasChemistry(ReactionMechanism(record).bind(state.gas), process.solver),
        GasChemistry(process.bound, replace(process.solver, method="rk23")),
        GasChemistry(
            process.bound.mechanism.bind(
                state.gas, {"gas:A": "B", "gas:B": "A"}
            ),
            process.solver,
        ),
    )
    for changed in alternatives:
        with pytest.raises(ValueError):
            changed.resume(saved, process_registry=registry)


@pytest.mark.parametrize("device", ["numpy", "cpu", "cuda:0"])
def test_serialized_rebind_preserves_closed_trajectory(tmp_path, device):
    """Run loaded JSON and YAML across reordered lanes and check the inert gas."""
    from particula.dynamics.chemistry import ReactionMechanism
    from particula.dynamics.chemistry.tests.process_test import (
        state_and_process,
    )
    from particula.execution.cpu_simulation import CPUSimulation

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    actual = []
    for suffix, order in (("json", [0, 1, 2, 3]), ("yaml", [3, 2, 0, 1])):
        state, base = state_and_process("arrhenius", "rk23")
        g = state.gas
        g.name = [g.name[i] for i in order]
        g.molar_mass = g.molar_mass[order]
        g.concentration = g.concentration[:, order].copy()
        g.partitioning = g.partitioning[order]
        path = tmp_path / f"mechanism.{suffix}"
        record = base.bound.mechanism.to_dict()
        record["species"]["gas:inert"] = {}
        g.molar_mass[g.name.index("inert")] = 0.029
        g.concentration[:, g.name.index("inert")] = 261.708813473543
        ReactionMechanism(record).save(path)
        bound = ReactionMechanism.load(path).bind(g)
        process = GasChemistry(bound, base.solver)
        if device == "numpy":
            owner = CPUSimulation(state, process, time_step=1.0)
            owner.step(1.0)
            sample = owner.snapshot()
        else:
            registry = ProcessRegistry()
            registry.register(GasChemistry, GasChemistryAdapter())
            with process.prepare(
                state, time_step=1.0, device=device, process_registry=registry
            ) as sim:
                sim.step()
                sample = sim.snapshot()
        lane = g.name.index("inert")
        np.testing.assert_array_equal(
            sample.gas.concentration[:, lane], g.concentration[:, lane]
        )
        actual.append(sample.gas.concentration[:, np.argsort(order)])
    np.testing.assert_array_equal(*actual)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stiff_tolerance_refinement(device):
    """Detect single-precision rounding of stiff solver coefficients."""
    from dataclasses import replace

    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    initial, base = state_and_process("stiff", "ros3")
    registry = ProcessRegistry()
    registry.register(GasChemistry, GasChemistryAdapter())
    exact = robertson_reference(np.array([100.0]), "Radau")[0]
    errors = []
    for rtol in (1e-5, 1e-8):
        process = GasChemistry(base.bound, replace(base.solver, rtol=rtol))
        with process.prepare(
            initial, device=device, time_step=100.0, process_registry=registry
        ) as sim:
            sim.step()
            y = sim.snapshot().gas.concentration[0, :3] / 0.1
        errors.append(np.abs(y - exact))
    assert np.all(errors[1] < errors[0])
    np.testing.assert_allclose(y, exact, rtol=1e-8, atol=1e-12)
