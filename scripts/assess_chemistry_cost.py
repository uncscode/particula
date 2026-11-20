"""Measure validated stiff gas chemistry ensembles on NumPy and resident CUDA.

Run python -m scripts.assess_chemistry_cost on a CUDA host. Each timed run
constructs and binds the state, prepares the owner, integrates, snapshots,
checkpoints and closes it. The independent references and comparison run
outside the timing window. Results include the range of per-box trial work.
"""

import json
from contextlib import nullcontext
from statistics import median
from time import perf_counter

import numpy as np
from particula import Aerosol, ProcessRegistry
from particula.dynamics.chemistry.adapter import GasChemistryAdapter
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry
from particula.execution.cpu_simulation import CPUSimulation
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from scipy.integrate import solve_ivp
from scripts.assess_gas_chemistry import benchmark
from scripts.assess_numerical_methods import cases

DURATION = 0.01


def ensemble(initial):
    """Make boxes with unequal initial molar inventories for variable work."""
    _, prototype = benchmark()
    n = len(initial)
    p = ParticleData(
        masses=np.full((n, 1, 1), 1e-18),
        concentration=np.ones((n, 1)),
        density=np.array([1000.0]),
        charge=np.zeros((n, 1)),
        volume=np.ones(n),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        ["A", "B", "C"],
        np.full(3, 0.1),
        np.column_stack((initial, np.zeros((n, 2)))) * 0.1,
        np.zeros(3, bool),
    )
    state = Aerosol(
        particles=p,
        gas=gas,
        environment=EnvironmentData(
            temperature=np.full(n, 298.0),
            pressure=np.full(n, 101325.0),
            saturation_ratio=np.zeros((n, 3)),
        ),
    )
    return state, GasChemistry(
        prototype.mechanism.bind(gas),
        ChemistrySolver(method="ros3", rtol=1e-6, atol=1e-13),
    )


def measure(initial, device):
    """Time the full workflow and read checkpoint work counts."""
    start = perf_counter()
    state, process = ensemble(initial)
    if device == "numpy":
        owner = nullcontext(CPUSimulation(state, process, time_step=DURATION))
    else:
        registry = ProcessRegistry()
        registry.register(GasChemistry, GasChemistryAdapter())
        owner = process.prepare(
            state, device=device, time_step=DURATION, process_registry=registry
        )
    with owner as sim:
        if device == "numpy":
            sim.step(DURATION)
        else:
            sim.step()
        result = sim.snapshot().gas.concentration / 0.1
        checkpoint = sim.checkpoint()
    elapsed = perf_counter() - start
    work = None
    if device != "numpy":
        field = next(
            p for p in checkpoint.extension_state if p.role == "work_counts"
        )
        counts = np.frombuffer(field.data, dtype=field.dtype).reshape(
            field.shape
        )
        work = dict(
            zip(
                ("accepted", "rejected", "factorizations", "solves"),
                [[int(c.min()), int(c.max())] for c in counts.T],
                strict=True,
            )
        )
    return elapsed, result, work


def main():
    """Compare to Radau and report warm median wall time."""
    rhs = next(c["rate"] for c in cases() if c["name"] == "robertson")
    rows = []
    for n in (1, 8, 32, 128):
        initial = np.geomspace(0.1, 2.0, n) if n > 1 else np.array([1.0])
        refs = []
        for amount in initial:
            sol = solve_ivp(
                rhs,
                (0, DURATION),
                [amount, 0.0, 0.0],
                method="Radau",
                rtol=1e-11,
                atol=[1e-14, 1e-17, 1e-14],
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            refs.append(sol.y[:, -1])
        for device in ("numpy", "cuda:0"):
            times = []
            for repeat in range(4):
                elapsed, result, work = measure(initial, device)
                np.testing.assert_allclose(result, refs, rtol=2e-6, atol=1e-12)
                np.testing.assert_allclose(
                    result.sum(axis=1), initial, rtol=1e-12
                )
                if repeat:
                    times.append(elapsed)
            rows.append(
                {
                    "boxes": n,
                    "device": device,
                    "median_s": median(times),
                    "trials_per_box": work,
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
