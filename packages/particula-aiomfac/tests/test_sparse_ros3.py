"""Physical ROS3 derivatives across sparse-table admission boundaries."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from particula_aiomfac.sparse import ActivityTable

ROOT = Path(__file__).resolve().parents[3]
TABLE = "mixed-two-vapor"


@pytest.fixture(scope="module")
def example():
    pytest.importorskip("particula.execution.condensation_ode")
    return runpy.run_path(
        str(
            ROOT
            / "docs/studies/examples/thermodynamics/activity/aiomfac_closed_box.py"
        )
    )


def boundary_box(example, table, water_only=False):
    """Put the reference class on the lower water-to-dry table face."""
    box = example["make_box"](table=table)
    recipe = ActivityTable.load(TABLE).recipe
    mass, _ = recipe.states(np.array([[0.0, 0.5, 0.5]]))
    box.aerosol.particles.masses[0, 0] = 5e-15 * mass[0] / mass.sum()
    if water_only:
        mapping = box.process.species_map
        box.process.species_map = replace(
            mapping, gas_to_particle=mapping.gas_to_particle[:1]
        )
    return box


def test_cpu_boundary_derivatives_and_forcing(example):
    from particula.execution.condensation_ode import CondensationODE

    recipe = ActivityTable.load(TABLE).recipe
    tmax = recipe.bounds[1][1]
    owners = []
    for table in (None, TABLE):
        box = boundary_box(example, table)
        owners.append(
            CondensationODE(
                box.aerosol,
                box.process,
                method="ros3",
                forcing=lambda time: (tmax + time, 101325.0),
            )
        )
    direct, hybrid = owners
    y = hybrid.solver.state
    provider = hybrid._strategy.activity_strategy
    inside, outside = y.copy(), y.copy()
    inside[0] *= 1.00001
    outside[0] *= 0.99999
    hybrid._rate(0.0, inside)
    before = provider.counts.copy()
    hybrid._rate(0.0, outside)
    assert before[0] > 0 and provider.counts[3] > before[3]
    before = provider.counts.copy()
    j = hybrid._jacobian(0.0, y)
    np.testing.assert_array_equal(j, direct._jacobian(0.0, y))
    assert j[0, 1] != 0
    ft = hybrid._partial_time(0.0, y)
    np.testing.assert_array_equal(ft, direct._partial_time(0.0, y))
    assert np.any(ft != 0)
    np.testing.assert_array_equal(provider.counts, before)


def test_native_ros3_trajectory(example):
    from particula.execution.condensation_ode import CondensationODE

    box = example["make_box"](table=TABLE, permuted=True)
    reference = example["radau"](example["make_box"](permuted=True))
    owner = CondensationODE(
        box.aerosol,
        box.process,
        method="ros3",
        rtol=1e-8,
        particle_atol=1e-30,
        gas_atol=1e-16,
        first_step=1e-5,
    )
    rows = [example["sample"](box, owner.snapshot())]
    for bound in example["TIMES"][1:]:
        while owner.time < bound:
            owner.step(bound)
        rows.append(example["sample"](box, owner.snapshot()))
    example["compare"](rows, reference, box.targets)
    counts = owner._strategy.activity_strategy.counts
    assert counts[0] > 0 and counts[2] > 0


@pytest.mark.parametrize("water_only", [True, False])
def test_cuda_boundary_derivatives(example, water_only):
    if not pytest.importorskip("warp").is_cuda_available():
        pytest.skip("requires NVIDIA CUDA")
    from particula.execution.gpu_condensation_ode import GPUCondensationODE

    buffers = []
    tmax = ActivityTable.load(TABLE).recipe.bounds[1][1]
    for table in (None, TABLE):
        box = boundary_box(example, table, water_only)
        owner = GPUCondensationODE(
            box.aerosol,
            box.process,
            method="ros3",
            device="cuda:0",
            forcing=lambda time: (tmax + time, 101325.0),
        )
        stages = owner._ros3
        stages.linearize(owner)
        buffers.append(
            [
                a.numpy()
                for a in (stages.local, stages.gas_column, stages.partial)
            ]
        )
    for exact, hybrid in zip(*buffers, strict=True):
        assert np.any(exact != 0)
        np.testing.assert_array_equal(exact, hybrid)
