"""Independent material-rate, coupled-linear-solve and trajectory checks."""

import numpy as np
import pytest
import warp as wp

from particula import _cpu_batch
from particula.aerosol import Aerosol
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.condensation_ode import CondensationODE
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.execution.tests.condensation_sparse_test import mixture
from particula.execution.tests.gpu_condensation_ode_test import advance
from particula.gas import GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityKappaParameter
from particula.particles.surface_strategies import SurfaceStrategyMass


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Separate mandatory Warp CPU from actual CUDA evidence."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    return request.param


def material_case(materials):
    """All lanes exchange; gas order is reversed with an interior inert lane."""
    original, strategy = mixture(7)
    rng = np.random.default_rng(158)
    density = np.linspace(900.0, 1800.0, materials)
    molar = np.linspace(0.018, 0.2, materials)
    masses = rng.uniform(1e-16, 2e-15, (2, 7, materials))
    masses[:, -1] = 0
    particles = ParticleData(
        masses=masses,
        concentration=original.particles.concentration.copy(),
        density=density,
        charge=original.particles.charge.copy(),
        volume=original.particles.volume.copy(),
        distribution_type="particle_resolved",
    )
    sources = np.insert(np.arange(materials)[::-1], 1, -1)
    gas_molar = np.array([molar[i] if i >= 0 else 0.029 for i in sources])
    gas = GasData(
        name=[f"gas_{i}" for i in range(materials + 1)],
        molar_mass=gas_molar,
        concentration=np.full((2, materials + 1), 1e-5),
        partitioning=sources >= 0,
    )
    original.environment.saturation_ratio = np.zeros((2, materials + 1))
    strategy.molar_mass = gas_molar
    strategy.activity_strategy = ActivityKappaParameter(
        kappa=np.linspace(0.01, 0.6, materials),
        density=density,
        molar_mass=molar,
        water_index=0,
    )
    strategy.surface_strategy = SurfaceStrategyMass(
        surface_tension=np.linspace(0.03, 0.08, materials),
        density=density,
        phase_index=np.arange(materials),
    )
    strategy.vapor_pressure_strategy = ConstantVaporPressureStrategy(
        np.linspace(0.1, 20, materials + 1)
    )
    strategy.species_map = CondensationSpeciesMap(
        tuple(gas.name),
        tuple((s, int(t)) for s, t in enumerate(sources) if t >= 0),
    )
    return Aerosol(
        particles=particles, gas=gas, environment=original.environment
    ), strategy


@pytest.mark.parametrize("materials", [3, 12])
@pytest.mark.parametrize("phase_mode", ["separate", "mixed", "static"])
def test_every_material_rate_matches_native_cpu(device, materials, phase_mode):
    """Check independent rates, trajectories or dense linear solves."""
    initial, strategy = material_case(materials)
    if phase_mode == "mixed":
        strategy.surface_strategy.phase_index = np.arange(materials) % 2
    elif phase_mode == "static":
        strategy.surface_strategy.phase_index = None
    owner = GPUCondensationODE(initial, strategy, device=device)
    actual = owner._mass_rate.numpy()
    expected = []
    for box in range(2):
        view = _cpu_batch.box_view(initial, box)
        expected.append(
            strategy._native_mass_transfer_rate(
                view.particles,
                view.gas,
                float(view.environment.temperature[0]),
                float(view.environment.pressure[0]),
            )
        )
    np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=1e-32)


@pytest.mark.parametrize("materials", [3, 12])
@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_multiple_exchanges_match_cpu_and_conserve_each_species(
    device, materials, method
):
    """Check independent rates, trajectories or dense linear solves."""
    initial, strategy = material_case(materials)
    controls = dict(
        method=method,
        rtol=1e-7,
        particle_atol=1e-27,
        gas_atol=1e-16,
        first_step=1e-5,
        max_step=0.01,
    )
    owner = GPUCondensationODE(initial, strategy, device=device, **controls)
    reference = CondensationODE(initial, strategy, **controls)
    actual, expected = advance(owner, 0.01), advance(reference, 0.01)
    np.testing.assert_allclose(
        actual.particles.masses,
        expected.particles.masses,
        rtol=5e-7,
        atol=1e-26,
    )
    np.testing.assert_allclose(
        actual.gas.concentration,
        expected.gas.concentration,
        rtol=5e-7,
        atol=1e-16,
    )
    for source, target in strategy.species_map.gas_to_particle:
        before = (
            initial.particles.species_mass_density[:, target]
            + initial.gas.concentration[:, source]
        )
        after = (
            actual.particles.species_mass_density[:, target]
            + actual.gas.concentration[:, source]
        )
        np.testing.assert_allclose(after, before, rtol=3e-13, atol=1e-30)
    np.testing.assert_array_equal(
        actual.gas.concentration[:, 1], initial.gas.concentration[:, 1]
    )
    np.testing.assert_array_equal(actual.particles.masses[:, -1], 0)


@pytest.mark.parametrize("materials, repeats", [(3, 1), (12, 1), (3, 10)])
def test_block_solve_matches_independent_dense_system(
    device, materials, repeats
):
    """Check independent rates, trajectories or dense linear solves."""
    initial, strategy = material_case(materials)
    initial.particles.masses = np.tile(
        initial.particles.masses, (1, repeats, 1)
    )
    initial.particles.concentration = np.tile(
        initial.particles.concentration, (1, repeats)
    )
    initial.particles.charge = np.tile(initial.particles.charge, (1, repeats))
    initial.particles.volume *= repeats
    slots = initial.particles.n_particles

    owner = GPUCondensationODE(initial, strategy, device=device, method="ros3")
    stages = owner._ros3
    rng = np.random.default_rng(194)
    shape = stages.local.shape
    local = rng.uniform(-0.2, 0.2, shape)
    for p in range(shape[0]):
        local[p] -= np.eye(materials) * rng.uniform(1, 1000, materials)
    gas_columns = rng.uniform(0, 2, shape)
    local[6::7] = gas_columns[6::7] = 0
    stages.local.assign(local)
    stages.gas_column.assign(gas_columns)
    rhs = rng.normal(size=owner._size)
    stages.rhs.assign(rhs)
    scale = owner._host_scale
    count = initial.particles.concentration
    # Assemble the complete independently indexed Jacobian, including inert rows.
    jac = np.zeros((owner._size, owner._size))
    weights = np.zeros((2 * materials, owner._size))
    pairs = stages.pairs
    for box in range(2):
        for row, (source, target) in enumerate(pairs):
            gi = owner._mass_size + box * (materials + 1) + source
            weights[box * materials + row, gi] = 1
            for slot in range(slots):
                p = box * slots + slot
                pi = p * materials + target
                weight = (
                    count[box, slot]
                    / initial.particles.volume[box]
                    * scale[pi]
                    / scale[gi]
                )
                weights[box * materials + row, pi] = weight
                for col, (gs, mt) in enumerate(pairs):
                    pj = p * materials + mt
                    gj = owner._mass_size + box * (materials + 1) + gs
                    jac[pi, pj] = local[p, row, col]
                    jac[pi, gj] = gas_columns[p, row, col]
                    jac[gi, pj] -= weight * local[p, row, col]
                    jac[gi, gj] -= weight * gas_columns[p, row, col]
    for q in (0.01, 10.0, 1e6):
        stages._factorized = False
        h = 1 / (q * 0.435866521508459)
        stages.solve(owner, h, owner._candidate)
        actual = owner._candidate.numpy()
        matrix = q * np.eye(owner._size) - jac
        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
        residual = np.linalg.norm(matrix @ actual - rhs, np.inf)
        normalized = residual / (
            np.linalg.norm(matrix, np.inf) * np.linalg.norm(actual, np.inf)
            + np.linalg.norm(rhs, np.inf)
        )
        assert normalized < 3e-15
        np.testing.assert_allclose(
            weights @ actual, weights @ rhs / q, rtol=2e-10, atol=2e-8
        )


def test_bat_ros3_mass_jacobian_at_table_knots(device):
    """Compare device slopes at knots and inside cells to host rate differences."""
    from particula.particles import ActivityBAT

    initial, strategy = material_case(2)
    bat = ActivityBAT(0.2, 0.5, 1243.4933333333333)
    strategy.activity_strategy = bat
    molar = np.array([0.01801528, 0.2])
    # Exact dyadic knots and interior points on the qualified 513-node grid.
    x = np.array([0.125, 0.25, 0.5, 0.75, 180.25 / 512, 200.75 / 512])
    initial.particles.masses[:, :6] = (
        np.column_stack((1 - x, x)) * molar * 2e-14
    )
    owner = GPUCondensationODE(initial, strategy, device=device, method="ros3")
    stages = owner._ros3
    stages.linearize(owner)
    local = stages.local.numpy().reshape(2, 7, 2, 2)
    scales = owner._host_scale[: owner._mass_size].reshape(2, 7, 2)
    table = np.array(owner._plan.values["bat_table"]).reshape((-1, 2))
    grid = np.linspace(0, 1, len(table))
    np.testing.assert_allclose(
        grid[np.rint(x[:4] * (len(grid) - 1)).astype(int)], x[:4]
    )

    class InterpolatedBAT(ActivityBAT):
        def activity(self, mass_concentration):
            moles = np.asarray(mass_concentration) / molar
            amount = moles.sum(axis=-1)
            organic = np.divide(
                moles[..., 1],
                amount,
                out=np.zeros_like(amount),
                where=amount > 0,
            )
            values = np.stack(
                [np.interp(organic, grid, col) for col in table.T], axis=-1
            )
            return np.where(amount[..., None] > 0, values, 0)

    strategy.activity_strategy = InterpolatedBAT(0.2, 0.5, 1243.4933333333333)

    def rates(view):
        return strategy._native_mass_transfer_rate(
            view.particles,
            view.gas,
            float(view.environment.temperature[0]),
            float(view.environment.pressure[0]),
        )

    for box in range(2):
        view = _cpu_batch.box_view(initial, box)

        for column, (_, material) in enumerate(stages.pairs):
            original = view.particles.masses.copy()
            delta = 1e-7 * original.sum(axis=-1)
            view.particles.masses[..., material] += delta
            plus = rates(view)
            view.particles.masses[..., material] -= 2 * delta
            minus = rates(view)
            view.particles.masses[:] = original
            for row, (_, target) in enumerate(stages.pairs):
                # CPU native rates are in mass per particle per second. ROS3
                # stores derivatives of scaled rates against scaled masses.
                slope = (plus[:6, target] - minus[:6, target]) / (
                    2 * delta[0, :6]
                )
                expected = (
                    slope * scales[box, :6, material] / scales[box, :6, target]
                )
                np.testing.assert_allclose(
                    local[box, :6, row, column], expected, rtol=2e-5, atol=1e-8
                )
