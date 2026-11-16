"""Sparse stiff stepping retains local physics and closed reservoir coupling."""

from dataclasses import replace

import numpy as np
import numpy.testing as npt
from scipy.linalg import expm
from scipy.sparse import csc_matrix, issparse

from particula import Aerosol
from particula.dynamics import CondensationIsothermal
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution._condensation_jacobian import sparse_jacobian
from particula.execution.condensation_ode import CondensationODE
from particula.execution.ode import AdaptiveSolver
from particula.execution.tests.condensation_ode_test import case
from particula.execution.tests.ode_test import linear_problem
from particula.gas import GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyVolume


def mixture(slots=3):
    """Two materials, reversed species mapping, inactive slot and two boxes."""
    original, _ = case(2)
    particles = ParticleData(
        masses=np.tile([[[2e-15, 1e-15]]], (2, slots, 1)),
        concentration=np.full((2, slots), 1e7),
        density=np.array([1000.0, 1500.0]),
        charge=np.zeros((2, slots)),
        volume=np.array([0.3, 0.8]),
        distribution_type="particle_resolved",
    )
    particles.masses[:, -1] = 0
    particles.concentration[:, -1] = 0
    gas = GasData(
        name=["b", "inert", "a"],
        molar_mass=np.array([0.018, 0.028, 0.18]),
        concentration=np.tile([[2e-5, 0.1, 1e-5]], (2, 1)),
        partitioning=np.array([True, False, True]),
    )
    original.environment.saturation_ratio = np.zeros((2, 3))
    strategy = CondensationIsothermal(
        molar_mass=gas.molar_mass,
        diffusion_coefficient=2e-5,
        activity_strategy=ActivityIdealMass(),
        surface_strategy=SurfaceStrategyVolume(density=particles.density),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(1e-5),
        species_map=CondensationSpeciesMap(tuple(gas.name), ((2, 0), (0, 1))),
    )
    return Aerosol(
        particles=particles, gas=gas, environment=original.environment
    ), strategy


def test_colored_jacobian_matches_individual_column_reference():
    """Cross-material, gas and per-box derivatives agree with dense differences."""
    initial, strategy = mixture()
    owner = CondensationODE(initial, strategy, method="ros2")
    values = owner.solver.state
    dense = owner._jacobian(0, values)
    sparse = sparse_jacobian(owner, 0, values)
    npt.assert_allclose(sparse.toarray(), dense, rtol=2e-6, atol=1e-8)
    # The weighted gas+particle invariant annihilates each Jacobian column.
    for box, pairs in enumerate(owner._maps):
        for source, target in pairs:
            weights = np.zeros(len(values))
            for index, (b, slot, material) in enumerate(owner._mass_indices):
                if (b, material) == (box, target):
                    weights[index] = (
                        owner._weights[b][slot] * owner._scale[index]
                    )
            gas_index = owner._mass_count + owner._gas_indices.index(
                (box, int(source))
            )
            weights[gas_index] = owner._scale[gas_index]
            npt.assert_allclose(weights @ sparse, 0, atol=1e-13)


def test_sparse_ros2_matches_independent_stiff_linear_solution():
    """Sparse factorization retains ROS2 accuracy and closed inventory."""
    matrix = np.array([[-100.0, 0.2], [100.0, -0.2]])
    problem = replace(
        linear_problem(matrix), jacobian=lambda t, y: csc_matrix(matrix)
    )
    solver = AdaptiveSolver(
        problem, [1.0, 0.0], method="ros2", rtol=1e-6, atol=1e-10
    )
    while solver.time < 1:
        solver.step(1)
    npt.assert_allclose(
        solver.state, expm(matrix) @ [1.0, 0.0], rtol=1e-6, atol=1e-9
    )
    npt.assert_allclose(solver.state.sum(), 1.0, rtol=0, atol=1e-13)


def test_large_native_owner_uses_sparse_without_particle_squared_work(
    monkeypatch,
):
    """Rate calls scale with material and gas width, not particle count."""
    initial, strategy = mixture(24)
    owner = CondensationODE(initial, strategy, method="ros2")
    calls = 0
    rate = owner._rate

    def counted(*args):
        nonlocal calls
        calls += 1
        return rate(*args)

    monkeypatch.setattr(owner, "_rate", counted)
    jac = owner._jacobian(0, owner.solver.state)
    assert issparse(jac)
    assert calls == 1 + 2 * 2 + 2 * 4
    assert jac.nnz < 12 * len(owner.solver.state)
    owner.step(1e-3)
    assert owner.time > 0


def test_custom_nonlocal_activity_retains_cross_particle_derivatives():
    """Custom models may couple particles; coloring must not erase that term."""

    class CollectiveActivity(ActivityIdealMass):
        def activity(self, mass_concentration):
            local = super().activity(mass_concentration)
            return local + 0.1 * np.mean(local, axis=0)

    initial, strategy = mixture(24)
    strategy.activity_strategy = CollectiveActivity()
    owner = CondensationODE(initial, strategy, method="ros2")
    values = owner.solver.state
    jac = owner._jacobian(0, values)
    assert not issparse(jac)
    row = owner._mass_indices.index((0, 0, 0))
    column = owner._mass_indices.index((0, 1, 0))
    assert abs(jac[row, column]) > 1e-10
    delta = 1e-6
    plus, minus = values.copy(), values.copy()
    plus[column] += delta
    minus[column] -= delta
    expected = (owner._rate(0, plus) - owner._rate(0, minus)) / (2 * delta)
    npt.assert_allclose(jac[:, column], expected, rtol=3e-4, atol=1e-8)


def test_kappa_dry_start_and_wet_derivatives_match_dense_reference():
    """The water-activity boundary used by cloud examples retains cross terms."""
    from particula.particles import ActivityKappaParameterBuilder

    initial, strategy = mixture()
    strategy.activity_strategy = (
        ActivityKappaParameterBuilder()
        .set_kappa(np.array([0.5, 0.0]))
        .set_density(initial.particles.density, "kg/m^3")
        .set_molar_mass(np.array([0.18, 0.018]), "kg/mol")
        .set_water_index(1)
        .build()
    )
    for dry in (False, True):
        if dry:
            initial.particles.masses[:, :, 1] = 0
        owner = CondensationODE(initial, strategy, method="ros2")
        values = owner.solver.state
        npt.assert_allclose(
            sparse_jacobian(owner, 0, values).toarray(),
            owner._jacobian(0, values),
            rtol=2e-6,
            atol=1e-8,
        )


def test_sparse_adaptive_disk_continuation_is_exact(tmp_path):
    """A restored large owner recreates its sparse solver without lost history."""
    from particula.execution.cpu_restart import (
        checkpoint_condensation,
        load_cpu_checkpoint,
        resume_condensation,
        save_cpu_checkpoint,
    )
    from particula.execution.recording import run_until

    initial, strategy = mixture(24)
    owner = CondensationODE(initial, strategy, method="ros2", rtol=1e-5)
    run_until(owner, 0.01)
    path = tmp_path / "sparse.restart"
    save_cpu_checkpoint(checkpoint_condensation(owner), path)
    restored = resume_condensation(load_cpu_checkpoint(path), strategy)
    run_until(owner, 0.02)
    run_until(restored, 0.02)
    npt.assert_array_equal(owner.solver.state, restored.solver.state)
    assert owner.solver.statistics == restored.solver.statistics
    assert owner.solver.next_step == restored.solver.next_step


def test_dry_relaxation_derivative_does_not_depend_on_initial_wet_scale():
    """A wet-to-dry mass change must not enlarge the physical FD perturbation."""
    from particula.particles import ActivityKappaParameterBuilder

    initial, strategy = mixture()
    strategy.activity_strategy = (
        ActivityKappaParameterBuilder()
        .set_kappa(np.array([0.5, 0.0]))
        .set_density(initial.particles.density, "kg/m^3")
        .set_molar_mass(np.array([0.18, 0.018]), "kg/mol")
        .set_water_index(1)
        .build()
    )
    active = initial.particles.concentration > 0
    initial.particles.masses[..., 0][active] = 2e-22
    initial.particles.masses[..., 1][active] = 1e-12
    wet = CondensationODE(initial, strategy, method="ros2")
    accepted = wet.solver.state
    for index, (_, _, material) in enumerate(wet._mass_indices):
        if material == 1:
            accepted[index] = 1e-22 / wet._scale[index]
    current = wet._unpack(0, accepted)
    fresh = CondensationODE(current, strategy, method="ros2")
    expected = fresh._jacobian(0, fresh.solver.state)
    transform = fresh._scale / wet._scale
    expected = transform[:, None] * expected / transform[None, :]
    npt.assert_allclose(
        sparse_jacobian(wet, 0, accepted).toarray(),
        expected,
        rtol=2e-5,
        atol=1e-7,
    )
