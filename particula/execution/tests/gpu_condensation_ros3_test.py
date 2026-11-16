"""Independent oracles for resident single-exchange ROS3 condensation."""

import numpy as np
import pytest
import warp as wp

from particula.execution import _condensation_ros3_kernels as kernels


@pytest.mark.parametrize("resolved", [False, True])
def test_arrowhead_solve_matches_independent_dense_matrix(device, resolved):
    """Check mapped coupling, scaled budgets and residuals across stiffness."""
    rng = np.random.default_rng(321)
    boxes, slots, materials, gases = 2, 17, 3, 5
    target, source = 2, 1
    count = rng.uniform(1, 8, (boxes, slots))
    count[:, ::4] = 0
    volume = np.array([0.25, 7.0])
    size = boxes * slots * materials + boxes * gases
    mass_size = boxes * slots * materials
    scale = np.exp(rng.uniform(-5, 5, size))
    a = -np.exp(rng.uniform(-5, 12, (boxes, slots)))
    b = np.exp(rng.uniform(-5, 5, (boxes, slots)))
    a[count == 0] = 0
    b[count == 0] = 0
    jac = np.zeros((size, size))
    weights = np.zeros((boxes, size))
    for box in range(boxes):
        gi = mass_size + box * gases + source
        weights[box, gi] = 1
        for slot in range(slots):
            pi = (box * slots + slot) * materials + target
            weight = count[box, slot] * scale[pi] / scale[gi]
            if resolved:
                weight /= volume[box]
            weights[box, pi] = weight
            jac[pi, pi] = a[box, slot]
            jac[pi, gi] = b[box, slot]
            jac[gi, pi] = -weight * a[box, slot]
            jac[gi, gi] -= weight * b[box, slot]
    arrays = [
        wp.array(x, dtype=wp.float64, device=device)
        for x in (a, b, count, volume, scale)
    ]
    for q in (0.01, 1.0, 100.0, 1e6):
        rhs = rng.normal(size=size)
        rhs_gpu = wp.array(rhs, dtype=wp.float64, device=device)
        result = wp.zeros(size, dtype=wp.float64, device=device)
        invalid = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            kernels.solve_gas,
            dim=boxes,
            inputs=[
                *arrays,
                rhs_gpu,
                materials,
                gases,
                target,
                source,
                int(resolved),
                q,
                result,
                invalid,
            ],
            device=device,
        )
        wp.launch(
            kernels.solve_remaining,
            dim=size,
            inputs=[
                arrays[0],
                arrays[1],
                rhs_gpu,
                materials,
                gases,
                target,
                source,
                q,
                result,
                invalid,
            ],
            device=device,
        )
        actual = result.numpy()
        assert invalid.numpy()[0] == 0
        matrix = q * np.eye(size) - jac
        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
        residual = np.linalg.norm(matrix @ actual - rhs, np.inf)
        normalized = residual / (
            np.linalg.norm(matrix, np.inf) * np.linalg.norm(actual, np.inf)
            + np.linalg.norm(rhs, np.inf)
        )
        assert normalized < 2e-15, normalized
        np.testing.assert_allclose(
            weights @ actual, weights @ rhs / q, rtol=2e-10, atol=2e-8
        )


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    """Separate Warp CPU evidence from actual CUDA qualification."""
    if request.param != "cpu" and not wp.is_cuda_available():
        pytest.skip("actual CUDA unavailable")
    return request.param


def test_ros3_growth_refines_against_inverse_radius_quadrature(device):
    """Float64 coefficients retain refinement far below the float32 floor."""
    from particula.execution.gpu_condensation_ode import GPUCondensationODE
    from particula.execution.tests.condensation_ode_test import case
    from particula.execution.tests.gpu_condensation_ode_test import (
        reference_time,
    )

    errors = []
    for tolerance in (1e-4, 1e-6, 1e-8):
        initial, strategy = case(2)
        owner = GPUCondensationODE(
            initial,
            strategy,
            method="ros3",
            device=device,
            rtol=tolerance,
            particle_atol=1e-26,
            gas_atol=1e-16,
        )
        bound = reference_time()
        while owner.time < bound:
            owner.step(bound)
        final = owner.snapshot()
        errors.append(
            float(np.max(abs(final.particles.radii[:, 0] / 1.2e-6 - 1)))
        )
        before = (
            initial.particles.species_mass_density[:, 0]
            + initial.gas.concentration[:, 1]
        )
        after = (
            final.particles.species_mass_density[:, 0]
            + final.gas.concentration[:, 1]
        )
        np.testing.assert_allclose(after, before, rtol=3e-13)
        np.testing.assert_array_equal(
            final.gas.concentration[:, 0], initial.gas.concentration[:, 0]
        )
        np.testing.assert_array_equal(final.particles.masses[:, 1], 0)
    assert errors[1] < errors[0] / 10
    assert errors[2] < errors[1] / 10
    assert errors[2] < 1e-9


@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_unchanged_native_coordinates_remain_exact(device, method):
    """Observation and stepping cannot change inert gas or dry material bits."""
    from particula.dynamics.condensation.species_mapping import (
        CondensationSpeciesMap,
    )
    from particula.execution.gpu_condensation_ode import GPUCondensationODE
    from particula.execution.tests.condensation_sparse_test import mixture
    from particula.particles.activity_strategies import ActivityIdealMolar
    from particula.particles.surface_strategies import SurfaceStrategyVolume

    initial, strategy = mixture(24)
    strategy.activity_strategy = ActivityIdealMolar(molar_mass=[0.18, 0.018])
    strategy.surface_strategy = SurfaceStrategyVolume(density=1000.0)
    rng = np.random.default_rng(43)
    initial.particles.masses[:, :-1] = 10 ** rng.uniform(
        -19, -13, initial.particles.masses[:, :-1].shape
    )
    strategy.species_map = CondensationSpeciesMap(
        tuple(initial.gas.name), ((2, 0),)
    )
    owner = GPUCondensationODE(initial, strategy, device=device, method=method)
    np.testing.assert_array_equal(
        owner.snapshot().particles.masses, initial.particles.masses
    )
    np.testing.assert_array_equal(
        owner.snapshot().gas.concentration, initial.gas.concentration
    )
    while owner.time < 1e-4:
        owner.step(1e-4)
    result = owner.snapshot()
    np.testing.assert_array_equal(
        result.particles.masses[:, :, 1], initial.particles.masses[:, :, 1]
    )
    np.testing.assert_array_equal(
        result.gas.concentration[:, :2], initial.gas.concentration[:, :2]
    )
    assert np.any(
        result.particles.masses[:, :, 0] != initial.particles.masses[:, :, 0]
    )


def test_ros3_multiple_exchange_rejects_before_device_access(monkeypatch):
    """Do not silently ignore cross-material derivatives or fall back to CPU."""
    from particula.execution.gpu_condensation_ode import GPUCondensationODE
    from particula.execution.tests.condensation_sparse_test import mixture
    from particula.gas.vapor_pressure_strategies import (
        ConstantVaporPressureStrategy,
    )
    from particula.particles.surface_strategies import SurfaceStrategyVolume

    initial, strategy = mixture()
    strategy.surface_strategy = SurfaceStrategyVolume(density=1000.0)
    strategy.vapor_pressure_strategy = ConstantVaporPressureStrategy(0.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported exchange reached the device")

    monkeypatch.setattr(wp, "get_device", forbidden)
    with pytest.raises(ValueError, match="one exchanging material"):
        GPUCondensationODE(initial, strategy, device="cuda:0", method="ros3")


def test_old_gpu_numerical_schema_rejects_before_upload(monkeypatch):
    """Do not reinterpret old coordinates under changed unscaling semantics."""
    from dataclasses import replace

    from particula.execution.gpu_condensation_ode import GPUCondensationODE
    from particula.execution.gpu_restart import (
        checkpoint_gpu_condensation,
        resume_gpu_condensation,
    )
    from particula.execution.tests.condensation_ode_test import case

    initial, strategy = case()
    owner = GPUCondensationODE(initial, strategy, device="cpu", method="ros3")
    saved = replace(checkpoint_gpu_condensation(owner), schema_version=2)

    def forbidden(*args, **kwargs):
        raise AssertionError("old schema reached the device")

    monkeypatch.setattr(wp, "get_device", forbidden)
    with pytest.raises(ValueError, match="schema/runtime"):
        resume_gpu_condensation(saved, strategy, device="cpu")
