"""Regression tests for charged kernel NaN issues."""

import numpy as np
import pytest
from particula.dynamics.coagulation import charged_dimensional_kernel as cdk
from particula.dynamics.coagulation.particle_resolved_step import (
    particle_resolved_method,
)
from particula.particles.properties import coulomb_enhancement

TEMPERATURE = 298.15
PRESSURE = 101325.0


def _build_kernel(
    kernel_radius: np.ndarray,
    kernel_mass: np.ndarray,
    kernel_charge: np.ndarray,
) -> np.ndarray:
    """Build a charged coagulation kernel for representative bins."""
    return cdk.get_hard_sphere_kernel_via_system_state(
        particle_radius=kernel_radius,
        particle_mass=kernel_mass,
        particle_charge=kernel_charge,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
    )


def _run_coagulation_step(
    particle_radius: np.ndarray,
    kernel_radius: np.ndarray,
    kernel_mass: np.ndarray,
    kernel_charge: np.ndarray,
    volume: float,
    time_step: float,
    seed: int,
) -> np.ndarray:
    """Run a deterministic particle-resolved coagulation step."""
    kernel = _build_kernel(kernel_radius, kernel_mass, kernel_charge)
    random_generator = np.random.default_rng(seed=seed)
    return particle_resolved_method.get_particle_resolved_coagulation_step(
        particle_radius=particle_radius,
        kernel=kernel,
        kernel_radius=kernel_radius,
        volume=volume,
        time_step=time_step,
        random_generator=random_generator,
    )


def test_monodisperse_opposite_sign_kernel_finite() -> None:
    """Opposite-sign monodisperse kernels should be finite and positive."""
    particle_radius = np.full(6, 50e-9)
    particle_mass = np.full(6, 1e-18)
    particle_charge = np.array([-6.0, 6.0, -6.0, 6.0, -6.0, 6.0])

    kernel = cdk.get_hard_sphere_kernel_via_system_state(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
    )

    assert np.isfinite(kernel).all()
    assert np.any(kernel > 0)


def test_monodisperse_same_sign_nan_regression() -> None:
    """Same-sign monodisperse kernels should be finite and near zero."""
    particle_radius = np.full(6, 50e-9)
    particle_mass = np.full(6, 1e-18)
    particle_charge = np.full(6, -30.0)

    kernel = cdk.get_hard_sphere_kernel_via_system_state(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
    )

    assert np.isfinite(kernel).all()
    assert np.allclose(kernel, 0.0, atol=1e-30)


@pytest.mark.parametrize(
    "kernel_func",
    [
        cdk.get_hard_sphere_kernel_via_system_state,
        cdk.get_coulomb_kernel_dyachkov2007_via_system_state,
        cdk.get_coulomb_kernel_gatti2008_via_system_state,
        cdk.get_coulomb_kernel_gopalakrishnan2012_via_system_state,
        cdk.get_coulomb_kernel_chahl2019_via_system_state,
    ],
)
def test_full_kernel_matrix_no_nan_inf(kernel_func) -> None:
    """Kernel matrices should remain finite for mixed-charge populations."""
    particle_radius = np.array([30e-9, 50e-9, 80e-9])
    particle_mass = np.array([1e-19, 1e-18, 5e-18])
    particle_charge = np.array([-3.0, 0.0, 3.0])

    kernel = kernel_func(
        particle_radius=particle_radius,
        particle_mass=particle_mass,
        particle_charge=particle_charge,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
    )

    assert np.isfinite(kernel).all()


def test_coulomb_kinetic_limit_extreme_negative() -> None:
    """Extreme negative potentials should remain finite and non-negative."""
    kinetic_limit = coulomb_enhancement.get_coulomb_kinetic_limit(-200)
    assert np.isfinite(kinetic_limit)
    assert kinetic_limit >= 0


def test_particle_resolved_step_opposite_sign_no_crash() -> None:
    """Particle-resolved step runs for opposite-sign monodisperse cases."""
    kernel_radius = np.linspace(30e-9, 70e-9, 10)
    kernel_mass = np.full_like(kernel_radius, 1e-18)
    kernel_charge = np.where(
        np.arange(kernel_radius.size) % 2 == 0,
        -6.0,
        6.0,
    )
    kernel = cdk.get_hard_sphere_kernel_via_system_state(
        particle_radius=kernel_radius,
        particle_mass=kernel_mass,
        particle_charge=kernel_charge,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
    )

    particle_radius = np.full(6, 50e-9)
    random_generator = np.random.default_rng(seed=1)

    loss_gain_index = (
        particle_resolved_method.get_particle_resolved_coagulation_step(
            particle_radius=particle_radius,
            kernel=kernel,
            kernel_radius=kernel_radius,
            volume=1e-6,
            time_step=1.0,
            random_generator=random_generator,
        )
    )

    assert loss_gain_index.shape[1] == 2


def test_particle_resolved_step_same_sign_zero_mergers() -> None:
    """Same-sign repulsion should produce zero coagulation events."""
    kernel_radius = np.linspace(30e-9, 70e-9, 10)
    kernel_mass = np.full_like(kernel_radius, 1e-18)
    kernel_charge = np.full_like(kernel_radius, -30.0)
    kernel = cdk.get_hard_sphere_kernel_via_system_state(
        particle_radius=kernel_radius,
        particle_mass=kernel_mass,
        particle_charge=kernel_charge,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
    )

    particle_radius = np.full(6, 50e-9)
    random_generator = np.random.default_rng(seed=2)

    loss_gain_index = (
        particle_resolved_method.get_particle_resolved_coagulation_step(
            particle_radius=particle_radius,
            kernel=kernel,
            kernel_radius=kernel_radius,
            volume=1e-6,
            time_step=1.0,
            random_generator=random_generator,
        )
    )

    assert loss_gain_index.size == 0


def test_spurious_calcite_calcite_with_ions() -> None:
    """Same-sign calcite should not merge when small ions are present."""
    particle_radius = np.concatenate(
        [np.full(100, 50e-9), np.full(6, 5e-9)]
    ).astype(np.float64)
    particle_charge = np.concatenate([np.full(100, -6.0), np.full(6, 1.0)])

    kernel_radius = np.array([5e-9, 50e-9], dtype=np.float64)
    kernel_mass = np.array([1e-21, 1e-18], dtype=np.float64)
    kernel_charge = np.array([1.0, -6.0], dtype=np.float64)

    loss_gain_index = _run_coagulation_step(
        particle_radius=particle_radius,
        kernel_radius=kernel_radius,
        kernel_mass=kernel_mass,
        kernel_charge=kernel_charge,
        volume=1e-12,
        time_step=0.05,
        seed=10,
    )

    calcite_mask = particle_charge < 0
    ion_mask = particle_charge > 0
    small_index = loss_gain_index[:, 0]
    large_index = loss_gain_index[:, 1]
    calcite_calcite = np.sum(
        calcite_mask[small_index] & calcite_mask[large_index]
    )
    ion_calcite = np.sum(
        (ion_mask[small_index] & calcite_mask[large_index])
        | (calcite_mask[small_index] & ion_mask[large_index])
    )

    assert calcite_calcite == 0
    assert ion_calcite > 0


def test_calcite_only_zero_mergers() -> None:
    """Calcite-only populations should not coagulate for same-sign charge."""
    particle_radius = np.full(200, 50e-9, dtype=np.float64)

    kernel_radius = np.array([50e-9, 55e-9], dtype=np.float64)
    kernel_mass = np.array([1e-18, 1.2e-18], dtype=np.float64)
    kernel_charge = np.array([-6.0, -6.0], dtype=np.float64)

    loss_gain_index = _run_coagulation_step(
        particle_radius=particle_radius,
        kernel_radius=kernel_radius,
        kernel_mass=kernel_mass,
        kernel_charge=kernel_charge,
        volume=1e-12,
        time_step=0.1,
        seed=11,
    )

    assert loss_gain_index.size == 0


def test_opposite_sign_attraction_preserved() -> None:
    """Opposite-sign populations should still show coagulation events."""
    particle_radius = np.concatenate(
        [np.full(40, 30e-9), np.full(40, 50e-9)]
    ).astype(np.float64)
    particle_charge = np.concatenate([np.full(40, -3.0), np.full(40, 3.0)])

    kernel_radius = np.array([30e-9, 50e-9], dtype=np.float64)
    kernel_mass = np.array([1e-19, 5e-19], dtype=np.float64)
    kernel_charge = np.array([-3.0, 3.0], dtype=np.float64)

    loss_gain_index = _run_coagulation_step(
        particle_radius=particle_radius,
        kernel_radius=kernel_radius,
        kernel_mass=kernel_mass,
        kernel_charge=kernel_charge,
        volume=1e-12,
        time_step=0.1,
        seed=12,
    )

    assert loss_gain_index.size > 0
    small_index = loss_gain_index[:, 0]
    large_index = loss_gain_index[:, 1]
    opposite_sign = np.sum(
        np.sign(particle_charge[small_index])
        != np.sign(particle_charge[large_index])
    )
    assert opposite_sign > 0


def test_mass_charge_conservation() -> None:
    """Mass and charge should be conserved in coagulation events."""
    particle_radius = np.concatenate(
        [np.full(40, 30e-9), np.full(40, 50e-9)]
    ).astype(np.float64)
    particle_charge = np.concatenate([np.full(40, -3.0), np.full(40, 3.0)])

    kernel_radius = np.array([30e-9, 50e-9], dtype=np.float64)
    kernel_mass = np.array([1e-19, 5e-19], dtype=np.float64)
    kernel_charge = np.array([-3.0, 3.0], dtype=np.float64)

    loss_gain_index = _run_coagulation_step(
        particle_radius=particle_radius,
        kernel_radius=kernel_radius,
        kernel_mass=kernel_mass,
        kernel_charge=kernel_charge,
        volume=1e-12,
        time_step=0.1,
        seed=13,
    )

    assert loss_gain_index.size > 0
    initial_volume = np.sum(particle_radius**3)
    loss_gain_index = loss_gain_index.astype(np.int64)
    updated_radius, _, _ = (
        particle_resolved_method.get_particle_resolved_update_step(
            particle_radius.copy(),
            np.zeros_like(particle_radius),
            np.zeros_like(particle_radius),
            loss_gain_index[:, 0],
            loss_gain_index[:, 1],
        )
    )
    final_volume = np.sum(updated_radius**3)
    np.testing.assert_allclose(initial_volume, final_volume, rtol=1e-12)

    charge_after = particle_charge.copy()
    for small_index, large_index in loss_gain_index:
        charge_after[large_index] += charge_after[small_index]
        charge_after[small_index] = 0.0
    np.testing.assert_allclose(
        np.sum(charge_after), np.sum(particle_charge), rtol=1e-12
    )
