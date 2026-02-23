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
