"""Test suites for particle class.
"""

import numpy as np
import pytest
from particula import u
from particula.aerosol_dynamics import environment, particle

small_particle = particle.Particle(
    name="small_particle",
    radius=1.0e-9 * u.m,
    density=1.0 * u.kg / u.m**3,  # need fixing, add e3
    charge=1,
)

large_particle = particle.Particle(
    name="large_particle",
    radius=1.0e-7 * u.m,
    density=1.8 * u.kg / u.m**3,  # need fixing, add e3
    charge=1,
)

standard_environment = environment.Environment(
    temperature=298 * u.K,
    pressure=101325 * u.Pa,
)

negative_ion = particle.Particle(
    name="negative_ion",
    radius=0.5e-9 * u.m,
    density=1.84e3 * u.kg / u.m**3,
    charge=-1,
)

positive_particle = particle.Particle(
    name="positive_particle",
    radius=3e-9 * u.m,
    density=1.7e3 * u.kg / u.m**3,
    charge=1,
)

standard_environment_ip = environment.Environment(
    temperature=300 * u.degK,
    pressure=101325 * u.Pa,
)


def test_getters():
    """Test that the getters work.
    """

    assert small_particle.name() == "small_particle"
    assert small_particle.radius() == 1.0e-9 * u.m
    assert small_particle.mass() == (
        4.0/3.0 * np.pi * (1.0e-9 ** 3 * u.m ** 3) * (
            1.0 * u.kg / u.m**3
        )
    )
    assert small_particle.charge() == 1.0


def test_knudsen_number():
    """Test that the knudsen number is calculated correctly.
    """

    assert small_particle.knudsen_number(
        standard_environment
    ) == pytest.approx(66.44, rel=1e-3)
    assert small_particle.knudsen_number(
        standard_environment
    ).check(["None"])
    assert large_particle.knudsen_number(
        standard_environment
    ) == pytest.approx(0.6644, rel=1e-3)


def test_slip_correction_factor():
    """"Test that the slip correction factor is calculated correctly.
    """

    assert small_particle.slip_correction_factor(
        standard_environment
    ) == pytest.approx(110.7, rel=1e-3)

    assert large_particle.slip_correction_factor(
        standard_environment
    ) == pytest.approx(1.886, rel=1e-3)

    assert small_particle.slip_correction_factor(
        standard_environment
    ).check(["None"])


def test_friction_factor():
    """Test that the friction factor is calculated correctly.
    """

    assert small_particle.friction_factor(
        standard_environment
    ).magnitude == pytest.approx(3.181e-15, rel=1e-3)

    assert large_particle.friction_factor(
        standard_environment
    ).magnitude == pytest.approx(1.84e-11)

    assert small_particle.friction_factor(
        standard_environment
    ).check("[mass]/[time]")


def test_reduced_mass():
    """Test that the reduced mass is calculated correctly.
    """

    reduced_mass_1_2 = small_particle.reduced_mass(large_particle)
    assert reduced_mass_1_2 == pytest.approx(4.189e-27, rel=1e-3)
    assert reduced_mass_1_2.check("[mass]")


def test_reduced_friction_factor():
    """Test that the reduced friction factor is calculated correctly.
    """

    reduced_friction_factor_1_2 = small_particle.reduced_friction_factor(
        large_particle, standard_environment)
    assert reduced_friction_factor_1_2 == pytest.approx(3.18e-15)
    assert reduced_friction_factor_1_2.check("[mass]/[time]")


def test_dimensionless_coagulation_kernel_parameterized():
    """Test that the paramaterized dimensionless coagulation kernel
    is calculated correctly.
    """

    assert small_particle.dimensionless_coagulation_kernel_parameterized(
        large_particle, standard_environment
    ) == pytest.approx(0.003, rel=10)
    assert small_particle.dimensionless_coagulation_kernel_parameterized(
        large_particle, standard_environment
    ).check(["None"])


def test_dimensioned_coagulation_kernel():
    """Tests dimensioned coagulation kernel is calculated correctly.
    """

    assert small_particle.dimensioned_coagulation_kernel(
        large_particle, standard_environment
    ) == pytest.approx(2.738e-10 * u.m**3 / u.s, rel=10)
    assert small_particle.dimensioned_coagulation_kernel(
        large_particle, standard_environment
    ).check("[length]**3/[time]")

    # FROM PHYSICS & FIRST PRINCIPLES:
    # when
    # negative ion with 1 charge and size 0.5e-9 m
    # with
    # positive particle with 1 charge and size 3e-9 m
    # then:
    # coagulation should be ~1e-12 m^3/s
    # (within 2x, or at most an order of magnitude)
    # (conditions assumed ~300 K, ~1 atm, but don't matter much)

    assert negative_ion.dimensioned_coagulation_kernel(
        positive_particle, standard_environment_ip
    ) == pytest.approx(1e-12 * u.m**3 / u.s, rel=10)
    # rel=10 means an order of magnitude
    assert negative_ion.dimensioned_coagulation_kernel(
        positive_particle, standard_environment_ip
    ).check("[length]**3/[time]")
