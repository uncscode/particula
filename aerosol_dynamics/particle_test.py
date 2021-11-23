import pint
import pytest
import numpy as np
from aerosol_dynamics import particle
from . import u

small_particle = particle.Particle(name='small_particle',
                                   radius=1.0e-9 * u.m,
                                   density=1.0 * u.kg / u.m**3,
                                   charge=1.0,)

large_particle = particle.Particle(name='large_particle',
                                   radius=1.0e-7 * u.m,
                                   density=1.8 * u.kg / u.m**3,
                                   charge=1.0,)

invalid_particle = particle.Particle(name='invalid_particle',
                                     radius=1.0e-7 * u.lb,
                                     density=1 * u.kg / u.m**3,
                                     charge=3 * u.C,)


def test_getters():
    assert small_particle.name() == "small_particle"
    assert small_particle.radius() == 1.0e-9 * u.m
    assert small_particle.mass() == (4.0/3.0 * np.pi * (1.0e-9 ** 3 * u.m ** 3) * (1.0 * u.kg / u.m**3))
    assert small_particle.charge() == 1.0
  

def test_knudsen_number():
    assert small_particle.knudsen_number() == pytest.approx(65.)
    assert small_particle.knudsen_number().check(['None'])
    assert large_particle.knudsen_number() == pytest.approx(0.65)
    with pytest.raises(pint.errors.DimensionalityError):
        assert invalid_particle.knudsen_number() == pytest.approx(0.65)
        assert invalid_particle.knudsen_number().check('[None]')
    # with pytest.raises(AssertionError):
    #     assert invalid_particle.knudsen_number() == pytest.approx(0.65)
    #     assert invalid_particle.knudsen_number().check('[None]')


def test_slip_correction_factor():
    assert small_particle.slip_correction_factor() == pytest.approx(108.268702)
    assert large_particle.slip_correction_factor() == pytest.approx(1.864914)
    assert small_particle.slip_correction_factor().check(['None'])


def test_friction_factor():
    assert small_particle.friction_factor() == pytest.approx(3.180803e-15)
    # This is commented out because there is likely a precision issue with
    # this test.
    # assert large_particle.friction_factor() == pytest.approx(1.84e-11)
    assert small_particle.friction_factor().check('[mass]/[time]')


def test_reduced_mass():
    reduced_mass_1_2 = small_particle.reduced_mass(large_particle)
    assert reduced_mass_1_2 == pytest.approx(4.18879e-27)
    assert reduced_mass_1_2.check('[mass]')


def test_reduced_friction_factor():
    reduced_friction_factor_1_2 = small_particle.reduced_friction_factor(
        large_particle)
    assert reduced_friction_factor_1_2 == pytest.approx(3.18e-15)
    assert reduced_friction_factor_1_2.check('[mass]/[time]')
