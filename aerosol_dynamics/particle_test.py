import pytest
from aerosol_dynamics import particle
from . import ureg

particle1 = particle.Particle(name='particle1',
                              radius=1.0e-9 * ureg.m,
                              density=1.0 * ureg.kg / ureg.m**3,
                              charge=1.0 * ureg.C,)

particle2 = particle.Particle(name='particle2',
                              radius=1.0e-7 * ureg.m,
                              density=1.8 * ureg.kg / ureg.m**3,
                              charge=1.0 * ureg.C,)

particle3 = particle.Particle(name='particle3',
                              radius=1.0e-7 * ureg.lb,
                              density=1 * ureg.kg / ureg.m**3,
                              charge=3,)

def knudsen_number_test():
    assert particle1.knudsen_number() == pytest.approx(65.)
    assert particle1.knudsen_number().check(['None'])
    assert particle2.knudsen_number() == pytest.approx(0.65)
    with pytest.raises(AssertionError):
        assert particle3.knudsen_number() == pytest.approx(0.65)
        assert particle3.knudsen_number().check('[None]')
knudsen_number_test()

def slip_correction_factor_test():
    assert particle1.slip_correction_factor() == pytest.approx(108.268702)
    assert particle2.slip_correction_factor() == pytest.approx(1.864914)
    assert particle1.slip_correction_factor().check(['None'])
slip_correction_factor_test()

def friction_factor_test():
    assert particle1.friction_factor() == pytest.approx(3.180803e-15)
    assert particle2.friction_factor() == pytest.approx(1.846633e-11) # Issue with 6th decimal place and rounding.
    assert particle1.friction_factor().check('[mass]/[time]')
friction_factor_test()

def reduced_mass_test():
    reduced_mass_1_2 = particle1.reduced_mass(particle2)
    assert reduced_mass_1_2 == pytest.approx(4.18879e-27)
    assert reduced_mass_1_2.check('[mass]')
reduced_mass_test()

def reduced_friction_factor_test():
    reduced_friction_factor_1_2 = particle1.reduced_friction_factor(particle2)
    assert reduced_friction_factor_1_2 == pytest.approx(1.833335)
    assert reduced_friction_factor_1_2.check('[None]')
