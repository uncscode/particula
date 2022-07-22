"""Test suites for particle class.
"""

import numpy as np
import pytest
from particula import environment, particle, u

small_particle = particle.Particle(
    particle_radius=1.0e-9 * u.m,
    particle_density=1.0 * u.kg / u.m**3,  # need fixing, add e3
    particle_charge=1,
)

large_particle = particle.Particle(
    name="large_particle",
    particle_radius=1.0e-7 * u.m,
    particle_density=1.8 * u.kg / u.m**3,  # need fixing, add e3
    particle_charge=1,
)

# invalid_particle = particle.Particle(
#     name="invalid_particle",
#     particle_radius=1.0e-7 * u.lb,
#     particle_density=1 * u.kg / u.m**3,
#     particle_charge=3 * u.C,
# )

standard_environment = environment.Environment(
    temperature=298 * u.K,
    pressure=101325 * u.Pa,
)

negative_ion = particle.Particle(
    name="negative_ion",
    particle_radius=0.5e-9 * u.m,
    particle_density=1.84e3 * u.kg / u.m**3,
    particle_charge=-1,
)

positive_particle = particle.Particle(
    name="positive_particle",
    particle_radius=3e-9 * u.m,
    particle_particle_density=1.7e3 * u.kg / u.m**3,
    particle_charge=1,
)

standard_environment_ip = environment.Environment(
    temperature=300 * u.K,
    pressure=101325 * u.Pa,
)

single_mode = {
    "mode": 500e-9,  # 500 nm median
    "nbins": 1000,  # 1000 bins
    "nparticles": 1e6,  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": 1.5,  # relatively narrow
    "spacing": "linspace",  # bin spacing,
}

multi_mode = {
    "mode": [500e-9, 500e-9],  # 500 nm median
    "nbins": 1000,  # 1000 bins
    "nparticles": [1e6, 1e6],  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": [1.2, 1.2],  # relatively narrow
    "spacing": "linspace",  # bin spacing,
}


def test_particle_distribution():
    """ Test that the particle distribution is calculated
        for single mode and multi mode.
    """
    def pdf_total(radius, pdf_distribution):
        return np.trapz(y=pdf_distribution, x=radius)

    # single mode
    particle_distribution_1 = particle.ParticleDistribution(
        **single_mode
    )
    total_number = pdf_total(
        particle_distribution_1.pre_radius().m,
        particle_distribution_1.pre_distribution().m,
    )
    assert total_number == pytest.approx(1e12, rel=1e10)

    # multi mode
    particle_distribution_2 = particle.ParticleDistribution(
        **multi_mode
    )
    total_number2 = pdf_total(
        particle_distribution_2.pre_radius().m,
        particle_distribution_2.pre_distribution().m
    )
    assert total_number2 == pytest.approx(2e12, rel=1e10)


def test_multi_mode_bins():
    """ testing the multi mode stuff for bins
    """

    assert particle.Particle(nbins=1000).particle_radius.shape == (1000,)
    assert particle.Particle(nbins=[2000]).particle_radius.shape == (2000,)
    assert particle.Particle(
        nbins=[1000, 2000]).particle_radius.shape == (3000,)


def test_getters():
    """ Test that the getters work.
    """

    assert small_particle.particle_radius == 1.0e-9 * u.m
    assert small_particle.particle_mass() == (
        4.0/3.0 * np.pi * (1.0e-9 ** 3 * u.m ** 3) * (
            1.0 * u.kg / u.m**3
        )
    )
    assert small_particle.particle_charge == 1.0


def test_individual_shapes():
    """ test individual shapes
    """
    assert particle.Particle(
        particle_radius=[1, 2]
    ).particle_radius.m.shape == (2,)
    assert particle.Particle(
        particle_radius=[1, 2]
    ).particle_mass().m.shape == (2,)
    assert particle.Particle(
        particle_radius=[1, 2]
    ).knudsen_number().m.shape == (2,)
    assert particle.Particle(
        particle_radius=[1, 2]).friction_factor().m.shape == (2,)


def test_knudsen_number():
    """ Test that the knudsen number is calculated correctly.
    """

    assert small_particle.knudsen_number() == pytest.approx(66.44, rel=1e-3)
    assert small_particle.knudsen_number().check(["None"])
    assert large_particle.knudsen_number() == pytest.approx(0.6644, rel=1e-3)
    # with pytest.raises(pint.errors.DimensionalityError):
    #     assert invalid_particle.knudsen_number() == pytest.approx(0.65)
    #     assert invalid_particle.knudsen_number().check("[None]")
    # # with pytest.raises(AssertionError):
    # #     assert invalid_particle.knudsen_number() == pytest.approx(0.65)
    # #     assert invalid_particle.knudsen_number().check("[None]")


def test_slip_correction_factor():
    """ Test that the slip correction factor is calculated correctly.
    """

    assert (
        small_particle.slip_correction_factor() ==
        pytest.approx(110.7, rel=1e-3)
    )

    assert (
        large_particle.slip_correction_factor() ==
        pytest.approx(1.886, rel=1e-3)
    )

    assert small_particle.slip_correction_factor().check(["None"])


def test_friction_factor():
    """ Test that the friction factor is calculated correctly.
    """

    assert small_particle.friction_factor(
    ).magnitude == pytest.approx(3.181e-15, rel=1e-3)

    assert (
        large_particle.friction_factor().magnitude ==
        pytest.approx(1.84e-11)
    )

    assert small_particle.friction_factor().check("[mass]/[time]")


def test_multiple_shapes():
    """ testing multiple shapes when two dists exist
    """

    assert particle.Particle(
        particle_radius=[1, 2],
    ).reduced_mass(
        other=particle.Particle(particle_radius=[1])
    ).m.shape == (2, 1)
    assert particle.Particle(
        particle_radius=[1],
    ).reduced_mass(
        other=particle.Particle(particle_radius=[1, 2])
    ).m.shape == (1, 2)

    assert particle.Particle(
        particle_radius=[1, 2],
    ).reduced_friction_factor(
        other=particle.Particle(particle_radius=[1])
    ).m.shape == (2, 1)
    assert particle.Particle(
        particle_radius=[1],
    ).reduced_friction_factor(
        other=particle.Particle(particle_radius=[1, 2])
    ).m.shape == (1, 2)

    assert particle.Particle(
        particle_radius=[1, 2],
    ).coulomb_potential_ratio(
        other=particle.Particle(particle_radius=[1])
    ).m.shape == (2, 1)
    assert particle.Particle(
        particle_radius=[1],
    ).coulomb_potential_ratio(
        other=particle.Particle(particle_radius=[1, 2])
    ).m.shape == (1, 2)

    assert particle.Particle(
        particle_radius=[1, 2],
    ).diffusive_knudsen_number(
        other=particle.Particle(particle_radius=[1])
    ).m.shape == (2, 1)
    assert particle.Particle(
        particle_radius=[1],
    ).diffusive_knudsen_number(
        other=particle.Particle(particle_radius=[1, 2])
    ).m.shape == (1, 2)

    assert particle.Particle(
        particle_radius=[1, 2],
    ).dimensionless_coagulation(
        other=particle.Particle(particle_radius=[1])
    ).m.shape == (2, 1)
    assert particle.Particle(
        particle_radius=[1],
    ).dimensionless_coagulation(
        other=particle.Particle(particle_radius=[1, 2])
    ).m.shape == (1, 2)

    assert particle.Particle(
        particle_radius=[1, 2],
    ).coagulation(
        other=particle.Particle(particle_radius=[1])
    ).m.shape == (2, 1)
    assert particle.Particle(
        particle_radius=[1],
    ).coagulation(
        other=particle.Particle(particle_radius=[1, 2])
    ).m.shape == (1, 2)


def test_big_shapes():
    """ testing big shapes
    """
    assert particle.Particle(
        particle_radius=np.linspace(1, 100, 100),
    ).dimensionless_coagulation(
        other=particle.Particle(particle_radius=[1])
    ).m.shape == (100, 1)
    assert particle.Particle(
        particle_radius=np.linspace(1, 100, 1000),
    ).dimensionless_coagulation(
        other=particle.Particle(particle_radius=np.linspace(1, 100, 500))
    ).m.shape == (1000, 500)


def test_reduced_mass():
    """ Test that the reduced mass is calculated correctly.
    """

    reduced_mass_1_2 = small_particle.reduced_mass(large_particle)
    assert reduced_mass_1_2 == pytest.approx(4.189e-27, rel=1e-3)
    assert reduced_mass_1_2.check("[mass]")


def test_reduced_friction_factor():
    """ Test that the reduced friction factor is calculated correctly.
    """

    reduced_friction_factor_1_2 = small_particle.reduced_friction_factor(
        large_particle,)
    assert reduced_friction_factor_1_2 == pytest.approx(3.18e-15)
    assert reduced_friction_factor_1_2.check("[mass]/[time]")


def test_coulomb_enh():
    """ testing coulomb enh
    """
    assert (
        small_particle.coulomb_potential_ratio(large_particle).u
        == u.dimensionless
    )
# def test_dimensionless_coagulation_kernel_parameterized():
#     """Test that the paramaterized dimensionless coagulation kernel
#     is calculated correctly.
#     """

#     assert small_particle.dimensionless_coagulation_kernel_parameterized(
#         large_particle,
#     ) == pytest.approx(0.003, rel=10)
#     assert small_particle.dimensionless_coagulation_kernel_parameterized(
#         large_particle,
#     ).check(["None"])


# def test_dimensioned_coagulation_kernel():
#     """Tests dimensioned coagulation kernel is calculated correctly.
#     """

#     assert small_particle.dimensioned_coagulation_kernel(
#         large_particle,
#     ) == pytest.approx(2.738e-10 * u.m**3 / u.s, rel=10)
#     assert small_particle.dimensioned_coagulation_kernel(
#         large_particle,
#     ).check("[length]**3/[time]")

#     # FROM PHYSICS & FIRST PRINCIPLES:
#     # when
#     # negative ion with 1 charge and size 0.5e-9 m
#     # with
#     # positive particle with 1 charge and size 3e-9 m
#     # then:
#     # coagulation should be ~1e-12 m^3/s
#     # (within 2x, or at most an order of magnitude)
#     # (conditions assumed ~300 K, ~1 atm, but don't matter much)

#     assert negative_ion.dimensioned_coagulation_kernel(
#         positive_particle,
#     ) == pytest.approx(1e-12 * u.m**3 / u.s, rel=10)
#     # rel=10 means an order of magnitude
#     assert negative_ion.dimensioned_coagulation_kernel(
#         positive_particle,
#     ).check("[length]**3/[time]")


def test_condensation_stuff():
    """ testing condensation stuff
    """

    simple_cond_kwargs = {
        "mode": 200e-9,  # 200 nm median
        "nbins": 1000,  # 1000 bins
        "nparticles": 2e6,  # 1e4 #
        "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
        "gsigma": 1.25,  # relatively narrow
        "cutoff": .99999,  # let's take it all lol
        "vapor_radius": 1.6e-9,  # 1.6 nm
        "vapor_density": 1400,  # 1400 kg/m^3
        "vapor_concentration": 1,  # 1 ug/m^3
        "vapor_attachment": 1,  # 1
        "vapor_molecular_weight": 200,  # 200 g/mol
        "something_something": None
    }

    two_cond_vaps = simple_cond_kwargs.copy()
    two_cond_vaps["vapor_radius"] = [1.6e-9, 1.6e-9]

    two_cond_oths = simple_cond_kwargs.copy()
    two_cond_oths["vapor_radius"] = [1.6e-9, 1.7e-9]
    two_cond_oths["vapor_density"] = [1400, 1500]
    two_cond_oths["vapor_concentration"] = [1, 2]
    two_cond_oths["vapor_attachment"] = [1, 2]
    two_cond_oths["vapor_molecular_weight"] = [200, 300]

    cond = particle.Particle(**simple_cond_kwargs)
    cond2vap = particle.Particle(**two_cond_vaps)
    cond2oth = particle.Particle(**two_cond_oths)

    assert cond.particle_radius.m.shape == (1000,)
    assert cond.driving_force().u == u.kg/u.m**3
    # assert cond.driving_force().m.shape == (1,)
    assert cond.molecular_enhancement().u == u.dimensionless
    assert cond.molecular_enhancement().m.shape == (1000,)
    assert cond.condensation_redmass().u == u.kg/u.mol
    assert cond.condensation_redmass().m.shape == (1000,)
    assert cond.vapor_speed().u == u.m/u.s
    assert cond.vapor_speed().m.shape == (1000,)
    assert cond.vapor_flux().u == u.kg/u.s
    assert cond.vapor_flux().m.shape == (1000,)
    assert cond.particle_growth().u == u.m/u.s
    assert cond.particle_growth().m.shape == (1000,)

    assert cond2vap.molecular_enhancement().m.shape == (1000, 2)
    assert cond2vap.vapor_flux().m.shape == (1000, 2)
    assert cond2vap.particle_growth().m.shape == (1000,)

    assert cond2oth.molecular_enhancement().m.shape == (1000, 2)
    assert cond2oth.condensation_redmass().m.shape == (1000, 2)
    assert cond2oth.vapor_speed().m.shape == (1000, 2)

    assert cond2oth.particle_area().m.shape == (1000,)
    assert cond2oth.vapor_attachment.m.shape == (2,)
    assert cond2oth.driving_force().m.shape == (1, 2)

    assert cond2oth.vapor_flux().m.shape == (1000, 2)
    assert cond2oth.particle_growth().m.shape == (1000,)


def test_likelike_coagulation():
    """ see https://github.com/uncscode/particula/issues/239
    """
    approx_list = ["gk2008", "gh2012", "dy2007", "cg2019", "hardsphere"]
    for approx in approx_list:
        ppart = particle.Particle(
            coagulation_approximation=approx,
            particle_radius=3e-9 * u.m,
            particle_density=1.7e3 * u.kg / u.m**3,
            particle_charge=1,
        )

        assert ppart.coagulation().m >= 0
