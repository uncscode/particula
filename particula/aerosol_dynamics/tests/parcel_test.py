"""tests for the parcel module.
"""

from particula import u
from particula.aerosol_dynamics import environment, parcel, particle

small_particle = particle.Particle(
    name="small_particle",
    radius=1.0e-9,
    density=1.0,
    charge=1,
)

large_particle = particle.Particle(
    name="large_particle",
    radius=1.0e-7,
    density=1.8,
    charge=1,
)

standard_environment = environment.Environment(
    temperature=298,
    pressure=101325,
)

# create a parcel
simple_parcel = parcel.Parcel('simple', standard_environment)

# add single particle to parcel
simple_parcel.add_particle(small_particle)
simple_parcel.add_particle(large_particle)
simple_parcel.create_and_add_particle('org1', 500)
# add multiple particles to parcel
simple_parcel.create_and_add_list_of_particle('org2', [1e-9, 2e-9, 3e-9])
simple_parcel.create_and_add_list_of_particle(
    'org2', [1e-9, 2e-9, 3e-9],
    [1.8, 1, 1], [1, 0, 2]
)


def test_getters():
    """
    Test that the getters work by confirming particle creation
    """
    assert len(simple_parcel.particle_classes_list()) == 9
    assert len(simple_parcel.particle_masses_list()) == 9
    assert len(simple_parcel.particle_radii_list()) == 9
    assert len(simple_parcel.particle_charges_list()) == 9
    assert len(
        simple_parcel.particle_dimensioned_coagulation_kernel_list(
            large_particle
        )
    ) == 9
    assert len(
        simple_parcel.particle_dimensionless_coagulation_kernel_list(
            large_particle
        )
    ) == 9


def test_particle_mass_units():
    '''
    Test that the mass of the particles is returned in kg
    '''
    assert (
        sum(
            i.mass().check('kg') for i in simple_parcel.particle_classes_list()
        )
        == 9
    )


def test_particle_radius_units():
    '''
    Test that the radius of the particles is returned in m
    '''
    assert sum(i.check('m') for i in simple_parcel.particle_radii_list()) == 9


def test_particle_density_units():
    '''
    Test that the density of the particles is returned in kg/m^3
    '''
    assert (
        sum(
            i.check(u.kg / u.m ** 3)
            for i in simple_parcel.particle_densities_list()
        )
        == 9
    )


def test_particle_charge_units():
    '''
    Test that the charge of the particles is returned in dimensionless
    '''
    assert (
        sum(
            i.check(u.dimensionless)
            for i in simple_parcel.particle_charges_list()
        )
        == 9
    )


def test_particle_knudsen_number():
    '''
    Test that the knudsen number is returned in dimensionless
    '''
    assert (
        sum(
            i.check(u.dimensionless)
            for i in simple_parcel.particle_knudsen_numbers_list()
        )
        == 9
    )


def test_particle_dimensioned_coagulation_kernel_list():
    '''
    Test that the dimensioned coagulation kernel list is returned in m^3/s
    '''
    assert sum(i.check(u.m ** 3 / u.s) for i in
                simple_parcel.particle_dimensioned_coagulation_kernel_list(
                    large_particle
                )) == 9


def test_particle_dimensionless_coagulation_kernel_list():
    '''
    Test that the dimensionless coagulation kernel list is returned
    in dimensionless
    '''
    assert sum(i.check(u.dimensionless) for i in
                simple_parcel.particle_dimensionless_coagulation_kernel_list(
                    large_particle
                )) == 9


def test_remove_particle():
    '''
    Test that the remove particle method works
    '''
    simple_parcel.remove_particle([0])
    assert len(simple_parcel.particle_classes_list()) == 8
    simple_parcel.remove_particle([2, 4])
    assert len(simple_parcel.particle_classes_list()) == 6


def test_remove_all_particles():
    '''
    Test that the remove all particles method works
    '''
    simple_parcel.remove_all_particles()
    assert len(simple_parcel.particle_classes_list()) == 0
