"""
Test suites for parcel class.
"""

from particula.aerosol_dynamics import parcel, particle

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

# create a parcel
simple_parcel = parcel.Parcel('simple', temperature=300, pressure=101325)

# add single particle to parcel
simple_parcel.add_particle(small_particle)
simple_parcel.add_particle(large_particle)
simple_parcel.create_and_add_particle('org1', 50e-9)
# add multiple particles to parcel
simple_parcel.create_and_add_list_of_particle('org2', [1e-9, 2e-9, 3e-9])


def test_getters():
    """
    Test that the getters work by confirming particle creation
    """
    assert len(simple_parcel.particle_classes()) == 6
    assert len(simple_parcel.particle_mass()) == 6
    assert len(simple_parcel.particle_radius()) == 6
