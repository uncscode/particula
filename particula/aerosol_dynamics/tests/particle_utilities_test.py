""" testing the new particle_utils
"""

import numpy as np
from particula import u
from particula.aerosol_dynamics import particle_utilities as pu
from particula.aerosol_dynamics import environment


standard_environment_ip = environment.Environment(
    temperature=300 * u.K,
    pressure=101325 * u.Pa,
)

charges_array = np.array([0, 0, 0, 0])
charge_other = charges_array[0]

radii_array = np.array([50, 75, 150, 10000]) * 1e-9
radius_other = radii_array[0]


mass_array = np.array([1, 1, 1, 1]) * 1e3
mass_other = mass_array[0]

TEMPERATURE = 300
mean_free_path_air = standard_environment_ip.mean_free_path_air().magnitude
dynamic_viscosity_air = standard_environment_ip.dynamic_viscosity_air().magnitude
AUTHORS = "cg2019"


# def test_file_work():
#     assert True

def test_dimensions_runs():
    """Test that the getters work.
    """
    test = pu.dimensioned_coagulation_kernel(
        charges_array,
        charge_other,
        radii_array,
        radius_other,
        mass_array,
        mass_other,
        TEMPERATURE,
        mean_free_path_air,
        dynamic_viscosity_air,
        AUTHORS,
    )
    assert (test > 0).all()
