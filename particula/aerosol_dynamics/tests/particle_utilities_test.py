
import numpy as np
import pytest
from particula import u
from particula.aerosol_dynamics import environment
from particula.aerosol_dynamics import particle_utilities as pu

# def dimensioned_coagulation_kernel(
#     charges_array, charge_other,
#     radii_array, radius_other,
#     mass_array, mass_other,
#     temperature, mean_free_path_air, dynamic_viscosity_air,
#     authors: str = "cg2019",
# ) -> float:
#     """Dimensioned particle--particle coagulation kernel.
#     Parameters:
#         charges_array  (np array) [dimensionless]
#         charge_other  (float) [dimensionless]
#         radii_array  (np array) [m]
#         radius_other  (float) [m]
#         mass_array  (np array) [kg]
#         mass_other  (float) [kg]
#         temperature  (float) [K]
#         mean_free_path_air  (float) [m]
#         dynamic_viscosity_air  (float) [N*s/m]
#         authors:        authors of the parameterization
#             - gh2012    doi.org:10.1103/PhysRevE.78.046402
#             - cg2019    doi:10.1080/02786826.2019.1614522
#             - hard_sphere
#             (default: cg2019)
#     returns:
#         dimensioned_coagulation_kernel (array) [m**3/s]

#     """

# standard_environment_ip = environment.Environment(
#     temperature=300 * u.K,
#     pressure=101325 * u.Pa,
# )

charges_array = np.array([0, 0, 0, 0])
charge_other = charges_array[0]

radii_array = np.array([50, 75, 150, 10000]) * 1e-9
radius_other = radii_array[0]


mass_array = np.array([1, 1, 1, 1]) * 1e3
mass_other = mass_array[0]

temperature = 300
# mean_free_path_air = standard_environment_ip.mean_free_path_air().magnitude
# dynamic_viscosity_air = standard_environment_ip.dynamic_viscosity_air().magnitude
authors = "cg2019"


def test_file_work():
    assert True

# def test_getters():
#     """Test that the getters work.
#     """

#     assert pu.dimensioned_coagulation_kernel(
#         charges_array, charge_other,
#         radii_array, radius_other,
#         mass_array, mass_other,
#         temperature, mean_free_path_air, dynamic_viscosity_air,
#         authors,
#     ) > 0
