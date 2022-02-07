#%%

# change directory up
import os
os.chdir('..')

#%%
import importlib

from particula.aerosol_dynamics import particle_distribution, environment, particle_utilities
import numpy as np

importlib.reload(particle_distribution)

#%%

standard_environment = environment.Environment(
    temperature=298,
    pressure=101325,
)

TEMPERATURE = 300
mean_free_path_air = particle_utilities.unitless(standard_environment.mean_free_path_air())
dynamic_viscosity_air = standard_environment.dynamic_viscosity_air().magnitude
# %%
charges_array = np.array([0, 0, 0, 0])
charge_other = charges_array[0]

radii_array = np.array([50, 75, 150, 10000]) * 1e-9
radius_other = radii_array[0]


mass_array = np.array([1, 1, 1, 1]) * 1e3
mass_other = mass_array[0]

density_array = np.array([1, 1, 1, 1]) * 1e3
number_array = density_array*1e6
AUTHORS = "cg2019"


# def test_file_work():
#     assert True


test = particle_utilities.dimensioned_coagulation_kernel(
    charges_array, charge_other,
    radii_array, radius_other,
    mass_array, mass_other,
    TEMPERATURE, mean_free_path_air, dynamic_viscosity_air,
    AUTHORS,
)
# %%

dist1 = particle_distribution.Particle_Distribution(radii_array, density_array, charges_array, number_array)



# %%
