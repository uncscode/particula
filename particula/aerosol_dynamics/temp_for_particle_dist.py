#%%

# change directory up
import os
os.chdir('..')

#%%
# import particula.utils.strip_units as strip_units
from particula.aerosol_dynamics import environment, particle_distribution, parcel
from particula.utils.strip_units import make_unitless
import numpy as np
import matplotlib.pyplot as plt
# importlib.reload(particula)

#%%

standard_environment = environment.Environment(
    temperature=298,
    pressure=101325,
)

# %%

dist_sample = np.random.poisson(100, size=10000)*10**-9 # m

bins_centers = np.linspace(0, 1000, 100)

particle_distribution(dist_sample, bins_centers)


def distribution_rasterization(particle_distribution, bins_centers):
    np.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)



np.logspace(10**-9, 10**-1, num=1000)
charges_array = np.array([0, 0, 0, 0])
charge_other = charges_array[0]
#%%

# %%
import matplotlib 
import matplotlib.cbook
import matplotlib.pyplot
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = '14'


fig, ax= matplotlib.pyplot.subplots(1,1,figsize=[6,6])

ax.hist(dist_sample, bins=50, density=True)


# ax.set_ylim(0,14000)
# ax.set_xlabel('particle 1, radi range (nm)', fontsize=12)
# ax.set_ylabel('coagulation kernel P12 [m^3/sec]', fontsize=12)
ax.grid(True, alpha=0.5)
# ax.legend(loc='best', fontsize=14)

# %%

dist1 = particle_distribution.Particle_Distribution(radii_array, density_array, charges_array, number_array)



# %%
