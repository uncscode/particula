#%%

# change directory up
import os
os.chdir('..')

#%%
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

bin_edges = np.linspace(0, 1000, 100)*10**-9 # m

particle_dist = particle_distribution.Particle_distribution( radii = dist_sample,
        density = np.ones(len(dist_sample))*1000,
        charge = np.zeros(len(dist_sample)),
        number = np.ones(len(dist_sample)),
    )
print(particle_dist.number_concentration())

# def rasterization(particle_distribution, bins):
#     """ 
    
#     """


#     # histogram method
#     bin_edges = np.histogram_bin_edges(particle_distribution.radii(), bins=bins)
#     particle_number, bin_edges = np.histogram(
#         particle_distribution.radii(),
#         bins=bin_edges,
#         weights = particle_distribution.number()
#     )

#     bin_centers = np.diff(bin_edges)+bin_edges[0:-1] # calculates bin centers and applies as the radii

#     # drop zero bins
#     non_zeros = particle_number>0
#     particle_number = particle_number[non_zeros]
#     bin_centers = bin_centers[non_zeros]

#     return particle_number, bin_centers

# number_raster, bin_centers = distribution_rasterization(particle_dist, 'auto')


# particle_dist.update_distribution(bin_centers, number_raster)
# print(particle_dist.number_concentration())


# np.logspace(10**-9, 10**-1, num=1000)
# charges_array = np.array([0, 0, 0, 0])
# charge_other = charges_array[0]
#%%

# %%
import matplotlib 
import matplotlib.cbook
import matplotlib.pyplot
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = '14'


fig, ax= matplotlib.pyplot.subplots(1,1,figsize=[6,6])

ax.plot(bin_centers, number_raster)


# ax.set_ylim(0,14000)
# ax.set_xlabel('particle 1, radi range (nm)', fontsize=12)
# ax.set_ylabel('coagulation kernel P12 [m^3/sec]', fontsize=12)
ax.grid(True, alpha=0.5)
# ax.legend(loc='best', fontsize=14)

# %%

dist1 = particle_distribution.Particle_Distribution(radii_array, density_array, charges_array, number_array)



# %%
