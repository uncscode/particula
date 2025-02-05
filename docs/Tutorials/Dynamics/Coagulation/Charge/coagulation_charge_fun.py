# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# particula imports
from particula.dynamics.coagulation import brownian_kernel, rate
from particula.particles.properties.lognormal_size_distribution import (
    lognormal_pmf_distribution,
)
from particula.dynamics.coagulation.transition_regime import coulomb_chahl2019
from particula.particles.properties import (
    coulomb_enhancement,
    diffusive_knudsen_number,
    friction_factor,
    cunningham_slip_correction,
    calculate_knudsen_number,
)
from particula.gas.properties import (
    get_dynamic_viscosity,
    molecule_mean_free_path,
)
from particula.util.reduced_quantity import reduced_self_broadcast

from particula.dynamics.coagulation import kernel
from particula.dynamics.coagulation import rate
from particula.particles.properties.lognormal_size_distribution import lognormal_pmf_distribution

# Create a size distribution for aerosol particles

# Define the bins for particle radius using a logarithmic scale
radius_bins = np.logspace(start=-9, stop=-5, num=5)  # m (1 nm to 10 μm)

# Calculate the mass of particles for each size bin
# The mass is calculated using the formula for the volume of a sphere (4/3 * π * r^3)
# and assuming a particle density of 1 g/cm^3 (which is 1000 kg/m^3 in SI units).
mass_bins = 4 / 3 * np.pi * radius_bins**3 * 1e3  # kg



# %%

charge_array = radius_bins * 0 +  # change me Ian
temperature = 298.15

coulomb_potential_ratio = coulomb_enhancement.ratio(
    radius_bins, charge_array, temperature=temperature
)
dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)
mol_free_path = molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)
knudsen_number = calculate_knudsen_number(
    mean_free_path=mol_free_path, particle_radius=radius_bins
)
slip_correction = cunningham_slip_correction(knudsen_number=knudsen_number)


friction_factor_value = friction_factor(
    radius=radius_bins,
    dynamic_viscosity=dynamic_viscosity,
    slip_correction=slip_correction,
)

diffusive_knudsen_values = diffusive_knudsen_number(
    radius=radius_bins,
    mass_particle=mass_bins,
    friction_factor=friction_factor_value,
    coulomb_potential_ratio=coulomb_potential_ratio,
    temperature=temperature,
)

non_dimensional_kernel = coulomb_chahl2019(
    diffusive_knudsen=diffusive_knudsen_values,
    coulomb_potential_ratio=coulomb_potential_ratio,
)


coulomb_potential_ratio = coulomb_enhancement.ratio(
    radius=radius_bins, charge=charge_array, temperature=temperature
)

coulomb_kinetic_limit = coulomb_enhancement.kinetic(coulomb_potential_ratio)
coulomb_continuum_limit = coulomb_enhancement.continuum(
    coulomb_potential_ratio
)

sum_of_radii = radius_bins[:, np.newaxis] + radius_bins[np.newaxis, :]
reduced_mass = reduced_self_broadcast(mass_bins)

dimensional_kernel = (
    non_dimensional_kernel
    * friction_factor_value
    * sum_of_radii**3
    * coulomb_kinetic_limit**2
    / (reduced_mass * coulomb_continuum_limit)
)

log_kernel = (
    np.log(non_dimensional_kernel)
    + np.log(friction_factor_value)
    + 3 * np.log(sum_of_radii)
    + 2 * np.log(coulomb_kinetic_limit)
    - (np.log(reduced_mass) + np.log(coulomb_continuum_limit))
)

# Convert back to real space while handling underflow
dimensional_kernel_logcalc = np.exp(log_kernel)

# %% plot the kernel

fig, ax = plt.subplots()
ax.plot(radius_bins, dimensional_kernel[:, :])
ax.plot(radius_bins, dimensional_kernel_logcalc[:, :], linestyle="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Coagulation kernel (m^3/s)")
ax.set_title("Coagulation kernel vs particle radius")
plt.show()

# %% Using strategies on just the kernel so it is easier to swap out

kernel_object = kernel.CoulumbChahl2019()

kernel_dimensionless = kernel_object.dimensionless(
    diffusive_knudsen=diffusive_knudsen_values,
    coulomb_potential_ratio=coulomb_potential_ratio,
)
kernel_dimensional = kernel_object.kernel(
    dimensionless_kernel=kernel_dimensionless,
    coulomb_potential_ratio=coulomb_potential_ratio,
    sum_of_radii=sum_of_radii,
    reduced_mass=reduced_mass,
    reduced_friction_factor=friction_factor_value,
)

fig, ax = plt.subplots()
ax.plot(radius_bins, kernel_dimensional[:, :])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Coagulation kernel (m^3/s)")
ax.set_title("Coagulation kernel vs particle radius")
plt.show()

# %% making a time step
# get rates of coagulation dn/dt

# make a number concentration distribution
number_concentration = lognormal_pmf_distribution(
    x_values=radius_bins,
    mode=np.array([200e-9]),  # m
    geometric_standard_deviation=np.array([1.5]),
    number_of_particles=np.array([1e12]),  # per m^3
)

gain_rate = rate.discrete_gain(
    radius=radius_bins,
    concentration=number_concentration,
    kernel=kernel_dimensional,
)
loss_rate = rate.discrete_loss(
    concentration=number_concentration,
    kernel=kernel_dimensional,
)

net_rate = gain_rate - loss_rate

# %% 
# plot distribution
fig, ax = plt.subplots()
ax.plot(radius_bins, number_concentration)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Number concentration (1/m^3)")
ax.set_title("Number concentration vs particle radius")
plt.show()

# plot the rates
fig, ax = plt.subplots()
ax.plot(radius_bins, gain_rate, label="Gain rate")
ax.plot(radius_bins, -1*loss_rate, label="Loss rate")
ax.plot(radius_bins, net_rate, label="Net rate", linestyle="--")
ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Coagulation rate 1/ (m^3 s)")
ax.set_title("Coagulation rate vs particle radius")
ax.legend()
plt.show()
# %%