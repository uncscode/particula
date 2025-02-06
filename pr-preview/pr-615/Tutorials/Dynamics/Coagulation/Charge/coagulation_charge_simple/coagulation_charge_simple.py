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

from particula.util.machine_limit import safe_exp, safe_log
# Create a size distribution for aerosol particles

# Define the bins for particle radius using a logarithmic scale
radius_bins = np.logspace(start=-9, stop=1, num=350)  # m (1 nm to 10 μm)

# Calculate the mass of particles for each size bin
# The mass is calculated using the formula for the volume of a sphere (4/3 * π * r^3)
# and assuming a particle density of 1 g/cm^3 (which is 1000 kg/m^3 in SI units).
mass_bins = 4 / 3 * np.pi * radius_bins**3 * 1e3  # kg


# %

charge_array = radius_bins * 0 + 10000 # change me Ian
# charge_array[5:] = -5000
temperature = 298.15

coulomb_potential_ratio = coulomb_enhancement.ratio(
    radius_bins, charge_array, temperature=temperature
)
print(coulomb_potential_ratio)
coulomb_kinetic_limit = coulomb_enhancement.kinetic(coulomb_potential_ratio)
coulomb_continuum_limit = coulomb_enhancement.continuum(
    coulomb_potential_ratio
)
dynamic_viscosity = get_dynamic_viscosity(temperature=temperature)
mol_free_path = molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)
knudsen_number = calculate_knudsen_number(
    mean_free_path=mol_free_path, particle_radius=radius_bins
)
slip_correction = cunningham_slip_correction(knudsen_number=knudsen_number)

coulomb_potential_ratio = np.clip(coulomb_potential_ratio, -200, np.finfo(np.float64).max)
fig, ax = plt.subplots()
ax.plot(radius_bins, coulomb_potential_ratio)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Coulomb potential ratio")
ax.set_title("Coulomb potential ratio vs particle radius")
plt.show()

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


fig, ax = plt.subplots()
ax.plot(radius_bins, diffusive_knudsen_values)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Diffusive Knudsen number")
ax.set_title("Diffusive Knudsen number vs particle radius")
plt.show()

sum_of_radii = radius_bins[:, np.newaxis] + radius_bins[np.newaxis, :]
reduced_mass = reduced_self_broadcast(mass_bins)


# % Using strategies on just the kernel so it is easier to swap out

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

# find any nan value or zeros
print(np.isnan(kernel_dimensional).any())
print(np.isinf(kernel_dimensional).any())
print(np.isneginf(kernel_dimensional).any())
print(np.isneginf(kernel_dimensional).any())


fig, ax = plt.subplots()
ax.plot(radius_bins, kernel_dimensional[:,:])
ax.plot(radius_bins, kernel_dimensional[0,:], label="small particle", marker="o")
ax.plot(radius_bins, kernel_dimensional[-1,:], label="large particle", marker="s")
# ax.plot(radius_bins, dimensional_kernel_logcalc[:, :], linestyle="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Coagulation kernel (m^3/s)")
ax.legend()
ax.set_title(f"Coagulation kernel vs particle radius, all charge = {charge_array[0]}")
plt.show()


# %%
