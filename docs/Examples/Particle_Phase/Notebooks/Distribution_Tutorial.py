"""Distribution strategy tutorial for particle property calculations.

This tutorial covers the distribution strategies implemented in particula
for calculating particle properties. Strategies define how to compute
derived properties like particle mass, radius, and total mass based on
the distribution type (mass-based, radius-based, or speciated-mass).

Strategies covered:
    - MassBasedMovingBin: Calculate radius from known particle mass
    - RadiiBasedMovingBin: Calculate mass from known particle radius
    - SpeciatedMassMovingBin: Multi-species mass-based calculations

Example:
    Calculate particle radius from mass::

        import numpy as np
        import particula as par
        mass = np.linspace(0, 10, 5)  # kg
        density = np.array([1000.0])  # kg/m^3
        radius = par.particles.MassBasedMovingBin().get_radius(mass, density)
"""
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: particula_dev312
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Distribution Strategy Tutorial
#
# The representation of particle distributions is core to the simulation, but it can vary depending on what you are trying to achieve. In this tutorial, we will cover the  distribution strategies currently implemented.
#
# The distribution strategies define how to calculate properties derived from the particle distribution. These include particle mass, radius, and total mass, all of which can have different methods depending on whether the distribution is mass-based, radius-based, or speciated-mass based.
#
# We will cover the following distribution strategies:
#
# - `MassBasedMovingBin`
# - `RadiiBasedMovingBin`
# - `SpeciatedMassMovingBin`
#
# As they are just operational strategies, they do not have any specific parameters to be set. They are just used to calculate the properties of the particles.

# %%
# In Colab uncomment the following command to install particula:
# #!pip install particula[extra] --quiet
import numpy as np
import particula as par

# %% [markdown]
# ## Strategy: MassBasedMovingBin
#
# The `MassBasedMovingBin` strategy is used when the distribution is mass-based. This
# means that the mass of the particles is known and the radius is calculated from the mass. The `MassBasedMovingBin` strategy calculates the radius of the particles using the following equation:
#
# $$
# r = \left(\frac{3m}{4\pi\rho}\right)^{1/3}
# $$
#
# where $r$ is the radius of the particle, $m$ is the mass of the particle, and $\rho$ is the density of the particle.
#

# %%
mass_distribution = np.linspace(0, 10, 5)  # kg
density = np.array([1000.0])  # kg/m^3

radius = par.particles.MassBasedMovingBin().get_radius(
    mass_distribution, density
)
print(f"Radius of the particles: {radius} m")

total_mass = par.particles.MassBasedMovingBin().get_total_mass(
    mass_distribution,
    concentration=np.ones_like(mass_distribution),
    density=density,
)
print(f"Total mass of the particles: {total_mass} kg")
print(f"Same as the sum*concentration: {np.sum(mass_distribution)} kg")

# %% [markdown]
# ## Builder: RadiiBasedMovingBin
#
# The `RadiiBasedMovingBin` strategy is used when the distribution is radius-based. This means that the radius of the particles is known and the mass is calculated from the radius. The `RadiiBasedMovingBin` strategy calculates the mass of the particles using the following equation:
#
# $$
# m = \frac{4\pi\rho r^3}{3}
# $$
#
# where $m$ is the mass of the particle, $r$ is the radius of the particle, and $\rho$ is the density of the particle.
#
# The builder does nothing in this case, as we just have no parameters to set. We use the builder pattern here to keep the code consistent with the other strategies.

# %%
radii_distribution = np.linspace(0, 0.1, 5)  # m
density_radii = np.array([1000.0])  # kg/m^3

radii_strategy = par.particles.RadiiBasedMovingBinBuilder().build()

mass_distribution = radii_strategy.get_mass(radii_distribution, density_radii)
print(f"Mass of the particles: {mass_distribution} kg")

total_mass = radii_strategy.get_total_mass(
    radii_distribution,
    concentration=np.ones_like(radii_distribution),
    density=density_radii,
)
print(f"Total mass of the particles: {total_mass} kg")

# %% [markdown]
# ## Factory: SpeciatedMassMovingBin
#
# The `SpeciatedMassMovingBin` strategy is used when the distribution is speciated-mass based. This means that the mass of the particles is known and the radius is calculated from the mass. The `SpeciatedMassMovingBin` has multiple species, and the mass of each species is known for that given bin or particle.

# %%
mass_distribution1 = np.linspace(0, 10, 5)  # kg
mass_distribution2 = np.linspace(0, 10, 5)  # kg
masses_combined = np.vstack((mass_distribution1, mass_distribution2)).T
density = np.array([1000.0, 2000.0])  # kg/m^3

speciated_mass = par.particles.DistributionFactory().get_strategy(
    "speciated_mass_moving_bin"
)

radius = speciated_mass.get_radius(masses_combined, density)

print(f"Radius of the particles: {radius} m")

total_mass = speciated_mass.get_total_mass(
    masses_combined,
    concentration=np.ones_like(mass_distribution1),
    density=density,
)
print(f"Total mass of the particles: {total_mass} kg")

# %% [markdown]
# ## Summary
#
# In this tutorial, we covered the distribution strategies implemented in the simulation. We covered the `MassBasedMovingBin`, `RadiiBasedMovingBin`, and `SpeciatedMassMovingBin` strategies. These strategies are used to calculate the properties of the particles based on the distribution type.
