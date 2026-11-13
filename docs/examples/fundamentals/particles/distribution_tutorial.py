# %%
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
# # Native Distribution Storage
#
# Native particles store material masses consistently, independent of whether
# initialization starts with mass or radius. Distribution metadata determines
# how population weights are integrated: discrete bins sum number densities,
# continuous PDFs integrate dN/dr on a radius grid, and resolved slots sum
# represented counts before division by physical volume.

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## Mass-based bins
#
# Retain five illustrative masses from zero to 10 kg at density 1000 kg/m^3.
# These large values demonstrate arithmetic, not a realistic aerosol spectrum.
# Each bin has number density 1 m^-3 at unit reference volume.

# %%
mass_distribution = np.linspace(0, 10, 5)
mass_bins = (
    par.particles.ParticleDataBuilder()
    .set_distribution_type("discrete")
    .set_masses(mass_distribution[:, None])
    .set_density(np.array([1000.0]))
    .set_concentration(np.ones(5))
    .set_charge(np.zeros(5))
    .set_volume(np.array([1.0]))
    .build()
)
print("Radii [m]:", mass_bins.radii)
print("Material mass density [kg/m^3]:", mass_bins.species_mass_density)
print("Material inventory [kg]:", mass_bins.species_mass_inventory)

# %% [markdown]
# ## Radius-based initialization
#
# Preserve the illustrative zero-radius endpoint as zero mass in the builder.
# The convenience radius initializer accepts strictly positive radii; direct
# mass construction also expresses this zero-material bin explicitly.

# %%
radii_distribution = np.linspace(0, 0.1, 5)
radius_masses = 4 * np.pi / 3 * 1000 * radii_distribution**3
radius_bins = (
    par.particles.ParticleDataBuilder()
    .set_distribution_type("discrete")
    .set_masses(radius_masses[:, None])
    .set_density(np.array([1000.0]))
    .set_concentration(np.ones(5))
    .set_charge(np.zeros(5))
    .set_volume(np.array([1.0]))
    .build()
)
print("Per-particle masses [kg]:", radius_bins.total_mass)
print("Material inventory [kg]:", radius_bins.species_mass_inventory)

# %% [markdown]
# ## Speciated mass bins
#
# For two materials, particle volume is the sum of mass/density over material
# lanes. Keep each material's inventory separately instead of losing species
# information in a total-mass scalar.

# %%
masses_combined = np.column_stack((mass_distribution, mass_distribution))
speciated_bins = (
    par.particles.ParticleDataBuilder()
    .set_distribution_type("discrete")
    .set_masses(masses_combined)
    .set_density(np.array([1000.0, 2000.0]))
    .set_concentration(np.ones(5))
    .set_charge(np.zeros(5))
    .set_volume(np.array([1.0]))
    .build()
)
print("Radii [m]:", speciated_bins.radii)
print("Material inventory [kg]:", speciated_bins.species_mass_inventory)

# %% [markdown]
# ## Resolved count/volume interpretation
#
# The same positive-material particles can represent weighted physical counts.
# At volume 0.25 m^3, counts of two give density eight per slot. Changing a
# distribution tag alone is not a conversion: construct the intended weights.

# %%
resolved = par.particles.create_resolved_particle_data(
    masses=masses_combined[1:],
    density=[1000.0, 2000.0],
    concentration=np.full(4, 2.0),
    volume=0.25,
)
print("Number density [m^-3]:", resolved.number_density)
print("Material inventory [kg]:", resolved.species_mass_inventory)
