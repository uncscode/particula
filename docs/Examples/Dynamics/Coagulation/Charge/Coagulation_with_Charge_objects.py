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

# %%
"""Show how to compose a charged coagulation strategy with the builder API.

The script creates a charged log-distributed particle population, calculates
supporting transport properties, builds the charged strategy via
ChargedCoagulationBuilder, and visualizes the resulting kernel on a log-log
grid.

Examples:
    >>> builder = par.dynamics.ChargedCoagulationBuilder()
    >>> builder.set_distribution_type("discrete").set_charge_kernel(...)
"""

# %% [markdown]
# # Coagulation with Charge (Builder pattern)
# This tutorial shows how to build a charged coagulation strategy using the public builder API in `particula.dynamics`. We:
# - define a particle size distribution with assigned charges,
# - compute supporting transport properties,
# - build a charged coagulation strategy with a selected kernel, and
# - visualize the resulting coagulation kernel.

# %%
import matplotlib.pyplot as plt
import numpy as np
import particula as par

plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# ## Define the particle size distribution We build a logarithmic radius grid (1 nm to 10 μm) and the corresponding mass bins assuming unit density.

# %%
radius_bins = np.logspace(start=-9, stop=-4, num=250)  # m (1 nm to 10 μm)
mass_bins = 4 / 3 * np.pi * radius_bins**3 * 1e3  # kg (ρ = 1000 kg/m³)
n_bins = len(radius_bins)

# %% [markdown]
# ## Assign particle charges
# Negative charges are applied to the first third of the distribution and positive charges to the remainder.

# %%
split_index = n_bins // 3
neg_charges = -np.logspace(np.log10(10), np.log10(1), num=split_index)
pos_charges = np.logspace(np.log10(1), np.log10(500), num=n_bins - split_index)
charge_array = np.concatenate((neg_charges, pos_charges))

fig, ax = plt.subplots()
ax.plot(radius_bins, charge_array, marker="o", linestyle="none")
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Particle charge (elementary charges)")
ax.set_title("Particle Charge vs. Radius")
ax.grid(True, which="both", ls="--")
plt.show()

# %% [markdown]
# ## Supporting properties
# We compute the Coulomb potential ratio, gas properties, Knudsen numbers, slip correction, friction factor, and helper matrices used by the kernel strategies.

# %%
temperature = 298.15  # K

coulomb_potential_ratio: np.ndarray = np.asarray(
    par.particles.get_coulomb_enhancement_ratio(
        radius_bins, charge_array, temperature=temperature
    ),
    dtype=float,
)
dynamic_viscosity = par.gas.get_dynamic_viscosity(temperature=temperature)
mol_free_path = par.gas.get_molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)
knudsen_number = par.particles.get_knudsen_number(
    mean_free_path=mol_free_path, particle_radius=radius_bins
)
slip_correction = par.particles.get_cunningham_slip_correction(
    knudsen_number=knudsen_number
)
friction_factor_value = par.particles.get_friction_factor(
    particle_radius=radius_bins,
    dynamic_viscosity=dynamic_viscosity,
    slip_correction=slip_correction,
)
diffusive_knudsen_values: np.ndarray = np.asarray(
    par.particles.get_diffusive_knudsen_number(
        particle_radius=radius_bins,
        particle_mass=mass_bins,
        friction_factor=friction_factor_value,
        coulomb_potential_ratio=coulomb_potential_ratio,
        temperature=temperature,
    ),
    dtype=float,
)

sum_of_radii = radius_bins[:, np.newaxis] + radius_bins[np.newaxis, :]
reduced_mass = par.util.get_reduced_self_broadcast(mass_bins.astype(float))
reduced_friction_factor = par.util.get_reduced_self_broadcast(
    np.asarray(friction_factor_value, dtype=float)
)

# %% [markdown]
# ## Build the charged coagulation strategy (builder)
# We use the public builder to compose a charged coagulation strategy with the Coulomb Gatti 2008 kernel. The builder enforces the required distribution type and kernel strategy.

# %%
kernel_strategy = par.dynamics.CoulombGatti2008KernelStrategy()
charged_coagulation = (
    par.dynamics.ChargedCoagulationBuilder()
    .set_distribution_type("discrete")
    .set_charged_kernel_strategy(kernel_strategy)
    .build()
)

# The ChargedCoagulationStrategy exposes the charged kernel strategy; call it
# directly for the dimensionless and dimensional kernels.
dimensionless_kernel: np.ndarray = kernel_strategy.dimensionless(
    diffusive_knudsen=diffusive_knudsen_values,
    coulomb_potential_ratio=coulomb_potential_ratio,
)
dimensional_kernel: np.ndarray = kernel_strategy.kernel(
    dimensionless_kernel=dimensionless_kernel,
    coulomb_potential_ratio=coulomb_potential_ratio,
    sum_of_radii=sum_of_radii,
    reduced_mass=reduced_mass,
    reduced_friction_factor=reduced_friction_factor,
)

# %% [markdown]
# ## Visualize the kernel
# Plot the dimensional coagulation kernel on a logarithmic grid.

# %%
fig, ax = plt.subplots()
mesh = ax.pcolormesh(
    radius_bins, radius_bins, np.log10(dimensional_kernel), shading="auto"
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius r_i (m)")
ax.set_ylabel("Particle radius r_j (m)")
ax.set_title("Coulomb Gatti 2008 Coagulation Kernel (Builder)")
fig.colorbar(mesh, ax=ax, label="log10(Kernel) (m³/s)")
plt.show()
