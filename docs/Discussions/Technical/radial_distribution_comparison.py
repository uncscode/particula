# %%
"""
This script compares the radial distribution function g₁₂ between DNS data
and the model prediction.

It uses the function `get_g12_radial_distribution_ao2008` from the `particula`
library to compute g₁₂ values over a range of particle radii.

The script then plots these computed values against the DNS dataset `r23_e100`
for visual comparison.

Usage:
- Run this script to generate and display the comparison graph.
"""

import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008 import (
    get_g12_radial_distribution_ao2008,
)
from particula.particles import properties
from particula.gas import properties as gas_properties
from particula.util.converting.units import convert_units


def g12_calc(particle_radius, turbulent_dissipation, reynolds_lambda):
    # Define constants and parameters
    temperature = 300  # Temperature in Kelvin
    particle_density = 1000  # Particle density in kg/m³
    fluid_density = 1.225  # Fluid (air) density in kg/m³
    relative_velocity = 1e-6  # Relative velocity in m/s

    # Basic fluid properties
    dynamic_viscosity = gas_properties.get_dynamic_viscosity(temperature)
    kinematic_viscosity = gas_properties.get_kinematic_viscosity(
        dynamic_viscosity=dynamic_viscosity, fluid_density=fluid_density
    )

    # Particle inertia and settling velocity
    particle_inertia_time = properties.get_particle_inertia_time(
        particle_radius=particle_radius,
        particle_density=particle_density,
        fluid_density=fluid_density,
        kinematic_viscosity=kinematic_viscosity,
        relative_velocity=relative_velocity,
    )

    # Kolmogorov parameters
    kolmogorov_time = gas_properties.get_kolmogorov_time(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    kolmogorov_length_scale = gas_properties.get_kolmogorov_length(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    normalized_accel_variance = (
        gas_properties.get_normalized_accel_variance_ao2008(
            re_lambda=reynolds_lambda
        )
    )
    kolmogorov_velocity = gas_properties.get_kolmogorov_velocity(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )

    stokes_number = properties.get_stokes_number(
        particle_inertia_time=particle_inertia_time,
        kolmogorov_time=kolmogorov_time,
    )

    # Compute g₁₂ Values
    g12_values = get_g12_radial_distribution_ao2008(
        particle_radius,
        stokes_number,
        kolmogorov_length_scale,
        reynolds_lambda,
        normalized_accel_variance,
        kolmogorov_velocity,
        kolmogorov_time,
    )

    return g12_values


# plot of radius 'a' (microns) vs g12 (r=2a)

# DNS datasets for radial distribution function
r23_e100 = np.array(
    [
        [9.937578027, 1.532846715],
        [19.98751561, 1.094890511],
        [29.91260924, 2.299270073],
        [40.02496879, 3.686131387],
        [49.95006242, 2.919708029],
        [60, 2.737226277],
    ]
)

# Define the particle radii and other required parameters
particle_radius = np.linspace(1e-6, 60e-6, 100)  # From 1 µm to 60 µm
temperature = 300  # Example value
particle_density = 1000  # Example value
fluid_density = 1.225  # Example value
relative_velocity = 1e-6  # Example value

# 1. Basic fluid properties
dynamic_viscosity = gas_properties.get_dynamic_viscosity(temperature)
kinematic_viscosity = gas_properties.get_kinematic_viscosity(
    dynamic_viscosity=dynamic_viscosity, fluid_density=fluid_density
)
mean_free_path = gas_properties.molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)

# 3. Particle inertia and settling velocity
particle_inertia_time = properties.get_particle_inertia_time(
    particle_radius=particle_radius,
    particle_density=particle_density,
    fluid_density=fluid_density,
    kinematic_viscosity=kinematic_viscosity,
    relative_velocity=relative_velocity,
)

# R_lambda = 23, turbulent_dissipation = 100 cm2/s3

reynolds_lambda = 23  # From the dataset
turbulent_dissipation = 100 * convert_units(
    "cm^2/s^3", "m^2/s^3"
)  # Example value

# encapsolate the following in a function:
kolmogorov_time = gas_properties.get_kolmogorov_time(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)
kolmogorov_length_scale = gas_properties.get_kolmogorov_length(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)
normalized_accel_variance = (
    gas_properties.get_normalized_accel_variance_ao2008(
        re_lambda=reynolds_lambda
    )
)
kolmogorov_velocity = gas_properties.get_kolmogorov_velocity(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)


stokes_number = properties.get_stokes_number(
    particle_inertia_time=particle_inertia_time,
    kolmogorov_time=kolmogorov_time,
)

# Convert turbulent dissipation from cm²/s³ to m²/s³
turbulent_dissipation_100 = 100 * convert_units("cm^2/s^3", "m^2/s^3")
turbulent_dissipation_400 = 400 * convert_units("cm^2/s^3", "m^2/s^3")

# Compute g₁₂ values for each case
g12_values_re23_e100 = g12_calc(
    particle_radius, turbulent_dissipation_100, reynolds_lambda=23
)
g12_values_re23_e400 = g12_calc(
    particle_radius, turbulent_dissipation_400, reynolds_lambda=23
)
g12_values_re72_4_e100 = g12_calc(
    particle_radius, turbulent_dissipation_100, reynolds_lambda=72.4
)
g12_values_re72_4_e400 = g12_calc(
    particle_radius, turbulent_dissipation_400, reynolds_lambda=72.4
)

# R_lambda = 23, turbulent_dissipation = 100 cm2/s3

# case: R_lambda = 23, turbulent_dissipation = 400 cm2 s−3
# r23_e400: 6 rows, 2 columns (X, Y)
r23_e400 = np.array(
    [
        [10.18726592, 1.094890511],
        [20.17478152, 3.248175182],
        [30.09987516, 8.175182482],
        [40.14981273, 8.686131387],
        [50.13732834, 7.226277372],
        [60.24968789, 5.620437956],
    ]
)

g12_values_re23_400 = ...

# case: R_lambda = 72.4, turbulent_dissipation = 100 cm2 s−3
# r72.4_e100: 6 rows, 2 columns (X, Y)
r72_4_e100 = np.array(
    [
        [10.12484395, 1.204379562],
        [19.92509363, 1.788321168],
        [29.97503121, 3.211678832],
        [40.08739076, 7.919708029],
        [50.01248439, 10.76642336],
        [59.93757803, 9.525547445],
    ]
)

g12_values_re72_4_e100 = ...

# case: R_lambda = 72.4, turbulent_dissipation = 400 cm2 s−3
# r72.4_e400: 6 rows, 2 columns (X, Y)
r72_4_e400 = np.array(
    [
        [10, 0.875912409],
        [20.11235955, 5.145985401],
        [30.03745318, 16.82481752],
        [40.08739076, 15.72992701],
        [50.01248439, 14.48905109],
        [60, 13.72262774],
    ]
)


# Plot the Comparison Graph for all cases
plt.figure(figsize=(10, 6))

# Plot DNS data
plt.scatter(
    r23_e100[:, 0],
    r23_e100[:, 1],
    label=r"DNS: $R_\lambda=23$, $\varepsilon=100$",
    color="blue",
    marker="o",
)
plt.scatter(
    r23_e400[:, 0],
    r23_e400[:, 1],
    label=r"DNS: $R_\lambda=23$, $\varepsilon=400$",
    color="green",
    marker="^",
)
plt.scatter(
    r72_4_e100[:, 0],
    r72_4_e100[:, 1],
    label=r"DNS: $R_\lambda=72.4$, $\varepsilon=100$",
    color="red",
    marker="s",
)
plt.scatter(
    r72_4_e400[:, 0],
    r72_4_e400[:, 1],
    label=r"DNS: $R_\lambda=72.4$, $\varepsilon=400$",
    color="purple",
    marker="d",
)

# Plot model predictions
plt.plot(
    particle_radius * 1e6,
    np.diagonal(g12_values_re23_e100),
    label=r"Model: $R_\lambda=23$, $\varepsilon=100$",
    color="blue",
)
plt.plot(
    particle_radius * 1e6,
    np.diagonal(g12_values_re23_e400),
    label=r"Model: $R_\lambda=23$, $\varepsilon=400$",
    color="green",
)
plt.plot(
    particle_radius * 1e6,
    np.diagonal(g12_values_re72_4_e100),
    label=r"Model: $R_\lambda=72.4$, $\varepsilon=100$",
    color="red",
)
plt.plot(
    particle_radius * 1e6,
    np.diagonal(g12_values_re72_4_e400),
    label=r"Model: $R_\lambda=72.4$, $\varepsilon=400$",
    color="purple",
)

plt.xlabel("Particle Radius (µm)")
plt.ylabel("Radial Distribution Function $g_{12}$")
plt.title("Radial Distribution Function Comparison")
plt.legend()
plt.grid(True)
plt.ylim(0, 30)
plt.show()
# %%
