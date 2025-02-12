# %%
from typing import Union
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_ao2008,
    get_radial_relative_velocity_dz2002,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008 import (
    get_relative_velocity_variance,
)
from particula.particles.properties.inertia_time import (
    get_particle_inertia_time,
)
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_with_drag,
)
from particula.gas.properties.kinematic_viscosity import (
    get_kinematic_viscosity,
)
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gas.properties.fluid_rms_velocity import get_fluid_rms_velocity
from particula.gas.properties.normalize_accel_variance import (
    get_normalized_accel_variance_ao2008,
)
from particula.gas.properties.kolmogorov_module import get_kolmogorov_time
from particula.gas.properties.taylor_microscale_module import (
    get_taylor_microscale,
    get_lagrangian_taylor_microscale_time,
)
from particula.gas.properties.integral_scale_module import (
    get_eulerian_integral_length,
    get_lagrangian_integral_time,
)
from particula.gas import properties as gas_properties
from particula.particles import properties
from particula.util.constants import STANDARD_GRAVITY
from particula.util.converting.units import convert_units

from particula.dynamics.coagulation.turbulent_dns_kernel.velocity_correlation_terms_ao2008 import (
    compute_b1,
    compute_b2,
    compute_c1,
    compute_c2,
    compute_d1,
    compute_d2,
    compute_e1,
    compute_e2,
    compute_z,
    compute_beta,
)


# %%
# DNS Data from Figure 13
data = np.array(
    [
        [10.06195787, 5.602409639],
        [15.01858736, 5.13253012],
        [19.97521685, 3.506024096],
        [25.11771995, 2.096385542],
        [27.53407683, 1.265060241],
        [30.01239157, 0.108433735],
        [32.49070632, 1.518072289],
        [40.04956629, 5.746987952],
        [49.96282528, 11.85542169],
        [60, 19.37349398],
    ]
)

# Define Particle Radii and Parameters
particle_radius = np.linspace(10e-6, 60e-6, 50)
temperature = 273  # Temperature in Kelvin
particle_density = 1000  # Particle density in kg/m³
fluid_density = 1.0  # Fluid (air) density in kg/m³

turbulent_dissipation = 400 * convert_units(
    "cm^2/s^3", "m^2/s^3"
)  # Example value in m²/s³
reynolds_lambda = 72.41  # Example value


# Calculate dynamic and kinematic viscosity
dynamic_viscosity = get_dynamic_viscosity(temperature)
kinematic_viscosity = get_kinematic_viscosity(dynamic_viscosity, fluid_density)
kolmogorov_time = get_kolmogorov_time(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)
# Calculate Particle Inertia Time
particle_inertia_time = get_particle_inertia_time(
    particle_radius=particle_radius,
    particle_density=particle_density,
    fluid_density=fluid_density,
    kinematic_viscosity=kinematic_viscosity,
)

# Calculate Particle Settling Velocity
mean_free_path = gas_properties.molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)
# 2. Slip correction factors
knudsen_number = properties.calculate_knudsen_number(
    mean_free_path=mean_free_path, particle_radius=particle_radius
)
slip_correction_factor = properties.cunningham_slip_correction(knudsen_number)
particle_settling_velocity = get_particle_settling_velocity_with_drag(
    particle_radius=particle_radius,
    particle_density=particle_density,
    fluid_density=fluid_density,
    dynamic_viscosity=dynamic_viscosity,
    slip_correction_factor=slip_correction_factor,
    re_threshold=0.1,
)

# Calculate Fluid RMS Velocity
fluid_rms_velocity = get_fluid_rms_velocity(
    re_lambda=reynolds_lambda,
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)

# Calculate Turbulence Scales
taylor_microscale = get_taylor_microscale(
    fluid_rms_velocity=fluid_rms_velocity,
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)
eulerian_integral_length = get_eulerian_integral_length(
    fluid_rms_velocity=fluid_rms_velocity,
    turbulent_dissipation=turbulent_dissipation,
)
lagrangian_integral_time = get_lagrangian_integral_time(
    fluid_rms_velocity=fluid_rms_velocity,
    turbulent_dissipation=turbulent_dissipation,
)
normalized_accel_variance = get_normalized_accel_variance_ao2008(
    re_lambda=reynolds_lambda
)
lagrangian_taylor_microscale_time = get_lagrangian_taylor_microscale_time(
    kolmogorov_time=kolmogorov_time,
    re_lambda=reynolds_lambda,
    accel_variance=normalized_accel_variance,
)


# Calculate Collisional Radius
collisional_radius = (
    particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :]
)

# %%
# Calculate Velocity Dispersion
velocity_dispersion = get_relative_velocity_variance(
    fluid_rms_velocity=fluid_rms_velocity,
    collisional_radius=collisional_radius,
    particle_inertia_time=particle_inertia_time,
    particle_velocity=particle_settling_velocity,
    taylor_microscale=taylor_microscale,
    eulerian_integral_length=eulerian_integral_length,
    lagrangian_integral_time=lagrangian_integral_time,
    lagrangian_taylor_microscale_time=lagrangian_taylor_microscale_time,
)

fig, ax = plt.subplots(figsize=(7, 9))
graph = ax.contourf(velocity_dispersion, cmap="viridis", origin="lower")
plt.xlabel("Particle Radius")
plt.ylabel("Particle Radius")
plt.title("Velocity Dispersion")
plt.colorbar(graph)
plt.show()


# %%
def radial_velocity_calc(velocity_dispersion, particle_inertia_time):
    # Debugging: Print the value of velocity_dispersion

    # Check if velocity_dispersion contains NaN
    if np.isnan(velocity_dispersion).any():
        print("Warning: velocity_dispersion contains NaN")

    # Compute Radial Relative Velocities
    radial_relative_velocity = get_radial_relative_velocity_dz2002(
        velocity_dispersion,
        particle_inertia_time,
    )

    return radial_relative_velocity


# Compute Radial Relative Velocities
radial_relative_velocity = radial_velocity_calc(
    np.abs(velocity_dispersion), particle_inertia_time
)

# %%
index = np.argmin(np.abs(particle_radius - 30e-6))
# Plot the Comparison Graph
plt.plot(
    particle_radius * 1e6,
    radial_relative_velocity[:, :] * 100,
    label="Model Prediction",
    color="brown",
    alpha=0.5,
)
plt.plot(
    particle_radius * 1e6,
    radial_relative_velocity[:, index] * 100,
    label="Model Prediction at 30 µm",
    color="blue",
    linestyle="--",
)
plt.scatter(data[:, 0], data[:, 1], label="DNS Data", color="purple")
plt.xlabel("Particle Radius (µm)")
plt.ylabel("Radial Relative Velocity (m/s)")
plt.title("Radial Relative Velocity Comparison")
# plt.legend()
plt.grid(True)
plt.show()

# # image plot of the radial relative velocity
fig, ax = plt.subplots(figsize=(5, 5))
graph = ax.contourf(radial_relative_velocity, cmap="viridis", origin="lower")
plt.xlabel("Particle Radius")
plt.ylabel("Particle Radius")
plt.title("Radial Relative Velocity")
plt.colorbar(graph)
plt.show()

"""
This script compares the radial relative velocities between DNS data and the model prediction.
"""

# %%
