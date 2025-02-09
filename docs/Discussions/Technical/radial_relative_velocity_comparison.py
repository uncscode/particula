# %%
import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_ao2008,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008 import (
    get_relative_velocity_variance,
)
from particula.particles.properties.inertia_time import (
    get_particle_inertia_time,
)
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_inertia,
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
particle_radius = np.linspace(1e-6, 60e-6, 100)
temperature = 293  # Temperature in Kelvin
particle_density = 1000  # Particle density in kg/m³
fluid_density = 1.225  # Fluid (air) density in kg/m³
relative_velocity = 1e-2  # Relative velocity in m/s

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
    relative_velocity=relative_velocity,
)

# Calculate dynamic and kinematic viscosity
dynamic_viscosity = get_dynamic_viscosity(temperature)
kinematic_viscosity = get_kinematic_viscosity(dynamic_viscosity, fluid_density)

# Calculate Particle Inertia Time
particle_inertia_time = get_particle_inertia_time(
    particle_radius=particle_radius,
    particle_density=particle_density,
    fluid_density=fluid_density,
    kinematic_viscosity=kinematic_viscosity,
    relative_velocity=relative_velocity,
)

# Calculate Particle Settling Velocity
slip_correction_factor = 1.0  # Assuming no slip correction for simplicity
particle_settling_velocity = get_particle_settling_velocity_via_inertia(
    particle_inertia_time=particle_inertia_time,
    gravitational_acceleration=STANDARD_GRAVITY,
    slip_correction_factor=slip_correction_factor,
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
z = compute_z(lagrangian_taylor_microscale_time, lagrangian_integral_time)
beta = compute_beta(taylor_microscale, eulerian_integral_length)

b1 = compute_b1(z)
array_b = np.linspace(1e-6, 0.75, 100)


b2 = compute_b2(z)
d1 = compute_d1(beta)
d2 = compute_d2(beta)
c1 = compute_c1(z, lagrangian_integral_time)
c2 = compute_c2(z, lagrangian_integral_time)
e1 = compute_e1(z, eulerian_integral_length)
e2 = compute_e2(z, eulerian_integral_length)

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


# %%
def radial_velocity_calc(velocity_dispersion, particle_inertia_time):
    # Debugging: Print the value of velocity_dispersion
    print(f"radial_velocity_calc - velocity_dispersion: {velocity_dispersion}")

    # Check if velocity_dispersion contains NaN
    if np.isnan(velocity_dispersion).any():
        print("Warning: velocity_dispersion contains NaN")

    # Compute Radial Relative Velocities
    radial_relative_velocity = get_radial_relative_velocity_ao2008(
        velocity_dispersion,
        particle_inertia_time,
    )

    # Debugging: Print the value of radial_relative_velocity
    print(
        f"radial_velocity_calc - radial_relative_velocity: {radial_relative_velocity}"
    )

    # Check if radial_relative_velocity contains NaN
    if np.isnan(radial_relative_velocity).any():
        print("Warning: radial_relative_velocity contains NaN")

    return radial_relative_velocity


# Compute Radial Relative Velocities
radial_relative_velocity = radial_velocity_calc(
    np.abs(velocity_dispersion), particle_inertia_time
)

# %%
index = np.argmin(np.abs(particle_radius - 30e-6))
# Plot the Comparison Graph
plt.scatter(data[:, 0], data[:, 1], label="DNS Data", color="purple")
plt.plot(
    particle_radius * 1e6,
    radial_relative_velocity[index, :],
    label="Model Prediction",
    color="brown",
)
plt.xlabel("Particle Radius (µm)")
plt.ylabel("Radial Relative Velocity (m/s)")
plt.title("Radial Relative Velocity Comparison")
plt.legend()
plt.grid(True)
plt.show()

"""
This script compares the radial relative velocities between DNS data and the model prediction.
"""

# %%
