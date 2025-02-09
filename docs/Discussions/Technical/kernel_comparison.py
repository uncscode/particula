import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.kernel_ao2008 import (
    get_kernel_ao2008,
)

from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_ao2008,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008 import (
    get_g12_radial_distribution_ao2008,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008 import (
    get_relative_velocity_variance,
)

from particula.particles import properties
from particula.gas import properties as gas_properties
from particula.util.converting.units import convert_units
from particula.util.constants import STANDARD_GRAVITY

# Case 1: Comparison of Collision Kernel

# DNS dynamic collision kernel and predicted collision kernel of
# sedimenting droplets in a turbulent ﬂow. (a) a1 = 30µm, Rλ = 72.41 and  =
# 400 cm2 s−3

# plot of radius (microns) vs kernel (cm3/s)

# Dataset for kernel comparison
data = np.array(
    [
        [10.06067961, 0.000581818],
        [14.97572816, 0.000654545],
        [19.8907767, 0.000642424],
        [25.1092233, 0.000581818],
        [27.53640777, 0.000484848],
        [29.96359223, 0.000315152],
        [32.51213592, 0.000666667],
        [40.03640777, 0.001963636],
        [50.04854369, 0.004618182],
        [60, 0.009127273],
    ]
)

# %% [markdown]
"""
## Define Particle Radii and Parameters

We define the particle radii range and other necessary parameters for the calculations.
"""
particle_radius = np.linspace(1e-6, 60e-6, 100)  # From 1 µm to 60 µm

# Convert turbulent dissipation from cm²/s³ to m²/s³
turbulent_dissipation = 400 * convert_units("cm^2/s^3", "m^2/s^3")
reynolds_lambda = 72.41  # Example value


def kernel_calc(particle_radius, turbulent_dissipation, reynolds_lambda):
    # Define constants and parameters
    temperature = 300  # Temperature in Kelvin
    particle_density = 1000  # Particle density in kg/m³
    fluid_density = 1.225  # Fluid (air) density in kg/m³
    relative_velocity = 1e-6  # Relative velocity in m/s

    # 1. Basic fluid properties
    dynamic_viscosity = gas_properties.get_dynamic_viscosity(temperature)
    kinematic_viscosity = gas_properties.get_kinematic_viscosity(
        dynamic_viscosity=dynamic_viscosity, fluid_density=fluid_density
    )
    mean_free_path = gas_properties.molecule_mean_free_path(
        temperature=temperature, dynamic_viscosity=dynamic_viscosity
    )

    # 2. Slip correction factors
    knudsen_number = properties.calculate_knudsen_number(
        mean_free_path=mean_free_path, particle_radius=particle_radius
    )
    slip_correction_factor = properties.cunningham_slip_correction(
        knudsen_number
    )

    # Handle radius addition properly for arrays
    collisional_radius = (
        particle_radius[:, np.newaxis] + particle_radius[np.newaxis, :]
        if isinstance(particle_radius, np.ndarray)
        else 2.0 * particle_radius
    )

    # 3. Particle inertia and settling velocity
    particle_inertia_time = properties.get_particle_inertia_time(
        particle_radius=particle_radius,
        particle_density=particle_density,
        fluid_density=fluid_density,
        kinematic_viscosity=kinematic_viscosity,
        relative_velocity=relative_velocity,
    )
    particle_settling_velocity = (
        properties.get_particle_settling_velocity_via_inertia(
            particle_inertia_time=particle_inertia_time,
            gravitational_acceleration=STANDARD_GRAVITY,
            slip_correction_factor=slip_correction_factor,
        )
    )

    # 4. Turbulence scales
    fluid_rms_velocity = gas_properties.get_fluid_rms_velocity(
        re_lambda=reynolds_lambda,
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    taylor_microscale = gas_properties.get_taylor_microscale(
        fluid_rms_velocity=fluid_rms_velocity,
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    eulerian_integral_length = gas_properties.get_eulerian_integral_length(
        fluid_rms_velocity=fluid_rms_velocity,
        turbulent_dissipation=turbulent_dissipation,
    )
    lagrangian_integral_time = gas_properties.get_lagrangian_integral_time(
        fluid_rms_velocity=fluid_rms_velocity,
        turbulent_dissipation=turbulent_dissipation,
    )

    # 5. Relative velocity variance
    velocity_dispersion = get_relative_velocity_variance(
        fluid_rms_velocity=fluid_rms_velocity,
        collisional_radius=collisional_radius,
        particle_inertia_time=particle_inertia_time,
        particle_velocity=particle_settling_velocity,
        taylor_microscale=taylor_microscale,
        eulerian_integral_length=eulerian_integral_length,
        lagrangian_integral_time=lagrangian_integral_time,
    )

    # 6. Additional turbulence-based quantities
    kolmogorov_time = gas_properties.get_kolmogorov_time(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    stokes_number = properties.get_stokes_number(
        particle_inertia_time=particle_inertia_time,
        kolmogorov_time=kolmogorov_time,
    )
    kolmogorov_length_scale = gas_properties.get_kolmogorov_length(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )
    reynolds_lambda = properties.get_particle_reynolds_number(
        particle_radius=particle_radius,
        particle_velocity=particle_settling_velocity,
        kinematic_viscosity=kinematic_viscosity,
    )
    normalized_accel_variance = (
        gas_properties.get_normalized_accel_variance_ao2008(
            re_lambda=reynolds_lambda,
        )
    )
    kolmogorov_velocity = gas_properties.get_kolmogorov_velocity(
        kinematic_viscosity=kinematic_viscosity,
        turbulent_dissipation=turbulent_dissipation,
    )

    # Compute Kernel Values
    kernel_values = get_kernel_ao2008(
        particle_radius,
        velocity_dispersion,
        particle_inertia_time,
        stokes_number,
        kolmogorov_length_scale,
        reynolds_lambda,
        normalized_accel_variance,
        kolmogorov_velocity,
        kolmogorov_time,
    )

    return kernel_values


# Compute Kernel Values
kernel_values = kernel_calc(
    particle_radius, turbulent_dissipation, reynolds_lambda
)

print(kernel_values)
# %% [markdown]
"""
## Plot the Comparison Graph

We plot the DNS data and their corresponding model predictions on the same graph for easy comparison.
"""

plt.scatter(data[:, 0], data[:, 1], label="DNS Data", color="cyan")
plt.plot(
    particle_radius * 1e6,
    kernel_values,
    label="Model Prediction",
    color="magenta",
)
plt.xlabel("Particle Radius (µm)")
plt.ylabel("Collision Kernel (cm³/s)")
plt.title("Collision Kernel Comparison")
plt.legend()
plt.grid(True)
plt.show()

"""
This script compares the collision kernel Γ₁₂ between DNS data and the model prediction.
"""
