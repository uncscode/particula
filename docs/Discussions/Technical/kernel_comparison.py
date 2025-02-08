import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.kernel_ao2008 import (
    get_kernel_ao2008,
)
from particula.particles import properties
from particula.gas import properties as gas_properties
from particula.util.converting.units import convert_units


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
