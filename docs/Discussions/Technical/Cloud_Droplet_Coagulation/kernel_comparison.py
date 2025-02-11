# %% [markdown]

"""
DNS Kernel Comparison


Reference: Figure 18a form

Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 2. Theory and parameterization. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075016


"""

# %%


import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.kernel_ao2008 import (
    get_kernel_ao2008,
    get_kernel_ao2008_via_system_state,
)

from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_dz2002,
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
particle_radius = np.linspace(1e-6, 60e-6, 200)  # From 1 µm to 60 µm

# Convert turbulent dissipation from cm²/s³ to m²/s³
turbulent_dissipation = 400 * convert_units("cm^2/s^3", "m^2/s^3")
reynolds_lambda = 72.41  # Example value

# %% [markdown]
"""
## Define the Kernel Calculation Function

This function calculates the collision kernel values using the specified parameters and the `particula` package implementations.
"""


def kernel_calc(particle_radius, turbulent_dissipation, reynolds_lambda):
    # Define constants and parameters
    temperature = 273  # Temperature in Kelvin
    particle_density = 1000  # Particle density in kg/m³
    fluid_density = 1.0  # Fluid (air) density in kg/m³

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
    )
    particle_settling_velocity = (
        properties.get_particle_settling_velocity_with_drag(
            particle_radius=particle_radius,
            particle_density=particle_density,
            fluid_density=fluid_density,
            dynamic_viscosity=dynamic_viscosity,
            slip_correction_factor=slip_correction_factor,
            re_threshold=0.1,
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
    lagrangian_taylor_microscale_time = (
        gas_properties.get_lagrangian_taylor_microscale_time(
            kolmogorov_time=kolmogorov_time,
            re_lambda=reynolds_lambda,
            accel_variance=normalized_accel_variance,
        )
    )

    # 5. Relative velocity variance
    velocity_dispersion = get_relative_velocity_variance(
        fluid_rms_velocity=fluid_rms_velocity,
        collisional_radius=collisional_radius,
        particle_inertia_time=particle_inertia_time,
        particle_velocity=np.abs(particle_settling_velocity),
        taylor_microscale=taylor_microscale,
        eulerian_integral_length=eulerian_integral_length,
        lagrangian_integral_time=lagrangian_integral_time,
        lagrangian_taylor_microscale_time=lagrangian_taylor_microscale_time,
    )

    # Compute Kernel Values
    kernel_values = get_kernel_ao2008(
        particle_radius=particle_radius,
        velocity_dispersion=np.abs(velocity_dispersion),
        particle_inertia_time=particle_inertia_time,
        stokes_number=stokes_number,
        kolmogorov_length_scale=kolmogorov_length_scale,
        reynolds_lambda=reynolds_lambda,
        normalized_accel_variance=normalized_accel_variance,
        kolmogorov_velocity=kolmogorov_velocity,
        kolmogorov_time=kolmogorov_time,
    )

    return kernel_values


# Compute Kernel Values
kernel_values = kernel_calc(
    particle_radius, turbulent_dissipation, reynolds_lambda
)

# %% [markdown]
"""
## Compute Kernel via System State

We compute the kernel using an alternative method that utilizes the system state directly.
"""

kernel_via_system_state = get_kernel_ao2008_via_system_state(
    particle_radius=particle_radius,
    particle_density=1000,
    fluid_density=1.0,
    temperature=273,
    turbulent_dissipation=400 * convert_units("cm^2/s^3", "m^2/s^3"),
    re_lambda=72.41,
    relative_velocity=0.0,
)
print(kernel_via_system_state)

# %% [markdown]
"""
## Plot the Comparison Graph

We plot the DNS data and their corresponding model predictions on the same graph for easy comparison.
"""

index = np.argmin(np.abs(particle_radius - 30e-6))

plt.scatter(data[:, 0], data[:, 1], label="DNS Data", color="cyan")
plt.plot(
    particle_radius * 1e6,
    kernel_values[:, index] * convert_units("m^3/s", "cm^3/s"),
    label="Model Prediction",
    color="magenta",
    alpha=0.5,
    linewidth=5,
)
plt.plot(
    particle_radius * 1e6,
    kernel_via_system_state[:, index] * convert_units("m^3/s", "cm^3/s"),
    label="Model Prediction via System State",
    color="green",
    alpha=0.5,
)
plt.xlabel("Particle Radius (µm)")
plt.ylabel("Collision Kernel (cm³/s)")
plt.title("Collision Kernel Comparison")
# plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
"""
## Calculate Percent Error

We calculate the percent error between the model predictions and the DNS data to assess the accuracy of our implementation.
"""

# Extract DNS data
dns_radii = data[:, 0] * 1e-6  # Convert from µm to meters
dns_kernels = data[:, 1] * convert_units("cm^3/s", "m^3/s")

# Interpolate model predictions at DNS radii
from scipy.interpolate import interp1d

interpolator = interp1d(
    particle_radius,
    kernel_values[:, np.argmin(np.abs(particle_radius - 30e-6))],
    kind="linear",
    fill_value="extrapolate",
)
model_kernels_at_dns = interpolator(dns_radii)

# Calculate percent error
percent_errors = (model_kernels_at_dns - dns_kernels) / dns_kernels * 100

# %% [markdown]
"""
## Display Comparison Table

We display the DNS data, model predictions, and percent errors in a table for comparison.
"""

import pandas as pd

results_df = pd.DataFrame(
    {
        "Radius (µm)": data[:, 0],
        "DNS Kernel (cm³/s)": data[:, 1],
        "Model Kernel (cm³/s)": model_kernels_at_dns
        * convert_units("m^3/s", "cm^3/s"),
        "Percent Error (%)": percent_errors,
    }
)

print(results_df)

"""
This script compares the collision kernel Γ₁₂ between DNS data and the model prediction.
"""

# %%
