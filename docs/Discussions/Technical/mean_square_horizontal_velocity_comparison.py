# %%
import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_ao2008,
)
from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008 import (
    get_relative_velocity_variance,
    _compute_rms_fluctuation_velocity,
    VelocityCorrelationTerms,
)
from particula.particles.properties.inertia_time import (
    get_particle_inertia_time,
)
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_inertia,
)
from particula.particles import properties
from particula.gas import properties as gas_properties
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


# %% DNS values


# Figure 12: Comparison of the predicted and simulated mean-square horizontal
# particle velocities for droplets falling in a turbulent ﬂow of Rλ = 72.41 and
# turbulent_dissipation = 400 cm2 s−3

# droplet radius (a2, microns) vs rms_velocity (cm2/s2)

# dns_10cm2/s3: 6 rows, 2 columns (X, Y)
dns_10cm2_s3 = np.array(
    [
        [9.938118812, 26.66666667],
        [20.02475248, 26.41975309],
        [30.04950495, 26.41975309],
        [40.01237624, 24.69135802],
        [50.16089109, 22.71604938],
        [60.06188119, 18.51851852],
    ]
)

# dns_100_cm2/s3: 6 rows, 2 columns (X, Y)
dns_100_cm2_s3 = np.array(
    [
        [9.938118812, 84.44444444],
        [20.02475248, 80.98765432],
        [29.98762376, 77.03703704],
        [39.95049505, 71.11111111],
        [49.97524752, 59.25925926],
        [60.06188119, 44.19753086],
    ]
)

# dns_400_cm2/s3: 6 rows, 2 columns (X, Y)
dns_400_cm2_s3 = np.array(
    [
        [9.876237624, 166.9135802],
        [20.08663366, 163.9506173],
        [30.11138614, 150.1234568],
        [40.07425743, 129.1358025],
        [50.03712871, 100.4938272],
        [60.06188119, 69.62962963],
    ]
)

# %% Model equations
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
mean_free_path = gas_properties.molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)

# 2. Slip correction factors
knudsen_number = properties.calculate_knudsen_number(
    mean_free_path=mean_free_path, particle_radius=particle_radius
)
slip_correction_factor = properties.cunningham_slip_correction(knudsen_number)
particle_settling_velocity = get_particle_settling_velocity_via_inertia(
    particle_inertia_time=particle_inertia_time,
    gravitational_acceleration=STANDARD_GRAVITY,
    slip_correction_factor=slip_correction_factor,
)


def calculate_horizontal_velocity(turbulent_dissipation, reynolds_lambda):
    """
    Helper function to calculate the mean-square horizontal velocity of particles
    for these specific cases.
    """
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

    z = compute_z(lagrangian_taylor_microscale_time, lagrangian_integral_time)
    beta = compute_beta(taylor_microscale, eulerian_integral_length)

    # Calculate v'² Values
    vel_corr_terms = VelocityCorrelationTerms(
        b1=compute_b1(z),
        b2=compute_b2(z),
        d1=compute_d1(beta),
        d2=compute_d2(beta),
        c1=compute_c1(z, lagrangian_integral_time),
        c2=compute_c2(z, lagrangian_integral_time),
        e1=compute_e1(z, eulerian_integral_length),
        e2=compute_e2(z, eulerian_integral_length),
    )

    return _compute_rms_fluctuation_velocity(
        fluid_rms_velocity, particle_inertia_time, vel_corr_terms
    )


# %%

model_rms_10cm2_s3 = calculate_horizontal_velocity(
    turbulent_dissipation=10 * convert_units("cm^2/s^3", "m^2/s^3"),
    reynolds_lambda=reynolds_lambda,
)

model_rms_100cm2_s3 = calculate_horizontal_velocity(
    turbulent_dissipation=100 * convert_units("cm^2/s^3", "m^2/s^3"),
    reynolds_lambda=reynolds_lambda,
)

model_rms_400cm2_s3 = calculate_horizontal_velocity(
    turbulent_dissipation=400 * convert_units("cm^2/s^3", "m^2/s^3"),
    reynolds_lambda=reynolds_lambda,
)


# Plot the Comparison Graph
fig, ax = plt.subplots(figsize=(6, 8))

# Case 1: R_lambda = 72.41, epsilon = 10 cm²/s³
ax.scatter(
    dns_10cm2_s3[:, 0],
    dns_10cm2_s3[:, 1],
    label=r"DNS: $R_\lambda=72.41$, $\varepsilon=10$",
    color="blue",
    marker="o",
)
ax.plot(
    particle_radius * 1e6,
    model_rms_10cm2_s3 * 1e4,
    label=r"Model: $R_\lambda=72.41$, $\varepsilon=10$",
    color="blue",
)

# Case 2: R_lambda = 72.41, epsilon = 100 cm²/s³
ax.scatter(
    dns_100_cm2_s3[:, 0],
    dns_100_cm2_s3[:, 1],
    label=r"DNS: $R_\lambda=72.41$, $\varepsilon=100$",
    color="green",
    marker="^",
)
ax.plot(
    particle_radius * 1e6,
    model_rms_100cm2_s3 * 1e4,
    label=r"Model: $R_\lambda=72.41$, $\varepsilon=100$",
    color="green",
)

# Case 3: R_lambda = 72.41, epsilon = 400 cm²/s³
ax.scatter(
    dns_400_cm2_s3[:, 0],
    dns_400_cm2_s3[:, 1],
    label=r"DNS: $R_\lambda=72.41$, $\varepsilon=400$",
    color="red",
    marker="s",
)
ax.plot(
    particle_radius * 1e6,
    model_rms_400cm2_s3 * 1e4,
    label=r"Model: $R_\lambda=72.41$, $\varepsilon=400$",
    color="red",
)

# Set labels, title, legend, etc.
ax.set_xlabel("Particle Radius (µm)")
ax.set_ylabel("Mean-Square Horizontal Velocity (cm²/s²)")
ax.set_title("Mean-Square Horizontal Velocity Comparison")
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=2
)
ax.grid(True)
plt.subplots_adjust(bottom=0.2)
plt.show()

"""
This script compares the mean-square horizontal velocities between DNS data and the model prediction.
"""

# %%
