# %%
from typing import Union
import numpy as np
from numpy.typing import NDArray

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
    get_particle_settling_velocity_with_drag,
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
particle_radius = np.linspace(10e-6, 60e-6, 6)
temperature = 273  # Temperature in Kelvin
particle_density = 1000  # Particle density in kg/m³
fluid_density = 1.0  # Fluid (air) density in kg/m³
air_velocity = 1e-9  # Relative velocity in m/s

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

# Calculate Particle Settling Velocity
mean_free_path = gas_properties.molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)

# 2. Slip correction factors
knudsen_number = properties.calculate_knudsen_number(
    mean_free_path=mean_free_path, particle_radius=particle_radius
)
slip_correction_factor = properties.cunningham_slip_correction(knudsen_number)
# iterative terminal settling velocity
iterative_settling_velocity = get_particle_settling_velocity_with_drag(
    particle_radius=particle_radius,
    particle_density=particle_density,
    fluid_density=fluid_density,
    dynamic_viscosity=dynamic_viscosity,
    slip_correction_factor=slip_correction_factor,
    gravitational_acceleration=STANDARD_GRAVITY,
)
settling_velocity = properties.particle_settling_velocity(
    particle_radius=particle_radius,
    particle_density=particle_density,
    slip_correction_factor=slip_correction_factor,
    dynamic_viscosity=dynamic_viscosity,
)
relative_velocity = iterative_settling_velocity - air_velocity
# Calculate Particle Inertia Time
particle_inertia_time = get_particle_inertia_time(
    particle_radius=particle_radius,
    particle_density=particle_density,
    fluid_density=fluid_density,
    kinematic_viscosity=kinematic_viscosity,
)

print(f"v: {kinematic_viscosity}")

velocity_kolmogorov = gas_properties.get_kolmogorov_velocity(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)
length_kolmogorov = gas_properties.get_kolmogorov_length(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)
timescale_kolmogorov = gas_properties.get_kolmogorov_time(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)
print(f"τ_k: {timescale_kolmogorov} s")
print(f"η: {length_kolmogorov * 100} cm")
print(f"v_k: {velocity_kolmogorov * 100} cm/s")


re_p = 2 * particle_radius * relative_velocity / kinematic_viscosity
f_re_p = 1 + 0.15 * re_p**0.687

manual_re_p = np.array([0.015, 0.116, 0.378, 0.851, 1.566, 2.537])
manual_t_p = np.array([0.0013, 0.0052, 0.0118, 0.0209, 0.0327, 0.0471])
manual_f_re_p = np.array([1.008, 1.034, 1.077, 1.134, 1.204, 1.284])
v_p = particle_inertia_time * STANDARD_GRAVITY / manual_f_re_p

# calculate relative velocity from re_p
manual_velocity = manual_re_p * kinematic_viscosity / (2 * particle_radius)


# print relative velocity in a table format
print(
    "Particle Radius (µm) | Relative Velocity (cm/s)| Manual Relative Velocity (cm/s)"
)
print("-" * 50)
for radius, velocity, manual_v, i_v in zip(
    particle_radius,
    relative_velocity,
    manual_velocity,
    iterative_settling_velocity,
):
    print(
        f"{radius * 1e6:.3f} \t\t | {velocity * 100:.3f} \t\t | {manual_v * 100:.3f} | {i_v * 100:.3f}"
    )


# print settling velocity in a table format
print("Particle Radius (µm) | Manual Settling Velocity (cm/s)| Re_p | f(Re_p)")
print("-" * 50)
for radius, v_index, re, f_re in zip(particle_radius, v_p, re_p, f_re_p):
    print(
        f"{radius * 1e6:.3f} \t\t | {v_index * 100:.2f} | {re:.3f} | {f_re:.3f}"
    )

# print manual re_p f_re_p in a table format vs calculated re_p f_re_p
print("Particle Radius (µm) | Re_p | f(Re_p) | Manual Re_p | Manual f(Re_p)")
print("-" * 50)
for radius, re, f_re, manual_re, manual_f_re in zip(
    particle_radius, re_p, f_re_p, manual_re_p, manual_f_re_p
):
    print(
        f"{radius * 1e6:.3f} \t\t | {re:.3f} | {f_re:.3f} | {manual_re:.3f} | {manual_f_re:.3f}"
    )

particle_settling_velocity = (
    properties.get_particle_settling_velocity_via_inertia(
        particle_inertia_time=particle_inertia_time,
        particle_radius=particle_radius,
        relative_velocity=iterative_settling_velocity,
        slip_correction_factor=slip_correction_factor,
        gravitational_acceleration=STANDARD_GRAVITY,
        kinematic_viscosity=kinematic_viscosity,
    )
)

particle_settling_velocity_via_manual = (
    particle_inertia_time * STANDARD_GRAVITY / manual_f_re_p
)


# print settling velocity in a table format
print("Radius (µm) | tp | Settling Velocity (cm/s)| Re_p | f(Re_p)")
print("-" * 50)
for radius, tp, settling_velocity, re, f_re in zip(
    particle_radius,
    particle_inertia_time,
    particle_settling_velocity,
    # particle_settling_velocity_via_manual,
    re_p,
    f_re_p,
):
    print(
        f"{radius * 1e6:.1f} \t | {tp:.4f} | {settling_velocity * 100:.2f} \t\t | {re:.3f} | {f_re:.3f}"
    )

# get stokes number

stokes_number = properties.get_stokes_number(
    particle_inertia_time=particle_inertia_time,
    kolmogorov_time=timescale_kolmogorov,
)
stokes_velocity = particle_settling_velocity / velocity_kolmogorov

# print stokes number in a table format
print("Particle Radius (µm) | Stokes Number | Stokes Velocity")
print("-" * 50)
for radius, stokes, stokes_v in zip(
    particle_radius, stokes_number, stokes_velocity
):
    print(f"{radius * 1e6:.2f} \t\t | {stokes:.2f} | {stokes_v:.2f}")


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

    print(f"Fluid RMS Velocity: {fluid_rms_velocity * 100} cm/s")
    print(f"Taylor Microscale: {taylor_microscale * 100} cm")
    print(f"Eulerian Integral Length: {eulerian_integral_length * 100} cm")
    print(f"Lagrangian Integral Time: {lagrangian_integral_time} s")
    print(
        f"Lagrangian Taylor Microscale Time: {lagrangian_taylor_microscale_time} s"
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

# %%
model_rms_100cm2_s3 = calculate_horizontal_velocity(
    turbulent_dissipation=100 * convert_units("cm^2/s^3", "m^2/s^3"),
    reynolds_lambda=reynolds_lambda,
)

model_rms_400cm2_s3 = calculate_horizontal_velocity(
    turbulent_dissipation=400 * convert_units("cm^2/s^3", "m^2/s^3"),
    reynolds_lambda=reynolds_lambda,
)


# # Plot the Comparison Graph
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
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.grid(True)
plt.subplots_adjust(bottom=0.2)
plt.show()

"""
This script compares the mean-square horizontal velocities between DNS data and the model prediction.
"""

# %%
