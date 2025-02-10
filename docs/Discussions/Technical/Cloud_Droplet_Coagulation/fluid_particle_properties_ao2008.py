# %% [markdown]

"""
# Fluid and Particle Properties for Cloud Droplet Coagulation

This notebook checks the `Particula` caculated values against the values from the paper "Aerosol Science and Technology" by Ayala and Rosa (2008).

Reference: Table 2 and Table 3

Ayala, O., Rosa, B., Wang, L. P., & Grabowski, W. W. (2008). Effects of turbulence on the geometric collision rate of sedimenting droplets. Part 1. Results from direct numerical simulation. New Journal of Physics, 10. https://doi.org/10.1088/1367-2630/10/7/075015

"""


# %%
import numpy as np

from particula.particles import properties
from particula.gas import properties as gas_properties

from particula.util.constants import STANDARD_GRAVITY
from particula.util.converting.units import convert_units


# %% [markdown]
"""
## Model Equations and Parameters

In this section, we define the particle radii and other parameters needed for the calculations. These include temperature, particle density, fluid density, and air velocity.
"""
# %%
particle_radius = np.linspace(10e-6, 60e-6, 6)
temperature = 273  # Temperature in Kelvin
particle_density = 1000  # Particle density in kg/m³
fluid_density = 1.0  # Fluid (air) density in kg/m³
air_velocity = 1e-9  # Relative velocity in m/s

turbulent_dissipation = 400 * convert_units(
    "cm^2/s^3", "m^2/s^3"
)  # Example value in m²/s³
reynolds_lambda = 72.41  # Example value


# %% [markdown]
"""
### Calculate Dynamic and Kinematic Viscosity

We calculate the dynamic and kinematic viscosity of the fluid using the temperature and fluid density.
"""
# %%
dynamic_viscosity = gas_properties.get_dynamic_viscosity(temperature)
kinematic_viscosity = gas_properties.get_kinematic_viscosity(
    dynamic_viscosity, fluid_density
)
kolmogorov_time = gas_properties.get_kolmogorov_time(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)

# %% [markdown]
"""
### Calculate Particle Settling Velocity

This section calculates the particle settling velocity using the slip correction factor and other parameters.
"""
# %%
mean_free_path = gas_properties.molecule_mean_free_path(
    temperature=temperature, dynamic_viscosity=dynamic_viscosity
)

# 2. Slip correction factors
knudsen_number = properties.calculate_knudsen_number(
    mean_free_path=mean_free_path, particle_radius=particle_radius
)
slip_correction_factor = properties.cunningham_slip_correction(knudsen_number)
# iterative terminal settling velocity
iterative_settling_velocity = (
    properties.get_particle_settling_velocity_with_drag(
        particle_radius=particle_radius,
        particle_density=particle_density,
        fluid_density=fluid_density,
        dynamic_viscosity=dynamic_viscosity,
        slip_correction_factor=slip_correction_factor,
        gravitational_acceleration=STANDARD_GRAVITY,
    )
)
settling_velocity = properties.particle_settling_velocity(
    particle_radius=particle_radius,
    particle_density=particle_density,
    slip_correction_factor=slip_correction_factor,
    dynamic_viscosity=dynamic_viscosity,
)
relative_velocity = iterative_settling_velocity - air_velocity
# %% [markdown]
"""
### Calculate Particle Inertia Time

We calculate the particle inertia time, which is a measure of how quickly a particle responds to changes in the surrounding fluid.
"""
# %%
particle_inertia_time = properties.get_particle_inertia_time(
    particle_radius=particle_radius,
    particle_density=particle_density,
    fluid_density=fluid_density,
    kinematic_viscosity=kinematic_viscosity,
)

re_p = 2 * particle_radius * relative_velocity / kinematic_viscosity
f_re_p = 1 + 0.15 * re_p**0.687

ao2008_re_p = np.array([0.015, 0.116, 0.378, 0.851, 1.566, 2.537])
ao2008_t_p = np.array([0.0013, 0.0052, 0.0118, 0.0209, 0.0327, 0.0471])
ao2008_f_re_p = np.array([1.008, 1.034, 1.077, 1.134, 1.204, 1.284])

# calculate relative velocity from re_p
ao2008_velocity = ao2008_re_p * kinematic_viscosity / (2 * particle_radius)


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

# %% [markdown]
"""
### Comparison of Paper Values and Computed Values

We compare the values from the paper with the computed values from the Particula library. This includes the Reynolds number, inertia time, and settling velocity.
"""
# %%
import matplotlib.pyplot as plt

# Calculate percent error
percent_error_re_p = 100 * (re_p - ao2008_re_p) / ao2008_re_p
percent_error_tp = 100 * (particle_inertia_time - ao2008_t_p) / ao2008_t_p
percent_error_velocity = (
    100 * (particle_settling_velocity - ao2008_velocity) / ao2008_velocity
)

# Plot comparison
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

ax[0].plot(
    particle_radius * 1e6,
    ao2008_re_p,
    "o-",
    color="blue",
    alpha=0.6,
    label="ao2008 Re_p",
)
ax[0].plot(
    particle_radius * 1e6, re_p, "x--", color="blue", label="Particula Re_p"
)
ax[0].set_title("Reynolds Number Comparison")
ax[0].set_xlabel("Radius (µm)")
ax[0].set_ylabel("Re_p")
ax[0].legend()

ax[1].plot(
    particle_radius * 1e6,
    ao2008_t_p,
    "o-",
    color="green",
    alpha=0.6,
    label="ao2008 t_p",
)
ax[1].plot(
    particle_radius * 1e6,
    particle_inertia_time,
    "x--",
    color="green",
    label="Particula t_p",
)
ax[1].set_title("Inertia Time Comparison")
ax[1].set_xlabel("Radius (µm)")
ax[1].set_ylabel("t_p (s)")
ax[1].legend()

ax[2].plot(
    particle_radius * 1e6,
    ao2008_velocity * 100,
    "o-",
    color="red",
    alpha=0.6,
    label="ao2008 Velocity",
)
ax[2].plot(
    particle_radius * 1e6,
    particle_settling_velocity * 100,
    "x--",
    color="red",
    label="Particula Velocity",
)
ax[2].set_title("Settling Velocity Comparison")
ax[2].set_xlabel("Radius (µm)")
ax[2].set_ylabel("Velocity (cm/s)")
ax[2].legend()

plt.tight_layout()
plt.show()

# Print percent error
print("Percent Error in Re_p:", percent_error_re_p)
print("Percent Error in t_p:", percent_error_tp)
print("Percent Error in Settling Velocity:", percent_error_velocity)
print("Paper Values From Table 2")
print("Radius (µm) | tp | Settling Velocity (cm/s)| Re_p | f(Re_p)")
print("-" * 50)
for radius, tp, settling_velocity, re, f_re in zip(
    particle_radius,
    ao2008_t_p,
    ao2008_velocity * 100,
    ao2008_re_p,
    ao2008_f_re_p,
):
    print(
        f"{radius * 1e6:.1f} \t | {tp:.4f} | {settling_velocity:.2f} \t\t | {re:.3f} | {f_re:.3f}"
    )

# print settling velocity in a table format
print("Particula Computed Values")
print("Radius (µm) | tp | Settling Velocity (cm/s)| Re_p | f(Re_p)")
print("-" * 50)
for radius, tp, settling_velocity, re, f_re in zip(
    particle_radius,
    particle_inertia_time,
    particle_settling_velocity,
    re_p,
    f_re_p,
):
    print(
        f"{radius * 1e6:.1f} \t | {tp:.4f} | {settling_velocity * 100:.2f} \t\t | {re:.3f} | {f_re:.3f}"
    )


# %% Compare Table 3 from paper with computed values
# Characteristic scales of cloud droplets.


# %% [markdown]
"""
### Stokes Number and Velocity Comparison

We calculate and compare the Stokes number and velocity for different turbulent dissipation rates.
"""
# %%
length_kolmogorov = gas_properties.get_kolmogorov_length(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=turbulent_dissipation,
)

timescale_kolmogorov_10 = gas_properties.get_kolmogorov_time(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=10 * convert_units("cm^2/s^3", "m^2/s^3"),
)
velocity_kolmogorov_10 = gas_properties.get_kolmogorov_velocity(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=10 * convert_units("cm^2/s^3", "m^2/s^3"),
)
stokes_number_10 = properties.get_stokes_number(
    particle_inertia_time=particle_inertia_time,
    kolmogorov_time=timescale_kolmogorov_10,
)
stokes_velocity_10 = particle_settling_velocity / velocity_kolmogorov_10

# 100 cm^2/s^3
timescale_kolmogorov_100 = gas_properties.get_kolmogorov_time(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=100 * convert_units("cm^2/s^3", "m^2/s^3"),
)
velocity_kolmogorov_100 = gas_properties.get_kolmogorov_velocity(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=100 * convert_units("cm^2/s^3", "m^2/s^3"),
)
stokes_number_100 = properties.get_stokes_number(
    particle_inertia_time=particle_inertia_time,
    kolmogorov_time=timescale_kolmogorov_100,
)
stokes_velocity_100 = particle_settling_velocity / velocity_kolmogorov_100

# 400 cm^2/s^3
timescale_kolmogorov_400 = gas_properties.get_kolmogorov_time(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=400 * convert_units("cm^2/s^3", "m^2/s^3"),
)
velocity_kolmogorov_400 = gas_properties.get_kolmogorov_velocity(
    kinematic_viscosity=kinematic_viscosity,
    turbulent_dissipation=400 * convert_units("cm^2/s^3", "m^2/s^3"),
)
stokes_number_400 = properties.get_stokes_number(
    particle_inertia_time=particle_inertia_time,
    kolmogorov_time=timescale_kolmogorov_400,
)
stokes_velocity_400 = particle_settling_velocity / velocity_kolmogorov_400


# from paper

st_ao2008 = np.array(
    [
        [0.010, 0.032, 0.063],
        [0.040, 0.127, 0.253],
        [0.090, 0.285, 0.570],
        [0.160, 0.507, 1.014],
        [0.250, 0.792, 1.585],
        [0.361, 1.141, 2.282],
    ]
)
sv_ao2008 = np.array(
    [
        [1.113, 0.626, 0.442],
        [4.343, 2.442, 1.727],
        [9.385, 5.278, 3.732],
        [15.841, 8.908, 6.299],
        [23.316, 13.111, 9.271],
        [31.478, 17.701, 12.516],
    ]
)

# %% [markdown]
"""
### Comparison of Stokes Number and Velocity

We compare the Stokes number and velocity from the paper with the computed values for different turbulent dissipation rates.
"""
# %%
print("Paper Values From Table 3")
print("-" * 80)
print("Radius (µm) | St (10 cm^2/s^3) | St (100 cm^2/s^3) | St (400 cm^2/s^3)")
print("-" * 80)
for radius, st_10, st_100, st_400 in zip(
    particle_radius,
    st_ao2008[:, 0],
    st_ao2008[:, 1],
    st_ao2008[:, 2],
):
    print(
        f"{radius * 1e6:.1f} \t | {st_10:.4f} \t\t | {st_100:.4f} \t\t | {st_400:.4f}"
    )

# Plot Stokes number comparison
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].plot(
    particle_radius * 1e6,
    st_ao2008[:, 0],
    "o-",
    color="blue",
    alpha=0.6,
    label="ao2008 St (10 cm^2/s^3)",
)
ax[0].plot(
    particle_radius * 1e6,
    stokes_number_10,
    "x--",
    color="blue",
    label="Particula St (10 cm^2/s^3)",
)
ax[0].plot(
    particle_radius * 1e6,
    st_ao2008[:, 1],
    "o-",
    color="green",
    alpha=0.6,
    label="ao2008 St (100 cm^2/s^3)",
)
ax[0].plot(
    particle_radius * 1e6,
    stokes_number_100,
    "x--",
    color="green",
    label="Particula St (100 cm^2/s^3)",
)
ax[0].plot(
    particle_radius * 1e6,
    st_ao2008[:, 2],
    "o-",
    color="red",
    alpha=0.6,
    label="ao2008 St (400 cm^2/s^3)",
)
ax[0].plot(
    particle_radius * 1e6,
    stokes_number_400,
    "x--",
    color="red",
    label="Particula St (400 cm^2/s^3)",
)
ax[0].set_title("Stokes Number Comparison")
ax[0].set_xlabel("Radius (µm)")
ax[0].set_ylabel("Stokes Number")
ax[0].legend()

# Plot Stokes velocity comparison
ax[1].plot(
    particle_radius * 1e6,
    sv_ao2008[:, 0],
    "o-",
    color="blue",
    alpha=0.6,
    label="ao2008 Sv (10 cm^2/s^3)",
)
ax[1].plot(
    particle_radius * 1e6,
    stokes_velocity_10,
    "x--",
    color="blue",
    label="Particula Sv (10 cm^2/s^3)",
)
ax[1].plot(
    particle_radius * 1e6,
    sv_ao2008[:, 1],
    "o-",
    color="green",
    alpha=0.6,
    label="ao2008 Sv (100 cm^2/s^3)",
)
ax[1].plot(
    particle_radius * 1e6,
    stokes_velocity_100,
    "x--",
    color="green",
    label="Particula Sv (100 cm^2/s^3)",
)
ax[1].plot(
    particle_radius * 1e6,
    sv_ao2008[:, 2],
    "o-",
    color="red",
    alpha=0.6,
    label="ao2008 Sv (400 cm^2/s^3)",
)
ax[1].plot(
    particle_radius * 1e6,
    stokes_velocity_400,
    "x--",
    color="red",
    label="Particula Sv (400 cm^2/s^3)",
)
ax[1].set_title("Stokes Velocity Comparison")
ax[1].set_xlabel("Radius (µm)")
ax[1].set_ylabel("Stokes Velocity")
ax[1].legend()

plt.tight_layout()
plt.show()

print("Particula Computed Values")
print("Radius (µm) | St (10 cm^2/s^3) | St (100 cm^2/s^3) | St (400 cm^2/s^3)")
print("-" * 80)
for radius, st_10, st_100, st_400 in zip(
    particle_radius,
    stokes_number_10,
    stokes_number_100,
    stokes_number_400,
):
    print(
        f"{radius * 1e6:.1f} \t | {st_10:.4f} \t\t | {st_100:.4f} \t\t | {st_400:.4f}"
    )


# print ao2008 values
print("Paper Values From Table 3")
print("-" * 80)
print("Radius (µm) | Sv (10 cm^2/s^3) | Sv (100 cm^2/s^3) | Sv (400 cm^2/s^3)")
print("-" * 80)
for radius, sv_10, sv_100, sv_400 in zip(
    particle_radius,
    sv_ao2008[:, 0],
    sv_ao2008[:, 1],
    sv_ao2008[:, 2],
):
    print(
        f"{radius * 1e6:.1f} \t | {sv_10:.4f} \t\t | {sv_100:.4f} \t\t | {sv_400:.4f}"
    )

# print stokes velocity in a table format
print(
    "Radius (µm)  | Sv (10 cm^2/s^3) | Sv (100 cm^2/s^3) | Sv (400 cm^2/s^3)"
)
print("-" * 80)
for radius, stv_10, stv_100, stv_400 in zip(
    particle_radius,
    stokes_velocity_10,
    stokes_velocity_100,
    stokes_velocity_400,
):
    print(
        f"{radius * 1e6:.1f} \t | {stv_10:.4f} \t\t | {stv_100:.4f} \t\t | {stv_400:.4f}"
    )
