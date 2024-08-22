"""
Example for converting aerodynamic size to a physical size
"""
# %%

import matplotlib.pyplot as plt
import numpy as np
from particula.next.particles import properties
from particula.next.gas.properties import mean_free_path
from scipy.optimize import fsolve


# %%

# define environmental conditions
temperature = 298  # K
pressure = 101325  # Pa

particle_radius = np.logspace(-7, -5, 100)  # m

# %% derived properties

mean_free_path_air = mean_free_path.molecule_mean_free_path(
    temperature=temperature, pressure=pressure
)


# Function to calculate aerodynamic length error
def cost_aerodynamic_length(
    guess_aerodynamic_length, mean_free_path_air, particle_radius
):
    physical_knudsen_number = properties.calculate_knudsen_number(
        mean_free_path_air, particle_radius
    )
    physical_slip_correction = properties.cunningham_slip_correction(
        knudsen_number=physical_knudsen_number
    )

    guess_aerodynamic_knudsen_number = properties.calculate_knudsen_number(
        mean_free_path_air, guess_aerodynamic_length
    )
    guess_aerodynamic_slip_correction = properties.cunningham_slip_correction(
        knudsen_number=guess_aerodynamic_knudsen_number
    )

    new_aerodynamic_length = properties.particle_aerodynamic_length(
        physical_length=particle_radius,
        physical_slip_correction_factor=physical_slip_correction,
        aerodynamic_slip_correction_factor=guess_aerodynamic_slip_correction,
        density=2000,
        reference_density=1000,
        aerodynamic_shape_factor=1.3,
    )

    # Error between guessed aerodynamic length and the guess for aerodynamic length
    return (new_aerodynamic_length - guess_aerodynamic_length)**2


def cost_physical_length(
    guess_phsycial_length, mean_free_path_air, aerodynamic_length
):
    guess_physical_knudsen_number = properties.calculate_knudsen_number(
        mean_free_path_air, guess_phsycial_length
    )
    guess_physical_slip_correction = properties.cunningham_slip_correction(
        knudsen_number=guess_physical_knudsen_number
    )

    aerodynamic_knudsen_number = properties.calculate_knudsen_number(
        mean_free_path_air, aerodynamic_length
    )
    aerodynamic_slip_correction = properties.cunningham_slip_correction(
        knudsen_number=aerodynamic_knudsen_number
    )

    new_aerodynamic_length = properties.particle_aerodynamic_length(
        physical_length=guess_phsycial_length,
        physical_slip_correction_factor=guess_physical_slip_correction,
        aerodynamic_slip_correction_factor=aerodynamic_slip_correction,
        density=2000,
        reference_density=1000,
        aerodynamic_shape_factor=1.3,
    )

    # Error between guessed aerodynamic length and the guess for aerodynamic length
    return (new_aerodynamic_length - aerodynamic_length) ** 2


# %% aerodynamic size root finding
initial_guess = particle_radius

optimal_aerodynamic_lengths = fsolve(
    cost_aerodynamic_length,
    initial_guess,
    args=(mean_free_path_air, particle_radius),
)

final_error = cost_aerodynamic_length(
    optimal_aerodynamic_lengths, mean_free_path_air, particle_radius
)

# inverse
initial_physical_guess = optimal_aerodynamic_lengths
optimal_physical_lengths = fsolve(
    cost_physical_length,
    initial_physical_guess,
    args=(mean_free_path_air, optimal_aerodynamic_lengths),
)

final_physical_error = cost_physical_length(
    optimal_physical_lengths, mean_free_path_air, optimal_aerodynamic_lengths
)

# %% plot

fig, ax = plt.subplots()
ax.plot(particle_radius, particle_radius, label="length")
ax.plot(particle_radius, optimal_aerodynamic_lengths, label="Aerodynamic length")
ax.plot(particle_radius, optimal_physical_lengths, label="Physical length", linestyle="--")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Aerodynamic length (m)")
ax.legend()
plt.show()

# %% ratio of aerodynamic to physical length
fig, ax = plt.subplots()
ax.plot(particle_radius, optimal_aerodynamic_lengths/optimal_physical_lengths)
ax.set_xscale("log")
ax.set_xlabel("Particle radius (m)")
ax.set_ylabel("Aerodynamic length / Physical length")
ax.set_title("Ratio of aerodynamic to physical length")
plt.show()
# %%
