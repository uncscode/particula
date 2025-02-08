import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_ao2008,
)
from particula.particles.properties.inertia_time import get_particle_inertia_time
from particula.gas.properties.kinematic_viscosity import get_kinematic_viscosity
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity

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
temperature = 300  # Temperature in Kelvin
particle_density = 1000  # Particle density in kg/m³
fluid_density = 1.225  # Fluid (air) density in kg/m³
relative_velocity = 1e-6  # Relative velocity in m/s

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
velocity_dispersion = 0.1  # Example value


def radial_velocity_calc(velocity_dispersion, particle_inertia_time):
    # Compute Radial Relative Velocities
    radial_relative_velocity = get_radial_relative_velocity_ao2008(
        velocity_dispersion,
        particle_inertia_time,
    )
    return radial_relative_velocity


# Compute Radial Relative Velocities
radial_relative_velocity = radial_velocity_calc(
    velocity_dispersion, particle_inertia_time
)

# Plot the Comparison Graph
plt.scatter(data[:, 0], data[:, 1], label="DNS Data", color="purple")
plt.plot(
    particle_radius * 1e6,
    radial_relative_velocity,
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
