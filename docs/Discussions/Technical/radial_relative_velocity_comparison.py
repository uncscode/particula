import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_ao2008,
)

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
particle_inertia_time = np.linspace(0.01, 0.1, 100)
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
