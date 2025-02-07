"""
This script compares the radial distribution function g₁₂ between DNS data and the model prediction.

It uses the function `get_g12_radial_distribution_ao2008` from the `particula` library to compute g₁₂ values over a range of particle radii.

The script then plots these computed values against the DNS dataset `r23_e100` for visual comparison.

Usage:
- Run this script to generate and display the comparison graph.
"""

import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008 import (
    get_g12_radial_distribution_ao2008,
)

# DNS datasets for radial distribution function
r23_e100 = np.array([
    [9.937578027, 1.532846715],
    [19.98751561, 1.094890511],
    [29.91260924, 2.299270073],
    [40.02496879, 3.686131387],
    [49.95006242, 2.919708029],
    [60, 2.737226277],
])

# Define the particle radii and other required parameters
particle_radius = np.linspace(1e-6, 60e-6, 100)  # From 1 µm to 60 µm
stokes_number = np.linspace(0.1, 1.0, 100)       # Example Stokes numbers
kolmogorov_length_scale = 1e-3                   # Example value
reynolds_lambda = 23                             # From the dataset
normalized_accel_variance = 0.5                  # Example value
kolmogorov_velocity = 0.1                        # Example value
kolmogorov_time = 0.01                           # Example value

# Compute g₁₂ Values
g12_values = get_g12_radial_distribution_ao2008(
    particle_radius,
    stokes_number,
    kolmogorov_length_scale,
    reynolds_lambda,
    normalized_accel_variance,
    kolmogorov_velocity,
    kolmogorov_time,
)

# Plot the Comparison Graph
plt.scatter(r23_e100[:, 0], r23_e100[:, 1], label='DNS Data', color='blue')
plt.plot(particle_radius * 1e6, g12_values, label='Model Prediction', color='red')
plt.xlabel('Particle Radius (µm)')
plt.ylabel('Radial Distribution Function g₁₂')
plt.title('Radial Distribution Function Comparison')
plt.legend()
plt.grid(True)
plt.show()

