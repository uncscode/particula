import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008 import (
    get_relative_velocity_variance,
)

# DNS dataset for mean-square horizontal velocities
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

# Define Parameters
particle_radius = np.linspace(1e-6, 60e-6, 100)
collisional_radius = particle_radius  # Adjust as needed
particle_inertia_time = np.linspace(0.01, 0.1, 100)
particle_velocity = np.linspace(0.1, 1.0, 100)
fluid_rms_velocity = 0.2
taylor_microscale = 1e-3
eulerian_integral_length = 1e-2
lagrangian_integral_time = 0.1

# Compute σ² Values
sigma_squared = get_relative_velocity_variance(
    fluid_rms_velocity,
    collisional_radius,
    particle_inertia_time,
    particle_velocity,
    taylor_microscale,
    eulerian_integral_length,
    lagrangian_integral_time,
)

# Plot the Comparison Graph
plt.scatter(
    dns_400_cm2_s3[:, 0], dns_400_cm2_s3[:, 1], label="DNS Data", color="green"
)
plt.plot(
    particle_radius * 1e6,
    sigma_squared,
    label="Model Prediction",
    color="orange",
)
plt.xlabel("Particle Radius (µm)")
plt.ylabel("Mean-Square Horizontal Velocity (cm²/s²)")
plt.title("Mean-Square Horizontal Velocity Comparison")
plt.legend()
plt.grid(True)
plt.show()

"""
This script compares the mean-square horizontal velocities between DNS data and the model prediction.
"""
