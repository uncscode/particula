import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008 import (
    get_relative_velocity_variance,
)

 Comparison of the radial distribution function g₁₂ between DNS data and model prediction

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
