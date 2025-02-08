import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.kernel_ao2008 import (
    get_kernel_ao2008,
)


# Case 1: Comparison of Collision Kernel

# DNS dynamic collision kernel and predicted collision kernel of
# sedimenting droplets in a turbulent ﬂow. (a) a1 = 30µm, Rλ = 72.41 and  =
# 400 cm2 s−3

# plot of radius (microns) vs kernel (cm3/s)

# Dataset for kernel comparison
data = np.array([
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
])

# Define Parameters
particle_radius = np.linspace(1e-6, 60e-6, 100)
particle_inertia_time = np.linspace(0.01, 0.1, 100)
stokes_number = np.linspace(0.1, 1.0, 100)
velocity_dispersion = 0.1
reynolds_lambda = 72.41  # Example value
turbulent_dissipation = 400  # cm²/s³


# Compute Kernel Values
kernel_values = get_kernel_ao2008(
    particle_radius,
    velocity_dispersion,
    particle_inertia_time,
    stokes_number,
    kolmogorov_length_scale,
    reynolds_lambda,
    normalized_accel_variance,
    kolmogorov_velocity,
    kolmogorov_time,
)

# Plot the Comparison Graph
plt.scatter(data[:, 0], data[:, 1], label='DNS Data', color='cyan')
plt.plot(particle_radius * 1e6, kernel_values, label='Model Prediction', color='magenta')
plt.xlabel('Particle Radius (µm)')
plt.ylabel('Collision Kernel (cm³/s)')
plt.title('Collision Kernel Comparison')
plt.legend()
plt.grid(True)
plt.show()

"""
This script compares the collision kernel Γ₁₂ between DNS data and the model prediction.
"""
