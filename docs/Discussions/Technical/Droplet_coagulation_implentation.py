# %% Comparison of DNS with the implementation in particula

import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.g12_radial_distribution_ao2008 import (
    get_g12_radial_distribution_ao2008,
)

"""
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
"""
# %% Data for the comparison


# Figure 11: Radial distribution function g12 for different Stokes numbers

# Comparison of the predicted and simulated radial relative velocities
# of nonsedimenting particles in a turbulent ﬂow. The particle radius is
# equal to half the Kolmogorov length scale (a = η/2). (a) Bidisperse
# system (similar to
# ﬁgure 12

# ploting tp2/Te vs radial relative velocity abs(wr)/u'

# tp1 = 1.0 Te
zhou_10te = np.array(
    [
        [0.018987342, 0.634615642],
        [0.044303797, 0.593406226],
        [0.101265823, 0.555331293],
        [0.151898734, 0.520430941],
        [0.196202532, 0.497152901],
        [0.398734177, 0.456812319],
        [0.598101266, 0.41858701],
        [0.803797468, 0.393026613],
        [1.0, 0.394931362],
        [1.202531646, 0.409501023],
        [1.39556962, 0.414577012],
        [1.601265823, 0.428087364],
        [1.797468354, 0.428936147],
        [1.993670886, 0.434008795],
        [2.202531646, 0.435900178],
        [2.401898734, 0.438857551],
        [2.607594937, 0.447088073],
        [2.803797468, 0.441601059],
        [2.996835443, 0.445621082],
    ]
)

# Tp1 = 0.2 Te
zhou_02te = np.array(
    [
        [0.015822785, 0.359011803],
        [0.034810127, 0.337872429],
        [0.050632911, 0.316736396],
        [0.066455696, 0.286096668],
        [0.082278481, 0.273408365],
        [0.098101266, 0.260720062],
        [0.151898734, 0.199417214],
        [0.202531646, 0.156069132],
        [0.291139241, 0.273187815],
        [0.405063291, 0.354376913],
        [0.803797468, 0.473280045],
        [1.0, 0.497360084],
        [1.199367089, 0.512989053],
        [1.392405063, 0.534960502],
        [1.601265823, 0.540019783],
        [1.803797468, 0.545085747],
        [2.006329114, 0.563879272],
        [2.205696203, 0.566836646],
        [2.405063291, 0.577185783],
        [2.60443038, 0.579087191],
        [2.797468354, 0.573603518],
        [3.006329114, 0.582886664],
    ]
)

# Tp1 = 0.1 Te
zhou01te = np.array(
    [
        [0.022151899, 0.231233208],
        [0.053797468, 0.169953751],
        [0.101265823, 0.090706161],
        [0.148734177, 0.178301231],
        [0.196202532, 0.254280673],
        [0.405063291, 0.404007325],
        [0.60443038, 0.474546536],
        [0.803797468, 0.514462727],
        [1.006329114, 0.550151712],
        [1.205696203, 0.571060511],
        [1.401898734, 0.584580888],
        [1.598101266, 0.591765469],
        [1.803797468, 0.611611618],
        [2.0, 0.614572334],
        [2.202531646, 0.623862163],
        [2.398734177, 0.631046743],
        [2.594936709, 0.631895526],
        [2.810126582, 0.627444428],
        [3.009493671, 0.632513734],
    ]
)

# %% Collision Kernel Comparison

import numpy as np
import matplotlib.pyplot as plt
from particula.dynamics.coagulation.turbulent_dns_kernel.kernel_ao2008 import (
    get_kernel_ao2008,
)

# Define necessary parameters
kernel_values = get_kernel_ao2008(
    particle_radius, velocity_dispersion, particle_inertia_time,
    stokes_number, kolmogorov_length_scale, reynolds_lambda,
    normalized_accel_variance, kolmogorov_velocity, kolmogorov_time,
)

# Using the provided kernel data
plt.scatter(r23_e100[:, 0], r23_e100[:, 1], label='DNS Data', color='cyan')

# Plot calculated kernel values
plt.plot(
    particle_radius * 1e6, kernel_values, label='Model Prediction',
    color='magenta'
)

plt.xlabel('Particle Radius (µm)')
plt.ylabel('Collision Kernel (cm³/s)')
plt.title('Collision Kernel Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Comparison of collision kernel Γ₁₂ between DNS data and model prediction

from particula.dynamics.coagulation.turbulent_dns_kernel.radial_velocity_module import (
    get_radial_relative_velocity_ao2008,
)

# Define necessary parameters
velocity_dispersion = 0.1  # Example value
particle_inertia_time = np.linspace(0.01, 0.1, 100)  # Example values

radial_relative_velocity = get_radial_relative_velocity_ao2008(
    velocity_dispersion,
    particle_inertia_time,
)

# Example using data from figure 13
plt.scatter(r23_e100[:, 0], r23_e100[:, 1], label='DNS Data', color='purple')

# Plot calculated radial relative velocities
plt.plot(
    particle_radius * 1e6, radial_relative_velocity,
    label='Model Prediction', color='brown'
)

plt.xlabel('Particle Radius (µm)')
plt.ylabel('Radial Relative Velocity (m/s)')
plt.title('Radial Relative Velocity Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Comparison of radial relative velocities between DNS data and model prediction

from particula.dynamics.coagulation.turbulent_dns_kernel.sigma_relative_velocity_ao2008 import (
    get_relative_velocity_variance,
)

# Define necessary parameters
fluid_rms_velocity = 0.2  # Example value
collisional_radius = np.linspace(1e-6, 60e-6, 100)  # Example values
particle_inertia_time = np.linspace(0.01, 0.1, 100)  # Example values
particle_velocity = np.linspace(0.1, 1.0, 100)  # Example values
taylor_microscale = 1e-3  # Example value
eulerian_integral_length = 1e-2  # Example value
lagrangian_integral_time = 0.1  # Example value

sigma_squared = get_relative_velocity_variance(
    fluid_rms_velocity,
    collisional_radius,
    particle_inertia_time,
    particle_velocity,
    taylor_microscale,
    eulerian_integral_length,
    lagrangian_integral_time,
)

# Plot DNS data
plt.scatter(
    dns_400_cm2_s3[:, 0], dns_400_cm2_s3[:, 1], label='DNS Data',
    color='green'
)

# Plot calculated mean-square velocities
plt.plot(
    particle_radius * 1e6, sigma_squared, label='Model Prediction',
    color='orange'
)

plt.xlabel('Particle Radius (µm)')
plt.ylabel('Mean-Square Horizontal Velocity (cm²/s²)')
plt.title('Mean-Square Horizontal Velocity Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Comparison of mean-square horizontal velocities between DNS data and model prediction

# Using the DNS datasets provided, e.g., r23_e100
particle_radius = np.linspace(
    1e-6, 60e-6, 100
)  # Example radii from 1 µm to 60 µm
stokes_number = np.linspace(0.1, 1.0, 100)  # Example Stokes numbers
kolmogorov_length_scale = 1e-3  # Example value
reynolds_lambda = 23  # Example value from the provided dataset
normalized_accel_variance = 0.5  # Example value
kolmogorov_velocity = 0.1  # Example value
kolmogorov_time = 0.01  # Example value

g12_values = get_g12_radial_distribution_ao2008(
    particle_radius,
    stokes_number,
    kolmogorov_length_scale,
    reynolds_lambda,
    normalized_accel_variance,
    kolmogorov_velocity,
    kolmogorov_time,
)

# Plot DNS data
plt.scatter(r23_e100[:, 0], r23_e100[:, 1], label='DNS Data', color='blue')

# Plot calculated g12 values
plt.plot(
    particle_radius * 1e6, g12_values, label='Model Prediction', color='red'
)

plt.xlabel('Particle Radius (µm)')
plt.ylabel('Radial Distribution Function g₁₂')
plt.title('Radial Distribution Function Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Comparison of the radial distribution function g₁₂ between DNS data and model prediction

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

# %% Commparison of radial relative velocity abs(wr) vs a2

# Figure 13. Comparison of the predicted and simulated radial relative velocity of
# sedimenting droplets in a turbulent ﬂow. (a) a1 = 30µm and the turbulent ﬂow
# parameters are: Rλ = 72.41 and turbulent_dissipation = 400 cm2 s−3.

# plot of radius (a2, microns) vs radial relative velocity

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

# %% radial distribution function g12

# DNS results and the modeled RDF of sedimenting droplets in a
# turbulent ﬂow. (a) Monodisperse case

# plot of radius 'a' (microns) vs g12 (r=2a)

# case: R_lambda = 23, turbulent_dissipation = 100 cm2 s−3
# r23_e100: 6 rows, 2 columns (X, Y)
r23_e100 = np.array(
    [
        [9.937578027, 1.532846715],
        [19.98751561, 1.094890511],
        [29.91260924, 2.299270073],
        [40.02496879, 3.686131387],
        [49.95006242, 2.919708029],
        [60, 2.737226277],
    ]
)

# case: R_lambda = 23, turbulent_dissipation = 400 cm2 s−3
# r23_e400: 6 rows, 2 columns (X, Y)
r23_e400 = np.array(
    [
        [10.18726592, 1.094890511],
        [20.17478152, 3.248175182],
        [30.09987516, 8.175182482],
        [40.14981273, 8.686131387],
        [50.13732834, 7.226277372],
        [60.24968789, 5.620437956],
    ]
)

# case: R_lambda = 72.4, turbulent_dissipation = 100 cm2 s−3
# r72.4_e100: 6 rows, 2 columns (X, Y)
r72_4_e100 = np.array(
    [
        [10.12484395, 1.204379562],
        [19.92509363, 1.788321168],
        [29.97503121, 3.211678832],
        [40.08739076, 7.919708029],
        [50.01248439, 10.76642336],
        [59.93757803, 9.525547445],
    ]
)

# case: R_lambda = 72.4, turbulent_dissipation = 400 cm2 s−3
# r72.4_e400: 6 rows, 2 columns (X, Y)
r72_4_e400 = np.array(
    [
        [10, 0.875912409],
        [20.11235955, 5.145985401],
        [30.03745318, 16.82481752],
        [40.08739076, 15.72992701],
        [50.01248439, 14.48905109],
        [60, 13.72262774],
    ]
)

# %% Comparison of the Kernel

# DNS dynamic collision kernel and predicted collision kernel of
# sedimenting droplets in a turbulent ﬂow. (a) a1 = 30µm, Rλ = 72.41 and  =
# 400 cm2 s−3

# plot of radius (microns) vs kernel (cm3/s)

data = np.array(
    [
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
    ]
)
