# %% Comparison of DNS with the implementation in particula

import numpy as np

"""
    - Ayala, O., Rosa, B., & Wang, L. P. (2008). Effects of turbulence on
        the geometric collision rate of sedimenting droplets. Part 2.
        Theory and parameterization. New Journal of Physics, 10.
        https://doi.org/10.1088/1367-2630/10/7/075016
"""
# %% Data for the comparison


# Figure 11: Radial distribution function g12 for different Stokes numbers

# Comparison of the predicted and simulated radial relative velocities
# of nonsedimenting particles in a turbulent ﬂow. The particle radius is equal to
# half the Kolmogorov length scale (a = η/2). (a) Bidisperse system (similar to
# ﬁgure 12

# ploting tp2/Te vs radial relative velocity

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

# %%

# Figure 12: Comparison of the predicted and simulated mean-square horizontal
# particle velocities for droplets falling in a turbulent ﬂow of Rλ = 72.41 and
# turbulent_dissipation = 400 cm2 s−3

# droplet radius vs rms_velocity

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
# parameters are: Rλ = 72.41 and eddy_dissipation = 400 cm2 s−3.