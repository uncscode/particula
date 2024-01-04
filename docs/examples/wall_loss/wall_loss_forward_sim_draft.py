# pylint: disable=all
# flake8: noqa

# %%
from particula import u
from particula.dynamics import Solver
from particula import particle
import numpy as np
from matplotlib import pyplot as plt

# %%
simple_dic_kwargs = {
    "mode": 100e-9,  # 200 nm median
    "nbins": 500,  # 1000 bins
    "nparticles": 8e4,  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": 1.5,  # relatively narrow
    "dilution_rate_coefficient": 0.1 * u.hour**-1,
    "wall_loss_approximation": "spherical",
    "chamber_dimension": 1 * u.m,
    "chamber_ktp_value": 0.5 * u.s**-1,
}


# pass the kwargs using ** prefix
particle_dist2 = particle.Particle(**simple_dic_kwargs)

# inital distribution coag kernel
time_array = np.linspace(0, 60*60*5, 100)


# call the solver
# solution = SimpleSolver(**problem).solution()

# call dynamics solver
rates_kwargs = {
    "particle": particle_dist2,
    "lazy": False,
}

solution2 = Solver(
    time_span=time_array,
    do_coagulation=False,
    do_condensation=False,
    do_nucleation=False,
    do_dilution=True,
    do_wall_loss=True,
    **rates_kwargs
).solution(method='odeint')

#%%
# plot
fig, ax = plt.subplots(1, 1, figsize=[9, 6])

radius = particle_dist2.particle_radius.m
ax.semilogx(
    radius,
    particle_dist2.particle_distribution().m,
    '-b',
    label='Inital')
ax.semilogx(radius, solution2.m[49, :], '--', label='t=50')
ax.semilogx(radius, solution2.m[-1, :], '-r', label='t=end')

ax.legend()
ax.set_ylabel(f"Number, {particle_dist2.particle_distribution().u}")
ax.set_xlabel(f"Radius, {particle_dist2.particle_radius.u}")
ax.grid(True, alpha=0.5)
plt.show()


# %%
