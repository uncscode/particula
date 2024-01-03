# pylint: disable=all
# flake8: noqa

# %%
from particula.util.simple_solver import SimpleSolver
from particula import particle, rates
import numpy as np
from matplotlib import pyplot as plt
from particula.util.coagulation_rate import CoagulationRate
from particula.util.dimensionless_coagulation import full_coag
from particula.util.input_handling import in_time

# %%
simple_dic_kwargs = {
    "mode": 200e-9,  # 200 nm median
    "nbins": 500,  # 1000 bins
    "nparticles": 1e6,  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": 1.5,  # relatively narrow
    ""
}
# pass the kwargs using ** prefix
particle_dist2 = particle.Particle(**simple_dic_kwargs)

# inital distribution coag kernel
coag_kernel = full_coag(radius=particle_dist2.particle_radius)
time_array = np.arange(0, 1000, 10)

# setup the inital state of the distribution
problem = {
    "distribution": particle_dist2.particle_distribution(),
    "radius": particle_dist2.particle_radius,
    "kernel": coag_kernel,
    "tspan": time_array
}

# call the solver
solution = SimpleSolver(**problem).solution()

# plot
fig, ax = plt.subplots(1, 1, figsize=[9, 6])

radius = particle_dist2.particle_radius.m
ax.semilogx(
    radius,
    particle_dist2.particle_distribution().m,
    '-b',
    label='Inital')
ax.semilogx(radius, solution.m[49, :], '--', label='t=50')
ax.semilogx(radius, solution.m[-1, :], '-r', label='t=end')

ax.legend()
ax.set_ylabel(f"Number, {particle_dist2.particle_distribution().u}")
ax.set_xlabel(f"Radius, {particle_dist2.particle_radius.u}")
ax.grid(True, alpha=0.5)


# %% [markdown]
# ## summary on stepping
# As we walked through, using the ODE solver is quite a nice way to get to the answer, without figuring out what time-step you might need.
#
