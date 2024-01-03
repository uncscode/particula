# pylint: disable=all
# flake8: noqa

# %%
from particula import u
from particula.util.simple_solver import SimpleSolver
from particula.dynamics import Solver
from particula.rates import Rates
from particula import particle
import numpy as np
from matplotlib import pyplot as plt
from particula.util.dimensionless_coagulation import full_coag

# %%
simple_dic_kwargs = {
    "mode": 200e-9,  # 200 nm median
    "nbins": 500,  # 1000 bins
    "nparticles": 1e6,  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": 1.5,  # relatively narrow
    "dilution_rate_coefficient": 0.1 * u.hour**-1,
}


# pass the kwargs using ** prefix
particle_dist2 = particle.Particle(**simple_dic_kwargs)

# inital distribution coag kernel
time_array = np.linspace(0, 1000, 100)


# call the solver
# solution = SimpleSolver(**problem).solution()

# call dynamics solver
rates_kwargs = {
    "particle": particle_dist2,
    "lazy": False,
}

solution2 = Solver(
    time_span=time_array,
    do_coagulation=True,
    do_condensation=False,
    do_nucleation=False,
    do_dilution=True,
    do_wall_loss=False,
    **rates_kwargs
).solution(method='solve_ivp')

print(solution2.shape)
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
