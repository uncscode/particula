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
    "nparticles": 5e4,  # 1e4 #
    "volume": 1e-6,  # per 1e-6 m^3 (or 1 cc)
    "gsigma": 1.5,  # 
    "dilution_rate_coefficient": 0.1 * u.hour**-1,
    "wall_loss_approximation": "rectangle",
    "chamber_dimension": [0.25 , 0.25 , 1.6 ]*u.m,
    "chamber_ktp_value": 0.2 * u.s**-1,
}

# pass the kwargs using ** prefix
particle_dist = particle.Particle(**simple_dic_kwargs)

# time to simulate
time_array = np.linspace(0, 60*60*1, 100)

# call dynamics solver
rates_kwargs = {
    "particle": particle_dist,
    "lazy": False,
}

solution_coag = Solver(
    time_span=time_array,
    do_coagulation=True,
    do_condensation=False,
    do_nucleation=False,
    do_dilution=False,
    do_wall_loss=False,
    **rates_kwargs
).solution(method='odeint')



#%%
# plot
fig, ax = plt.subplots(1, 1, figsize=[9, 6])

radius = particle_dist.particle_radius.m
ax.semilogx(
    radius,
    particle_dist.particle_distribution().m,
    '-b',
    label='Inital')
ax.semilogx(radius, solution_coag.m[49, :], '--', label='t=middle')
ax.semilogx(radius, solution_coag.m[-1, :], '-r', label='t=end')

ax.legend()
ax.set_ylabel(f"Number, {particle_dist.particle_distribution().u}")
ax.set_xlabel(f"Radius, {particle_dist.particle_radius.u}")
ax.grid(True, alpha=0.5)
plt.show()


# %%
# run with dilution only,

solution_coag_dil = Solver(
    time_span=time_array,
    do_coagulation=True,
    do_condensation=False,
    do_nucleation=False,
    do_dilution=True,
    do_wall_loss=False,
    **rates_kwargs
).solution(method='odeint')

solution_coag_dil_wall = Solver(
    time_span=time_array,
    do_coagulation=True,
    do_condensation=False,
    do_nucleation=False,
    do_dilution=True,
    do_wall_loss=True,
    **rates_kwargs
).solution(method='odeint')

solution_dil_wall = Solver(
    time_span=time_array,
    do_coagulation=False,
    do_condensation=False,
    do_nucleation=False,
    do_dilution=True,
    do_wall_loss=True,
    **rates_kwargs
).solution(method='odeint')

# %% plot final comparison

fig, ax = plt.subplots(1, 1, figsize=[9, 6])
ax.semilogx(
    radius,
    particle_dist.particle_distribution().m,
    '--b',
    label='Inital')
ax.semilogx(
    radius,
    solution_coag.m[-1, :],
    '-r',
    label='coagulation')
ax.semilogx(
    radius,
    solution_coag_dil.m[-1, :],
    '-g',
    label='coagulation+dilution')
ax.semilogx(
    radius,
    solution_coag_dil_wall.m[-1, :],
    '-m',
    label='coagulation+dilution+wall')
ax.semilogx(
    radius,
    solution_dil_wall.m[-1, :],
    '-k',
    label='dilution+wall')
ax.legend()
ax.set_ylabel(f"Number, {particle_dist.particle_distribution().u}")
ax.set_xlabel(f"Radius, {particle_dist.particle_radius.u}")
ax.grid(True, alpha=0.5)
ax.set_title('Comparison of coagulation, dilution, and wall loss')
plt.show()



# %%
