# %%
# linting disabled until reformatting of this file
# pylint: disable=all
# flake8: noqa
# pytype: skip-file

# need to convert this to a notebook and add some more explanation

import numpy as np
import matplotlib.pyplot as plt
from particula.activity import phase_separation

plt.rcParams.update({'text.color': "#333333",
                     'axes.labelcolor': "#333333",
                     "figure.figsize": (6, 4),
                     "font.size": 14,
                     "axes.edgecolor": "#333333",
                     "axes.labelcolor": "#333333",
                     "xtick.color": "#333333",
                     "ytick.color": "#333333",
                     "pdf.fonttype": 42,
                     "ps.fonttype": 42})
# %%

org_mole_fraction = np.linspace(0, 1, 100)
molarmass_ratio = 18.016/250
O2C = 0.2
H2C = 0
N2C = 0


O2C, molarmass_ratio = phase_separation.convert_to_OH_equivalent(
        O2C,
        molarmass_ratio,
        BAT_functional_group=None
    )

# check the limits of possible mole fractions
org_mole_fraction = np.where(org_mole_fraction>1, 1, org_mole_fraction)
org_mole_fraction = np.where(org_mole_fraction<=0, 10**-20, org_mole_fraction)

density = phase_separation.organic_density_estimate(
    18.016/molarmass_ratio, O2C, H2C, N2C)


#%%

O2C_array = np.linspace(0, 0.6, 100)
weights_matrix = np.zeros((100, 3))

for index, O2C in enumerate(O2C_array):
    weights_matrix[index, :] = phase_separation.bat_blending_weights(
        molarmass_ratio, O2C)

fig, ax = plt.subplots()
ax.plot(
    O2C_array,
    weights_matrix,
    )

ax.set_xlabel("O:C")
ax.set_ylabel("weights")
# ax.legend()
fig.show


gibbs_mix, dervative_gibbs = phase_separation.gibbs_of_mixing(
    molarmass_ratio,
    org_mole_fraction,
    O2C,
    density,
    phase_separation.FIT_LOW,
)

organic_molecular_weight = np.linspace(50, 500, 500)

o2c_value = phase_separation.organic_water_single_phase(
    18.01528 / organic_molecular_weight)

fig, ax = plt.subplots()
ax.plot(
    organic_molecular_weight,
    o2c_value,
    label="initial",
    linestyle='dashed'
    )
ax.set_xlabel("Organic molecular weight (g/mol)")
ax.set_ylabel("O:C ratio")
ax.set_title("Organic Phase separation")
ax.legend()


# %%
O2C=0.225
molarmass_ratio = 18.016/100
density = phase_separation.organic_density_estimate(
    18.016/molarmass_ratio, O2C)

activity_water, activity_organic, mass_water, mass_organic = \
    phase_separation.activity_coefficients(
        molarmass_ratio=molarmass_ratio,
        org_mole_fraction=org_mole_fraction,
        O2C=O2C,
        density=density,
        BAT_functional_group=None,)

fig, ax = plt.subplots()
ax.plot(
    1-org_mole_fraction,
    activity_water,
    label="water",
    linestyle='dashed'
    )

ax.plot(
    1-org_mole_fraction,
    activity_organic,
    label="organic",
    )
ax.set_ylim()
ax.set_xlabel("water mole fraction")
ax.set_ylabel("activity")
ax.legend()
fig.show





#%%

aw = np.linspace(0, 1, 100)
phase_sep_aw = phase_separation.find_phase_separation(
    activity_water, activity_organic)

q_alpha = phase_separation.phase_separation_q_alpha(
    a_w_sep=phase_sep_aw['upper_a_w_sep'],
    aw_series=aw,
)

fig, ax = plt.subplots()

plt.plot(aw, q_alpha)
plt.xlabel('water activity')
plt.ylabel('q_alpha')
plt.show()


# %%
o2c = [0.1, 0.2, 0.3, 0.4, 0.5]
mweight = [200, 200, 200, 200, 200]
mratio = phase_separation.to_molarmass_ratio(mweight)
H2C = [2, 2, 2, 2, 2]
RH_cross_point = phase_separation.biphasic_to_single_phase_RH_point(
    o2c,
    H2C,
    mratio,
    BAT_functional_group=None
)
print(f'RH_cross_point: {RH_cross_point}')
# %%
