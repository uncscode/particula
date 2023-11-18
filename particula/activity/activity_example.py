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
molar_mass_ratio = 18.016 / 250
oxygen2carbon = 0.2
hydrogen2carbon = 0
nitrogen2carbon = 0


oxygen2carbon, molar_mass_ratio = phase_separation.convert_to_oh_equivalent(
    oxygen2carbon,
    molar_mass_ratio,
    functional_group=None
)

# check the limits of possible mole fractions
org_mole_fraction = np.where(org_mole_fraction > 1, 1, org_mole_fraction)
org_mole_fraction = np.where(
    org_mole_fraction <= 0,
    10**-20,
    org_mole_fraction)

density = phase_separation.organic_density_estimate(
    18.016 / molar_mass_ratio, oxygen2carbon, hydrogen2carbon, nitrogen2carbon)


# %%

oxygen2carbon_array = np.linspace(0, 0.6, 100)
weights_matrix = np.zeros((100, 3))

for index, oxygen2carbon in enumerate(oxygen2carbon_array):
    weights_matrix[index, :] = phase_separation.bat_blending_weights(
        molar_mass_ratio, oxygen2carbon)

fig, ax = plt.subplots()
ax.plot(
    oxygen2carbon_array,
    weights_matrix,
)

ax.set_xlabel("O:C")
ax.set_ylabel("weights")
# ax.legend()
fig.show


gibbs_mix, dervative_gibbs = phase_separation.gibbs_of_mixing(
    molar_mass_ratio,
    org_mole_fraction,
    oxygen2carbon,
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
oxygen2carbon = 0.225
molar_mass_ratio = 18.016 / 100
density = phase_separation.organic_density_estimate(
    18.016 / molar_mass_ratio, oxygen2carbon)

activity_water, activity_organic, mass_water, mass_organic = \
    phase_separation.activity_coefficients(
        molar_mass_ratio=molar_mass_ratio,
        org_mole_fraction=org_mole_fraction,
        oxygen2carbon=oxygen2carbon,
        density=density,
        functional_group=None,)

fig, ax = plt.subplots()
ax.plot(
    1 - org_mole_fraction,
    activity_water,
    label="water",
    linestyle='dashed'
)

ax.plot(
    1 - org_mole_fraction,
    activity_organic,
    label="organic",
)
ax.set_ylim()
ax.set_xlabel("water mole fraction")
ax.set_ylabel("activity")
ax.legend()


# %%

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


# %%
o2c = [0.1, 0.2, 0.3, 0.4, 0.5]
mweight = [200, 200, 200, 200, 200]
mratio = phase_separation.to_molar_mass_ratio(mweight)
hydrogen2carbon = [2, 2, 2, 2, 2]
RH_cross_point = phase_separation.biphasic_to_single_water_activity(
    o2c,
    hydrogen2carbon,
    mratio,
    functional_group=None
)
print(f'RH_cross_point: {RH_cross_point}')
# %%
