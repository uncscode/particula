# %% [markdown]
# # What passes through a 100 nm mobility channel?
#
# In the NaCl chamber experiment of [Cai et al. (2022)](https://doi.org/10.5194/acp-22-11529-2022),
# the collectors were size-selected before they passed through a ⁶³Ni bipolar
# neutralizer. Two different physical diameters can pass the same classifier
# setting if their charges differ. Subsequent ion attachment changes the charge
# while preserving the selected diameter.
#
# ## Estimate the neutralizer outlet fractions
#
# The Wiedensohler approximation gives a separate fit for each signed charge.
# Use metres for diameter and elementary charges for charge. The raw fractions
# below are evaluated independently. The population model in the
# [study record](../../../../contribute/validation/coagulation/nacl-scavenging.md)
# also represents higher states and normalizes before drawing particles.

# %%
import matplotlib.pyplot as plt
import numpy as np
from particula import gas, particles
from scipy.optimize import brentq

for diameter_nm, states in [(4, [-1, 0, 1]), (100, [-2, -1, 0, 1, 2])]:
    fractions = particles.get_bipolar_charge_probability(
        diameter_nm * 1e-9, states
    )
    print(f"{diameter_nm} nm")
    for state, fraction in zip(states, fractions, strict=True):
        print(f"  {state:+d} e   {fraction:7.3%}")

# %% [markdown]
# ## Find the true diameters admitted on each charge branch
#
# Electrical mobility has magnitude $|z|e B(d)$, where $B(d)$ is the
# slip-corrected aerodynamic mobility. The elementary charge cancels when
# two mobilities are equal. Solve for the equivalent physical diameter at
# the center of each channel in air at 293.15 K and 101325 Pa.

# %%
viscosity = gas.get_dynamic_viscosity(293.15)
mean_free_path = gas.get_molecule_mean_free_path(
    temperature=293.15, pressure=101325.0, dynamic_viscosity=viscosity
)


def mobility(diameter_nm):
    """Return slip-corrected mechanical mobility in seconds per kilogram."""
    radius = diameter_nm * 0.5e-9
    slip = particles.get_cunningham_slip_correction(mean_free_path / radius)
    return particles.get_aerodynamic_mobility(radius, slip, viscosity)


def equivalent_diameter(nominal, state):
    """Solve for equal electrical mobility on a positive charge branch."""
    return brentq(
        lambda d: state * mobility(d) - mobility(nominal), nominal, 4 * nominal
    )


for target in (4, 100):
    centers = [equivalent_diameter(target, z) for z in (1, 2, 3)]
    print(f"{target} nm channel, +1/+2/+3 centers (nm): {np.round(centers, 2)}")

# %% [markdown]
# ## Compare the steady count retrieval with the measurements
#
# The solid curves below load the archived five-size study results. The
# published-attraction calculation anchors the collision rate at a 100 nm
# collector to the authors' curve and uses the model for its dependence on
# collector size and charge. At 4 nm, larger transmitted collectors can
# account for the residual under the assumed generator spectrum. The same
# assumptions give higher rates than measured at 3.6 and 3.7 nm.

# %%
rates = np.genfromtxt(
    "docs/data/nacl-scavenging/comparison.csv", names=True, delimiter=","
)
fig, ax = plt.subplots(layout="constrained")
for field, label, style in [
    ("measured_cm3_s", "Cai measurements", "ko"),
    ("native_steady_cm3_s", "Charged hard spheres", "v-"),
    ("native_plus_vdw_steady_cm3_s", "Particula + dispersion", "s-"),
    ("anchored_published_cm3_s", "Published attraction with transfer", "^-"),
]:
    ax.plot(rates["d_nm"], rates[field] * 1e7, style, label=label)
ax.set(
    xlabel="Nominal tracer diameter (nm)",
    ylabel="Capture coefficient (10⁻⁷ cm³/s)",
)
ax.legend(frameon=False)
plt.show()
