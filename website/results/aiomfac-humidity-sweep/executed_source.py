# %% [markdown]
# # When does drying split an alcohol–salt droplet into two liquids?
#
# Remove water from a solution while retaining all its 2-propanol and
# ammonium sulfate. At first the water content of a single liquid falls.
# Once two liquids coexist, the overall composition no longer describes
# either liquid. Here we bracket the appearance of the second liquid and
# follow both compositions at 303.15 K. Feng et al. measured tie lines in
# this ternary system at this temperature. Their measurements constrain
# liquid compositions. The humidities below are model predictions.
# [Feng et al. (2012)](https://doi.org/10.1016/j.fluid.2011.12.023)
#
# To run these two examples from this feature checkout, install the local
# backend and plotting dependencies with
# `python -m pip install -e ./packages/particula-aiomfac -e ".[plot]"`.
# Both worked calculations use the CPU.
#
# ## Fix the dry inventory
#
# We retain 70 µg/m³ of alcohol and 30 µg/m³ of salt. Only water can enter
# or leave the liquid. This constraint isolates water uptake even though
# 2-propanol itself is volatile. The water activity equals ambient RH over a
# flat surface. Here 0.80 means 80% RH. Pure-water pressure is held at
# 4246 Pa at 303.15 K. The calculation assumes an ideal vapor, dissolved
# salt, and instantaneous equilibration among at most two liquids.
# [Equilibrium equations](../../../../theory/thermodynamics/equilibria-theory.md)
# %%
import matplotlib.pyplot as plt
import numpy as np
from particula.equilibria import EquilibriaFactory
from particula.particles.activity_factories import ActivityFactory
from particula.util.constants import GAS_CONSTANT
from particula_aiomfac import MixtureSpec

TEMPERATURE = 303.15  # K
WATER_CSTAR = 4246.0 * 0.01801528 / (GAS_CONSTANT * TEMPERATURE) * 1e9
GROUPS = {
    "water": {16: 1},
    "2-propanol": {141: 2, 151: 1, 153: 1},
    "ammonium sulfate": {204: 2, 261: 1},
}


def make_activity():
    """One dissolved mixture in water, alcohol, salt material order."""
    return ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": MixtureSpec(GROUPS),
            "material_mapping": dict(enumerate(GROUPS)),
            "phase_mode": "stable",
        },
    )


# %% [markdown]
# ## Solve for each humidity
#
# The initial water inventory sets the reference for reservoir exchange. RH
# determines the final amount. At each point the two dry inventories must
# close and water activity must recover the imposed humidity. A missing beta
# phase appears as a gap in its composition curve and a zero mass share.
# %%
def simulate():
    """Find liquid compositions and phase masses at prescribed RH."""
    activity = make_activity()
    solver = EquilibriaFactory().get_strategy(
        "liquid_vapor", {"activity_model": activity}
    )
    inventory = np.array([100.0, 70.0, 30.0])
    humidity = [0.88, 0.85, 0.835, 0.825, 0.815, 0.805, 0.79, 0.77]
    alpha, beta, share = [], [], []
    for rh in humidity:
        result = solver.solve(
            inventories=inventory,
            temperature=TEMPERATURE,
            saturation_concentrations={"water": WATER_CSTAR},
            relative_humidity=rh,
        )
        liquids = result.liquids
        first, second = liquids.alpha_masses, liquids.beta_masses
        alpha.append((first / first.sum()).tolist())
        water_index = liquids.alpha.neutral_ids.index("water")
        np.testing.assert_allclose(
            liquids.alpha.neutral_activities[water_index], rh, atol=2e-6, rtol=0
        )
        if second is None:
            total = first
            beta.append([None, None, None])
            share.append(0.0)
        else:
            total = first + second
            beta.append((second / second.sum()).tolist())
            share.append(float(second.sum() / total.sum()))
        np.testing.assert_allclose(total[1:], inventory[1:], rtol=1e-8)
    split = np.flatnonzero(np.asarray(share) > 0)
    if not len(split) or split[0] == 0:
        raise ArithmeticError("Humidity sweep did not bracket phase separation")
    return {
        "humidity": humidity,
        "alpha": alpha,
        "beta": beta,
        "beta_share": share,
        "onset_bracket": [humidity[split[0]], humidity[split[0] - 1]],
    }


# %% [markdown]
# ## Read the phase compositions
#
# Follow the panels from high to low humidity. Dashed curves start where
# the organic-rich liquid first appears on this grid. Its composition
# and its share of total liquid mass answer different questions. A small
# organic-rich phase can already differ sharply from the aqueous phase.
# %%
def plot_results(data):
    """Plot component mass fractions and the extent of the phase split."""
    rh = 100 * np.asarray(data["humidity"])
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.2, 3.8), constrained_layout=True
    )
    colors = ["#17678a", "#a24426", "#487b43"]
    for j, (name, color) in enumerate(zip(GROUPS, colors, strict=True)):
        left.plot(
            rh, np.array(data["alpha"])[:, j], "o-", color=color, label=name
        )
        left.plot(
            rh, np.array(data["beta"], dtype=float)[:, j], "x--", color=color
        )
    left.set(ylabel="Component mass fraction", ylim=(0, 1))
    left.legend(frameon=False)
    left.set_title("Water-rich (solid) and organic-rich (dashed)")
    right.plot(rh, data["beta_share"], "s-", color="#a24426")
    right.set(ylabel="Organic-rich share of liquid mass", ylim=(0, 1))
    right.set_title("How much of the droplet separates?")
    for ax in (left, right):
        ax.axvspan(
            *(100 * np.asarray(data["onset_bracket"])),
            color="grey",
            alpha=0.15,
        )
        ax.set_xlabel("Relative humidity (%)")
        ax.invert_xaxis()
    return fig


def run_example():
    """Report the bracket and draw the two phases along the drying path."""
    result = simulate()
    lower, upper = 100 * np.array(result["onset_bracket"])
    print(f"Two liquids appear between {lower:g} and {upper:g}% RH.")
    plot_results(result)
    return result


if __name__ == "__main__":
    run_example()
    plt.show()

# %% [markdown]
# The model gives one liquid at 82.5% RH and two at 81.5% RH. The shaded
# interval locates separation to the resolution of this grid. The
# [tie-line comparison](../../../../theory/thermodynamics/equilibria-theory.md)
# tests how closely the two compositions reproduce Feng et al.'s measurements.
# The thermodynamic transition contains no clock and cannot predict
# the delay before an experimental droplet splits. The
# [closed-box example](aiomfac_evaporating_droplet.py) follows the time for
# water vapor to leave the same mixture while each condensed state
# equilibrates internally.
