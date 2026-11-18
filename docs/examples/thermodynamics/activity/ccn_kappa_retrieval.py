# %% [markdown]
# # Fit hygroscopicity to an activation measurement
#
# Pair a mobility-selected dry diameter with a critical supersaturation
# inferred from a CCN scan. What can that pair tell us about other sizes?
# Here the binary activity thermodynamics (BAT) model supplies a synthetic
# reference for a miscible organic. Fit κ at 100 nm, leave the other dry
# diameters out of the fit, and predict their critical supersaturations.
# The [experiment guide](../../../guide/processes/ccn-retrieval.md) explains
# how finite growth time and organic uptake change the measured pair.

# %% [markdown]
# ## Keep the particle and its environment fixed
#
# Use the oxygen-to-carbon ratio (0.50), molar mass (200 g/mol), and organic
# density of the miscible Figure 9 substance from Gorkowski et al. (2019).
# Fix the water surface tension at 0.072 N/m and temperature at 298.15 K.
# Particula's Kelvin radius is a length in metres. Doubling it gives the
# coefficient used with wet **diameter** in the equilibrium equation.

# %%
import matplotlib.pyplot as plt
import numpy as np
from particula.activity import bat_activity_coefficients
from particula.particles import get_kelvin_radius
from scipy.optimize import brentq, minimize_scalar

WATER_MOLAR_MASS = 0.01801528
ORGANIC_MOLAR_MASS = 0.200
WATER_DENSITY = 1000.0
ORGANIC_DENSITY = 1243.493333
KELVIN_DIAMETER = 2 * get_kelvin_radius(
    0.072, WATER_DENSITY, WATER_MOLAR_MASS, 298.15
)


# %% [markdown]
# ## Generate the activation point
#
# At each trial wet diameter, dry volume fixes the organic amount and the
# remaining volume is water. Convert both to moles before evaluating BAT.
# Multiplying water activity by the Kelvin factor gives the equilibrium
# saturation ratio. Locate its maximum over wet diameter. Use the same
# geometry to calculate a second curve with a prescribed constant κ.


# %%
def critical_supersaturation(dry_m, kappa=None):
    """Return critical water supersaturation in percent at fixed dry mass."""

    def equilibrium(log_ratio):
        water_volume = np.expm1(3 * log_ratio)  # Per unit dry organic volume
        if kappa is None:
            organic_moles = ORGANIC_DENSITY / ORGANIC_MOLAR_MASS
            water_moles = water_volume * WATER_DENSITY / WATER_MOLAR_MASS
            organic_fraction = organic_moles / (water_moles + organic_moles)
            water_activity = bat_activity_coefficients(
                WATER_MOLAR_MASS / ORGANIC_MOLAR_MASS,
                organic_fraction,
                0.50,
                ORGANIC_DENSITY,
            )[0]
        else:
            water_activity = water_volume / (water_volume + kappa)
        return water_activity * np.exp(
            KELVIN_DIAMETER / (dry_m * np.exp(log_ratio))
        )

    grid = np.linspace(0, np.log(256), 513)
    peak = np.argmax(equilibrium(grid))
    if not 0 < peak < len(grid) - 1:
        raise ValueError("Extend the wet-diameter range to bracket activation.")
    solution = minimize_scalar(
        lambda x: -equilibrium(x),
        bounds=(grid[peak - 1], grid[peak + 1]),
        method="bounded",
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return 100 * (-solution.fun - 1)


# %% [markdown]
# ## Fit locally, then transfer one calibration across sizes
#
# The full inversion solves for the κ whose curve has the BAT maximum.
# The shortcut uses Eq. 10 of Petters and Kreidenweis (2007) on the same
# activation point. They suggest that shortcut for κ above 0.2. Here we
# deliberately test its extrapolation to a weakly hygroscopic organic.
# Reuse only the 100 nm full fit for the held-out predictions.


# %%
def simulate():
    """Calculate local retrievals and the held-out critical supersaturations."""
    diameters = np.arange(10.0, 201.0)
    dry = diameters * 1e-9
    bat = np.array([critical_supersaturation(d) for d in dry])

    def invert(diameter, observed):
        return brentq(
            lambda k: critical_supersaturation(diameter, k) - observed,
            0.015,
            0.5,
        )

    full = np.array([invert(d, sc) for d, sc in zip(dry, bat, strict=True)])
    approximate = (
        4 * (KELVIN_DIAMETER / dry) ** 3 / (27 * np.log1p(bat / 100) ** 2)
    )
    fitted = float(full[diameters == 100].item())
    predicted = np.array([critical_supersaturation(d, fitted) for d in dry])
    error = 100 * (predicted / bat - 1)
    held_out = (diameters >= 50) & (diameters != 100)
    return {
        "diameter_nm": diameters.tolist(),
        "bat_sc_percent": bat.tolist(),
        "full_kappa": full.tolist(),
        "shortcut_kappa": approximate.tolist(),
        "prediction_error_percent": error.tolist(),
        "fit_at_100_nm": fitted,
        "held_out_rms_percent": float(np.sqrt(np.mean(error[held_out] ** 2))),
    }


# %% [markdown]
# ## Inspect the two errors
#
# The left panel asks how the **retrieved coefficient** changes if we fit
# each diameter separately. The right panel asks how well **supersaturation**
# is predicted from a single calibration. A zero residual at the calibration
# diameter follows from fitting that point.


# %%
def plot_results(result):
    """Show size-resolved κ and threshold error from a 100 nm calibration."""
    size = result["diameter_nm"]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 4), layout="constrained"
    )
    left.plot(size, result["full_kappa"], label="Full κ-Köhler inversion")
    left.plot(size, result["shortcut_kappa"], "--", label="Eq. 10 shortcut")
    left.set_ylabel("Effective κ")
    left.legend()
    right.plot(size, result["prediction_error_percent"], color="#087f78")
    right.axhline(0, color="0.5", linewidth=0.8)
    right.plot(100, 0, "o", color="black", label="Calibration")
    right.set_ylabel("Critical supersaturation error (%)")
    right.legend()
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xticks(
            [10, 20, 50, 100, 200], labels=["10", "20", "50", "100", "200"]
        )
        ax.set_xlabel("Dry diameter (nm)")
    return fig


def run_example():
    """Run the calculation and print the coefficient and prediction error."""
    result = simulate()
    print(f"Full fit at 100 nm: κ = {result['fit_at_100_nm']:.5f}")
    print(
        f"Relative RMS error at held-out 50–200 nm sizes: {result['held_out_rms_percent']:.2f}%"
    )
    plot_results(result)
    plt.show()
    return result


if __name__ == "__main__":
    run_example()

# %% [markdown]
# The 100 nm fit gives κ = 0.11037. For this organic, it predicts activation
# over the other 50–200 nm diameters with 0.43% relative RMS error. At 10 nm
# the prediction falls about 9% below BAT and the required supersaturation
# exceeds 11%. To test the fitted κ on a different observable, measure wet
# diameter below saturation and compare growth at the same dry size and RH.
# The [technical note](../../../guide/processes/ccn-retrieval.md) carries
# that comparison through to an experimental decision.
#
# ## Choose a supersaturation scan
#
# These four figures come from the companion kinetic simulations. Start with
# the expected κ and the median dry mobility diameter. The line gives the
# predicted 50% counting point after 5 s and the band spans 10–90% counted.
# For 100 nm particles with κ = 0.030, increasing the geometric standard
# deviation from 1.10 to 1.20 widens the interval from 0.54–0.77% to
# 0.46–0.90% supersaturation.
#
# ![Five-second supersaturation scan planner for 10–200 nm particles at two selection widths.](../../../images/thermodynamics/ccn-retrieval/lab-scan-planner.png)
#
# ## Check whether residence time changes the fit
#
# Compare the same size and κ in the 1 s and 5 s panels. Color gives the
# signed relative error in the Eq. 10 retrieval. In the gray region a droplet
# can reach the 1 µm optical gate before crossing its equilibrium maximum.
#
# ![Signed error in retrieved κ at the optical midpoint for one- and five-second exposures.](../../../images/thermodynamics/ccn-retrieval/lab-kappa-error.png)
#
# ## Test the effect of organic vapor
#
# The cross marks 100 nm and 0.35% supersaturation. With organic mass fixed,
# changing from ideal mixing to BAT lowers the count by about 2 percentage
# points. Supplying 10 µg/m³ of organic vapor raises the BAT count by 36
# percentage points. Read each panel against its own color scale.
#
# ![Count differences from BAT nonideality and from organic co-condensation at five seconds.](../../../images/thermodynamics/ccn-retrieval/bat-design-contrast.png)
#
# ## Compare an ambient population
#
# In this separate four-component simulation, 44.57% of individually
# composed particles reach the optical gate at 0.40% supersaturation in 5 s.
# Averaging composition within narrow size bands gives 45.25%. The right
# panel shows where the count is most sensitive to that assumption.
#
# ![Particle-resolved ambient scan and change in counts when size-band composition is averaged.](../../../images/thermodynamics/ccn-retrieval/ambient-cuda-reference.png)
#
# An experiment can test the mixing assumption by measuring composition
# within the mobility bands where these count curves diverge. See the
# [study methods](../../../contribute/validation/thermodynamics/ccn-retrieval.md)
# for the population, numerical checks and scripts behind these figures.
