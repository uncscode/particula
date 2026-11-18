# %% [markdown]
# # How accurately does BAT predict citric-acid droplet water activity?
#
# Marsh et al. inferred water activity and solute mass fraction for aqueous
# citric acid droplets at 293.15 K. We compare all 12 points in their
# [Table S9.2](https://doi.org/10.5194/acp-17-5583-2017-supplement) with the
# parameters used for the citric-acid BAT curve in [Gorkowski et al.,
# Figure 2b](https://doi.org/10.5194/acp-19-13383-2019). The parameters are
# fixed before evaluating the measurements. The final calculation asks whether
# finite-rate exchange approaches the same water content from either direction.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula.particles import ActivityBAT
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

DATA = Path("docs/data/thermodynamics/marsh2017-citric-acid.csv")
M_WATER = 0.01801528
M_ORGANIC = M_WATER / 0.0938
DENSITY = 1469.1849580794367


# %% [markdown]
# ## Evaluate particle activity at measured composition
#
# Use the authors' mass ratio and O:C (1.16667). The common mass scale
# cancels when BAT forms a mole fraction. Evaluate both edges of the
# composition error bar and compare that prediction interval with the
# asymmetric experimental water-activity interval. The observed mass-fraction
# range lies on the monotonic homogeneous branch for this compound.


# %%
def water_activity(fraction, density=DENSITY):
    """Evaluate particle water activity at citric acid mass fraction."""
    mass = np.stack((1 - fraction, fraction), axis=-1)
    return ActivityBAT(M_ORGANIC, 1.16667, density).activity(mass)[..., 0]


def read_points():
    """Locate the CSV from a checkout or extracted example bundle."""
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.append(Path(__file__).resolve().parent)
    for start in starts:
        for parent in (start, *start.parents):
            candidate = parent / DATA
            if candidate.is_file():
                return np.loadtxt(candidate, delimiter=",", skiprows=1).T
    raise FileNotFoundError(f"Extract the example bundle containing {DATA}")


# %% [markdown]
# ## Relax a droplet in a fixed water-vapor reservoir
#
# Use isothermal continuum diffusion, with water mass rate
# $4\pi r D\rho_{v,sat}(a_{w,ambient}-a_{w,particle})$. The saturation
# vapor density is 0.0173 kg/m³ at 293.15 K and D is 2.5 × 10⁻⁵ m²/s.
# Dry radius is 5 µm. These choices set the time scale of this diagnostic.
# The flat-interface pressure, prescribed reservoir and nonvolatile solute
# isolate the long-time composition. The measurement supplies no relaxation
# time for this test. Integrating the log water-to-solute mass ratio preserves
# positive masses. Each reservoir receives the opposite of the droplet's
# water-mass change.


# %%
def relax(target, duration, tolerance):
    """Integrate 12 uncoupled droplets from each of two water loadings."""
    dry_mass = 4 * np.pi / 3 * (5e-6) ** 3 * DENSITY
    ambient = np.tile(target, 2)
    initial = np.repeat([0.05, 80.0], len(target))

    def rate(_, log_ratio):
        ratio = np.exp(log_ratio)
        radius = np.cbrt(
            3 * dry_mass * (1 / DENSITY + ratio / 998.2) / (4 * np.pi)
        )
        delta = ambient - water_activity(1 / (1 + ratio))
        return 4 * np.pi * radius * 2.5e-5 * 0.0173 * delta / (dry_mass * ratio)

    sol = solve_ivp(
        rate,
        (0, duration),
        np.log(initial),
        method="BDF",
        rtol=tolerance,
        atol=tolerance * 0.01,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    final = np.exp(sol.y[:, -1])
    return (1 / (1 + final)).reshape(2, -1), (
        dry_mass * (initial - final)
    ).reshape(2, -1)


def simulate():
    """Compute residuals, sensitivity and kinetic convergence."""
    aw, plus, minus, mfs, dm = read_points()
    calc = water_activity(mfs)
    error = calc - aw
    lower, upper = water_activity(mfs + dm), water_activity(mfs - dm)
    covered = (lower <= aw + plus) & (upper >= aw - minus)
    inverse = np.array(
        [
            brentq(lambda f, target=a: water_activity(f) - target, 0, 1)
            for a in aw
        ]
    )
    coarse, _ = relax(aw, 2000.0, 1e-7)
    fine, reservoir = relax(aw, 4000.0, 1e-9)
    residual = np.max(np.abs(water_activity(fine) - aw))
    if (
        max(
            residual,
            np.max(np.abs(water_activity(coarse) - water_activity(fine))),
        )
        > 1e-4
    ):
        raise ArithmeticError("Kinetic endpoints exceed the activity tolerance")
    return dict(
        activity=aw.tolist(),
        plus=plus.tolist(),
        minus=minus.tolist(),
        mfs=mfs.tolist(),
        dm=dm.tolist(),
        predicted=calc.tolist(),
        bias=float(error.mean()),
        rms=float(np.sqrt(np.mean(error**2))),
        maximum=float(np.max(np.abs(error))),
        covered=int(covered.sum()),
        density_rms=float(
            np.sqrt(np.mean((water_activity(mfs, 1550.0) - aw) ** 2))
        ),
        root_mfs=inverse.tolist(),
        kinetic_mfs=fine.tolist(),
        reservoir_exchange_kg=reservoir.tolist(),
        max_mfs_root_error=float(np.max(np.abs(fine - inverse))),
        max_activity_error=float(residual),
    )


# %% [markdown]
# ## Interpret the residual
#
# The left panel compares the observed and calculated activities. The right
# panel subtracts the observed value at each composition. Vertical and
# horizontal bars show the experimental activity and composition errors.


# %%
def plot_results(result):
    """Plot published observations and the signed BAT discrepancies."""
    f = np.array(result["mfs"])
    aw = np.array(result["activity"])
    pred = np.array(result["predicted"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].errorbar(
        f,
        aw,
        xerr=result["dm"],
        yerr=[result["minus"], result["plus"]],
        fmt="o",
        capsize=3,
        label="Marsh et al.",
    )
    axes[0].plot(f, pred, "s--", label="BAT")
    axes[0].set_ylabel("Particle water activity")
    axes[0].legend()
    axes[1].axhline(0, color="0.45", lw=1)
    axes[1].errorbar(
        f,
        pred - aw,
        xerr=result["dm"],
        yerr=[result["plus"], result["minus"]],
        fmt="o",
        capsize=3,
    )
    axes[1].set_ylabel("Predicted − measured activity")
    for ax in axes:
        ax.set_xlabel("Citric acid mass fraction")
    return fig


def run_example():
    """Run the complete comparison and display the publication figure."""
    result = simulate()
    print(
        f"Bias {result['bias']:+.5f}, RMS {result['rms']:.5f}, "
        f"overlapping uncertainty intervals {result['covered']}/12"
    )
    print(
        f"Maximum kinetic endpoint activity residual: {result['max_activity_error']:.2e}"
    )
    plot_results(result)
    plt.show()
    return result


if __name__ == "__main__":
    run_example()

# %% [markdown]
# BAT overpredicts measured water activity most strongly at high solute mass
# fraction. Using the melt-density estimate from the experimental study
# increases RMS error. Comparing additional compounds at matched temperature
# would help distinguish a citric-acid-specific discrepancy from a systematic
# bias in the reduced description of organic composition.
