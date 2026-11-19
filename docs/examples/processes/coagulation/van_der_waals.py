# %% [markdown]
# # How does attraction change a Brownian collision rate?
#
# A small particle can miss another sphere geometrically and still be drawn
# into contact. Compare the rate for neutral spheres with a hard-sphere
# control, then test that prediction against two chamber experiments. The
# calculation assumes compact homogeneous spheres, air at 298 K and a
# sticking probability of one. Every size specified below is a diameter.
#
# ## Construct material and particle states
#
# The factory takes a Hamaker constant in joules. The state takes individual
# particle masses in kilograms, material density in kg/m³ and charge in
# elementary-charge units. Concentrations do not enter this pair calculation.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import particula as par


def matrix(diameters, density, charge, hamaker, epsilon):
    """Evaluate one native kernel in order of the supplied diameters."""
    radius = np.asarray(diameters) * 0.5e-9
    mass = 4 * np.pi / 3 * radius**3 * density
    particles = par.particles.ParticleData(
        masses=mass[None, :, None],
        density=np.array([float(density)]),
        charge=np.asarray(charge, dtype=float)[None, :],
        concentration=np.ones((1, radius.size)),
        volume=np.ones(1),
        distribution_type="discrete",
    )
    strategy = par.dynamics.CoagulationFactory().get_strategy(
        "van_der_waals",
        {
            "distribution_type": "discrete",
            "hamaker_constant": hamaker,
            "relative_permittivity": epsilon,
        },
    )
    return strategy.kernel(particles, 298.0, 101325.0)


# %% [markdown]
# ## Evaluate the rate and the chamber comparisons
#
# For the sulfuric-acid droplets, use Chan and Mozurkewich's fitted material
# constant and uncertainty and assume a density of 1650 kg/m³. Their reported
# dimer-production constant must be doubled to obtain a pair kernel. The
# supplied data already include that conversion. The NaCl calculation adopts
# 2160 kg/m³ and the 8.9 × 10⁻²⁰ J scenario from Cai et al., Supplement Fig. S1.
# Use a relative permittivity of 5.9 as an illustrative dry-salt input for
# both Boren series. The 20 nm particle carries the charge in the second
# series. The saved data identify the published figures and digitizing error.


# %%
def simulate():
    """Retain numerical values and assumptions needed to redraw the panels."""
    filename = "van-der-waals-measurements.json"
    path = Path(__file__).with_name(filename)
    if not path.is_file():
        path = Path(filename)
    if not path.is_file():
        path = (
            Path(__file__).resolve().parents[3] / "data/coagulation" / filename
        )
    observations = json.loads(path.read_text())
    grid = np.geomspace(10, 200, 26)
    control = matrix(grid, 2160, np.zeros_like(grid), 0, 1)
    enhanced = matrix(grid, 2160, np.zeros_like(grid), 8.9e-20, 5.9)
    report = {"grid_nm": grid.tolist(), "ratio": (enhanced / control).tolist()}
    for key, dataset in observations.items():
        points = dataset["points"]
        diameters = np.array([p["diameter_nm"] for p in points])
        if key == "acid":
            charge = np.zeros_like(diameters)
            central, lower, upper, hard = [
                matrix(diameters, 1650, charge, a, 1).diagonal().tolist()
                for a in (6.4e-20, 3.8e-20, 9e-20, 0)
            ]
        else:
            both = np.r_[20.0, diameters]
            charge = np.zeros_like(both)
            charge[0] = int(key == "borenc")
            central = matrix(both, 2160, charge, 8.9e-20, 5.9)[0, 1:].tolist()
            hard = matrix(both, 2160, np.zeros_like(both), 0.0, 1.0)[
                0, 1:
            ].tolist()
            lower = upper = central
        report[key] = dict(
            diameter_nm=diameters.tolist(),
            prediction=central,
            lower=lower,
            upper=upper,
            hard=hard,
            points=points,
        )
    return report


# %% [markdown]
# ## Read the size dependence
#
# On the left, each square is a pair of diameters. On the right, absolute
# rates retain information that dividing by a hard-sphere prediction would
# hide. The shaded band propagates the published Hamaker uncertainty only.
# Error bars reproduce the plotted experimental bars.


# %%
def plot_results(report):
    """Plot the pair map, absolute chamber rates and relative residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    size = report["grid_nm"]
    mesh = axes[0, 0].pcolormesh(
        size, size, report["ratio"], shading="auto", cmap="viridis"
    )
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        xlabel="Diameter i (nm)",
        ylabel="Diameter j (nm)",
        title="Neutral salt · pair enhancement",
    )
    fig.colorbar(mesh, ax=axes[0, 0], label="K / hard-sphere K")
    for key, label, color in [
        ("acid", "Acid droplets", "#884968"),
        ("borenn", "Neutral pair", "#176a84"),
        ("borenc", "Charged–neutral", "#a55119"),
    ]:
        data = report[key]
        x = np.asarray(data["diameter_nm"])
        y = np.array([p["kernel_m3_s"] for p in data["points"]]) * 1e15
        dy = np.array([p["error_m3_s"] for p in data["points"]]) * 1e15
        model = np.array(data["prediction"]) * 1e15
        ax = axes[0, 1] if key == "acid" else axes[1, 0]
        ax.errorbar(
            x, y, yerr=dy, fmt="o", ms=3, color=color, capsize=2, label=label
        )
        ax.plot(x, model, color=color)
        ax.plot(
            x, np.array(data["hard"]) * 1e15, color=color, ls=":", alpha=0.7
        )
        if key == "acid":
            ax.fill_between(
                x,
                np.array(data["lower"]) * 1e15,
                np.array(data["upper"]) * 1e15,
                color=color,
                alpha=0.14,
            )
        axes[1, 1].scatter(
            x, (model / y - 1) * 100, label=label, color=color, s=13
        )
    for ax, title in [
        (axes[0, 1], "Equal-size acid droplets"),
        (axes[1, 0], "20 nm probe and larger partner"),
    ]:
        ax.set(
            title=title,
            xlabel="Partner diameter (nm)",
            ylabel="Pair kernel (10⁻¹⁵ m³/s)",
        )
        ax.legend(fontsize=8)
    axes[1, 1].axhline(0, color=".4", lw=1)
    axes[1, 1].set(
        title="Model residual",
        xlabel="Partner diameter (nm)",
        ylabel="(model / measured − 1) × 100 (%)",
    )
    axes[1, 1].legend(fontsize=8)
    plt.show()


if __name__ == "__main__":
    plot_results(simulate())


# %% [markdown]
# ## Interpretation
#
# Small equal-size particles and unequal-size pairs sample different parts
# of the enhancement surface. The solid curves use one material constant per
# experiment and the dotted curves show hard spheres. Compare the residual
# patterns against the measurement bars before interpreting an apparent size
# trend. The acid comparison reuses a fitted constant. The salt comparisons
# add sensitivity to material and shape assumptions, especially when one
# particle polarizes its neutral partner. The next experiment is to vary the
# assumed salt permittivity while holding the neutral-series inputs fixed.
#
# [Chan and Mozurkewich](https://doi.org/10.1016/S0021-8502(00)00081-1) ·
# [Boren et al.](https://doi.org/10.1016/j.jaerosci.2026.106780) ·
# [Cai et al., supplement](https://doi.org/10.5194/acp-22-11529-2022-supplement)
