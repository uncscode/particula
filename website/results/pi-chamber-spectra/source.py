# %% [markdown]
# # When do two Π Chamber size distributions converge?
#
# Prabhakaran and colleagues injected 130 nm sodium chloride seeds at
# 33 cm⁻³ min⁻¹ into a turbulent chamber. The fresh-water and salt-water
# boundaries produced different steady-state haze and cloud populations.
# Here we start the same model twice, once with dry residents and once with
# wet residents inferred from the experiment. Only the resident water differs.
# The published observations describe steady state. Elapsed times below are
# predictions of this independent Particula calculation.
# [Paper and experimental conditions](https://doi.org/10.1073/pnas.2006426117)
#
# ## Follow residents and incoming seeds
#
# The CUDA calculation samples wet size in the interior of a two-region
# chamber. Particles exchange between regions and can leave by settling or
# background removal. All new arrivals are dry. The restart assigns the same
# initial total number to both regions per unit local volume. The shared
# interior humidity was fitted to haze in the earlier steady-state analysis.
# Run a second ensemble with a smaller timestep and independent seed to check
# whether the timing is sensitive to the integration and sampling choices.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from particula import plotting
from scripts.pi_chamber.retrieval import fit_humidity
from scripts.pi_chamber.temporal import simulate_history


def simulate():
    """Run two time resolutions with independent particle histories."""
    fit = fit_humidity()
    return {
        "humidity_fit_pct": fit,
        "main": simulate_history(0.0625, 19273, fit["shared"]),
        "refined": simulate_history(0.03125, 89117, fit["shared"]),
    }


# %% [markdown]
# ## Read the spectra at each sampled time
#
# Each column is an instantaneous population sampled in the interior. Color
# shows number per cm³ per unit log₁₀ diameter. The white line marks the
# 1.8 µm cloud threshold. The thin gray line marks the 0.6 µm instrument
# limit. Sizes below it are accessible in the model. Blank cells have zero
# sampled particles. Columns are evenly spaced for legibility.


# %%
@plt.style.context("particula.default")
def plot_results(report):
    """Plot the time-dependent spectra and separation between starts."""
    main = report["main"]
    d = np.array(main["edges_um"])
    times = np.array(main["seconds"])
    spectra = np.array(main["bin_cm3"]) / np.diff(np.log10(d))
    fig, axes = plotting.subplots(2, 2, size="wide", sharex=True, sharey=True)
    ticks = [0, 1, 3, 5, 7, 9, 11, 13, 15]
    for i, ax in enumerate(axes.flat):
        data = np.ma.masked_equal(spectra[:, i].T, 0)
        ax.grid(False)
        mesh = ax.pcolormesh(
            np.arange(len(times) + 1) - 0.5,
            d,
            data,
            shading="flat",
            cmap=plotting.colormap("magnitude"),
            norm=LogNorm(1, 3e4),
            rasterized=True,
        )
        ax.axhline(0.6, color="#aaa", lw=0.9, ls=":")
        ax.axhline(1.8, color="white", lw=1.1, ls="--")
        ax.set(
            yscale="log",
            ylim=(d[0], 40),
            xticks=ticks,
            xticklabels=[f"{times[k]:,}" for k in ticks],
        )
        ax.tick_params(axis="x", labelrotation=38)
        ax.set_title(
            ("Fresh water" if i < 2 else "Salt water")
            + " · "
            + ("dry start" if i % 2 == 0 else "wet restart"),
            loc="left",
        )
    fig.supxlabel("Elapsed model time (s)")
    fig.supylabel("Wet diameter (µm)")
    fig.colorbar(
        mesh, ax=axes.ravel().tolist(), label="dN/dlog₁₀D (cm⁻³)", extend="both"
    )
    plotting.fit(fig)
    return fig


# %% [markdown]
# ## Compare the starts at the same time
#
# The second figure sums the absolute difference between the two binned
# spectra and divides by their combined number. Zero means identical spectra.
# A difference of 0.05 corresponds to 5% of one population for two
# equal-sized populations differing only by redistribution. Points show the
# sampled times. The lines connect those points to guide the eye.


# %%
@plt.style.context("particula.default")
def plot_separation(report):
    """Compare the sensitivity to initialization at two timestep settings."""
    fig, ax = plotting.subplots(size="wide")
    for method, style in [("main", "-"), ("refined", "--")]:
        result = report[method]
        for i, (name, color) in enumerate(
            [
                ("Fresh water", plotting.COLORS["teal"]),
                ("Salt water", plotting.COLORS["rust"]),
            ]
        ):
            ax.plot(
                result["seconds"],
                100 * np.array(result["relative_spectrum_difference"])[:, i],
                style,
                color=color,
                marker=".",
                label=f"{name}, Δt = {result['dt']:g} s",
            )
    ax.axhline(5, color="#666", lw=0.8, ls=":")
    ax.set(
        xscale="symlog",
        xlim=(0, 17000),
        ylim=(-2, 103),
        xlabel="Elapsed model time (s)",
        ylabel="Difference between spectra (%)",
    )
    plotting.legend(ax)
    plotting.fit(fig)
    return fig


# %% [markdown]
# At both timesteps, the first sampled spectrum difference below 5%
# occurred at 8 s for fresh water and 16 s for salt water. Both differences
# fell below 1% at 32 s. In the finer run, fresh-water cloud number from the
# wet restart rose from about 479 cm⁻³ at 32 s to 611 cm⁻³ at 16,384 s.
# Injection and removal continued to change the population after it lost
# most of its sensitivity to the starting wet sizes. Agreement between starts can coexist
# with a bias against measured haze. A time-resolved wet spectrum following an injection change would
# test the relaxation time. The [chamber comparison and uncertainty
# analysis](../../contribute/validation/thermodynamics/pi-chamber.md)
# gives the steady-state counts, the mixing-state extension and the
# assumptions used to reconstruct the initial population.

# %%
if __name__ == "__main__":
    result = simulate()
    plot_results(result)
    plot_separation(result)
    plt.show()
