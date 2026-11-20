# %% [markdown]
# # Recovering initial particle sizes after condensation
#
# Can a measurement after 1.4 s of growth distinguish the initial sizes of
# two particle populations? Each box contains two modes with known particle
# counts that draw from the same vapor. We fit only their initial radii.
# The observations are synthetic. An independent radius equation generates
# the observations and an fp64 fixed-step mass-transfer model fits them.
#
# ## From final radius to an initial size
#
# Begin with a single mode. The CPU reference uses a bracketed inversion and
# bounded least squares. Near a physical search bound, repeated observations
# with 10 nm Gaussian noise produce asymmetric recovery errors.

# %%
import matplotlib.pyplot as plt
from matplotlib import style as mplstyle
from particula import plotting
from scripts.inverse_studies.distribution import assess_fit
from scripts.inverse_studies.scalar_campaign import assess

# %% [markdown]
# ## Resolve two modes with finite-width channels
#
# Twelve synthetic channels measure size-weighted counts after growth.
# Their Gaussian radius acceptance has a standard deviation of 0.04 µm
# in the resolved case. Broaden it to 1.5 µm to test what happens when
# nearly every channel counts both modes with similar efficiency.
# Each noisy run uses the same 1% absolute count-fraction noise realization.
# Compare the smallest singular value of the noise-scaled sensitivity matrix
# and the population recovered in the initially empty 0.95–1.05 µm band.


# %%
def simulate():
    """Run scalar reference fits and the paired CUDA distribution study."""
    scalar = assess()
    fits = [
        assess_fit("cuda:0", method)
        for method in ("ad", "fd", "powell", "global")
    ]
    if not all(run["success"] and run["max_error_nm"] < 2 for run in fits):
        raise ArithmeticError("resolved noiseless recovery failed")
    resolution = [
        assess_fit("cuda:0", "ad", width=width, noise_seed=318)
        for width in (0.04, 0.25, 1.5)
    ]
    return dict(scalar=scalar, fits=fits, resolution=resolution)


# %% [markdown]
# The left panel shows the spread of repeated single-mode fits. The right
# panel uses the two-mode fit to show the effect of channel width on the
# largest initial-radius error. Counts, transport and initial vapor are
# prescribed in every inversion.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot scalar uncertainty and the two-mode resolution experiment."""
    fig, axes = plotting.subplots(1, 2)
    rows = report["scalar"]["cases"]
    axes[0].errorbar(
        [r["initial_um"] for r in rows],
        [r["bias_nm"] for r in rows],
        yerr=[r["spread_nm"] for r in rows],
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set(
        xlabel="Initial radius (µm)",
        ylabel="Recovery bias ± 1 SD (nm)",
        title="Single-mode recovery",
    )
    scans = report["resolution"]
    axes[1].plot(
        [r["width_um"] for r in scans], [r["max_error_nm"] for r in scans], "o-"
    )
    axes[1].set(
        xscale="log",
        xlabel="Channel width (µm)",
        ylabel="Largest radius error (nm)",
        title="Two modes sharing vapor",
    )
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    result = simulate()
    figure = plot_results(result)
    plt.show()

# %% [markdown]
# A narrow channel can distinguish growth histories that a broad counting
# response averages together. Optimizer convergence alone gives no measure
# of this information loss. Inspect the sensitivity spectrum and fitted
# empty-band population before interpreting a recovered distribution.
# To adapt this calculation, replace the synthetic efficiencies and noise
# with calibrated instrument responses and measured uncertainties.
