# %% [markdown]
# # Can bulk optics resolve aerosol mixing state?
#
# A chemical sample can tell us how much of each constituent was present.
# The same dry mass and size distribution can describe particles that each
# contain both constituents or a population of separate pure particles.
# Can light distinguish those cases? This synthetic study compares a
# whole-population optical fit with a fit that also uses two ideal size
# cuts. It uses homogeneous spheres, Lorentz–Lorenz volume mixing,
# subsaturated κ-Köhler growth, and the [Particula optics guide's
# conventions](../../../../guide/optics/index.md). The two materials have
# prescribed, illustrative values for complex index, density, and κ.
# The inverse calculations use a CUDA device and the Warp optical adjoint.
#
# ## Prescribe dry inventory and optical measurements
#
# Start with a bimodal dry distribution and convert dry mass concentrations
# and material densities into volume fractions. Scale the size distribution
# to the resulting total dry volume. This makes the sum of the mass in the
# size bins agree with the bulk chemistry, in kg/m³. Measurement groups
# specify vacuum wavelength, RH as a fraction, and a dry diameter cut.
# Use an ideal transmission of one inside each selected band and zero
# outside. These channels represent integrated optical coefficients within
# the cut, in m⁻¹. The last three channels are held out of every fit.

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula.optics import get_mie_efficiencies
from particula.optics.humidity import hydrate_index, subsaturated_growth
from particula.optics.retrieval import (
    MixingStateModel,
    OpticalGrid,
    OpticalObservations,
    dry_volume_fractions,
    fit_optical,
    profile_optical,
)
from particula.optics.state import mix_refractive_index

BOUNDARY = 235e-9  # dry diameter [m]
MASS = np.array([2.2e-9, 6.3e-9])  # absorbing, hygroscopic [kg/m³]
DENSITY = np.array([1740.0, 1470.0])  # material density [kg/m³]
WAVELENGTH = np.array([450, 700, 550, 700, 450, 700, 633, 633, 633])
RH = np.array([0.0, 0.0, 0.83, 0.83, 0.0, 0.83, 0.72, 0.72, 0.72])
CUT = ("all", "all", "all", "all", "small", "large", "all", "small", "large")
CHANNELS = (
    "absorption",
    "absorption",
    "scattering",
    "scattering",
    "absorption",
    "scattering",
    "scattering",
    "absorption",
    "absorption",
)
TRUTH = np.array([0.18, 0.82])  # internal number fractions
SIGMA = 0.01  # assumed relative standard deviation, synthetic exercise
CUTOFF = 3.841459  # nominal Gaussian 95% profile cutoff for one coordinate


def dry_population():
    """Return the source distribution, optical size cuts, and dry inventories."""
    edges = np.geomspace(73e-9, 750e-9, 25)
    diameter = np.sqrt(edges[:-1] * edges[1:])
    small = diameter < BOUNDARY
    source = 6.0 * np.exp(
        -0.5 * (np.log(diameter / 156e-9) / 0.205) ** 2
    ) + np.exp(-0.5 * (np.log(diameter / 380e-9) / 0.21) ** 2)
    volume = (np.pi / 6) * diameter**3
    number = source * (MASS / DENSITY).sum() / (source @ volume)
    by_bin = number * volume
    fractions = dry_volume_fractions(MASS, DENSITY)
    np.testing.assert_allclose(
        (by_bin @ np.broadcast_to(fractions, (diameter.size, 2))) * DENSITY,
        MASS,
        rtol=1e-14,
    )
    # Move absorbing material to small bins while preserving bulk inventories.
    shift = 0.06 * np.where(
        small, 1.0, -by_bin[small].sum() / by_bin[~small].sum()
    )
    reassigned = fractions + shift[:, None] * np.array([1.0, -1.0])
    np.testing.assert_allclose(
        (by_bin @ reassigned) * DENSITY, MASS, rtol=1e-14
    )
    masks = {"all": np.ones(small.size), "small": small, "large": ~small}
    counts = number * np.stack([masks[cut] for cut in CUT])
    grid = OpticalGrid.from_diameter_bins(
        edges,
        counts,
        WAVELENGTH * 1e-9,
        measure="bin_density",
    )
    return grid, diameter, number, fractions, reassigned


# %% [markdown]
# ## Generate observations from independent population sums
#
# If a dry bin contains $N$ particles per cubic metre, an internal
# fraction $f$ assigns $Nf$ particles to the homogeneous mixture and
# $N(1-f)\phi_j$ particles to each pure material $j$, with dry
# volume fraction $\phi_j$. Every subpopulation starts at the same dry
# diameter. Therefore the dry bin number and each dry species inventory
# are independent of $f$. Verify this at the pure internal and pure
# external endpoints and at the specified truth. Then let each
# subpopulation equilibrate separately before summing its Mie cross
# sections. This reference uses NumPy with explicit number and mass
# accounting. Both forward models share the same physical assumptions.
#
# For the strong-contrast case, assign the first material a higher index,
# stronger absorption, and weaker hygroscopicity than the second. For
# the weak case, assign similar indices and hygroscopicities to both.
# Use noiseless central values with an assumed 1% independent standard
# deviation per channel. The fit then measures sensitivity to that error
# scale without depending on one random noise draw.


# %%
def synthetic_optics(grid, diameter, fractions, indices, kappa):
    """Calculate observations with explicit number and mass checks."""
    is_small = diameter < BOUNDARY
    population_composition = np.vstack((fractions, np.eye(2)))
    dry_volume = (np.pi / 6) * diameter**3
    inventory = (
        (grid.number_density * dry_volume)[:, :, None] * fractions * DENSITY
    )

    def count_by_type(split):
        internal = np.where(is_small, split[0], split[1])
        number_share = np.column_stack(
            (internal, (1 - internal)[:, None] * fractions)
        )
        return grid.number_density[..., None] * number_share

    for split in (np.zeros(2), np.ones(2), TRUTH):
        population_number = count_by_type(split)
        np.testing.assert_allclose(
            population_number.sum(axis=2), grid.number_density, rtol=1e-14
        )
        actual = (population_number @ population_composition) * dry_volume[
            None, :, None
        ]
        np.testing.assert_allclose(actual * DENSITY, inventory, rtol=1e-14)

    dry_index = np.r_[mix_refractive_index(indices, fractions), indices]
    kappas = np.r_[fractions @ kappa, kappa]
    wet_radius, water_ratio = subsaturated_growth(
        grid.radius[..., None],
        RH[:, None, None],
        kappas,
        1.05e-9,
    )
    wet_index = hydrate_index(dry_index, 1.333, water_ratio)
    mie = get_mie_efficiencies(
        wet_radius, grid.wavelength[:, None, None], wet_index
    )
    area = np.pi * wet_radius**2 * count_by_type(TRUTH)
    return np.array(
        [
            np.sum(getattr(mie, name)[i] * area[i])
            for i, name in enumerate(CHANNELS)
        ]
    )


# %% [markdown]
# ## Fit the same measurements with one or two fractions
#
# Fit one common fraction first. Then fit two fractions, using the
# prescribed 235 nm boundary. Compare the effects of adding dry size
# selection or changing the allocation of dry material. For each two-band
# fit, scan one fraction from zero to one and refit the other. The width
# of the range within the displayed profile cutoff reflects the stated
# uncertainty scale and model. Near physical bounds this nominal cutoff
# is a sensitivity diagnostic. The unused 633 nm channels test predictions
# at another RH and wavelength.


# %%
def compare(grid, fractions, indices, kappa, values, rows, *, boundary=None):
    """Fit selected channels and evaluate the three withheld coefficients."""
    model = MixingStateModel(
        grid,
        volume_fractions=fractions,
        species_index=indices,
        species_kappa=kappa,
        relative_humidity=RH,
        dry_boundary=boundary,
        device="cuda:0",
    )
    selectors = [(i, CHANNELS[i]) for i in rows]
    observations = OpticalObservations.channels(
        len(CHANNELS),
        selectors,
        values[rows],
        standard_deviation=SIGMA * values[rows],
    )
    size = len(model.parameters)
    bounds = np.zeros(size), np.ones(size)
    starts = np.array([[0.2] * size, [0.8] * size])
    fit = fit_optical(model, observations, bounds=bounds, starts=starts)
    if not fit.success:
        raise RuntimeError("Mixing-state fit failed")
    withheld_rows = np.arange(len(CHANNELS) - 3, len(CHANNELS))
    withheld = OpticalObservations.channels(
        len(CHANNELS),
        [(i, CHANNELS[i]) for i in withheld_rows],
        values[withheld_rows],
        standard_deviation=SIGMA * values[withheld_rows],
    )
    predicted = withheld.project(model.predictions(fit.parameters))
    check = (predicted - withheld.values) / (SIGMA * withheld.values)
    result = {
        "estimate": fit.parameters.tolist(),
        "chi_square": float(fit.chi_square),
        "singular_values": fit.singular_values.tolist(),
        "held_out_sigma": check.tolist(),
    }
    if boundary is not None:
        profiles = {}
        for label, estimate in zip(
            model.parameters, fit.parameters, strict=True
        ):
            coordinates = np.unique(np.r_[np.linspace(0, 1, 31), estimate])
            samples, cost, intervals = profile_optical(
                model,
                observations,
                fit,
                label,
                coordinates,
                bounds=bounds,
                delta_chi_square=CUTOFF,
            )
            profiles[label] = {
                "fraction": samples.tolist(),
                "delta_chi_square": np.maximum(
                    cost - fit.chi_square, 0
                ).tolist(),
                "intervals": intervals,
            }
        result["profiles"] = list(profiles.values())
    return result


def simulate():
    """Compare optical fits and retain compact evidence for interpretation."""
    grid, diameter, number, uniform, reassigned = dry_population()
    kappa = np.array([0.01, 0.68])
    index = np.array([1.83 + 0.28j, 1.46 + 0.002j])
    first = synthetic_optics(grid, diameter, uniform, index, kappa)
    full = np.arange(6)
    whole = np.arange(4)
    weak_index = np.array([1.520 + 0.015j, 1.525 + 0.016j])
    weak_kappa = np.array([0.23, 0.25])
    second = synthetic_optics(grid, diameter, uniform, weak_index, weak_kappa)
    scenarios = {
        "One fraction": compare(grid, uniform, index, kappa, first, full),
        "Whole population": compare(
            grid, uniform, index, kappa, first, whole, boundary=BOUNDARY
        ),
        "Two fractions": compare(
            grid, uniform, index, kappa, first, full, boundary=BOUNDARY
        ),
        "Reassigned species": compare(
            grid, reassigned, index, kappa, first, full, boundary=BOUNDARY
        ),
        "Weak material contrast": compare(
            grid,
            uniform,
            weak_index,
            weak_kappa,
            second,
            full,
            boundary=BOUNDARY,
        ),
    }
    rich = scenarios["Two fractions"]
    np.testing.assert_allclose(rich["estimate"], TRUTH, rtol=0, atol=5e-7)
    np.testing.assert_allclose(rich["held_out_sigma"], 0, atol=1e-5)
    weak = scenarios["Weak material contrast"]
    if any(p["intervals"] != ((0.0, 1.0),) for p in weak["profiles"]):
        raise ArithmeticError(
            "Weak-contrast fractions exceeded the profile cutoff"
        )
    return {
        "diameter_nm": (diameter * 1e9).tolist(),
        "number_per_cm3_bin": (number * 1e-6).tolist(),
        "boundary_nm": BOUNDARY * 1e9,
        "truth": TRUTH.tolist(),
        "sigma_fraction": SIGMA,
        "profile_cutoff": CUTOFF,
        "cases": scenarios,
    }


# %% [markdown]
# ## Read the profiles and predictions
#
# The whole-population fit recovers the prescribed fractions from the
# noiseless signals. At the assumed 1% precision, its small-band profile
# still spans 0–1. With size cuts, the nominal profile ranges are about
# 0.015–0.35 for the small band and 0.76–0.88 for the large band.
# The weak-contrast case admits the full range in both bands even with
# size selection. The dash-dot line shows the nominal cutoff, and the
# vertical dotted lines locate the generating fractions.
#
# A single common fraction underpredicts withheld small-band absorption
# by 5.5 assumed standard deviations. Reassigning the same bulk species
# mass by size shifts the fit to the upper small-band bound and
# overpredicts the same held-out coefficient by 16.3 standard deviations.
# The poorer predictions also show up in the training residuals, with
# χ² about 62 for one fraction and 226 for the reassigned composition
# over six fitted channels. All comparisons use synthetic observations from
# the two-band, uniform-composition model.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot the physical population and independent evidence of resolution."""
    cases = report["cases"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), layout="constrained")
    size_ax, small_ax, large_ax, residual_ax = axes.ravel()
    size_ax.semilogx(
        report["diameter_nm"],
        report["number_per_cm3_bin"],
        "o-",
        ms=3,
    )
    size_ax.axvline(
        report["boundary_nm"],
        color="0.35",
        linestyle="--",
        label="Dry size cut",
    )
    size_ax.set(
        xlabel="Dry diameter [nm]",
        xticks=(100, 200, 400, 700),
        xticklabels=("100", "200", "400", "700"),
        ylabel="Number per bin [cm⁻³]",
        title="Spherical dry population",
    )
    size_ax.legend()
    styles = (
        ("Whole population", "--", "Whole population only"),
        ("Two fractions", "-", "With size cuts"),
        ("Weak material contrast", ":", "Weak material contrast"),
    )
    for i, (ax, title) in enumerate(
        ((small_ax, "Small band"), (large_ax, "Large band"))
    ):
        for name, linestyle, label in styles:
            profile = cases[name]["profiles"][i]
            ax.plot(
                profile["fraction"],
                profile["delta_chi_square"],
                linestyle=linestyle,
                label=label,
            )
        ax.axhline(report["profile_cutoff"], color="0.4", linestyle="-.")
        ax.axvline(report["truth"][i], color="0.5", linestyle=":", linewidth=1)
        ax.set(
            xlabel="Internally mixed number fraction",
            ylabel="Profile Δχ²",
            xlim=(0, 1),
            ylim=(-0.25, 12),
            title=title,
        )
    small_ax.legend(loc="upper right", fontsize=9)
    rows = ("One fraction", "Two fractions", "Reassigned species")
    offsets = (-0.2, 0, 0.2)
    channels = np.arange(3)
    for name, offset, marker in zip(
        rows, offsets, ("o", "D", "s"), strict=True
    ):
        residual_ax.scatter(
            channels + offset,
            cases[name]["held_out_sigma"],
            label=name,
            marker=marker,
            s=30,
        )
    residual_ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    residual_ax.set(
        xticks=channels,
        xticklabels=(
            "Whole\nscattering",
            "Small band\nabsorption",
            "Large band\nabsorption",
        ),
        ylabel="Prediction error / standard deviation",
        title="Held out at 633 nm and 72% RH",
    )
    residual_ax.legend(fontsize=9)
    return fig


if __name__ == "__main__":
    report = simulate()
    figure = plot_results(report)
    plt.show()

# %% [markdown]
# Optical size selection gives the two-band fit different sensitivities
# to each population. The dry size distribution and bulk species
# inventories alone leave the within-bin mixtures open. In a follow-up
# instrument study, replace the ideal cuts with calibrated transmission
# curves, compare predictions at withheld RH conditions, and use
# size-resolved chemistry to assess how much inferred mixing depends on
# allocating species by size.
