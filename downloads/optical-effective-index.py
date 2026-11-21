# %% [markdown]
# # Retrieve effective refractive index and hygroscopic growth from optics
#
# A size distribution supplies particle counts and radii. Extinction and
# absorption can then constrain a common complex refractive index for spherical
# particles. If the sizes are measured in a dryer but the optics are measured
# at higher humidity, water changes both radius and refractive index. We fit
# an effective dry index and κ together, hold out a wavelength, and compare
# the fit with two measured dry–humidified sequences at Prague-Suchdol.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style as mplstyle
from particula import plotting
from particula.optics import get_mie_efficiencies
from particula.optics.retrieval import (
    EffectiveIndexModel,
    OpticalGrid,
    OpticalObservations,
    fit_optical,
    profile_optical,
)
from particula.util.constants import GAS_CONSTANT

DATA = Path("docs/data/optical-retrieval/suchdol.json")
WET_BOUNDS = ([1.32, 0.0], [1.9, 0.25])
WET_STARTS = [[1.4, 0.02], [1.7, 0.1]]
DRY_BOUNDS = ([1.33, 0.0, 0.0], [2.0, 0.45, 1.3])
DRY_STARTS = [[1.43, 0.02, 0.6], [1.72, 0.13, 0.15]]


# %% [markdown]
# ## Look for an extinction ridge at one wavelength
#
# Generate extinction and absorption for a measured-at-RH size spectrum
# with known index. Each synthetic channel has an assumed 2% standard
# deviation. The objective surface shows where changing absorption and
# changing the real index give a similar extinction. The second channel
# must be independently measured to supply a second constraint.


# %%
def synthetic_grid(wavelength):
    """Make a prescribed lognormal distribution on explicit diameter bins."""
    edges = np.geomspace(50e-9, 850e-9, 24)
    centre = np.sqrt(edges[:-1] * edges[1:])
    density = 4e9 * np.exp(-0.5 * (np.log(centre / 275e-9) / 0.48) ** 2)
    return OpticalGrid.from_diameter_bins(
        edges,
        density,
        wavelength,
        measure="dN_dlog10D",
    )


def effective_index():
    """Fit both channels and compute the extinction-only surface."""
    wave = 550e-9
    grid = synthetic_grid(wave)
    cpu = EffectiveIndexModel(grid)
    gpu = EffectiveIndexModel(grid, device="cuda:0")
    truth = np.array([1.56, 0.039])
    target = cpu.predictions(truth)[0, [0, 2]]
    kwargs = dict(bounds=WET_BOUNDS, starts=WET_STARTS)
    channels = OpticalObservations.channels(
        1,
        [(0, "extinction"), (0, "absorption")],
        target,
        standard_deviation=0.02 * target,
    )
    both = fit_optical(gpu, channels, **kwargs)
    extinction = OpticalObservations.channels(
        1,
        [(0, "extinction")],
        target[:1],
        standard_deviation=0.02 * target[:1],
    )
    one = fit_optical(gpu, extinction, **kwargs)
    if not both.success or np.max(np.abs(both.parameters - truth)) > 1e-7:
        raise ArithmeticError("Known effective-index recovery failed")
    n = np.linspace(1.37, 1.8, 46)
    k = np.linspace(0.002, 0.15, 48)
    surface = n[None, :, None] + 1j * k[:, None, None]
    q = get_mie_efficiencies(grid.radius, wave, surface)
    weights = np.pi * grid.radius**2 * grid.number_density
    ext = (q.extinction * weights).sum(axis=-1)
    absorb = (q.absorption * weights).sum(axis=-1)
    ext_residual = (ext - target[0]) / (0.02 * target[0])
    abs_residual = (absorb - target[1]) / (0.02 * target[1])
    return {
        "effective_n": n.tolist(),
        "effective_k": k.tolist(),
        "extinction_cost": (ext_residual**2).tolist(),
        "joint_cost": (ext_residual**2 + abs_residual**2).tolist(),
        "truth": truth.tolist(),
        "fit": both.parameters.tolist(),
        "singular_extinction": one.singular_values.tolist(),
        "singular_joint": both.singular_values.tolist(),
    }


# %% [markdown]
# ## Add independent humidity information
#
# Use a dry distribution and compute equilibrium water on the liquid
# κ-Köhler branch. Optical constants use a volume-based Lorentz–Lorenz
# mixture of the dry component and water. We fit dry extinction and
# absorption along with humidified scattering at two RHs. The 700 nm
# humidified scattering is held out. A second fit uses extinction and
# absorption at one humidified RH, supplying two measurements for three
# fitted quantities.


# %%
def hygroscopicity():
    """Recover known dry index and kappa and compare uncertainty profiles."""
    grid = synthetic_grid(np.array([550, 550, 550, 700]) * 1e-9)
    rh = [0.0, 0.55, 0.85, 0.85]
    truth = np.array([1.57, 0.04, 0.28])
    cpu = EffectiveIndexModel(grid, relative_humidity=rh)
    gpu = EffectiveIndexModel(grid, relative_humidity=rh, device="cuda:0")
    bulk = cpu.predictions(truth)
    broad = OpticalObservations.channels(
        4,
        [
            (0, "extinction"),
            (0, "absorption"),
            (1, "scattering"),
            (2, "scattering"),
        ],
        bulk[[0, 0, 1, 2], [0, 2, 1, 1]],
        standard_deviation=0.02 * bulk[[0, 0, 1, 2], [0, 2, 1, 1]],
    )
    humid_only = OpticalObservations.channels(
        4,
        [(2, "extinction"), (2, "absorption")],
        bulk[2, [0, 2]],
        standard_deviation=0.02 * bulk[2, [0, 2]],
    )
    kwargs = dict(bounds=DRY_BOUNDS, starts=DRY_STARTS)
    fit = fit_optical(gpu, broad, **kwargs)
    weak = fit_optical(gpu, humid_only, **kwargs)
    grid_k = np.sort(np.r_[np.linspace(0, 0.8, 31), truth[2]])
    _, cost, intervals = profile_optical(
        gpu,
        broad,
        fit,
        "kappa",
        grid_k,
        bounds=DRY_BOUNDS,
    )
    _, cost_weak, intervals_weak = profile_optical(
        gpu,
        humid_only,
        weak,
        "kappa",
        grid_k,
        bounds=DRY_BOUNDS,
    )
    blind = gpu.predictions(fit.parameters)[3, 1]
    if np.max(np.abs(fit.parameters - truth)) > 1e-7 or not fit.success:
        raise ArithmeticError("Known hygroscopicity recovery failed")
    return dict(
        kappa=grid_k.tolist(),
        cost=cost.tolist(),
        cost_weak=cost_weak.tolist(),
        intervals=intervals,
        intervals_weak=intervals_weak,
        fitted=fit.parameters.tolist(),
        held_out_percent=float(100 * (blind / bulk[3, 1] - 1)),
    )


# %% [markdown]
# ## Compare with two Suchdol measurement windows
#
# Use the archived hourly mobility spectra and sizing RH from EBAS. The
# paired optical-cell RH comes from the Suchdol nephelometer campaign.
# Its dry and humidified measurements alternate, so each optical hour
# uses its own size spectrum. The 550 nm dry absorption datum is the
# campaign's processed three-hour estimate from the AE33 spectrum. The
# instrument measures at 520 and 590 nm on either side of 550 nm.
# We compare that estimate with the average
# of the two dry-hour model predictions. Both instruments' published
# concentrations use a standard-volume basis.
#
# Published diameters are centers. Approximate bin edges by adjacent log
# midpoints and extrapolate the end widths. Use corrected total scattering
# at 450 and 550 nm and leave every 700 nm value out of the fit. The assumed
# channel errors combine 5% for scattering and 15% for absorption with
# 0.3 Mm⁻¹ in quadrature. They serve as weights for optical closure. The
# size spectrum spans 8.8–836 nm, whereas the nephelometer samples PM₁₀.
# Repeat with ±20% number scale and ±3% diameter
# scale to inspect the dependence on sizing.


# %%
def _fixture():
    # The example ZIP contains the same relative data path as the repository.
    bases = [Path.cwd()]
    if "__file__" in globals():
        bases += list(Path(__file__).resolve().parents)
    for base in bases:
        path = base / DATA
        if path.is_file():
            return json.loads(path.read_text())
    raise FileNotFoundError(DATA)


def _window_grid(window, document, *, number_scale=1.0, diameter_scale=1.0):
    rows = window["rows"]
    waves = np.tile(np.asarray(document["wavelength_nm"]) * 1e-9, len(rows))
    diameter = np.asarray(document["diameter_centres_nm"]) * 1e-9
    logs = np.log10(diameter)
    edges = np.r_[
        logs[0] - np.diff(logs)[0] / 2,
        (logs[1:] + logs[:-1]) / 2,
        logs[-1] + np.diff(logs)[-1] / 2,
    ]
    count = np.asarray([r["dN_dlog10D_cm3"] for r in rows])
    count = np.repeat(count * np.diff(edges) * 1e6 * number_scale, 3, axis=0)
    grid = OpticalGrid(diameter * diameter_scale / 2, count, waves)
    optics_rh = (
        np.repeat(
            [r["optical_cell_relative_humidity_percent"] for r in rows], 3
        )
        / 100
    )
    sizing_rh = (
        np.repeat([r["sizing_relative_humidity_percent"] for r in rows], 3)
        / 100
    )
    temp = np.repeat([r["optical_cell_temperature_K"] for r in rows], 3)
    kelvin = 2 * 0.072 * 0.01801528 / (GAS_CONSTANT * temp * 997)
    return EffectiveIndexModel(
        grid,
        relative_humidity=optics_rh,
        sizing_relative_humidity=sizing_rh,
        kelvin_length=kelvin,
        device="cuda:0",
    )


def _field_fit(window, document):
    observed = np.array(
        [r["scattering_Mm_inverse"] for r in window["rows"]]
    ).ravel()
    selection = np.array([0, 1, 3, 4, 6, 7])
    response = np.zeros((7, 9, 3))
    response[np.arange(6), selection, 1] = 1
    response[-1, [1, 7], 2] = 0.5
    measured = (
        1e-6
        * np.r_[observed[selection], window["dry_absorption_at_550_Mm_inverse"]]
    )
    error = np.sqrt(
        (measured * np.r_[np.full(6, 0.05), 0.15]) ** 2 + (0.3e-6) ** 2
    )
    observations = OpticalObservations(
        response, measured, standard_deviation=error
    )
    outcomes = []
    for number, diameter in ((1, 1), (0.8, 1), (1.2, 1), (1, 0.97), (1, 1.03)):
        model = _window_grid(
            window, document, number_scale=number, diameter_scale=diameter
        )
        fit = fit_optical(
            model, observations, bounds=DRY_BOUNDS, starts=DRY_STARTS
        )
        if not fit.success:
            raise ArithmeticError("Suchdol retrieval did not converge")
        all_scattering = model.predictions(fit.parameters)[:, 1] * 1e6
        outcomes.append(
            dict(
                n_k_kappa=fit.parameters.tolist(),
                active_bounds=fit.active_bounds.tolist(),
                singular_values=fit.singular_values.tolist(),
                chi_square=float(fit.chi_square),
                predicted_scattering_Mm_inverse=all_scattering.tolist(),
                held_out_percent=(
                    100 * (all_scattering[2::3] / observed[2::3] - 1)
                ).tolist(),
            )
        )
    return {
        "start_utc": window["start_utc"],
        "optical_rh_percent": [
            row["optical_cell_relative_humidity_percent"]
            for row in window["rows"]
        ],
        "measured_scattering_Mm_inverse": observed.tolist(),
        "outcomes": outcomes,
    }


def simulate():
    """Run the controlled recovery and measured optical closure cases."""
    document = _fixture()
    field = [_field_fit(window, document) for window in document["windows"]]
    for window in field:
        values = np.array([case["n_k_kappa"] for case in window["outcomes"]])
        print(
            window["start_utc"],
            "Dry–wet–dry RH (%)",
            np.round(window["optical_rh_percent"], 1),
        )
        print("Fitted n, k, κ", np.round(values[0], 3))
        print(
            "Calibration sensitivity",
            np.round(values.min(axis=0), 3),
            "to",
            np.round(values.max(axis=0), 3),
        )
    return {
        "index": effective_index(),
        "growth": hygroscopicity(),
        "field": field,
    }


# %% [markdown]
# The contours show the increase in χ² from the best fit using synthetic
# extinction alone and using both independent channels. The κ profiles
# compare a two-channel humidified spectrum with a larger set of independent
# dry and humidified observations.
# In the field panels, each point pairs a simulated scattering coefficient
# with an observed value. Circles show fit channels and triangles show
# the held-out 700 nm channel.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot the objective ridge, humidity profiles, and field closure."""
    index, growth, field = (report[key] for key in ("index", "growth", "field"))
    fig, axes = plotting.subplots(2, 2)
    ax = axes[0, 0]
    for name, label, color, linestyle in (
        ("extinction_cost", "Extinction only", "tab:blue", "-"),
        ("joint_cost", "Extinction + absorption", "tab:orange", "--"),
    ):
        cost = np.asarray(index[name])
        ax.contour(
            index["effective_n"],
            index["effective_k"],
            cost,
            levels=[5.991],
            colors=[color],
            linestyles=[linestyle],
        )
        ax.plot([], [], color=color, linestyle=linestyle, label=label)
    ax.plot(*index["truth"], "k+", label="Prescribed truth")
    ax.set(
        xlabel="Real index n",
        ylabel="Imaginary index k",
        title="One wavelength",
    )
    ax = axes[0, 1]
    for key, label in (
        ("cost_weak", "One humidified RH"),
        ("cost", "Dry + two humidified RHs"),
    ):
        cost = np.asarray(growth[key])
        ax.plot(growth["kappa"], cost - cost.min(), label=label)
    ax.axhline(3.841459, color="grey", linestyle=":")
    ax.set(
        xlabel="Hygroscopicity κ",
        ylabel="Profile Δχ²",
        ylim=(-0.1, 20),
        title="Growth uncertainty",
    )
    for ax, case in zip(axes[1], field, strict=True):
        y = np.array(case["outcomes"][0]["predicted_scattering_Mm_inverse"])
        x = np.array(case["measured_scattering_Mm_inverse"])
        select = np.arange(x.size) % 3 != 2
        ax.plot(x[select], y[select], "o", label="Fit channels")
        ax.plot(x[~select], y[~select], "^", label="700 nm withheld")
        limits = [min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1]
        ax.plot(limits, limits, ":", color="grey")
        ax.set(
            xlabel="Measured scattering (Mm⁻¹)",
            ylabel="Predicted scattering (Mm⁻¹)",
            title=case["start_utc"][:10],
        )
    for ax in axes[0]:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.21),
            borderaxespad=0,
        )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2)
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    output = simulate()
    figure = plot_results(output)
    plt.show()

# %% [markdown]
# The synthetic κ interval narrows from about 0.15–0.43 at one humidified
# condition to 0.26–0.30 with dry and two humidified conditions at the
# assumed 2% errors. The field comparison has larger discrepancies. The
# November fit overpredicts 700 nm scattering by 12–17%. The August
# underprediction reaches about 7%. In November, shifting the size
# distribution's number scale by ±20% shifts the fitted real index
# from 1.73 to a range of about 1.62–1.88.
#
# Joint optical channels constrain an effective index for the prescribed size
# spectrum. In the field, the fitted values are conditional on spherical
# particles, one effective composition across the window, and a common dry
# index across the visible band. Inspect the held-out channel residuals and
# the response to calibration changes. The 700 nm residuals can reflect
# coarser particles outside the measured size range, index dispersion,
# shape, or aerosol changes within each hour. The two windows demonstrate
# the workflow. Independent wet size measurements would test the growth
# calculation directly.
#
# Field data are from [Suchánková et al. (2026)](https://amt.copernicus.org/articles/19/1611/2026/),
# [campaign files v2](https://doi.org/10.17632/8ds98t2f3x.2), and Ondráček
# et al.'s [EBAS particle size spectra](https://doi.org/10.48597/GHVT-PE5Z).
