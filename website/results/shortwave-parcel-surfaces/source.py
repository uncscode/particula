# %% [markdown]
# # How does the ground change the sunlight reflected by an aerosol layer?
#
# An aerosol parcel above dark water scatters sunlight back toward space.
# Over bright soil, light reflected from the ground can cross the parcel
# again and be absorbed. How much of the change escapes to space, and how
# much warms the ground or the air? We use a uniform 1 km layer and compare
# two surfaces under the same Sun and ideal clear atmosphere. Each surface
# has its own aerosol-free control.
#
# ## Convert a particle population to a vertical layer
#
# We prescribe spherical particles with three radii and a weakly absorbing
# refractive index. The native discrete population contains the number in a
# representative cubic metre. A second, empty box supplies the particle-free
# atmosphere above the layer. The layer thickness sets the optical path.
# The Mie calculation gives extinction, scattering and angular asymmetry
# at each wavelength. The transfer calculation uses the associated
# Henyey–Greenstein phase function.

# %%
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import numpy as np
from particula.optics import OpticalConfig, population_optics
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import solve_shortwave
from particula.optics.solar import earth_reference
from particula.optics.surface import dusty_ground, ocean_surface
from particula.particles import ParticleData


def aerosol_layer(wavelength_m, interfaces):
    """Map the empty and populated boxes into the upper and lower layers."""
    radii = np.array([75.0, 150.0, 250.0]) * 1e-9
    density = 1500.0
    masses = (4 * np.pi / 3 * radii**3 * density)[:, None]
    counts = np.array([500.0, 450.0, 50.0]) * 1e6  # per cubic metre
    particles = ParticleData(
        masses=np.stack((np.zeros_like(masses), masses)),
        population=np.stack((np.zeros_like(counts), counts)),
        density=np.array([density]),
        charge=np.zeros((2, 3)),
        volume=np.ones(2),
        distribution_type="discrete",
    )
    config = OpticalConfig(wavelength_m, particle_index=1.50 + 0.025j)
    population = population_optics(particles, config)
    return SpectralColumn.from_population(wavelength_m, interfaces, population)


# %% [markdown]
# ## Compare matched columns over ocean and soil
#
# At 20° N on the June solstice, we apply the same sunlight at noon UTC on
# the Greenwich meridian. The two hypothetical surfaces share the same site
# geometry to isolate their effect. Run the column with and without particles
# above each surface, subtract flux at each wavelength, then integrate. The
# changes in reflected, surface-absorbed, and air-absorbed flux should sum
# to zero because the incoming solar energy is fixed.


# %%
def simulate():
    heights = [120_000.0, 1_000.0, 0.0]
    earth = earth_reference(
        20, 0, datetime(2022, 6, 21, 12, tzinfo=UTC), heights
    )
    wave = earth.column.wavelength_m
    particles = aerosol_layer(wave, heights)
    case = earth.column.plus(particles)
    bounds = np.unique(earth.band_bounds_m, axis=0)
    edges_nm = np.r_[bounds[:, 0], bounds[-1, 1]] * 1e9
    report = {"band_edges_nm": edges_nm.tolist(), "surfaces": {}}
    for name, surface in (
        ("Ocean", ocean_surface(wave)),
        ("Dusty ground", dusty_ground(wave)),
    ):
        reference = solve_shortwave(earth.column, earth.sun, surface)
        result = solve_shortwave(case, earth.sun, surface)
        delta = result.difference(reference)
        reflected = float(delta.integrate(delta.up[0]))
        ground = float(delta.integrate(delta.net_down[-1]))
        air = float(delta.integrate(delta.absorption.sum(axis=0)))
        np.testing.assert_allclose(reflected + ground + air, 0.0, atol=1e-8)
        band_energy = np.array(
            [
                delta.integrate(delta.up[0], band_m=interval)
                for interval in bounds
            ]
        )
        np.testing.assert_allclose(band_energy.sum(), reflected)
        report["surfaces"][name] = {
            "reflected_w_m2_nm": (band_energy / np.diff(edges_nm)).tolist(),
            "reflected_w_m2": reflected,
            "surface_w_m2": ground,
            "atmosphere_w_m2": air,
        }
    return report


# %% [markdown]
# ## Follow the energy by wavelength
#
# Positive values on the left mean more sunlight returned to space with
# the aerosol present. Each horizontal step gives the band energy divided
# by its width in nm. The right panel shows the energy budget across the
# modeled solar spectrum, including absorption throughout the atmosphere.
# A second figure sums the energy of successive bands from the shortest
# wavelength upward. Each marker includes the full band ending there.


# %%
def plot_results(report):
    fig, (spectra, budget) = plt.subplots(
        1, 2, figsize=(9, 3.8), layout="constrained"
    )
    categories = ["Reflected\nto space", "Ground\nuptake", "Air\nabsorption"]
    location = np.arange(len(categories))
    colors = ["#28718a", "#ae6534"]
    for i, name in enumerate(("Ocean", "Dusty ground")):
        field = report["surfaces"][name]
        print(
            f"{name}: reflected {field['reflected_w_m2']:+.2f}, "
            f"surface uptake {field['surface_w_m2']:+.2f}, "
            f"air absorption {field['atmosphere_w_m2']:+.2f} W/m² "
            "(case minus aerosol-free)"
        )
        spectra.stairs(
            field["reflected_w_m2_nm"],
            report["band_edges_nm"],
            baseline=None,
            linestyle=["-", "--"][i],
            color=colors[i],
            label=name,
        )
        budget.bar(
            location + (i - 0.5) * 0.34,
            [
                field[key]
                for key in ("reflected_w_m2", "surface_w_m2", "atmosphere_w_m2")
            ],
            width=0.34,
            color=colors[i],
            label=name,
            hatch=["", "//"][i],
        )
    spectra.set(
        xlabel="Wavelength (nm)",
        ylabel="Change in reflected flux (W m⁻² nm⁻¹)",
        xscale="log",
    )
    spectra.set_xticks(
        [400, 700, 1000, 2000, 4000],
        labels=["400", "700", "1000", "2000", "4000"],
    )
    spectra.legend()
    budget.set(
        xticks=location, xticklabels=categories, ylabel="Change in flux (W m⁻²)"
    )
    for ax in (spectra, budget):
        ax.axhline(0, color="0.45", lw=0.8)
    edges = np.asarray(report["band_edges_nm"])
    _, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    ax.axhline(0, color="0.45", lw=0.9)
    for i, name in enumerate(("Ocean", "Dusty ground")):
        field = report["surfaces"][name]
        energy = np.asarray(field["reflected_w_m2_nm"]) * np.diff(edges)
        cumulative = np.r_[0.0, np.cumsum(energy)]
        ax.plot(
            edges,
            cumulative,
            color=colors[i],
            marker=["o", "s"][i],
            markersize=3.5,
            linestyle=["-", "--"][i],
            label=f"{name} ({cumulative[-1]:+.2f} W/m²)",
        )
    ax.set(
        xlabel="Upper wavelength of included bands (nm)",
        ylabel="Cumulative change in reflected flux (W/m²)",
        xscale="log",
        xlim=(275, 4200),
    )
    ax.set_xticks(
        [300, 500, 700, 1000, 2000, 4000],
        labels=["300", "500", "700", "1000", "2000", "4000"],
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# %% [markdown]
# The same particles brighten the ocean scene and darken the dusty-ground
# scene when viewed in total reflected sunlight. The aerosol reduces ground
# uptake in both runs. Upward light from the brighter soil also passes back
# through the absorbing air, increasing atmospheric uptake. The surface
# uptake includes both direct sunlight and scattered skylight. Field
# comparisons require surface albedo, refractive index, and particle
# populations representative of the scene. The [surface reflection derivation](../../../../theory/optics/surface-and-sensor.md)
# connects hemispheric flux to the directional radiance seen by a sensor.
# The [satellite comparisons](../../../validation/spectral-radiation/observations.md)
# use the measured viewing geometry over ocean and dry land.

# %%
if __name__ == "__main__":
    plot_results(simulate())
    plt.show()
