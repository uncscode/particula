# %% [markdown]
# # Does moving an absorbing layer change where sunlight is lost?
#
# A surface radiometer sees the light that reaches the ground. Several
# vertical aerosol profiles can give similar readings. We prescribe the
# same aerosol optical depth in either of two 2 km layers, then inspect
# the net sunlight that crosses each interface. A NASA POWER grid-cell
# record supplies an independent estimate of the sunlight reaching the
# surface near Sodankylä, Finland. The experiment uses one ideal atmosphere.
#
# ## Place the same optical depth at two heights
#
# The prescribed aerosol has optical depth 0.12 at 550 nm, an Ångström
# exponent of 1.2, single-scattering albedo 0.88, and asymmetry 0.60.
# These illustrative parameters determine the spectrum and a
# Henyey–Greenstein angular distribution. Holding them fixed when moving
# the aerosol isolates the effect of its location in the molecular column.

# %%
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula.data.power import read_power_hour
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import solve_shortwave
from particula.optics.solar import earth_reference
from particula.optics.surface import dusty_ground


def elevated_layer(wave, height, layer):
    depth = np.zeros((len(height) - 1, len(wave)))
    depth[layer] = 0.12 * (wave / 550e-9) ** -1.2
    moments = np.broadcast_to(0.60 ** np.arange(33), (*depth.shape, 33))
    return SpectralColumn(
        wave, height, depth, np.full_like(depth, 0.88), moments
    )


# %% [markdown]
# ## Compare one hour of ideal clear sky with a gridded all-sky estimate
#
# The NASA record labels the start of a UTC hour. Compute four model
# snapshots at quarter-hour midpoints to approximate its average. The
# latitude, longitude, and elevation come from the saved hourly point
# response. The chosen soil reflectance and gas and aerosol profiles
# describe hypothetical columns at those coordinates. Clouds and actual
# site reflectance can strongly affect the surface irradiance.
#
# The example download contains the original one-day source and manifest.
# Their checksum identifies the input to this replay. To retrieve the same
# UTC day into a new directory, run the following command from a checkout.
# Change the coordinates and dates for your own site, then pass the new
# directory and a timezone-aware `start` to `simulate`. [Retrieval and
# provenance](https://github.com/uncscode/particula/tree/main/studies/shortwave-radiation)
#
# ```bash
# python studies/shortwave-radiation/download_power.py ./my-solar-data \
#   --latitude 67.36663 --longitude 26.62901 --start 2022-07-15 --end 2022-07-15
# ```


# %%
def simulate(data_dir=None, *, start=datetime(2022, 7, 15, 10, tzinfo=UTC)):
    if data_dir is None:
        # Input files accompany the downloadable script. In a checkout they
        # also live in the documentation data directory.
        data_dir = Path("sodankyla-20220715")
        if not data_dir.is_dir() and "__file__" in globals():
            source = Path(__file__).resolve()
            bundled = source.parent / data_dir
            if bundled.is_dir():
                data_dir = bundled
            elif len(source.parents) > 4 and source.parents[4].name == "docs":
                data_dir = source.parents[4] / "data" / "shortwave" / data_dir
            else:
                raise FileNotFoundError(
                    "Use the example ZIP with the POWER data, or pass data_dir."
                )
    folder = Path(data_dir)
    manifest = json.loads((folder / "manifest.json").read_text())
    hour = read_power_hour(
        folder / manifest["file"], start, sha256=manifest["sha256"]
    )
    elevation = hour.elevation_m
    height = np.array([120e3, elevation + 4000, elevation + 2000, elevation])
    names = ["Reference", "Aerosol at 0–2 km", "Aerosol at 2–4 km"]
    samples = {name: [] for name in names}
    for minutes in (7.5, 22.5, 37.5, 52.5):
        earth = earth_reference(
            hour.latitude_deg,
            hour.longitude_deg,
            hour.start_utc + timedelta(minutes=minutes),
            height,
        )
        wave = earth.column.wavelength_m
        for name, layer in zip(names, (None, 2, 1), strict=True):
            column = earth.column
            if layer is not None:
                column = column.plus(elevated_layer(wave, height, layer))
            flux = solve_shortwave(column, earth.sun, dusty_ground(wave))
            net = flux.integrate(flux.net_down)
            down = flux.integrate(flux.direct_down + flux.diffuse_down)
            samples[name].append(np.append(net, down[-1]))
    averages = {
        name: np.mean(value, axis=0).tolist() for name, value in samples.items()
    }
    return {
        "height_km": ((height - elevation) / 1000).tolist(),
        "mean": averages,
        "power_ghi_w_m2": hour.global_horizontal,
        "power_sha256": hour.sha256,
        "power_api_version": hour.api_version,
    }


# %% [markdown]
# ## Locate the loss of net solar flux
#
# Subtract the paired aerosol-free calculation from each profile at the same
# interface. The first panel locates the added net flux loss vertically.
# The second compares downward surface flux against POWER's all-sky hourly
# global horizontal irradiance. The latter comparison includes differences in
# the entire atmospheric column, surface, and spectral coverage.


# %%
def plot_results(report):
    fig, (profile, surface) = plt.subplots(
        1, 2, figsize=(9, 4), layout="constrained"
    )
    fields = report["mean"]
    for name, mean in fields.items():
        print(f"{name}: mean downwelling at ground {mean[-1]:.1f} W/m²")
    print(
        "NASA POWER: hourly mean global horizontal "
        f"{report['power_ghi_w_m2']:.1f} W/m²"
    )
    control = np.asarray(fields["Reference"][:-1])
    heights = report["height_km"]
    layers = ["Aerosol at 0–2 km", "Aerosol at 2–4 km"]
    colors = ["#247b80", "#b26532"]
    for i, name in enumerate(layers):
        delta = np.asarray(fields[name][:-1]) - control
        profile.plot(
            delta,
            np.arange(len(heights)),
            ["o-", "s--"][i],
            color=colors[i],
            label=name,
        )
    profile.set(
        yticks=np.arange(len(heights)),
        yticklabels=[f"{h:g}" if h < 10 else f"{h:.0f} (top)" for h in heights],
        xlabel="Change in net downward flux (W m⁻²)",
        ylabel="Layer interface (km above surface)",
    )
    profile.invert_yaxis()
    profile.legend(fontsize=8)
    values = [report["power_ghi_w_m2"]] + [
        fields[name][-1] for name in ("Reference", *layers)
    ]
    surface.bar(np.arange(4), values, color=["#555555", "#889599", *colors])
    surface.set(
        xticks=np.arange(4),
        xticklabels=["POWER\nall sky", "Clear\nreference", "0–2 km", "2–4 km"],
        ylabel="Downward surface irradiance (W m⁻²)",
    )
    return fig


# %% [markdown]
# Moving the layer through 2 km changes the hourly downward flux at the
# surface by less than 1 W/m² in this case, while the flux differences at
# the internal interfaces reveal where the air absorbed more sunlight. Both
# clear-sky calculations exceed the POWER all-sky estimate by hundreds of
# W/m². That discrepancy alone cannot identify the aerosol height. A
# co-located extinction profile and independent cloud information could
# test the two height assumptions at the same overpass.
# The [spectral transfer equations](../../../../theory/optics/spectral-transport.md)
# give the sign convention and the relation between flux and layer absorption.

# %%
if __name__ == "__main__":
    plot_results(simulate())
    plt.show()
