# %% [markdown]
# # Sphere spectra, polarization and cloud optical depth
#
# Particula 0.3.0 development optics uses independent CPU functions and SI units.
# This example needs NumPy and Matplotlib. It calculates optical properties;
# it does not evolve temperature, liquid mass or gas composition.
#
# The constant indices below are illustrative, not measured spectral material
# data. The cloud is a dilute population of independent homogeneous spheres.
# Direct-beam transmission excludes multiple scattering and diffuse radiation.
# See the [optics guide](../../guide/optics/index.md) for conventions.

# %%
import matplotlib.pyplot as plt
import numpy as np
from particula.optics import (
    apply_mueller_matrix,
    get_column_optics,
    get_cross_sections,
    get_differential_cross_sections,
    get_mie_coefficients,
    get_mie_efficiencies,
    get_mueller_matrix,
    get_population_optics,
    get_population_weights,
    get_scattering_amplitudes,
)


# %%
def simulate():
    """Calculate compact spectral, angular and column observations."""
    wavelength = np.linspace(400e-9, 1000e-9, 121)
    radius = 200e-9
    efficiency = get_mie_efficiencies(radius, wavelength, 1.5 + 0.02j)
    cross = get_cross_sections(efficiency, radius)
    np.testing.assert_allclose(
        cross.extinction, cross.scattering + cross.absorption, rtol=1e-12
    )

    angle = np.linspace(0, np.pi, 361)
    angular_wave = 550e-9
    angular_radius = 500e-9
    coefficients = get_mie_coefficients(
        1.33 + 1e-8j, 2 * np.pi * angular_radius / angular_wave
    )
    amplitudes = get_scattering_amplitudes(coefficients, angle)
    differential = get_differential_cross_sections(*amplitudes, angular_wave)
    scattered = apply_mueller_matrix(
        get_mueller_matrix(*amplitudes), [1.0, 0.0, 0.0, 0.0]
    )
    polarization = -scattered[:, 1] / scattered[:, 0]
    assert np.all(abs(polarization) <= 1 + 1e-12)

    # Explicit discrete bin densities: normalized lognormal weights per bin.
    cloud_radius = np.geomspace(1e-6, 30e-6, 64)
    bin_shape = np.exp(-0.5 * (np.log(cloud_radius / 8e-6) / 0.35) ** 2)
    number_density = 1e8 * bin_shape / bin_shape.sum()  # particles / m³
    cloud_q = get_mie_efficiencies(cloud_radius, 550e-9, 1.33 + 1e-8j)
    cloud_cross = get_cross_sections(cloud_q, cloud_radius)
    bulk = get_population_optics(cloud_cross, cloud_q.asymmetry, number_density)
    # Represented counts / representative volume give the same optical field.
    volume = 2e-6  # m³; this does not define an optical path length.
    resolved = get_population_weights(
        number_density * volume, "particle_resolved", volume=volume
    )
    np.testing.assert_allclose(resolved, number_density, rtol=1e-14)
    path_length = np.linspace(0, 500, 101)
    columns = get_column_optics(
        np.full((len(path_length), 1), bulk.extinction),
        path_length[:, None],
    )
    np.testing.assert_allclose(
        columns.optical_depth, bulk.extinction * path_length, rtol=1e-14
    )
    assert columns.transmission[0] == 1
    assert np.all(np.diff(columns.transmission) <= 0)
    return {
        "wavelength_nm": (wavelength * 1e9).tolist(),
        "cross_sections_m2": {
            "extinction": cross.extinction.tolist(),
            "scattering": cross.scattering.tolist(),
            "absorption": cross.absorption.tolist(),
        },
        "angle_degrees": np.rad2deg(angle).tolist(),
        "differential_m2_sr": {
            "perpendicular": differential.perpendicular.tolist(),
            "parallel": differential.parallel.tolist(),
            "unpolarized": differential.unpolarized.tolist(),
        },
        "linear_polarization": polarization.tolist(),
        "path_length_m": path_length.tolist(),
        "optical_depth": columns.optical_depth.tolist(),
        "transmission": columns.transmission.tolist(),
        "cloud_extinction_m_inv": float(bulk.extinction),
        "cloud_single_scattering_albedo": float(bulk.single_scattering_albedo),
        "cloud_asymmetry": float(bulk.asymmetry),
    }


def plot_results(report):
    """Render compact saved optical observations without rerunning physics."""
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, values in report["cross_sections_m2"].items():
        ax.plot(report["wavelength_nm"], np.array(values) * 1e12, label=label)
    ax.set(
        xlabel="Vacuum wavelength [nm]",
        ylabel="Cross section [µm²]",
        title="200 nm radius sphere · illustrative index 1.5 + 0.02i",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for label, values in report["differential_m2_sr"].items():
        axes[0].semilogy(report["angle_degrees"], values, label=label)
    axes[0].set(
        xlabel="Scattering angle [degrees]",
        ylabel="Differential cross section [m²/sr]",
        title="500 nm radius sphere · 550 nm light",
    )
    axes[0].legend()
    axes[1].plot(report["angle_degrees"], report["linear_polarization"])
    axes[1].set(
        xlabel="Scattering angle [degrees]",
        ylabel="−Q / I",
        ylim=(-1.05, 1.05),
        title="Unpolarized incident light",
    )
    for ax in axes:
        ax.grid(alpha=0.2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, label in zip(
        axes,
        ["optical_depth", "transmission"],
        ["Optical depth τ [1]", "Direct transmission [1]"],
        strict=True,
    ):
        ax.plot(report["path_length_m"], report[key])
        ax.set(xlabel="Path length [m]", ylabel=label)
        ax.grid(alpha=0.2)
    fig.suptitle("Illustrative cloud · 100 droplets/cm³ · 550 nm light")
    print(f"Cloud extinction: {report['cloud_extinction_m_inv']:.6g} m⁻¹")
    print(
        f"Single-scattering albedo: {report['cloud_single_scattering_albedo']:.8f}"
    )
    print(f"Scattering-weighted g: {report['cloud_asymmetry']:.6f}")
    return report


def run_example():
    """Calculate once and plot the resulting optical observations."""
    return plot_results(simulate())


# %% [markdown]
# The particle absorption spectrum can supply a later energy calculation.
# A heating model also needs incident spectral irradiance, thermal properties,
# emission and an energy balance. Evaporation additionally needs mass transfer
# and latent heat. None of those state changes are implied by this calculation.

# %%
if __name__ == "__main__":
    result = run_example()
    plt.show()
