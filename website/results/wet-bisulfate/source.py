# %% [markdown]
# # Calculate scattering as a salt droplet takes up water
#
# Tang and Munkelwitz measured the density and refractive index of ammonium
# bisulfate solutions at 25 °C. Here we use two of their liquid compositions
# to predict how much more 633 nm light an aerosol scatters after taking up
# water. We run the same calculation twice so we can see what happens if we
# update the wet diameters but keep the initial refractive index.
#
# ## Start with two measured solutions
#
# The CSV in the example download contains the ammonium bisulfate rows of
# [Table 3](https://patarnott.com/atms749/pdf/TangMunkelwitz.pdf#page=7).
# Load the most concentrated and most dilute rows. Each has a measured salt
# weight percentage, density in g/cm³, and dimensionless real refractive
# index. The bulk refractometer wavelength is unspecified. We use its
# indices as approximations at 633 nm. [Original paper](https://doi.org/10.1029/94JD01345)

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from particula.optics import get_mie_efficiencies

DATA_DIRECTORY = Path("docs/data/wet-bisulfate")


def read_table(filename):
    """Read a named numerical table from the checkout or example ZIP."""
    relative_path = DATA_DIRECTORY / filename
    roots = [Path.cwd()]
    if "__file__" in globals():
        roots.insert(0, Path(__file__).resolve().parent)
    for root in roots:
        for parent in (root, *root.parents):
            path = parent / relative_path
            if path.is_file():
                return np.genfromtxt(path, delimiter=",", names=True)
    raise FileNotFoundError(
        f"Extract the example ZIP containing {relative_path}"
    )


# %% [markdown]
# ## Check the published material properties
#
# Tang and Munkelwitz inferred density and molal refraction from a levitated
# droplet. Their separate bulk measurements give us a way to check our
# transcription of those relations. Equations 3–6 use salt weight percent
# for the density polynomial and salt mass fraction for the optical relation.
# The molal refractions are in cm³/mol and the molar masses in g/mol.
# A blank in Table 3 means that property was not reported at that composition.


# %%
def check_solution_properties(rows):
    """Return the largest absolute residual for each measured property."""
    percent = rows["wt_percent"]
    density = (
        0.9971
        + 0.00587 * percent
        - 1.89e-6 * percent**2
        + 1.763e-7 * percent**3
    )
    fraction = percent / 100
    refractivity = density * (
        fraction * 18.38 / 115.109 + (1 - fraction) * 3.717 / 18.01528
    )
    index = np.sqrt((1 + 2 * refractivity) / (1 - refractivity))
    return {
        "max_index_residual": float(
            np.nanmax(abs(index - rows["refractive_index"]))
        ),
        "max_density_residual_g_cm3": float(
            np.nanmax(abs(density - rows["density_g_cm3"]))
        ),
    }


# %% [markdown]
# ## Calculate wet radius at constant salt mass
#
# Salt mass equals salt mass fraction times solution density times droplet
# volume. For a sphere the volume scales with the cube of its diameter.
# The ratio of salt mass concentrations in the two solutions therefore
# gives the volume growth factor. Its cube root gives the diameter growth
# factor. At the same two endpoints, read the refractive index directly
# from the measurements. This approach works when you know the index of
# the whole wet particle at each composition.
#
# Use seven initial wet number median diameters and the same lognormal
# width (geometric standard deviation 1.3) for each. Integrate over
# log diameter with the number distribution as weights. Particula's Mie
# function takes radius and wavelength in metres. Multiplying scattering
# efficiency by geometric cross section gives m² per particle. The second
# CSV contains cross sections saved by the CUDA study at the same seven
# sizes. Check the CPU result against all 21 saved values.


# %%
def simulate():
    """Calculate scattering growth for two liquid compositions on CPU."""
    rows = read_table("bulk.csv")
    initial = rows[rows["wt_percent"].argmax()]
    final = rows[rows["wt_percent"].argmin()]
    growth = (
        initial["wt_percent"]
        * initial["density_g_cm3"]
        / (final["wt_percent"] * final["density_g_cm3"])
    ) ** (1 / 3)
    n_initial = initial["refractive_index"]
    n_final = final["refractive_index"]

    median_nm = np.array([100, 150, 200, 300, 400, 600, 800])
    z = np.linspace(-8, 8, 1025)
    number_weight = np.exp(-0.5 * z**2)
    initial_radius_m = 0.5e-9 * median_nm[:, None] * 1.3**z
    final_radius_m = initial_radius_m * growth

    def mean_scattering(radius_m, index):
        efficiency = get_mie_efficiencies(radius_m, 633e-9, index)
        area_m2 = efficiency.scattering * np.pi * radius_m**2
        return np.trapezoid(area_m2 * number_weight, z, axis=-1) / np.trapezoid(
            number_weight, z
        )

    initial_area = mean_scattering(initial_radius_m, n_initial)
    fixed_area = mean_scattering(final_radius_m, n_initial)
    wet_area = mean_scattering(final_radius_m, n_final)
    reference = read_table("cuda-reference.csv")
    np.testing.assert_array_equal(median_nm, reference["median_nm"])
    for area, name in [
        (initial_area, "initial_m2"),
        (fixed_area, "fixed_m2"),
        (wet_area, "varying_m2"),
    ]:
        np.testing.assert_allclose(area, reference[name], rtol=1e-8, atol=0)
    fixed_gain = fixed_area / initial_area
    wet_gain = wet_area / initial_area
    return {
        **check_solution_properties(rows),
        "median_nm": median_nm.tolist(),
        "diameter_ratio": float(growth),
        "initial_index": float(n_initial),
        "final_index": float(n_final),
        "fixed_index_gain": fixed_gain.tolist(),
        "wet_index_gain": wet_gain.tolist(),
    }


# %% [markdown]
# ## Compare the two predictions at each initial size
#
# Each plotted value is the mean scattering cross section after dilution
# divided by its value before dilution. Both curves use the same enlarged
# droplets. The distance between the curves shows the effect of the
# refractive index alone. The two printed cases show where their ordering
# reverses.


# %%
def plot_results(report):
    """Plot the two scattering gains using the compact numerical report."""
    diameters = report["median_nm"]
    fixed = report["fixed_index_gain"]
    wet = report["wet_index_gain"]
    figure, axis = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    axis.plot(
        diameters, fixed, "s--", color="#975033", label="Keep initial index"
    )
    axis.plot(
        diameters, wet, "o-", color="#156656", label="Use final wet index"
    )
    axis.set(
        xlabel="Initial wet number median diameter (nm)",
        ylabel="Final / initial mean scattering cross section",
        title="Predicted 633 nm scattering after dilution",
        xticks=diameters,
    )
    axis.legend(frameon=False)
    axis.spines[["right", "top"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    print(f"Wet diameter growth factor: {report['diameter_ratio']:.3f}")
    for diameter in (200, 800):
        i = diameters.index(diameter)
        print(
            f"{diameter} nm: wet index {wet[i]:.2f}x, fixed index {fixed[i]:.2f}x"
        )
    return figure


def main():
    """Run the CPU calculation, display its figure and return the results."""
    report = simulate()
    plot_results(report)
    plt.show()
    return report


if __name__ == "__main__":
    main()


# %% [markdown]
# At 200 nm, updating the wet index gives 7.31-fold scattering growth,
# while retaining the initial index gives 9.66-fold growth. At 800 nm
# the same two calculations give 2.34 and 2.12. Scattering efficiency
# depends on both refractive index and particle size relative to the
# wavelength, so a fixed index can shift the predicted gain in either
# direction.
#
# The paper provides the material properties. The seven size distributions
# and their scattering gains are model predictions. A chamber test could
# measure the wet size distribution and scattering per particle at both
# compositions, using the same optical wavelength for refractometry and
# scattering. When only the component indices are available, the
# [optics guide](../../../../guide/optics/index.md) explains how to supply
# water and solute indices for volume-fraction mixing as composition evolves.
