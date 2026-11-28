# %% [markdown]
# # Count particles in a represented air volume
#
# Two idealized counters sample the same well-mixed aerosol. One samples 0.25 m³ of
# air and the other 4 m³. Their reported concentrations should agree. How
# many particles and how much of each species should a model store for each
# sample? This CPU example follows two discrete size bins, two resolved
# particle slots and a separate continuous radius distribution through
# the conversion. All three use prescribed concentrations.
#
# `ParticleData` stores each sample as a row. A column holds a bin, a
# resolved slot or a radius node. The [0.4.0 migration guide](../../../../guide/migration/v0.4.0/index.md)
# maps the earlier field names to these quantities. Start with NumPy and the
# CPU particle container and builder.

# %%
from importlib.util import find_spec

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
from particula import plotting
from particula.particles import ParticleData, ParticleDataBuilder

# %% [markdown]
# ## Count the same two sizes in different samples
#
# The first size contributes 4 particles/m³ and the second contributes 8.
# For discrete bins and resolved slots, multiply each density by the air
# volume once to get the number represented by that slot. For resolved data
# this is the multiplicity of the computational particle. A multiplicity
# already stored as a count in 0.3.6 carries over directly.
#
# Each row of the mass table describes one particle, with one column per
# species (A, B). The two species each have material density 1000 kg/m³.
# Dividing a component mass by that density gives its material volume.
# The volume passed to `ParticleData` is the sampled air volume. Extrapolating
# to a chamber or grid cell would additionally require its physical volume
# and an assumption that the sample represents the air throughout it.


# %%
def make_samples(kind, density, masses_kg, radius_m=None):
    """Create independent samples with equal concentrations at two volumes."""
    volumes_m3 = np.array([0.25, 4.0])
    population = volumes_m3[:, None] * density
    return ParticleData(
        population=population,
        masses=np.stack([masses_kg, masses_kg]),
        density=np.array([1000.0, 1000.0]),
        charge=np.zeros_like(population),
        volume=volumes_m3,
        radius_grid=radius_m,
        distribution_type=kind,
    )


def budget(particles):
    """Record density and represented inventories in the same box order."""
    return {
        "volume_m3": particles.volume.tolist(),
        "population": particles.population.tolist(),
        "slot_density": particles.slot_concentration_density.tolist(),
        "number_per_m3": particles.number_density.tolist(),
        "represented_number": (
            particles.number_density * particles.volume
        ).tolist(),
        "species_kg_per_m3": particles.species_mass_density.tolist(),
        "species_kg": particles.species_mass_inventory.tolist(),
    }


def discrete_and_resolved():
    """Compare two discrete bins with two individually weighted slots."""
    density = np.array([4.0, 8.0])  # particles/m³ per bin or slot
    masses_kg = np.array([[1.0, 3.0], [2.0, 4.0]]) * 1e-21
    return {
        kind: budget(make_samples(kind, density, masses_kg))
        for kind in ("discrete", "particle_resolved")
    }


# %% [markdown]
# Both representations store [1, 2] particles in the small sample and
# [16, 32] in the large one. Both report 12 particles/m³. The abundance of
# species A is $(4×1 + 8×2)×10^{-21}$ = 2.0 × 10⁻²⁰ kg/m³. Species B gives
# 4.4 × 10⁻²⁰ kg/m³. Multiplying either species density by 4 m³ gives
# sixteen times its inventory in the 0.25 m³ sample.
#
# ## Integrate a radius distribution on an uneven grid
#
# The continuous PDF specifies a number density per unit radius, $dn/dr$,
# at radii 1, 2 and 4 nm. Its SI units are m⁻⁴. Multiplication by sampled air
# volume produces the stored ordinate $dN/dr$ in m⁻¹. Integrate over radius
# to count particles. Linear interpolation gives trapezoidal node widths of
# half the adjacent interval at each endpoint and half the sum of adjacent
# intervals at each interior node.
#
# To check species mass on this grid, consider spheres with the same material
# density as above. Species A accounts for 25%, 50% and 75% of particle mass
# at the three successive radii. Weight each species' per-particle mass by
# the node's contribution to number density.


# %%
def radius_distribution():
    """Integrate radius ordinates and report both samples' species budgets."""
    radius_m = np.array([1.0, 2.0, 4.0]) * 1e-9
    ordinates = np.array([2.0, 4.0, 2.0]) * 1e9  # particles/(m³ air m radius)
    sphere_mass = (4 * np.pi / 3) * 1000.0 * radius_m**3
    fraction_a = np.array([0.25, 0.50, 0.75])
    masses_kg = sphere_mass[:, None] * np.column_stack(
        [fraction_a, 1 - fraction_a]
    )
    particles = make_samples("continuous_pdf", ordinates, masses_kg, radius_m)
    widths_m = np.diff(radius_m, prepend=radius_m[0]) / 2
    widths_m += np.diff(radius_m, append=radius_m[-1]) / 2
    density = particles.slot_concentration_density[0]
    return {
        **budget(particles),
        "radius_nm": (radius_m * 1e9).tolist(),
        "dn_dr_per_m3_per_nm": (density * 1e-9).tolist(),
        "node_width_nm": (widths_m * 1e9).tolist(),
        "node_number_per_m3": (density * widths_m).tolist(),
    }


# %% [markdown]
# The middle node contributes most of the number density. Its 1.5 nm width
# gives 6 particles/m³. The endpoint widths are 0.5 and 1 nm and contribute
# 1 and 2 particles/m³. Together they give 9 particles/m³, or 2.25 and 36
# represented particles in the two samples. A fractional inventory is possible
# because this example represents a continuous distribution.
#
# ## Compare the PDF with a diameter-binned measurement
#
# Suppose a counter covers diameters of 3–6 nm. Its edges fall inside two
# radius intervals. Convert every diameter edge to radius ($r=D/2$),
# interpolate the ordinate there, and integrate each interval. The edge list
# includes every interior PDF node, so each trapezoid integrates one linear
# segment exactly. On an arbitrary grid, split at any intervening PDF nodes.
# The bin's density divided by $\Delta\log_{10}D$ is its mean density per
# diameter decade. The bin density itself is the quantity to sum.


# %%
def diameter_spectrum(pdf):
    """Resolve an instrument band and both tails on the PDF support."""
    edges_nm = np.array([2.0, 3.0, 4.0, 6.0, 8.0])
    radius_edges_nm = edges_nm / 2
    heights = np.interp(
        radius_edges_nm, pdf["radius_nm"], pdf["dn_dr_per_m3_per_nm"]
    )
    bins = 0.5 * (heights[1:] + heights[:-1]) * np.diff(radius_edges_nm)
    return {
        "diameter_edges_nm": edges_nm.tolist(),
        "bin_number_per_m3": bins.tolist(),
        "per_m3_per_decade": (bins / np.diff(np.log10(edges_nm))).tolist(),
        "below_band_per_m3": float(bins[0]),
        "in_band_per_m3": float(bins[1:3].sum()),
        "above_band_per_m3": float(bins[-1]),
    }


# %% [markdown]
# ## Replace a particle measurement
#
# A separate one-bin sample at 2 particles/cm³ corresponds to 2 × 10⁶
# particles/m³, or 500,000 particles in 0.25 m³. On a base installation,
# supply the SI value to the builder. With Pint installed, the builder can
# also take the original units. The following calculation checks their
# agreement when that conversion is available.
#
# The CPU instance setter takes SI density. A new reading of 400 particles/m³
# represents 100 particles at the current 0.25 m³, or 200 if it came from
# 0.5 m³ of air. Record the state after each replacement. Then try a negative
# measurement with a positive new volume and check that both previous
# values survive the rejected update.


# %%
def import_measurement(value, units):
    """Build one sample from a density in the supplied units."""
    return (
        ParticleDataBuilder()
        .set_distribution_type("discrete")
        .set_masses(np.array([[1e-21]]))
        .set_density(np.array([1000.0]))
        .set_number_density(np.array([value]), units)
        .set_volume(0.25)
        .build()
    )


def replace_measurement():
    """Read the count and air volume after each attempted update."""
    sample = import_measurement(2e6, "1/m^3")
    if find_spec("pint") is not None:
        original_units = import_measurement(2.0, "1/cm^3")
        np.testing.assert_allclose(original_units.population, sample.population)

    def reading():
        return [float(sample.volume[0]), float(sample.population[0, 0])]

    readings = {"imported": reading()}
    sample.set_slot_concentration_density([400.0])
    readings["current_volume"] = reading()
    sample.set_slot_concentration_density([400.0], volume=0.5)
    readings["new_volume"] = reading()
    try:
        sample.set_slot_concentration_density([-1.0], volume=4.0)
    except ValueError:
        readings["after_rejection"] = reading()
    else:
        raise AssertionError("Negative number density should be rejected")
    assert readings["after_rejection"] == readings["new_volume"]
    return readings


# %% [markdown]
# ## Read the two spectra
#
# The full shaded area on the radius plot gives 9 particles/m³.
# The bars show the identical distribution in diameter intervals on a
# logarithmic horizontal axis. Multiplying each bar height by its width in
# decades recovers its share of that area. The text below the figure reports
# the number and species inventories for both samples.


# %%
@plt.style.context("particula.default")
def plot_results(report):
    """Plot the saved radius and diameter spectra, then print the budgets."""
    pdf = report["continuous_pdf"]
    spectrum = report["diameter_spectrum"]
    edges = np.array(spectrum["diameter_edges_nm"])
    fig, (left, right) = plotting.subplots(1, 2)
    left.fill_between(
        pdf["radius_nm"],
        pdf["dn_dr_per_m3_per_nm"],
        color=plotting.COLORS["teal"],
        alpha=0.2,
    )
    left.plot(
        pdf["radius_nm"],
        pdf["dn_dr_per_m3_per_nm"],
        "o-",
        color=plotting.COLORS["teal"],
    )
    left.set(
        title="Density per unit radius",
        xlabel="Radius (nm)",
        ylabel="dn/dr (particles m⁻³ nm⁻¹)",
        ylim=(0, 5),
        xticks=pdf["radius_nm"],
    )
    right.bar(
        edges[:-1],
        spectrum["per_m3_per_decade"],
        width=np.diff(edges),
        align="edge",
        color=["#d1d5db", "#08786c", "#08786c", "#d1d5db"],
        edgecolor="#343a40",
        linewidth=0.8,
    )
    right.set(
        title="Diameter bins",
        xlabel="Diameter (nm)",
        ylabel="Particles m⁻³ decade⁻¹",
        xscale="log",
    )
    right.set_xticks(edges, labels=[f"{edge:g}" for edge in edges])
    right.xaxis.set_minor_locator(NullLocator())
    right.text(
        0.04,
        0.96,
        "Teal bars span 3–6 nm",
        transform=right.transAxes,
        va="top",
        fontsize=8,
    )
    right.set_ylim(0, max(spectrum["per_m3_per_decade"]) * 1.25)
    plotting.fit(fig)
    print("Volume (m³)   Number    Species A (kg)   Species B (kg)")
    for kind in ("discrete", "particle_resolved", "continuous_pdf"):
        print(kind)
        sample = report[kind]
        for volume, count, (mass_a, mass_b) in zip(
            sample["volume_m3"],
            sample["represented_number"],
            sample["species_kg"],
            strict=True,
        ):
            print(
                f"{volume:>11g}   {count:>6g}    {mass_a:>14.4g}   {mass_b:>14.4g}"
            )
    return report


def run_example():
    """Compute the three physical comparisons and draw their spectra."""
    pdf = radius_distribution()
    return plot_results(
        {
            **discrete_and_resolved(),
            "continuous_pdf": pdf,
            "diameter_spectrum": diameter_spectrum(pdf),
            "measurement": replace_measurement(),
        }
    )


# %% [markdown]
# Running the example prints the per-species mass in each air sample and
# draws both spectra from the same three radius ordinates.

# %%
if __name__ == "__main__":
    report = run_example()
    plt.show()

# %% [markdown]
# ## What would the counter report?
#
# Both sampled volumes give the same concentration for a given distribution.
# The larger sample represents sixteen times the number and mass of particles.
# For the continuous PDF, the 3–6 nm diameter band captures 5.25 of the
# 9 particles/m³. The smaller-diameter tail contributes 1.25 and the
# larger-diameter tail 2.5 particles/m³, closing the budget over 2–8 nm.
# These are ideal number densities for a counter with sharp size limits and unit
# efficiency. For a counter with a known size response, a next step is to
# weight the radius distribution by detection efficiency before integration.
