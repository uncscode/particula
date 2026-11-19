# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: notebookname
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
#   notebookname: Particle_Surface_Tutorial
# ---

"""Tutorial on surface strategies and Kelvin effect calculations for particles."""

# %% [markdown]
# # How strongly does curvature raise equilibrium vapor pressure?
#
# A small droplet needs a higher vapor pressure to remain in equilibrium than
# a flat liquid surface. How much of that increase comes from radius, and how
# much comes from the condensing species and the assumed surface composition?
# We compare glycerol, squalane, and glycerol in an illustrative mixed phase at
# 300 K. This is a Kelvin-factor calculation, not a droplet-growth simulation.
#
# The dimensionless factor is $K_i=\exp(r_{K,i}/r)$, where
# $r_{K,i}=2\sigma M_i/(RT\rho_i)$. Here $r$ and $r_K$ are in m,
# $\sigma$ in N/m, $M_i$ in kg/mol, and $\rho_i$ in kg/m³.
# The strategy uses each species' $M_i/\rho_i$ as its liquid molar-volume
# approximation. It can average surface tension within assigned phases; it does
# not replace species densities with a bulk mixture density.
#
# Read this alongside the [activity comparison](activity_tutorial.py): activity
# and curvature are separate factors in equilibrium pressure. Material constants
# below are illustrative fixed inputs, without temperature dependence or a fitted
# surface-enrichment model.

# %%
# In Colab uncomment the following command to install particula:
# Install from the repository root: `python -m pip install -e ".[extra]"`
import matplotlib.pyplot as plt
import numpy as np
import particula as par
from particula import plotting

plt.style.use("particula.default")

# %% [markdown]
# ## Establish a pure-component reference
#
# For one species, mole-, mass-, and volume-weighted surface tensions coincide.
# Use named SI arguments to keep molar mass and surface tension distinct. The
# printed Kelvin radius is a characteristic length, not the physical droplet
# radius: at $r=r_K$, the Kelvin factor is $e$.

# %%
glycerol_molar_mass = 0.092093  # kg/mol
glycerol_density = 1261  # kg/m^3
glycerol_surface_tension = 0.063  # N/m

# Create a surface strategy
glycerol_surface_strategy = par.particles.SurfaceStrategyMolar(
    surface_tension=glycerol_surface_tension,
    density=glycerol_density,
    molar_mass=glycerol_molar_mass,
)

glycerol_kelvin_radius = float(
    np.squeeze(
        glycerol_surface_strategy.kelvin_radius(
            molar_mass=glycerol_molar_mass,
            mass_concentration=0.1,  # does not matter for a single species mixture
            temperature=300,
        )
    )
)

print(f"kelvin radius of glycerol {glycerol_kelvin_radius} m")

# %% [markdown]
# ### Read the radius dependence
#
# The curve approaches one at large radius and rises rapidly as radius shrinks.
# The dashed line marks $r_K$. The grid extends from 0.1 nm to 1 µm to expose the
# formula's behavior; its molecular-scale end is an extrapolation of a continuum
# surface model, not a reliable prediction for molecular clusters. The
# implementation caps the exponent at 100; the smallest-radius squalane plateau
# in the following plots is that numerical cap, not a physical saturation.

# %%
radii = np.logspace(-10, -6, 100)

glycerol_kelvin_term = np.atleast_1d(
    glycerol_surface_strategy.kelvin_term(
        radius=radii,
        molar_mass=glycerol_molar_mass,
        mass_concentration=0.1,
        temperature=300,
    )
).squeeze()

fig, ax = plotting.subplots(size="wide")
ax.plot(radii, glycerol_kelvin_term)
ax.plot(
    np.array([glycerol_kelvin_radius, glycerol_kelvin_radius]),
    np.array([glycerol_kelvin_term.min(), glycerol_kelvin_term.max()]),
    color=plotting.COLORS["rust"],
    linestyle="--",
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title("Kelvin term of glycerol")
ax.set_xlabel("radius (m)")
ax.set_ylabel("Kelvin term")
plotting.legend(ax, labels=["Kelvin term", "Kelvin radius"])
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Separate surface tension from molecular volume
#
# The builder converts 0.81 g/cm³ to 810 kg/m³ and 28 mN/m to 0.028 N/m for
# squalane. Its surface tension is lower than glycerol's, but its $M/\rho$ is
# larger. Compare the product $\sigma M/\rho$, rather than surface tension
# alone: these inputs give a larger Kelvin radius and factor for squalane at
# the same particle radius. The builder changes input handling, not the equation.

# %%
squalane_surface = (
    par.particles.SurfaceStrategyMassBuilder()
    .set_density(0.81, density_units="g/cm^3")  # call with parameter name
    .set_surface_tension(28, "mN/m")  # call without parameter name
    .build()
)

# create plot
squalane_kelvin_radius = float(
    np.squeeze(
        squalane_surface.kelvin_radius(
            molar_mass=0.422, mass_concentration=0.1, temperature=300
        )
    )
)
squalane_kelvin_term = np.atleast_1d(
    squalane_surface.kelvin_term(
        radius=radii,
        molar_mass=0.422,
        mass_concentration=0.1,
        temperature=300,
    )
).squeeze()

fig, ax = plotting.subplots(size="wide")
ax.plot(radii, squalane_kelvin_term)
ax.plot(
    np.array([squalane_kelvin_radius, squalane_kelvin_radius]),
    np.array([squalane_kelvin_term.min(), squalane_kelvin_term.max()]),
    color=plotting.COLORS["rust"],
    linestyle="--",
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title("Kelvin term of squalane")
ax.set_xlabel("radius (m)")
ax.set_ylabel("Kelvin term")
plotting.legend(ax, labels=["Kelvin term", "Kelvin radius"])
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## Put two species in one assumed phase
#
# The factory exposes the volume-weighted builder. Assign both species to phase
# zero explicitly: without a phase assignment, the strategy retains separate
# species surface tensions. Equal mass contributions of 0.1 kg/m³ give unequal
# volume fractions because the densities differ.
#
# This is an imposed homogeneous phase, not a prediction that glycerol and
# squalane are miscible. Volume averaging sets one surface tension for that
# phase. Each species still has its own molar mass and density. Calculate both
# species lanes, then select glycerol's lane for comparison with pure glycerol.
# That comparison isolates the chosen surface-tension mixing rule. The pure
# squalane curve also changes the condensing species and is labeled separately.

# %%
parameters = {  # glycerol, squalane
    "density": np.array([1261, 810]),  # kg/m^3
    "density_units": "kg/m^3",
    "surface_tension": np.array([0.063, 0.028]),  # N/m
    "surface_tension_units": "N/m",
}

mixture_surface = (
    par.particles.SurfaceFactory()
    .get_builders()["volume"]
    .set_parameters(parameters)
    .set_phase_index(np.array([0, 0]))
    .build()
)

mixture_kelvin_radius = mixture_surface.kelvin_radius(
    molar_mass=np.array([0.092093, 0.422]),
    mass_concentration=np.array([0.1, 0.1]),
    temperature=300,
)
mixture_kelvin_radius = float(np.asarray(mixture_kelvin_radius).reshape(-1)[0])
print(f"kelvin radius of glycerol in mixture {mixture_kelvin_radius} m")

mixture_kelvin_term = np.atleast_1d(
    mixture_surface.kelvin_term(
        radius=radii,
        molar_mass=np.array([0.092093, 0.422]),
        mass_concentration=np.array([0.1, 0.1]),
        temperature=300,
    )
)[:, 0]

# Independently check SI conversion and the chosen phase-average formula.
from particula.util.constants import GAS_CONSTANT

volume_contributions = np.array([0.1, 0.1]) / parameters["density"]
volume_fractions = volume_contributions / volume_contributions.sum()
expected_surface_tension = np.dot(
    volume_fractions, parameters["surface_tension"]
)
np.testing.assert_allclose(
    glycerol_kelvin_radius,
    2
    * glycerol_surface_tension
    * glycerol_molar_mass
    / (GAS_CONSTANT * 300 * glycerol_density),
)
np.testing.assert_allclose(
    squalane_kelvin_radius, 2 * 0.028 * 0.422 / (GAS_CONSTANT * 300 * 810)
)
np.testing.assert_allclose(
    mixture_kelvin_radius,
    2
    * expected_surface_tension
    * glycerol_molar_mass
    / (GAS_CONSTANT * 300 * glycerol_density),
)
np.testing.assert_allclose(
    mixture_kelvin_term, np.exp(mixture_kelvin_radius / radii)
)

fig, ax = plotting.subplots(size="wide")
ax.plot(radii, mixture_kelvin_term, label="Glycerol in mixed phase")
ax.plot(
    np.array([mixture_kelvin_radius, mixture_kelvin_radius]),
    np.array([mixture_kelvin_term.min(), mixture_kelvin_term.max()]),
    color=plotting.COLORS["rust"],
    linestyle="--",
)
ax.plot(radii, glycerol_kelvin_term, label="Pure glycerol")
ax.plot(
    np.array([glycerol_kelvin_radius, glycerol_kelvin_radius]),
    np.array([glycerol_kelvin_term.min(), glycerol_kelvin_term.max()]),
    color=plotting.COLORS["rust"],
    linestyle="--",
)
ax.plot(radii, squalane_kelvin_term, label="Pure squalane")
ax.plot(
    np.array([squalane_kelvin_radius, squalane_kelvin_radius]),
    np.array([squalane_kelvin_term.min(), squalane_kelvin_term.max()]),
    color=plotting.COLORS["rust"],
    linestyle="--",
)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_title("Kelvin term of mixture")
ax.set_xlabel("radius (m)")
ax.set_ylabel("Kelvin term")
plotting.legend(ax)
plotting.fit(fig)
plt.show()

# %% [markdown]
# ## What the comparison tells us
#
# For these inputs, adding the lower-surface-tension component reduces the
# Kelvin factor for **glycerol** relative to pure glycerol. Pure squalane has the
# larger factor despite its lower surface tension, because its molar volume is
# larger. Thus surface tension alone cannot rank curvature effects across species.
# The equation checks verify parameter conversion and the imposed averaging rule;
# they do not establish that the mixture is physically homogeneous.
#
# For a research extension, vary the squalane mass fraction and compare glycerol's
# $K_i$ at a fixed 20 nm radius under mass-, mole-, and volume-weighted surface
# tension rules. Keep the condensing species fixed. Measured mixture surface
# tension and phase behavior would be needed to choose among those approximations
# before using them in a [condensation calculation](../../processes/condensation/condensation_1_bin.py).
