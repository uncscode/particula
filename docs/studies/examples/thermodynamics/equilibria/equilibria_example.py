# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Separate water uptake from organic partitioning
#
# Can humidity add substantial particle water while barely changing how much
# organic material is condensed? This equilibrium example solves three low-volatility
# organic classes at prescribed water activity and inspects phase inventories.
# It teaches how to read partition coefficients, associated water, and an optimizer
# residual without confusing them with a time-dependent growth calculation.

# %%
import numpy as np
import particula as par

# %% [markdown]
# ## Keep concentration units and volatility scales consistent
#
# The dry C* values are 10⁻⁶, 10⁻⁴, and 10⁻² µg m⁻³, in the same concentration
# units as the total organic inputs 1, 5, and 10 µg m⁻³. They are all small relative
# to the loading, so near-complete organic condensation is a plausible outcome.
# This interface takes molar mass in g mol⁻¹, unlike many native process interfaces.
# The O:C ratios and densities specify illustrative organic descriptors.

# %%
# Three species with different volatilities (C* values)
c_star_j_dry = np.array([1e-6, 1e-4, 1e-2])  # saturation concentrations
concentration_organic = np.array([1.0, 5.0, 10.0])  # ug/m^3
molar_mass = np.array([200.0, 200.0, 200.0])  # g/mol
o2c_ratio = np.array([0.2, 0.3, 0.5])  # O:C ratios
density = np.array([1200.0, 1200.0, 1200.0])  # kg/m^3

print("=== Organic Species Properties ===")
print(f"C* (dry): {c_star_j_dry}")
print(f"Total concentration: {concentration_organic} ug/m^3")
print(f"O:C ratios: {o2c_ratio}")

# %% [markdown]
# ## Prescribe water activity, with its equilibrium assumptions
#
# The first solve sets a_w = 0.75. Interpreting that as 75% relative humidity
# assumes equilibrium with the gas and neglects curvature. No radius, Kelvin
# correction, finite water reservoir, or equilibration timescale is solved here.

# %%
strategy = par.equilibria.LiquidVaporPartitioningStrategy(
    water_activity=0.75,  # 75% relative humidity
)

print(f"Strategy: {type(strategy).__name__}")
print(f"Water activity (RH): {strategy.water_activity * 100:.0f}%")

# %% [markdown]
# ## Solve an endpoint rather than a trajectory
#
# The strategy computes phase partitioning at the imposed activity. Its return
# value contains the modeled endpoint and numerical residual; the printed success
# message alone is not the acceptance test. The explicit assertions later check
# bounds, finite residual, and organic material accounting.

# %%
result = strategy.solve(
    c_star_j_dry=c_star_j_dry,
    concentration_organic_matter=concentration_organic,
    molar_mass=molar_mass,
    oxygen2carbon=o2c_ratio,
    density=density,
)

print("Equilibrium solved successfully!")

# %% [markdown]
# ## Interpret phase entries before summing them
#
# Partition coefficients give condensed organic fractions. Phase species entries
# include species-associated water, so subtract phase water before comparing with
# organic-only input. At a_w = 0.75, the saved solution has about 4.51 µg m⁻³
# water in the alpha phase and zero in beta, with organic partition coefficients
# near one. The presence of a beta result object does not mean a populated second
# phase has formed.

# %%
print("=== Liquid-Vapor Partitioning ===")
print(f"Water activity (RH): {strategy.water_activity * 100:.0f}%\n")

print("Input organic species:")
print(f"  C* (dry): {c_star_j_dry}")
print(f"  Total concentration: {concentration_organic} ug/m^3")
print(f"  O:C ratios: {o2c_ratio}")

print("\n=== Equilibrium Results ===")
print(f"Partition coefficients: {result.partition_coefficients}")
print("  (fraction in condensed phase)")

print("\nAlpha phase (water-rich):")
print(
    f"  Species-associated organic + water: {result.alpha_phase.species_concentrations}"
)
print(
    "  Water concentration: "
    f"{result.alpha_phase.water_concentration:.2f} ug/m^3"
)
print(
    "  Total concentration: "
    f"{result.alpha_phase.total_concentration:.2f} ug/m^3"
)

if result.beta_phase is not None:
    print("\nBeta phase (organic-rich):")
    print(
        f"  Species-associated organic + water: {result.beta_phase.species_concentrations}"
    )
    print(
        "  Water concentration: "
        f"{result.beta_phase.water_concentration:.2f} ug/m^3"
    )

print(
    f"\nWater content: alpha={result.water_content[0]:.2f}, "
    f"beta={result.water_content[1]:.2f} ug/m^3"
)
print(f"Optimization error: {result.error:.2e}")

# %% [markdown]
# ## Compare humidity sensitivity with the right diagnostic
#
# Across imposed activities 0.3–0.9, saved alpha-phase water rises from about
# 0.66 to 13.42 µg m⁻³ while the mean organic partition coefficient remains near
# one. The unweighted mean is not the total condensed mass fraction; species have
# different loadings. The final inventory assertion applies to the first 0.75
# solution, not every point in this sweep.

# %%
print("=== Effect of RH on Partitioning ===\n")
rh_values = [0.3, 0.5, 0.75, 0.9]
for rh in rh_values:
    strat = par.equilibria.LiquidVaporPartitioningStrategy(
        water_activity=rh,
    )
    res = strat.solve(
        c_star_j_dry=c_star_j_dry,
        concentration_organic_matter=concentration_organic,
        molar_mass=molar_mass,
        oxygen2carbon=o2c_ratio,
        density=density,
    )
    mean_partition = np.mean(res.partition_coefficients)
    print(
        f"RH={rh * 100:.0f}%: mean partition coeff = {mean_partition:.4f}, "
        f"water = {res.alpha_phase.water_concentration:.2f} ug/m^3"
    )

# The phase species entries include associated water. Subtract total water
# before comparing with the organic-only partitioned inventory.
phases = [p for p in (result.alpha_phase, result.beta_phase) if p is not None]
organic_condensed = sum(
    phase.total_concentration - phase.water_concentration for phase in phases
)
np.testing.assert_allclose(
    organic_condensed,
    np.sum(result.partition_coefficients * concentration_organic),
    rtol=1e-12,
    atol=1e-12,
)
assert np.all(
    (result.partition_coefficients >= 0) & (result.partition_coefficients <= 1)
)
assert np.isfinite(result.error) and result.error < 1e-8

# %% [markdown]
# ## Choose a case that can answer your research question
#
# Under these low-C* inputs, humidity sensitivity is much more apparent in water
# than in the rounded organic partition coefficients. That outcome should not
# be generalized to all volatile mixtures. The small optimizer residual is a
# numerical endpoint check, not validation of the activity model.
#
# To probe humidity-driven changes in organic mass, choose C* values spanning the
# actual organic loading, preserve consistent concentration units, and track each
# species' gas and condensed fractions plus total water. Check residuals and
# inventories at every sweep point. Compare with a kinetic growth calculation only
# after matching activity, volatility, reservoir, and curvature assumptions.
