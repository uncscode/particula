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
# # Wall Loss Tutorial: Spherical vs Rectangular Chambers
#
# Learn how to configure wall loss strategies, compare chamber geometries,
# and chain wall loss with coagulation and condensation. All examples run
# quickly (≤30 s) with small grids and reuse helper functions to avoid
# recomputing simulations.

# %% [markdown]
# ## Setup
# - Uses particula public APIs from `par.dynamics` re-exports.
# - Time grid: 0–30 min with 30 s spacing (61 points).
# - Distribution: preset discrete radius bins (small and fast).
# - Plots: minutes on x-axis; log-scale concentrations.

# %%
import numpy as np
import particula as par
from matplotlib import pyplot as plt

np.set_printoptions(precision=3, suppress=True)
TIME_STEP = 30.0  # seconds
TOTAL_TIME = 1800.0  # seconds (30 minutes)
N_STEPS = int(TOTAL_TIME / TIME_STEP)
TIME_GRID = np.linspace(0.0, TOTAL_TIME, N_STEPS + 1)
TEMPERATURE = 298.15  # K
PRESSURE = 101325.0  # Pa


# %%
def build_aerosol() -> par.Aerosol:
    """Create a small discrete aerosol with an empty atmosphere."""
    particles = par.particles.PresetParticleRadiusBuilder().build()
    partitioning_species = par.gas.GasSpecies(
        name="PlaceholderPartitioning",
        molar_mass=0.018,
        partitioning=True,
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(1e3),
    )
    gas_only_species = par.gas.GasSpecies(
        name="PlaceholderGasOnly",
        molar_mass=0.028,
        partitioning=False,
        vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(1e3),
    )
    atmosphere = par.gas.Atmosphere(
        temperature=TEMPERATURE,
        total_pressure=PRESSURE,
        partitioning_species=partitioning_species,
        gas_only_species=gas_only_species,
    )
    return par.Aerosol(atmosphere=atmosphere, particles=particles)


def run_wall_loss(strategy: par.dynamics.WallLossStrategy, sub_steps: int = 1):
    """Execute wall loss over the shared time grid."""
    aerosol = build_aerosol()
    wall_loss = par.dynamics.WallLoss(wall_loss_strategy=strategy)
    series = np.zeros_like(TIME_GRID)
    series[0] = aerosol.particles.get_total_concentration()
    for idx in range(1, TIME_GRID.size):
        aerosol = wall_loss.execute(
            aerosol, time_step=TIME_STEP, sub_steps=sub_steps
        )
        series[idx] = aerosol.particles.get_total_concentration()
    return series, aerosol


def plot_decay(times_min, *series_with_labels):
    fig, ax = plt.subplots(figsize=(6, 4))
    for series, label, style in series_with_labels:
        ax.plot(times_min, series / series[0], style, label=label, linewidth=2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Normalized concentration")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig, ax


times_minutes = TIME_GRID / 60.0
baseline_aerosol = build_aerosol()
print("Initial bins:", baseline_aerosol.particles.get_radius().size)
print(
    "Initial total conc (1/m^3):",
    f"{baseline_aerosol.particles.get_total_concentration():.3e}",
)

# %% [markdown]
# ## Basic spherical wall loss
# Spherical chamber with modest eddy diffusivity and radius. Uses two
# sub-steps to demonstrate clamping behavior in the runnable.

# %%
spherical_strategy = par.dynamics.SphericalWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_radius=0.5,
    distribution_type="discrete",
)
spherical_series, spherical_aerosol = run_wall_loss(
    spherical_strategy, sub_steps=2
)
print("Final conc (spherical):", f"{spherical_series[-1]:.3e} 1/m^3")

# %%
_ = plot_decay(
    times_minutes,
    (spherical_series, "Spherical (0.5 m)", "b-"),
)

# %% [markdown]
# ## Geometry comparison: spherical vs rectangular
# Equal-volume comparison: a 0.5 m radius sphere (~0.52 m^3) vs a rectangular
# chamber with dimensions (1.0, 0.72, 0.72) m.

# %%
rectangular_strategy = par.dynamics.RectangularWallLossStrategy(
    wall_eddy_diffusivity=1e-3,
    chamber_dimensions=(1.0, 0.72, 0.72),
    distribution_type="discrete",
)
rect_series, _ = run_wall_loss(rectangular_strategy)
fig_cmp, _ = plot_decay(
    times_minutes,
    (spherical_series, "Spherical (0.5 m)", "b-"),
    (rect_series, "Rectangular (1.0x0.72x0.72 m)", "r--"),
)
fig_cmp.axes[0].set_title("Geometry comparison")

# %% [markdown]
# ## Advanced chaining: coagulation + wall loss
# Demonstrates runnable composition with the `|` operator. Uses the same time
#     grid and small bin counts to keep runtime low.

# %%
coag_strategy = par.dynamics.BrownianCoagulationStrategy(
    distribution_type="discrete"
)
coagulation = par.dynamics.Coagulation(coagulation_strategy=coag_strategy)
coag_wall = coagulation | par.dynamics.WallLoss(
    wall_loss_strategy=spherical_strategy
)

aerosol_wall_only = build_aerosol()
aerosol_coag_wall = build_aerosol()
wall_only_series = np.zeros_like(TIME_GRID)
coag_wall_series = np.zeros_like(TIME_GRID)
wall_only_series[0] = aerosol_wall_only.particles.get_total_concentration()
coag_wall_series[0] = aerosol_coag_wall.particles.get_total_concentration()

for idx in range(1, TIME_GRID.size):
    aerosol_wall_only = par.dynamics.WallLoss(
        wall_loss_strategy=spherical_strategy
    ).execute(aerosol_wall_only, time_step=TIME_STEP)
    aerosol_coag_wall = coag_wall.execute(
        aerosol_coag_wall, time_step=TIME_STEP
    )
    wall_only_series[idx] = (
        aerosol_wall_only.particles.get_total_concentration()
    )
    coag_wall_series[idx] = (
        aerosol_coag_wall.particles.get_total_concentration()
    )

_ = plot_decay(
    times_minutes,
    (wall_only_series, "Wall loss only", "b-"),
    (coag_wall_series, "Coagulation | Wall loss", "g--"),
)

# %% [markdown]
# ### Concise condensation + wall loss snippet
# Shows a lightweight condensation strategy chained with wall loss over a
# short window (10 min). Parameters are intentionally small to keep runtime
# ≤1 s.

# %%
short_time = np.linspace(0.0, 600.0, 21)  # 10 min, 30 s spacing
cond_series = np.zeros_like(short_time)
try:
    condensation_strategy = par.dynamics.CondensationIsothermalBuilder()
    condensation_strategy.set_molar_mass(0.12, "kg/mol")
    condensation_strategy.set_diffusion_coefficient(1e-5, "m^2/s")
    condensation_strategy.set_accommodation_coefficient(0.9)
    condensation_strategy.set_update_gases(False)
    condensation = par.dynamics.MassCondensation(
        condensation_strategy=condensation_strategy.build()
    )
    wall_loss_runnable = par.dynamics.WallLoss(
        wall_loss_strategy=spherical_strategy
    )
    cond_wall = condensation | wall_loss_runnable

    aerosol_cond = build_aerosol()
    cond_series[0] = aerosol_cond.particles.get_total_concentration()

    for idx in range(1, short_time.size):
        aerosol_cond = cond_wall.execute(
            aerosol_cond, time_step=short_time[1] - short_time[0]
        )
        cond_series[idx] = aerosol_cond.particles.get_total_concentration()

    _ = plot_decay(
        short_time / 60.0,
        (cond_series, "Condensation | Wall loss (10 min)", "m-"),
    )
except ValueError as exc:
    print(f"Skipping condensation wall loss snippet: {exc}")

# %% [markdown]
# ## Assertions and sanity checks
# Quick guards to keep the notebook honest during headless execution.


# %%
# Geometry parameter validation
try:
    par.dynamics.SphericalWallLossStrategy(
        wall_eddy_diffusivity=1e-3, chamber_radius=-1.0
    )
    raise AssertionError("Expected ValueError for negative radius")
except ValueError:
    pass

try:
    par.dynamics.RectangularWallLossStrategy(
        wall_eddy_diffusivity=1e-3, chamber_dimensions=(0.0, 1.0, 1.0)
    )
    raise AssertionError("Expected ValueError for non-positive dimension")
except ValueError:
    pass

# Concentration non-negativity after simulations
for name, series in [
    ("spherical", spherical_series),
    ("rectangular", rect_series),
    ("wall only", wall_only_series),
    ("coagulation | wall", coag_wall_series),
    ("condensation | wall", cond_series),
]:
    if not np.all(series >= 0.0):
        raise AssertionError(f"Non-negative concentrations failed for {name}")

print("Assertions passed: parameter guards and non-negativity checks.")

# %% [markdown]
# ## Summary
# - Built reusable helpers to construct aerosols and run wall loss quickly.
# - Compared spherical vs rectangular chambers over a shared time grid.
# - Demonstrated chaining with coagulation and a concise condensation variant.
# - Added inline assertions for invalid geometry and non-negative concentrations.
