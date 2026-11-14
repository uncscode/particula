# %% [markdown]
# # Build on the CPU, evolve on CUDA, analyze on the CPU
#
# A prepared simulation owns the complete aerosol state on one GPU. Construct
# particles, gas and environment normally, compose supported processes with
# `|`, then call `prepare` once. `run` keeps state resident. `snapshot` returns
# detached NumPy arrays for analysis and plotting.
#
# This example contains eight independent boxes, 1,024 resolved particles per
# box, two particle materials, and three gas species. Water is gas lane 2 and
# particle material 0: an explicit species map preserves that distinction.
# Temperature varies between boxes. After one millisecond, every box cools by 5 K.
# We capture the subsequent schedule and verify exact checkpoint continuation.
# A second run halves the timestep at the same forcing time and checks final
# particle masses and vapor. This is a short, illustrative cooling experiment.
# CUDA is required; there is no CPU fallback or GPU timing claim here.

# %%
"""A complete native CPU → prepared CUDA → CPU analysis workflow."""

import numpy as np
from particula import Aerosol
from particula.dynamics import Dilution, DilutionStrategy, MassCondensation
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution import Backend
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
    WaterBuckStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityIdealMolar
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT


def build_aerosol():
    """Construct float64 native state with unit particle multiplicities."""
    boxes, slots = 8, 1024
    rng = np.random.default_rng(42)
    radius = rng.lognormal(np.log(35e-9), np.log(1.6), (boxes, slots))
    total_mass = 4 * np.pi / 3 * 1000 * radius**3
    masses = total_mass[..., None] * np.array([0.2, 0.8])
    temperature = np.linspace(278.15, 308.15, boxes)
    molar_mass = np.array([0.028, 0.2, 0.018])
    gas = np.full((boxes, 3), 1e-8)
    celsius = temperature - 273.15
    pressure_water = 611.21 * np.exp(
        (18.678 - celsius / 234.5) * celsius / (257.14 + celsius)
    )
    gas[:, 2] = (
        1.3 * pressure_water * molar_mass[2] / (GAS_CONSTANT * temperature)
    )
    return Aerosol(
        particles=ParticleData(
            masses=masses,
            concentration=np.ones((boxes, slots)),  # counts, not m^-3
            charge=np.zeros((boxes, slots)),
            density=np.array([1000.0, 1000.0]),
            volume=np.full(boxes, slots / 5e13),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            name=["inert", "tracer", "water"],
            molar_mass=molar_mass,
            concentration=gas,  # kg m^-3
            partitioning=np.array([False, False, True]),
        ),
        environment=EnvironmentData(
            temperature=temperature,
            pressure=np.full(boxes, 101325.0),
            saturation_ratio=np.ones((boxes, 3)),
        ),
    )


# %% [markdown]
# ## Bind supported physics explicitly
#
# The condensation configuration supplies the water vapor-pressure law,
# ideal molar activity, surface tension and transport coefficients. Each CUDA
# condensation call performs four equal coupled substeps. Dilution removes
# particle counts and gas with the same exponential factor.
#
# Preparation validates this configuration before upload and freezes it for
# the session. Editing the original process object later does not change the
# prepared calculation. Unsupported strategies raise an error.


# %%
def build_process(aerosol):
    """Compose water condensation and dilution without backend-specific physics."""
    condensation = MassCondensation(
        CondensationIsothermal(
            molar_mass=aerosol.gas.molar_mass.copy(),
            diffusion_coefficient=np.full(3, 2e-5),
            accommodation_coefficient=np.full(3, 0.5),
            activity_strategy=ActivityIdealMolar(np.array([0.018, 0.2])),
            surface_strategy=SurfaceStrategyVolume(
                surface_tension=0.072, density=1000.0
            ),
            vapor_pressure_strategy=(
                ConstantVaporPressureStrategy(1.0),
                ConstantVaporPressureStrategy(1.0),
                WaterBuckStrategy(),
            ),
            species_map=CondensationSpeciesMap(
                tuple(aerosol.gas.name), ((2, 0),)
            ),
        )
    )
    return condensation | Dilution(DilutionStrategy(0.02))


def water_inventory(aerosol):
    """Compute extensive water inventory independently from CPU snapshot arrays."""
    p = aerosol.particles
    return np.sum(p.masses[..., 0] * p.concentration, axis=1) + (
        aerosol.gas.concentration[:, 2] * p.volume
    )


# %% [markdown]
# ## Run, change forcing, capture and restart
#
# `update_environment` is an explicit synchronization and upload boundary.
# `capture()` records the schedule without taking a scientific timestep.
# A snapshot is for inspection; a checkpoint additionally retains process
# configuration, random continuation and canonical state for exact restart.
# A resumed session can capture its own new graph if desired.


# %%
def run_example():
    """Execute the complete three-millisecond CUDA trajectory and exact continuation."""
    import matplotlib.pyplot as plt
    import warp as wp

    if not wp.is_cuda_available():
        raise RuntimeError("This example requires CUDA; no CPU fallback")
    initial = build_aerosol()
    process = build_process(initial)
    original_masses = initial.particles.masses.copy()
    with process.prepare(
        initial,
        backend=Backend.WARP,
        device="cuda:0",
        time_step=1e-6,
        root_seed=7,
        logical_box_ids=tuple(f"box-{i}" for i in range(8)),
    ) as simulation:
        simulation.run(steps=1000)
        warm = simulation.snapshot()
        simulation.update_environment(
            temperature=initial.environment.temperature - 5.0
        )
        simulation.capture()
        simulation.run(steps=1000)
        saved = simulation.checkpoint()
        simulation.run(steps=1000)
        final = simulation.snapshot()
        diagnostics = simulation.diagnostics()
    with process.resume(saved) as resumed:
        resumed.run(steps=1000)
        continued = resumed.snapshot()
    for field in ("masses", "concentration", "charge", "volume"):
        np.testing.assert_array_equal(
            getattr(final.particles, field), getattr(continued.particles, field)
        )
    np.testing.assert_array_equal(
        final.gas.concentration, continued.gas.concentration
    )
    np.testing.assert_array_equal(initial.particles.masses, original_masses)
    expected = water_inventory(initial) * np.exp(-0.02 * 0.003)
    np.testing.assert_allclose(
        water_inventory(final), expected, rtol=2e-12, atol=1e-30
    )
    if not np.any(final.particles.masses != original_masses):
        raise AssertionError("No condensation occurred")
    verify_refinement(initial, process, final, expected)
    plot_snapshots(initial, warm, final)
    print("CUDA:", wp.get_device("cuda:0").name)
    print("Physical time: 3 ms; 8 boxes × 1024 particle slots; float64.")
    print(
        "Final number concentrations (m^-3):", diagnostics.number_concentration
    )
    print(
        "Water inventory matches dilution; restart is exact; CPU input is unchanged."
    )
    plt.show()
    return final


def verify_refinement(initial, process, coarse, expected_inventory):
    """Check final-state timestep sensitivity at identical forcing times."""
    with process.prepare(initial, device="cuda:0", time_step=5e-7) as refined:
        refined.capture()
        refined.run(steps=2000)
        refined.update_environment(
            temperature=initial.environment.temperature - 5.0
        )
        refined.run(steps=4000)
        fine = refined.snapshot()
    coarse_mass = coarse.particles.masses.sum(axis=2)
    fine_mass = fine.particles.masses.sum(axis=2)
    np.testing.assert_allclose(coarse_mass, fine_mass, rtol=0.01, atol=0)
    np.testing.assert_allclose(
        coarse.gas.concentration[:, 2],
        fine.gas.concentration[:, 2],
        rtol=0.001,
        atol=0,
    )
    np.testing.assert_allclose(
        water_inventory(fine), expected_inventory, rtol=2e-12, atol=1e-30
    )
    print(
        "Maximum final particle-mass change with half timestep:",
        np.max(np.abs(coarse_mass / fine_mass - 1)),
    )


# %% [markdown]
# ## Analyze detached snapshots
#
# The left panel compares size distributions before and after cooling in two
# boxes. The right panel shows the final distribution in every box. Histogram
# weights use represented counts divided by each box's volume and by bin width
# in log10 radius. These are particle-resolved histograms, not a PDF solver.


# %%
def plot_snapshots(initial, warm, final):
    """Plot distributions using only CPU NumPy arrays."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    edges = np.geomspace(5e-9, 2e-6, 55)

    def histogram(state):
        p = state.particles
        radius = np.cbrt(
            np.sum(p.masses / p.density, axis=-1) * 3 / (4 * np.pi)
        )
        return np.array(
            [
                np.histogram(r, bins=edges, weights=c / v)[0]
                for r, c, v in zip(
                    radius, p.concentration, p.volume, strict=True
                )
            ]
        ) / np.diff(np.log10(edges))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for box, color in ((0, "#287a90"), (7, "#be6c39")):
        for state, style, label in (
            (initial, ":", "initial"),
            (warm, "--", "1 ms"),
            (final, "-", "3 ms"),
        ):
            axes[0].stairs(
                histogram(state)[box],
                edges * 1e9,
                color=color,
                linestyle=style,
                label=f"Box {box + 1}, {label}",
            )
    axes[0].set(
        xscale="log", xlabel="Radius (nm)", ylabel="dN / dlog₁₀ r (m$^{-3}$)"
    )
    axes[0].legend(fontsize=8)
    values = histogram(final)
    mesh = axes[1].pcolormesh(
        edges * 1e9,
        np.arange(9) + 0.5,
        np.ma.masked_less_equal(values, 0),
        norm=LogNorm(),
        cmap="viridis",
        shading="flat",
    )
    axes[1].set(
        xscale="log",
        xlabel="Radius (nm)",
        ylabel="Box",
        yticks=np.arange(8) + 1,
    )
    fig.colorbar(mesh, ax=axes[1], label="dN / dlog₁₀ r (m$^{-3}$)")
    fig.suptitle(
        "Prepared CUDA simulation · cooling eight particle-resolved boxes"
    )


if __name__ == "__main__":
    run_example()
