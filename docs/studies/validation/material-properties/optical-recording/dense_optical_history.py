# %% [markdown]
# # Record aerosol extinction at 10 Hz during hygroscopic growth
#
# Can a simulation save the optical transient without saving each aerosol
# state? We follow water uptake in a closed, isothermal box and record bulk
# extinction, scattering, and absorption every 0.1 s at 450, 550, and 700 nm.
# Sparse aerosol snapshots from separate runs provide an offline calculation
# at matching times. A second comparison measures the cost of writing the
# same number of optical and native-array observations.
#
# The case uses 192 weighted particle-resolved slots with dry radii between
# 80 and 260 nm and a total number density of 2 × 10¹⁰ m⁻³. Each slot starts
# with water volume equal to 35% of its solute volume. The gas starts at 83%
# relative humidity and can lose water to particles. Temperature and pressure
# stay at 298.15 K and 101325 Pa. A small accommodation coefficient
# (3 × 10⁻⁶) spreads the hygroscopic transient over several seconds so that
# the observation cadence is visible. The parameters illustrate the recording
# workflow and do not describe a specific aerosol or instrument.
#
# The setup uses native Particula arrays and the isothermal condensation
# runnable. The optical configuration combines illustrative species indices
# by volume with the Lorentz–Lorenz rule. The result assumes independent
# scattering by homogeneous spheres. This example runs NumPy and native
# CUDA explicitly and requires a CUDA device. See the
# [condensation theory](../../../../theory/dynamics/condensation-equations.md)
# for transport and the [optics guide](../../../../guide/optics/index.md)
# for the optical mixing and reduction conventions.

# %%
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import Aerosol, plotting
from particula.dynamics import MassCondensation
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.analysis_io import save_analysis
from particula.execution.cpu_simulation import CPUSimulation
from particula.execution.optical_recording import (
    OpticalHistory,
    OpticalObserver,
    read_optical_history,
)
from particula.execution.prepared_recording import (
    PreparedObserver,
    run_prepared,
)
from particula.execution.recording import Recorder, run_until
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import WaterBuckStrategy
from particula.optics import OpticalConfig, get_mie_efficiencies
from particula.particles import create_resolved_particle_data
from particula.particles.activity_strategies import ActivityKappaParameter
from particula.particles.surface_strategies import SurfaceStrategyVolume

STEP_S = 0.1
STEPS = 180
SLOTS = 192
BATCH_SIZE = 32
WAVELENGTH_M = np.array([450, 550, 700]) * 1e-9
SPECIES_INDEX = np.array(
    [[1.34, 1.57 + 0.024j], [1.333, 1.56 + 0.020j], [1.33, 1.55 + 0.017j]]
)
CHANNELS = ("extinction", "scattering", "absorption")
TEMPERATURE_K = 298.15
INITIAL_RH = 0.83
WATER_MOLAR_MASS = 0.01801528
ACCOMMODATION = 3e-6


def chamber():
    """Construct one closed box with subsaturated water vapor and wet seeds."""
    dry_radius = np.geomspace(80e-9, 260e-9, SLOTS)
    count = np.exp(-0.5 * (np.log(dry_radius / 135e-9) / 0.27) ** 2)
    number = 2e10 * count / count.sum()
    dry_volume = 4 * np.pi / 3 * dry_radius**3
    material_density = np.array([997.0, 1550.0])
    particle = create_resolved_particle_data(
        masses=np.column_stack((0.35 * dry_volume, dry_volume))
        * material_density,
        density=material_density,
        concentration=number,
        concentration_units="1/m^3",
        volume=0.5,
    )
    water = WaterBuckStrategy()
    vapor = INITIAL_RH * water.saturation_concentration(
        WATER_MOLAR_MASS, TEMPERATURE_K
    )
    initial = Aerosol(
        particles=particle,
        gas=GasData(["water"], np.array([WATER_MOLAR_MASS]), [[vapor]], [True]),
        environment=EnvironmentData(
            [TEMPERATURE_K], [101325.0], [[INITIAL_RH]]
        ),
    )
    process = MassCondensation(
        CondensationIsothermal(
            molar_mass=np.array([WATER_MOLAR_MASS]),
            diffusion_coefficient=2.4e-5,
            accommodation_coefficient=ACCOMMODATION,
            activity_strategy=ActivityKappaParameter(
                kappa=np.array([0.0, 0.4]),
                density=material_density,
                molar_mass=np.array([WATER_MOLAR_MASS, 0.150]),
                water_index=0,
            ),
            surface_strategy=SurfaceStrategyVolume(
                density=997.0, surface_tension=0.072
            ),
            vapor_pressure_strategy=water,
            species_map=CondensationSpeciesMap(("water",), ((0, 0),)),
        )
    )
    return initial, process


# %% [markdown]
# ## Give optics and native state different output schedules
#
# The CUDA owner advances one 0.1 s cycle using four internal condensation
# substeps. On NumPy we set a 0.025 s maximum step and land on the same
# observation grid. The NumPy time driver can shorten a step to reach a
# requested time. For each backend we use the same physics schedule with and
# without recording.
#
# The optical observer evaluates the current population and writes reduced
# coefficients in chunks of 32 rows. Dense optical recording needs no
# detached native state, and the CUDA path transfers only bulk coefficients
# and status for the observer. The native-array comparison records particle
# masses, represented counts, charge, gas water, and fixed environmental
# coordinates. Its uncompressed analysis archive retains arrays for
# reanalysis. It uses the same 10 Hz output grid.


# %%
def bind(initial, process, backend):
    """Use the same observation grid with explicit backend choice."""
    if backend == "NumPy":
        return nullcontext(
            CPUSimulation(initial, process, time_step=STEP_S / 4)
        )
    return process.prepare(initial, time_step=STEP_S, device="cuda:0")


def advance(owner, backend, grid, *, optics=None, native=None):
    """Advance over grid and record the requested payload."""
    if backend == "NumPy":
        run_until(
            owner,
            grid[-1],
            output_times=grid,
            recorder=native,
            optics=optics,
            optical_output_times=grid if optics is not None else None,
        )
    else:
        observer = (
            PreparedObserver(owner, native, download="snapshot")
            if native is not None
            else None
        )
        run_prepared(
            owner,
            grid[-1],
            output_times=grid if native is not None else (),
            observer=observer,
            optics=optics,
            optical_initial=optics is not None,
        )


def optical_provenance(backend):
    """Record the RH convention and initial conditions beside the spectra."""
    return {
        "backend": backend,
        "initial_rh_fraction": INITIAL_RH,
        "rh_basis": "Gas water partial pressure / Buck flat-water saturation",
        "temperature_k": TEMPERATURE_K,
        "vapor_reservoir": "Closed gas plus particle water",
        "step_s": STEP_S,
    }


def timed_run(initial, process, backend, mode, grid, folder):
    """Measure one complete run, keeping setup and output costs visible."""
    with bind(initial, process, backend) as owner:
        optics = None
        native = None
        path = folder / mode
        setup_start = perf_counter()
        if mode == "optical":
            history = OpticalHistory(
                path,
                OpticalConfig(WAVELENGTH_M, species_index=SPECIES_INDEX),
                variables=CHANNELS,
                provenance=optical_provenance(backend),
            )
            optics = OpticalObserver(
                owner,
                history,
                batch_size=BATCH_SIZE,
                particle_tile=128,
                coefficient_capacity=32,
            )
        elif mode == "native":
            native = Recorder(
                (
                    "particle_masses",
                    "particle_concentration",
                    "particle_charge",
                    "gas_concentration",
                    "temperature",
                    "pressure",
                    "volume",
                )
            )
        advance(owner, backend, grid, optics=optics, native=native)
        if native is not None:
            save_analysis(
                native,
                path,
                configuration=optical_provenance(backend),
                numerical_settings={"output_interval_s": STEP_S},
                source_identity={"study": "dense_optical_history"},
                compressed=False,
            )
        elapsed = perf_counter() - setup_start
    return elapsed, path.stat().st_size if path.exists() else 0


# %% [markdown]
# ## Reconstruct extinction from independent snapshots
#
# We advance a second owner from the same initial state, detach aerosol
# states at 0, 0.5, 1.5, 4, 9, and 18 s, and recompute optical efficiencies on
# NumPy. We form each particle's mixed dielectric constant explicitly and
# sum its extinction, scattering, and absorption cross sections weighted
# by represented counts divided by box volume. This comparison exercises live-to-optical binding, number
# weighting and the output clock. It shares the same physical sphere optics
# assumptions as the recorder.
#
# For each sparse state we calculate gas RH from the water concentration
# and temperature of that state. We also verify that the gain in particle
# water equals the gas-water loss. This transfer comparison is sensitive
# to the small condensate gain in a relatively large vapor reservoir.


# %%
def snapshot_optics(snapshot):
    """Evaluate bulk coefficients [m⁻¹] by explicit number weighting."""
    particles = snapshot.particles
    volume = particles.masses[0] / particles.density
    total_volume = volume.sum(axis=-1)
    radius = np.cbrt(3 * total_volume / (4 * np.pi))
    fractions = volume / total_volume[:, None]
    dielectric = SPECIES_INDEX**2
    polar = fractions @ ((dielectric - 1) / (dielectric + 2)).T
    index = np.sqrt((1 + 2 * polar) / (1 - polar))
    efficiencies = get_mie_efficiencies(
        radius[:, None], WAVELENGTH_M[None, :], index
    )
    q = np.stack(
        [getattr(efficiencies, channel) for channel in CHANNELS], axis=-1
    )
    number = particles.concentration[0] / particles.volume[0]
    return (q * (np.pi * radius**2 * number)[:, None, None]).sum(axis=0)


def sparse_reference(initial, process, backend, grid):
    """Take detached snapshots and check water transfer and gas RH."""
    rows = np.array([0, 5, 15, 40, 90, STEPS])
    spectra, rh = [], []
    water = WaterBuckStrategy()
    start = 0
    initial_vapor = initial.gas.concentration[0, 0]
    initial_mass = initial.particles.masses[0, :, 0]
    number = initial.particles.concentration[0] / initial.particles.volume[0]
    with bind(initial, process, backend) as owner:
        for row in rows:
            advance(owner, backend, grid[start : row + 1])
            snapshot = owner.snapshot()
            spectra.append(snapshot_optics(snapshot))
            vapor = snapshot.gas.concentration[0, 0]
            rh.append(
                water.saturation_ratio(
                    vapor, WATER_MOLAR_MASS, snapshot.environment.temperature[0]
                )
            )
            gain = number @ (snapshot.particles.masses[0, :, 0] - initial_mass)
            np.testing.assert_allclose(
                gain, initial_vapor - vapor, rtol=3e-9, atol=2e-17
            )
            start = row
    return rows, np.asarray(spectra), np.asarray(rh)


# %% [markdown]
# Three repetitions follow a short warm-up. Timing starts after construction
# of the simulation owner and includes output setup, synchronized execution,
# and file writing. Deriving optics from the native archive requires a
# separate Mie calculation after this timed interval.
#
# The memory comparison counts the numeric payload retained for output.
# The optical observer holds at most one batch, including status values on
# CUDA. The native recorder retains all observation rows until the archive
# is written. Mie workspaces, temporary copies, serialization, and allocator
# overhead contribute additional memory beyond these payload counts. The
# CUDA memory-pool high-water mark is process-wide across the warm-up and
# all modes. It measures pool allocations and assigns no peak to a mode.


# %%
def simulate():
    """Check both backends and collect spectra, runtime and output size."""
    if not wp.is_cuda_available():
        raise RuntimeError("This example requires a native CUDA device")
    initial, process = chamber()
    grid = STEP_S * np.arange(STEPS + 1)
    device = wp.get_device("cuda:0")
    report = {
        "wavelength_nm": (WAVELENGTH_M * 1e9).tolist(),
        "cuda_device": device.name,
        "cuda_pool_high_water_bytes": None,
        "backends": {},
    }
    with TemporaryDirectory() as directory:
        for backend in ("NumPy", "CUDA"):
            folder = Path(directory)
            metrics, bulk, times = {}, None, None
            for mode in ("physics", "optical", "native"):
                warm = folder / f"{backend}-{mode}-warm"
                warm.mkdir()
                timed_run(initial, process, backend, mode, grid[:3], warm)
                runs = []
                for rep in range(3):
                    output = folder / f"{backend}-{mode}-{rep}"
                    output.mkdir()
                    elapsed, size = timed_run(
                        initial, process, backend, mode, grid, output
                    )
                    runs.append(elapsed)
                metrics[mode] = {
                    "seconds": float(np.median(runs)),
                    "file_bytes": size,
                }
                if mode == "optical":
                    chunks = list(read_optical_history(output / mode))
                    times = np.concatenate([part["time"] for _, part in chunks])
                    bulk = np.stack(
                        [
                            np.concatenate(
                                [part[channel][:, 0] for _, part in chunks]
                            )
                            for channel in CHANNELS
                        ],
                        axis=-1,
                    )
                    if chunks[0][0]["provenance"] != optical_provenance(
                        backend
                    ):
                        raise ValueError("Optical archive lost RH provenance")
            rows, offline, rh = sparse_reference(
                initial, process, backend, grid
            )
            np.testing.assert_allclose(times, grid, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(
                bulk[rows], offline, rtol=5e-10, atol=1e-12
            )
            ext = bulk[..., 0]
            offline_ext = offline[..., 0]
            if not np.all(ext[-1] > ext[0]):
                raise ValueError("Expected growth to increase extinction")
            optical_payload = BATCH_SIZE * (8 + WAVELENGTH_M.size * 9 * 8)
            if backend == "CUDA":
                optical_payload += BATCH_SIZE * WAVELENGTH_M.size * 4
            native_payload = len(grid) * (4 * SLOTS + 5) * 8
            report["backends"][backend] = {
                "time_s": times.tolist(),
                "extinction_m_inv": ext.tolist(),
                "snapshot_time_s": times[rows].tolist(),
                "snapshot_extinction_m_inv": offline_ext.tolist(),
                "snapshot_rh_fraction": rh.tolist(),
                "max_snapshot_error_m_inv": float(
                    np.max(np.abs(ext[rows] - offline_ext))
                ),
                "retained_observation_bytes": {
                    "optical": optical_payload,
                    "native": native_payload,
                },
                "metrics": metrics,
            }
    high_water = getattr(wp, "get_mempool_used_mem_high", None)
    if high_water is not None and wp.is_mempool_enabled(device):
        wp.synchronize_device(device)
        report["cuda_pool_high_water_bytes"] = int(high_water(device))
    numpy_extinction = report["backends"]["NumPy"]["extinction_m_inv"]
    cuda_extinction = report["backends"]["CUDA"]["extinction_m_inv"]
    np.testing.assert_allclose(
        cuda_extinction, numpy_extinction, rtol=5e-10, atol=1e-12
    )
    return report


# %% [markdown]
# ## Read the transient and compare storage
#
# Lines show 10 Hz extinction from the resident CUDA observer. Symbols show
# extinction recomputed after downloading sparse states from a separate
# CUDA run. The blue-to-red spectral dependence evolves as particles grow
# and water changes their refractive index. The right panel compares file
# sizes for the same 181 observation times. The printed timings include
# writing the files and apply to this chamber and device.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Plot extinction and output sizes and print measured workflow costs."""
    fig, axes = plotting.subplots(1, 2, size="wide")
    cuda = report["backends"]["CUDA"]
    colors = ("#00669f", "#247657", "#b53d29")
    markers = ("o", "s", "^")
    dense = np.asarray(cuda["extinction_m_inv"]) * 1e6
    reference = np.asarray(cuda["snapshot_extinction_m_inv"]) * 1e6
    for j, wavelength in enumerate(report["wavelength_nm"]):
        axes[0].plot(
            cuda["time_s"],
            dense[:, j],
            color=colors[j],
            label=f"{wavelength:.0f} nm",
        )
        axes[0].plot(
            cuda["snapshot_time_s"],
            reference[:, j],
            linestyle="none",
            marker=markers[j],
            color=colors[j],
            markerfacecolor="white",
        )
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Bulk extinction (Mm⁻¹)",
        title="Water uptake changes extinction",
    )
    axes[0].legend(title="Vacuum wavelength", fontsize=10, title_fontsize=10)
    axes[0].grid(alpha=0.25)

    x = np.arange(2)
    for i, (backend, case) in enumerate(report["backends"].items()):
        metrics = case["metrics"]
        sizes = [
            metrics[mode]["file_bytes"] / 1024 for mode in ("optical", "native")
        ]
        bars = axes[1].bar(
            x + (i - 0.5) * 0.32,
            sizes,
            width=0.32,
            label=backend,
            color=colors[i],
        )
        axes[1].bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
        print(f"{backend} median elapsed time (s)")
        for mode, metric in metrics.items():
            print(f"  {mode:<8} {metric['seconds']:.4f}")
        print(
            f"  retained observation payload (bytes) {case['retained_observation_bytes']}"
        )
        print(
            f"  largest sparse extinction difference (m^-1) {case['max_snapshot_error_m_inv']:.3e}"
        )
    pool_peak = report["cuda_pool_high_water_bytes"]
    if pool_peak is not None:
        print(f"CUDA process pool high-water across modes (bytes) {pool_peak}")
    axes[1].set_xticks(x, ("Bulk optics", "Native arrays"))
    axes[1].set(
        yscale="log",
        ylabel="Output file (KiB)",
        title="Same output cadence",
        ylim=(10, 3200),
    )
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    result = simulate()
    figure = plot_results(result)
    plt.show()

# %% [markdown]
# At 550 nm, simulated extinction rises from about 2,784 to 4,256 Mm⁻¹
# while RH stays near 83%. Both backends agree with the sparse checks.
# The optical file occupies about 40 KiB, compared with about 1,097 KiB
# for the uncompressed native arrays. The ratio depends on particle slot
# count and archive encoding. Calculating Mie efficiencies during the run
# adds work before saving. The chosen accommodation coefficient controls
# the timescale in this illustrative calculation.
#
# To choose a cadence for your experiment, rerun the same physical trajectory
# at a coarser optical sampling interval and compare the largest missed
# extinction change with your measurement uncertainty. Save full snapshots
# separately when your downstream analysis needs the underlying size or
# composition distributions.
