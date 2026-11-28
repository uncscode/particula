# %% [markdown]
# # How many particles remain eligible to activate as a chamber is diluted?
#
# A clean-air inlet removes particles at a common rate across all sizes. At
# fixed composition and temperature the minimum dry diameter for activation
# stays fixed, yet the number above that diameter falls. Record that number
# live at two supersaturations, then ask about a third setting after the run.
# An optical observer calculates how the same removal changes light extinction.
#
# This example runs the same resolved population on NumPy and native CUDA.
# The equilibrium criterion assumes complete wetting, spherical particles,
# constant hygroscopicity and surface tension. The selected supersaturations
# describe a hypothetical CCN counter. For the dry-diameter equation and its
# limits, see the [κ-Köhler derivation](../../../../theory/thermodynamics/kappa-kohler-activation.md).
#
# ## A chamber with five dry sizes
#
# Assign each dry size a known abundance in 0.5 m³ of sampled air. All five
# sizes carry liquid water equal to their dry volume. This water changes the
# optical radius but contributes no dry material to the activation threshold.
# A size-independent dilution rate of ln(2)/10 s⁻¹ halves each abundance
# after ten seconds. Hold per-particle masses fixed to isolate removal.

# %%
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import particula as par
import warp as wp
from particula import plotting
from particula.execution.analysis_io import load_analysis, save_analysis
from particula.execution.ccn_recording import CCNObserver, read_ccn_history
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
from particula.optics import OpticalConfig
from particula.particles.properties.ccn import CCNConfig, evaluate_ccn_history

DRY_NM = np.array([48, 75, 90, 130, 200])
NUMBER_CM3 = np.array([100, 200, 100, 60, 40])
LIVE_RATIOS = [1.001, 1.002]
OFFLINE_RATIOS = [*LIVE_RATIOS, 1.003]
SNAPSHOT_TIMES = [0.0, 5.0, 10.0]


def chamber():
    """Construct wet seeds with externally specified equivalent dry sizes."""
    volume = 0.5
    dry_volume = np.pi / 6 * (DRY_NM * 1e-9) ** 3
    density = np.array([997.0, 1500.0])  # water, dry solute [kg/m³]
    particles = par.particles.create_resolved_particle_data(
        masses=dry_volume[:, None] * density,
        density=density,
        population=NUMBER_CM3 * 1e6 * volume,
        volume=volume,
    )
    aerosol = par.Aerosol(
        particles=particles,
        gas=par.gas.GasData(["tracer"], np.array([0.028]), [[0.0]], [False]),
        environment=par.gas.EnvironmentData([298.15], [101325.0], [[0.0]]),
    )
    process = par.dynamics.Dilution(
        par.dynamics.DilutionStrategy(np.log(2.0) / 10.0)
    )
    return aerosol, process


def counter(ratios):
    """Set hypothetical counter saturation ratios and the dry-seed material."""
    return CCNConfig(
        saturation_ratios=ratios,
        kappa=0.3,
        kelvin_diameter=2.1e-9,
        water_index=0,
    )


# %% [markdown]
# ## Choose what to record while the particles are in memory
#
# Advance in steps of 0.5 s. Observe equilibrium CCN every step and
# extinction at 550 nm every 2.5 s. The optical calculation uses a
# homogeneous effective refractive index of 1.45. Save particle masses, represented counts and box volume at 0, 5 and 10 s
# for later evaluation. Each fresh simulation follows the same
# physical steps and snapshot schedule.
#
# To quantify the extra work, run a control with optics alone and a second
# run with optics and CCN. Synchronize at both ends of each measured interval
# on CUDA. Measure advancing, observing and taking sparse snapshots together.
# Exclude constructing the observers and serializing the CCN history.


# %%
def run_chamber(backend, directory, *, record_ccn):
    """Run on the requested backend, returning spectra and sparse states."""
    initial, process = chamber()
    cpu = backend == "NumPy"
    owner = (
        nullcontext(CPUSimulation(initial, process, time_step=0.5))
        if cpu
        else process.prepare(initial, time_step=0.5, device="cuda:0")
    )
    optical_path = directory / "optics.sqlite"
    particles = Recorder(("particle_masses", "particle_population", "volume"))
    with owner as simulation:
        optics = OpticalObserver(
            simulation,
            OpticalHistory(
                optical_path,
                OpticalConfig([550e-9], particle_index=1.45),
                variables=("extinction",),
            ),
            coefficient_capacity=32,
        )
        ccn = (
            CCNObserver(simulation, counter(LIVE_RATIOS))
            if record_ccn
            else None
        )
        sampling = dict(
            output_times=SNAPSHOT_TIMES,
            optics=optics,
            optical_output_times=[0.0, 2.5, 5.0, 7.5, 10.0],
            ccn=ccn,
            ccn_initial=record_ccn,
        )
        if not cpu:
            sparse = PreparedObserver(
                simulation, particles, download="snapshot"
            )
            simulation.synchronize()
        start = perf_counter()
        if cpu:
            run_until(simulation, 10.0, recorder=particles, **sampling)
        else:
            run_prepared(simulation, 10.0, observer=sparse, **sampling)
        elapsed = perf_counter() - start
        if ccn is None:
            return elapsed, None
        ccn_path = directory / "live.ccn"
        ccn.save(ccn_path)
    history = read_ccn_history(ccn_path).arrays
    chunks = [row for _, row in read_optical_history(optical_path)]
    offline_path = directory / "particles.analysis"
    save_analysis(
        particles,
        offline_path,
        configuration={
            "kappa": 0.3,
            "water_index": 0,
            "kelvin_diameter_m": 2.1e-9,
        },
        numerical_settings={
            "time_step_s": 0.5,
            "snapshot_times_s": SNAPSHOT_TIMES,
        },
        source_identity={
            "example": "equilibrium-ccn-spectrum",
            "backend": backend,
        },
    )
    offline = evaluate_ccn_history(
        load_analysis(offline_path), counter(OFFLINE_RATIOS)
    )
    return elapsed, {
        "time_s": history["time"].tolist(),
        "ccn_m3": history["equilibrium_ccn_number_concentration"][
            :, 0
        ].tolist(),
        "threshold_m": history["critical_dry_diameter"][0, 0].tolist(),
        "offline_m3": offline.equilibrium_ccn_number_concentration[
            :, 0
        ].tolist(),
        "particle_file_bytes": offline_path.stat().st_size,
        "optical_time_s": np.concatenate([r["time"] for r in chunks]).tolist(),
        "extinction_m_inv": np.concatenate(
            [r["extinction"][:, 0, 0] for r in chunks]
        ).tolist(),
        "ccn_file_bytes": ccn_path.stat().st_size,
    }


# %% [markdown]
# ## Check the number concentrations before interpreting the histories
#
# The 200 nm class alone is eligible at 0.1%, the 130 and 200 nm classes
# at 0.2%, and the 90, 130 and 200 nm classes at 0.3%. Their initial
# concentrations are therefore 40, 100 and 200 cm⁻³. Compare every live
# observation with exponential dilution of those known counts. At the three
# saved times compare the CPU post-processing with the independent
# dilution prediction and with the two live channels. Both backends should
# also recover the same extinction for the same population.
#
# The timed pairs follow one untimed warm-up per mode. Three repetitions
# give a median duration for this small chamber on the machine running it.


# %%
def check_run(result):
    """Compare live, offline and hand-counted CCN in physical units."""
    time = np.array(result["time_s"])
    np.testing.assert_array_equal(time, np.arange(21) / 2)
    diluted = np.exp2(-time[:, None] / 10)
    np.testing.assert_allclose(
        result["ccn_m3"], diluted * [4e7, 1e8], rtol=1e-11, atol=0
    )
    np.testing.assert_allclose(
        result["offline_m3"],
        diluted[[0, 10, 20]] * [4e7, 1e8, 2e8],
        rtol=1e-11,
        atol=0,
    )
    np.testing.assert_allclose(
        np.array(result["ccn_m3"])[[0, 10, 20]],
        np.array(result["offline_m3"])[:, :2],
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_array_equal(
        result["optical_time_s"], [0.0, 2.5, 5.0, 7.5, 10.0]
    )
    extinction = np.array(result["extinction_m_inv"])
    np.testing.assert_allclose(
        extinction,
        extinction[0] * np.exp2(-np.arange(5) / 4),
        rtol=1e-10,
        atol=0,
    )


def simulate():
    """Run and validate both execution backends on a native CUDA system."""
    if not wp.is_cuda_available():
        raise RuntimeError("This comparison requires a native CUDA device.")
    report = {"cuda_device": wp.get_device("cuda:0").name, "backends": {}}
    for backend in ("NumPy", "CUDA"):
        durations = {False: [], True: []}
        result = None
        for rep in range(4):
            for enabled in durations:
                with TemporaryDirectory() as tmp:
                    elapsed, values = run_chamber(
                        backend, Path(tmp), record_ccn=enabled
                    )
                if enabled:
                    check_run(values)
                    result = values
                if rep:
                    durations[enabled].append(elapsed)
        result["optics_only_s"] = float(np.median(durations[False]))
        result["optics_and_ccn_s"] = float(np.median(durations[True]))
        report["backends"][backend] = result
    for name in ("ccn_m3", "offline_m3", "extinction_m_inv", "threshold_m"):
        np.testing.assert_allclose(
            report["backends"]["NumPy"][name],
            report["backends"]["CUDA"][name],
            rtol=1e-10,
            atol=0,
        )
    return report


# %% [markdown]
# ## Compare the counter settings and the optical response
#
# The left panel shows the two live CUDA number concentrations. Circles
# mark the same settings evaluated later from particle snapshots. Triangles
# show a third counter setting calculated only after the run. The right
# panel gives extinction at the independent optical sample times. The
# printed durations and archive sizes describe this run on the reported
# device and include filesystem writes made by the optical observer.


# %%
@plt.style.context("particula.default")
def plot_results(report):
    """Plot live and offline number concentrations with the optical record."""
    result = report["backends"]["CUDA"]
    fig, (number, optics) = plotting.subplots(1, 2, size="wide")
    live = np.array(result["ccn_m3"]) / 1e6
    offline = np.array(result["offline_m3"]) / 1e6
    for j, (color, style) in enumerate([("#296499", "-"), ("#c14d24", "--")]):
        number.plot(
            result["time_s"],
            live[:, j],
            style,
            color=color,
            label=f"{[0.1, 0.2][j]}% live",
        )
        number.plot(
            SNAPSHOT_TIMES,
            offline[:, j],
            "o",
            color=color,
            markerfacecolor="white",
            label="Snapshot" if j == 0 else None,
        )
    number.plot(
        SNAPSHOT_TIMES,
        offline[:, 2],
        "^:",
        color="#337447",
        label="0.3% offline",
    )
    number.set(
        xlabel="Time (s)",
        ylabel="Equilibrium CCN (cm⁻³)",
        title="Counter supersaturation",
    )
    number.set_ylim(bottom=0)
    number.legend()
    optics.plot(
        result["optical_time_s"],
        1e6 * np.array(result["extinction_m_inv"]),
        "D-",
        color="#80558b",
    )
    optics.set(
        xlabel="Time (s)", ylabel="Extinction (Mm⁻¹)", title="Light at 550 nm"
    )
    optics.set_ylim(bottom=0)
    plotting.fit(fig)
    print("Native CUDA device:", report["cuda_device"])
    for backend, row in report["backends"].items():
        print(
            f"{backend}: optics {row['optics_only_s'] * 1000:.1f} ms; "
            f"optics + CCN {row['optics_and_ccn_s'] * 1000:.1f} ms; "
            f"CCN archive {row['ccn_file_bytes']} bytes"
        )
    return fig


if __name__ == "__main__":
    plot_results(simulate())
    plt.show()

# %% [markdown]
# After ten seconds the 0.2% channel drops from 100 to 50 cm⁻³ while
# its cutoff remains 104.54 nm. Sampling at 0.3% after the run gives
# 100 cm⁻³ at ten seconds because the 90 nm dry seeds also qualify.
# Extinction halves over that interval as dilution leaves the optical
# cross section of each surviving particle unchanged.
#
# Choose live output for densely sampled counter settings. Keep sparse
# states if the settings, material κ values or dry composition assumptions
# might need revisiting. A next experiment could let water condense without
# diluting the chamber. Wet growth can then alter extinction while the
# dry-size count remains fixed. Predicting counts at the outlet of a real
# instrument also requires its growth time and optical detection threshold.
