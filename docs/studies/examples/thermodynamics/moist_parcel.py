# %% [markdown]
# # How does a faster updraft change cloud water at a fixed time?
#
# Follow one closed parcel of warm air through cloud base. As the parcel
# expands and cools, its vapor can reach saturation. Water condensing on the
# aerosol releases heat and removes vapor from the gas. The resulting
# supersaturation peak depends on how quickly the droplets can grow.
#
# We ask how much liquid water one kilogram of dry air carries at the end
# of the ascent, and how that quantity responds to a small change in updraft
# speed. Holding elapsed time fixed lets the faster parcel reach a higher
# altitude. A comparison at a fixed height would ask a different question.
#
# ## Set up the closed parcel
#
# Give each size class a dry radius and a represented particle count. The
# three classes represent a single soluble mode with a common κ of 0.5.
# Their counts and the dry-air mass follow the parcel as its physical volume
# changes. Initial relative humidity is about 97.65% at 288.15 K and 90 kPa.
# The solver equilibrates the initial wet radii with the vapor using
# κ-dependent water activity and the
# [Kelvin correction](../../../theory/dynamics/condensation-equations.md#kelvin-effect-correction-factor).
# Terminal liquid water includes this initial water. Every perturbed run
# starts from the same equilibrium. The calculation assumes continuum vapor
# diffusion and heat conduction around each droplet and constant latent heat.
# This example requires a CUDA device for the forward and reverse passes.

# %%
import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
from particula.dynamics.parcel import ParcelInitialState, run_parcel

UPDRAFT = 1.0  # m/s
TIMESTEP = 0.05  # s
STEPS = 2000


def initial_parcel():
    """Build one soluble mode in a parcel containing 1 kg of dry air."""
    return ParcelInitialState(
        height=0.0,
        pressure=90000.0,
        temperature=288.15,
        dry_air_mass=1.0,
        vapor=0.0118,
        dry_radius=np.array([45, 70, 110]) * 1e-9,
        number=np.array([2.5e7, 5.0e7, 2.5e7]),
        kappa=0.5,
    )


# %% [markdown]
# ## Differentiate the ascent and repeat it at nearby speeds
#
# Warp records the CUDA calculation and propagates the sensitivity of the
# final liquid-water mixing ratio back to the prescribed speed. Check that
# derivative independently by increasing and decreasing the speed by the same
# amount and dividing the difference in terminal liquid water by the
# difference in speed. Three perturbation sizes show whether that estimate
# stabilizes. An independent CPU integration supplies a forward comparison at
# the same output times.


# %%
def simulate():
    """Compare a CUDA ascent, a CPU trajectory and centered perturbations."""
    initial = initial_parcel()
    schedule = dict(timestep=TIMESTEP, steps=STEPS)
    cuda = run_parcel(
        initial, updraft=UPDRAFT, device="cuda:0", gradient=True, **schedule
    )
    cpu = run_parcel(initial, updraft=UPDRAFT, device="cpu", **schedule)
    perturbations = [1e-2, 1e-3, 1e-4]  # m/s
    differences = []
    liquid_increments = []
    for delta in perturbations:
        endpoints = [
            run_parcel(
                initial, updraft=UPDRAFT + shift, device="cuda:0", **schedule
            ).liquid[-1]
            for shift in (-delta, delta)
        ]
        differences.append((endpoints[1] - endpoints[0]) / (2 * delta))
        liquid_increments.append(endpoints[1] - cuda.liquid[-1])

    peak = int(np.argmax(cuda.supersaturation))
    samples = np.unique(np.append(np.linspace(0, STEPS, 151, dtype=int), peak))
    first_saturated = np.flatnonzero(cuda.supersaturation >= 0)
    cloud_base = None
    if first_saturated.size:
        i = first_saturated[0]
        cloud_base = float(
            np.interp(
                0.0,
                cuda.supersaturation[i - 1 : i + 1],
                cuda.height[i - 1 : i + 1],
            )
        )
    fields = (
        "height",
        "pressure",
        "temperature",
        "volume",
        "vapor",
        "liquid",
        "wet_radius",
        "supersaturation",
    )
    errors = {
        name: float(np.max(np.abs(getattr(cuda, name) - getattr(cpu, name))))
        for name in fields
    }
    total_water = cuda.vapor + cuda.liquid
    return {
        "device": wp.get_device("cuda:0").name,
        "precision": "float64",
        "max_cpu_cuda_error_si": errors,
        "max_total_water_drift_kg_kg": float(
            np.max(np.abs(total_water - total_water[0]))
        ),
        "cloud_base_height_m": cloud_base,
        "updraft_m_s": UPDRAFT,
        "timestep_s": TIMESTEP,
        "steps": STEPS,
        "duration_s": float(cuda.time[-1]),
        "dry_radius_nm": (initial.dry_radius * 1e9).tolist(),
        "height_m": cuda.height[samples].tolist(),
        "temperature_k": cuda.temperature[samples].tolist(),
        "volume_m3": cuda.volume[samples].tolist(),
        "supersaturation_percent": (
            100 * cuda.supersaturation[samples]
        ).tolist(),
        "wet_radius_um": (1e6 * cuda.wet_radius[samples]).tolist(),
        "peak_height_m": float(cuda.height[peak]),
        "peak_supersaturation_percent": float(100 * cuda.supersaturation[peak]),
        "initial_liquid_kg_kg": float(cuda.liquid[0]),
        "positive_liquid_increment_kg_kg": liquid_increments,
        "terminal_liquid_kg_kg": float(cuda.liquid[-1]),
        "updraft_gradient": float(cuda.updraft_gradient),
        "perturbation_m_s": perturbations,
        "finite_difference": differences,
    }


# %% [markdown]
# ## Read cloud base and the droplet response together
#
# The dashed zero line marks saturation over liquid water. Follow the wet
# radii through that crossing and look for continued growth after the
# supersaturation maximum marked by the dot. The upper panels show the change in the parcel's
# temperature and physical volume over the same ascent.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Show parcel expansion, cooling, supersaturation and wet growth."""
    height = report["height_m"]
    fig, axes = plotting.subplots(2, 2)
    temperature, volume, saturation, radius = axes.flat
    temperature.plot(height, report["temperature_k"])
    temperature.set_ylabel("Temperature (K)")
    volume.plot(height, report["volume_m3"])
    volume.set_ylabel("Parcel volume (m³)")
    saturation.axhline(0, color="0.45", linestyle="--", linewidth=0.9)
    saturation.plot(height, report["supersaturation_percent"])
    saturation.plot(
        report["peak_height_m"], report["peak_supersaturation_percent"], "o"
    )
    saturation.set_ylabel("Supersaturation (%)")
    wet = np.asarray(report["wet_radius_um"])
    for i, dry_nm in enumerate(report["dry_radius_nm"]):
        radius.plot(
            height,
            wet[:, i],
            linestyle=("-", "--", ":", "-.")[i % 4],
            label=f"{dry_nm:g} nm dry",
        )
    radius.set_yscale("log")
    radius.set_ylabel("Wet radius (µm)")
    plotting.legend(radius)
    saturation.set_xlabel("Height (m)")
    radius.set_xlabel("Height (m)")
    plotting.fit(fig)
    return fig


# %% [markdown]
# The printed derivative is in kg liquid water per kg dry air per (m/s).
# Multiply it by a small change in updraft speed to estimate the corresponding
# change in terminal water. The CPU comparison reports maximum differences
# over the whole trajectory.


# %%
def run_example():
    """Print the signed sensitivity beside its numerical checks."""
    report = simulate()
    print(
        f"After {report['duration_s']:g} s at {UPDRAFT:g} m/s, "
        f"liquid water = {report['terminal_liquid_kg_kg']:.6g} kg/kg dry air."
    )
    print(
        f"Peak supersaturation = {report['peak_supersaturation_percent']:.5g}% "
        f"at {report['peak_height_m']:.4g} m."
    )
    if report["cloud_base_height_m"] is not None:
        print(
            f"Liquid-water saturation reached at {report['cloud_base_height_m']:.3g} m."
        )
    print(
        f"Maximum CPU/CUDA discrepancies on {report['device']} (float64): "
        f"{report['max_cpu_cuda_error_si']['liquid']:.3g} kg/kg liquid water, "
        f"{100 * report['max_cpu_cuda_error_si']['supersaturation']:.3g} "
        "percentage points supersaturation."
    )
    print(f"CUDA derivative = {report['updraft_gradient']:+.7g} (kg/kg)/(m/s)")
    for delta, estimate in zip(
        report["perturbation_m_s"], report["finite_difference"], strict=True
    ):
        print(f"Centered at ±{delta:g} m/s: {estimate:+.7g} (kg/kg)/(m/s)")
    delta = report["perturbation_m_s"][0]
    sensitivity = report["updraft_gradient"]
    measured = report["positive_liquid_increment_kg_kg"][0]
    print(
        f"For +{delta:g} m/s the derivative predicts {1e6 * sensitivity * delta:+.5g} "
        f"mg liquid/kg dry air, and the rerun gives {1e6 * measured:+.5g} mg/kg."
    )
    plot_results(report)
    plt.show()
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# The parcel first reaches saturation near 45 m. Supersaturation peaks near
# 0.53% at 62 m, and the droplets keep growing above that height. After
# 100 s the parcel carries about 0.106 g liquid water per kg dry air.
# Increasing the updraft from 1.00 to 1.01 m/s adds about 2.12 mg/kg.
# The local derivative predicts the same increment to three significant
# figures. This sensitivity applies to this initial aerosol and elapsed time.
# To examine how aerosol abundance changes the competition for water, vary
# the represented particle counts while keeping
# the dry sizes, initial humidity and ascent unchanged. Track both the
# supersaturation peak and the resulting sensitivity.
