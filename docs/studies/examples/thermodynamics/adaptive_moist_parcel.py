# %% [markdown]
# # Which parcel solver reaches the same accuracy targets faster?
#
# In a rising parcel, small wet particles can return to equilibrium much
# faster than the parcel cools. Compare two dry size spectra under the same
# ascent. Explicit RK23 resolves the fastest droplet relaxation directly.
# Linearly implicit ROS3 solves for the coupled stage increments. Which takes
# less elapsed time on one GPU when both must reproduce the cloud response
# and its sensitivity to updraft?
#
# ## Hold the ascent and observation times fixed
#
# Begin with the 45, 70 and 110 nm dry **radii** from the
# [fixed-step parcel](moist_parcel.py). For the wider spectrum, move the
# outer classes to 30 and 150 nm. Both have the same counts and hygroscopicity.
# Changing both tails changes dry volume as well as the spread in sizes.
# In this closed parcel, 1 kg of dry air rises for 100 s at 1 m/s. Only the
# observation cadence is fixed. Each CUDA solver chooses its own internal
# steps. The controller tests the embedded pressure, class liquid volume,
# vapor and temperature errors against the same relative tolerance. The
# scale floors are 10⁻⁸ for normalized pressure and liquid/dry volume ratio,
# and 10⁻¹² kg/kg dry air for vapor. For RK23, a stability check also caps the step
# duration times the estimated relaxation rate at two. Both methods use the default
# minimum retry step of 10⁻⁹ s and limit of 100,000 total trials.
# SciPy's implicit Radau solver supplies an independent CPU trajectory
# sampled at the same times. A CUDA-capable GPU is required to run this example.

# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
from particula.dynamics.parcel import (
    CP,
    LATENT,
    RHO_W,
    RV,
    SIGMA,
    ParcelInitialState,
    ParcelRK23,
    ParcelROS3,
    run_parcel,
)

SPECTRA = {"Original": [45, 70, 110], "Wider": [30, 70, 150]}
SOLVERS = {"RK23": ParcelRK23, "ROS3": ParcelROS3}
TOLERANCES = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
REPETITIONS = 4
# Common observable limits for selecting the first qualifying tolerance.
LIMITS = {
    "max_radius_error_nm": 0.1,
    "absolute_peak_error_percentage_points": 1e-5,
    "max_activation_fraction_error": 0.0,
    "gradient_relative_error_adaptive_fd": 1e-4,
    "gradient_relative_error_radau": 1e-4,
    "water_drift_kg_kg": 1e-14,
    "energy_residual_j_kg": 1e-4,
}
OBSERVATION = 0.5  # s
STEPS = 200
UPDRAFT = 1.0  # m/s
DELTA = 1e-3  # m/s, for centered differences


def initial_parcel(dry_radius_nm):
    """Initialize the three weighted classes at subsaturated humidity."""
    return ParcelInitialState(
        dry_radius=1e-9 * np.asarray(dry_radius_nm),
        number=np.array([2.5e7, 5e7, 2.5e7]),
        kappa=0.5,
        temperature=288.15,
        pressure=90000.0,
        vapor=0.0118,
        dry_air_mass=1.0,
        height=0.0,
    )


# %% [markdown]
# ## Diagnose activation, water and pressure work
#
# A class counts as activated when its wet radius passes the maximum of the
# full κ-Köhler equilibrium curve at the current temperature. With wet-to-dry
# radius ratio $x$, let $u=x^3-1$ and let $a$ be the Kelvin length
# divided by dry radius. The critical point satisfies
# $3κx^4 = a u(u+κ)$. On the large-radius side the right side is larger.
# Weight classes by represented number to obtain the fraction above critical
# radius at each observation.
# [Kelvin correction](../../../theory/dynamics/condensation-equations.md#kelvin-effect-correction-factor)
#
# Reconstruct liquid water from the class radii to check it against vapor
# loss. At initial equilibrium the restoring rate for a small radius
# perturbation at fixed ambient conditions is proportional to the slope of
# the κ-Köhler curve divided by the wet radius. The vapor-diffusion and heat
# resistance is common to the two spectra at the shared initial temperature
# and pressure, so the ratio of these slopes gives a relative relaxation
# rate. Compare the parcel's sensible and latent energy changes with the
# pressure work integrated from its volume and pressure at the saved times.
# The same trapezoidal quadrature is applied to all three integrators.


# %%
def diagnostics(state, initial):
    """Return activation and conservation checks at observed states."""
    dry = initial.dry_radius
    x = state.wet_radius / dry
    u = x**3 - 1
    a = 2 * SIGMA / (RV * RHO_W * state.temperature[:, None] * dry)
    initial_slope = (
        (1 + state.supersaturation[0])
        / dry
        * (
            3 * initial.kappa * x[0] ** 2 / (u[0] * (u[0] + initial.kappa))
            - a[0] / x[0] ** 2
        )
    )
    activated = 3 * initial.kappa * x**4 <= a * u * (u + initial.kappa)
    fractions = np.average(activated, axis=1, weights=initial.number)

    wet = state.wet_radius
    water_volume = (4 * np.pi / 3) * (wet - dry) * (wet**2 + wet * dry + dry**2)
    liquid = RHO_W * (water_volume @ initial.number) / initial.dry_air_mass
    total_water = state.vapor + liquid
    specific_volume = state.volume / initial.dry_air_mass
    work = np.cumsum(
        0.5
        * (specific_volume[:-1] + specific_volume[1:])
        * np.diff(state.pressure)
    )
    energy = CP * (state.temperature[1:] - state.temperature[0]) - LATENT * (
        liquid[1:] - liquid[0]
    )
    peak = np.argmax(state.supersaturation)
    return {
        "max_restoring_slope_per_radius_m2": float(
            np.max(initial_slope / state.wet_radius[0])
        ),
        "peak_percent": float(100 * state.supersaturation[peak]),
        "peak_height_m": float(state.height[peak]),
        "fraction_at_50_s": float(fractions[round(50 / OBSERVATION)]),
        "fraction_at_100_s": float(fractions[-1]),
        "terminal_liquid_kg_kg": float(liquid[-1]),
        "water_drift_kg_kg": float(
            np.max(np.abs(total_water - total_water[0]))
        ),
        "energy_residual_j_kg": float(np.max(np.abs(energy - work))),
    }, fractions


# %% [markdown]
# ## Perturb the updraft and let the controller choose steps again
#
# For each tolerance, differentiate terminal liquid water at 100 s along
# the baseline accepted CUDA steps. Check it by running the ascent at 0.999 and
# 1.001 m/s. Each perturbed forward calculation can accept and reject different
# steps. Also compute a Radau finite difference, which uses an independent
# time integrator. All three calculations are compared after the same 100 s.


# %%
def integrate(initial, device, speed=UPDRAFT, adaptive=None, gradient=False):
    """Sample the parcel at 0.5 s intervals through 100 s."""
    return run_parcel(
        initial,
        device=device,
        updraft=speed,
        timestep=OBSERVATION,
        steps=STEPS,
        adaptive=adaptive,
        gradient=gradient,
    )


def finite_difference(initial, device, adaptive=None):
    """Reintegrate both perturbed updrafts and difference terminal liquid."""
    endpoints = [
        integrate(initial, device, UPDRAFT + shift, adaptive=adaptive).liquid[
            -1
        ]
        for shift in (-DELTA, DELTA)
    ]
    return float((endpoints[1] - endpoints[0]) / (2 * DELTA))


def measure_accuracy(
    initial, reference, activation, reference_peak, radau_fd, control
):
    """Compare all saved observations and the terminal updraft derivative."""
    result = integrate(initial, "cuda:0", adaptive=control, gradient=True)
    summary, active = diagnostics(result, initial)
    adaptive_fd = finite_difference(initial, "cuda:0", control)
    radii = 1e9 * np.max(
        np.abs(result.wet_radius - reference.wet_radius), axis=1
    )
    stats = result.statistics
    summary.update(
        rtol=control.rtol,
        accepted=stats.accepted_steps,
        rejected=stats.rejected_steps,
        evaluated=stats.evaluated_trials,
        min_step_s=stats.minimum_step,
        max_step_s=stats.maximum_step,
        max_activation_fraction_error=float(
            np.max(np.abs(active - activation))
        ),
        absolute_peak_error_percentage_points=abs(
            summary["peak_percent"] - reference_peak
        ),
        max_radius_error_nm=float(np.max(radii)),
        radius_error_nm=radii.tolist(),
        gradient=result.updraft_gradient,
        adaptive_fd=adaptive_fd,
        gradient_relative_error_radau=abs(
            result.updraft_gradient / radau_fd - 1
        ),
        gradient_relative_error_adaptive_fd=abs(
            result.updraft_gradient / adaptive_fd - 1
        ),
    )
    summary["qualified"] = all(
        summary[key] <= limit for key, limit in LIMITS.items()
    )
    return summary


# %% [markdown]
# ## Time the solvers at a common accuracy
#
# Compare the sampled supersaturation peak within 0.00001 percentage points,
# every class radius within 0.1 nm, and the number-weighted activated fraction
# at all 201 times exactly.
# Require the updraft derivative to agree within 0.01% with each finite
# difference. Check water drift within 10⁻¹⁴ kg/kg dry air and the first-law
# residual within 10⁻⁴ J/kg dry air. Test each decade from 10⁻³ through 10⁻⁵ to compare
# convergence. Continue tightening if all three miss any of the limits.
#
# Time the first qualifying tolerance for each method and spectrum. The
# preceding accuracy runs warm up both the forward and derivative paths.
# Synchronize CUDA before and after each complete call, including allocations,
# adaptive decisions and transfer of the observed trajectory back to the host.
# For each use case, alternate solver order across four repetitions. Save
# individual wall times and report their median. The derivative timing includes
# the forward integration and the reverse pass.


# %%
def benchmark(initial, runs):
    """Record synchronized whole-call elapsed times at qualifying tolerances."""
    controls = {
        method: SOLVERS[method](rtol=series["selected_rtol"])
        for method, series in runs.items()
    }
    order = list(controls)
    for gradient, mode in ((False, "forward"), (True, "with_gradient")):
        for repetition in range(REPETITIONS):
            for method in order if repetition % 2 == 0 else reversed(order):
                wp.synchronize_device("cuda:0")
                start = perf_counter()
                integrate(
                    initial,
                    "cuda:0",
                    adaptive=controls[method],
                    gradient=gradient,
                )
                wp.synchronize_device("cuda:0")
                runs[method]["elapsed_s"][mode].append(perf_counter() - start)


def simulate():
    """Qualify both methods against Radau, then time their production calls."""
    cases = {}
    for name, dry in SPECTRA.items():
        initial = initial_parcel(dry)
        reference = integrate(initial, "cpu")
        ref, activation = diagnostics(reference, initial)
        radau_fd = finite_difference(initial, "cpu")
        runs = {}
        for method, solver in SOLVERS.items():
            accuracy = []
            for tolerance in TOLERANCES:
                accuracy.append(
                    measure_accuracy(
                        initial,
                        reference,
                        activation,
                        ref["peak_percent"],
                        radau_fd,
                        solver(rtol=tolerance),
                    )
                )
                if tolerance <= 1e-5 and any(
                    run["qualified"] for run in accuracy
                ):
                    break
            selected = next((run for run in accuracy if run["qualified"]), None)
            if selected is None:
                raise RuntimeError(
                    f"{method} missed parcel accuracy targets for {name}"
                )
            runs[method] = {
                "accuracy": accuracy,
                "selected_rtol": selected["rtol"],
                "elapsed_s": {"forward": [], "with_gradient": []},
            }
        benchmark(initial, runs)
        cases[name] = {
            "dry_radius_nm": dry,
            "reference": ref,
            "reference_gradient_fd": radau_fd,
            "runs": runs,
        }
    return {
        "device": wp.get_device("cuda:0").name,
        "precision": "float64",
        "time_s": reference.time.tolist(),
        "observation_s": OBSERVATION,
        "updraft_m_s": UPDRAFT,
        "fd_half_width_m_s": DELTA,
        "limits": LIMITS,
        "repetitions": REPETITIONS,
        "cases": cases,
    }


# %% [markdown]
# ## Compare runtime and droplet growth
#
# Each curve follows the largest absolute radius error across the three
# classes for a timing-qualified run. The dotted horizontal line is the shared
# 0.1 nm limit. The right panel shows the median and the full range of the
# synchronized timings on a logarithmic scale. Compare methods within each
# spectrum separately for a forward trajectory and a trajectory with a
# terminal liquid-water derivative.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Show trajectory error and synchronized timings at qualifying accuracy."""
    fig, (error, runtime) = plotting.subplots(1, 2)
    colors = {"RK23": "#087f78", "ROS3": "#b44b24"}
    for i, (name, case) in enumerate(report["cases"].items()):
        for j, (method, run) in enumerate(case["runs"].items()):
            selected = next(
                r for r in run["accuracy"] if r["rtol"] == run["selected_rtol"]
            )
            error.plot(
                report["time_s"][1:],
                np.ma.masked_equal(selected["radius_error_nm"][1:], 0),
                color=colors[method],
                linestyle=("-", "--")[i],
                label=f"{method}, {name.lower()}",
            )
            for k, mode in enumerate(("forward", "with_gradient")):
                times = run["elapsed_s"][mode]
                median = np.median(times)
                runtime.errorbar(
                    2 * i + k + (-0.11, 0.11)[j],
                    median,
                    yerr=[[median - min(times)], [max(times) - median]],
                    color=colors[method],
                    marker=("o", "s")[j],
                    capsize=3,
                    label=method if i == 0 and k == 0 else None,
                )
    error.axhline(
        report["limits"]["max_radius_error_nm"], color="0.45", linestyle=":"
    )
    error.set(
        xlabel="Elapsed ascent time (s)",
        ylabel="Maximum wet-radius error (nm)",
        yscale="log",
    )
    plotting.legend(error, placement="inside", fontsize=8)
    runtime.set_xticks(
        [0, 1, 2, 3],
        ["Original\nfwd", "Original\n+ grad", "Wider\nfwd", "Wider\n+ grad"],
    )
    runtime.set(
        xlim=(-0.5, 3.5), ylabel="Wall time per ascent (s)", yscale="log"
    )
    plotting.legend(runtime, placement="inside")
    plotting.fit(fig)
    return fig


# %% [markdown]
# Read the peak and gradient errors alongside the timing ratios below.
# Evaluated trials include any speculative trials that the controller
# discarded. One ROS3 trial includes three coupled linear solves. The timed
# ratios compare work at common observable limits on this GPU.


# %%
def run_example():
    """Print numerical qualification and elapsed times for the two ascents."""
    report = simulate()
    for name, case in report["cases"].items():
        ref = case["reference"]
        print(
            f"{name} spectrum. Radau peak {ref['peak_percent']:.6f}% at {ref['peak_height_m']:.1f} m. "
            f"Activation at 50/100 s {ref['fraction_at_50_s']:.0%}/{ref['fraction_at_100_s']:.0%}. "
            f"Terminal updraft derivative {case['reference_gradient_fd']:.8g} (kg/kg)/(m/s)."
        )
        restoring_ratio = (
            ref["max_restoring_slope_per_radius_m2"]
            / report["cases"]["Original"]["reference"][
                "max_restoring_slope_per_radius_m2"
            ]
        )
        print(
            f"  Initial fastest-class restoring rate is {restoring_ratio:.2f}× the original spectrum's rate."
        )
        for method, series in case["runs"].items():
            for run in series["accuracy"]:
                print(
                    f"  {method} rtol={run['rtol']:g} "
                    f"({'timed' if run['rtol'] == series['selected_rtol'] else 'accuracy only'}), "
                    f"{run['accepted']:,} accepted, "
                    f"{run['rejected']:,} rejected, {run['evaluated']:,} evaluated. "
                    f"Max radius error {run['max_radius_error_nm']:.4g} nm, "
                    f"peak error {run['absolute_peak_error_percentage_points']:.3g} percentage points, "
                    f"max activation fraction error {run['max_activation_fraction_error']:g}.\n"
                    f"    Water drift {run['water_drift_kg_kg']:.2e} kg/kg, "
                    f"energy residual {run['energy_residual_j_kg']:.2e} J/kg dry air. "
                    f"AD derivative {run['gradient']:.8g}, re-adapted FD {run['adaptive_fd']:.8g}. "
                    f"Relative errors {run['gradient_relative_error_adaptive_fd']:.2e} against "
                    f"re-adapted FD and {run['gradient_relative_error_radau']:.2e} against Radau."
                )
        for mode in ("forward", "with_gradient"):
            rk, ros = [
                np.median(case["runs"][m]["elapsed_s"][mode])
                for m in ("RK23", "ROS3")
            ]
            print(
                f"  {mode}: RK23 {rk:.3f} s, ROS3 {ros:.3f} s. RK23/ROS3 time ratio {rk / ros:.2f}."
            )
    plot_results(report)
    plt.show()
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# ROS3 met the observable limits in less elapsed time for both spectra.
# At 10⁻⁴ relative tolerance both methods qualified on the original spectrum.
# On the wider spectrum, the maximum RK23 wet-radius error reached 0.214 nm
# at that tolerance, so it required 10⁻⁵. ROS3 met the same 0.1 nm target at
# 10⁻⁴, with a maximum error of 0.027 nm. Its 552 evaluated trials required
# 1,656 forward linear solves, compared with 35,933 evaluated RK23 trials.
# Both methods reproduced Radau's activated fraction at all 201 sample times.
# At 50 s the wider spectrum had 50% of particles activated and the original
# had 75%. Both reached 100% at 100 s.
#
# On the shared RTX 5060, ROS3 shortened the median forward time by a factor
# of 5.18 for the original spectrum and 6.42 for the wider spectrum.
# Including the terminal updraft derivative gave factors of 9.32 and 11.05.
# Other CUDA workloads ran during the benchmark. Wider-spectrum times with
# the gradient ranged from 24.81 to 48.57 s for RK23 and 2.81 to 3.93 s
# for ROS3. For these three-class ascents the ranges support choosing ROS3.
# To transfer that choice to a finer size distribution, repeat the comparison
# at the intended number of classes and output cadence.
