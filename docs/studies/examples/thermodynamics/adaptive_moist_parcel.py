# %% [markdown]
# # How does a wider aerosol spectrum change adaptive parcel stepping?
#
# In a rising parcel, small wet particles can return to equilibrium much
# faster than the parcel cools. Compare two dry size spectra under the same
# ascent. How much work does an explicit adaptive solver do to track each
# spectrum, and does a tighter tolerance improve its predictions of
# supersaturation, activation and the response of liquid water to the updraft?
#
# ## Hold the ascent and observation times fixed
#
# Begin with the 45, 70 and 110 nm dry **radii** from the
# [fixed-step parcel](moist_parcel.py). For the wider spectrum, move the
# outer classes to 30 and 150 nm. Both have the same counts and hygroscopicity.
# Changing both tails changes dry volume as well as the spread in sizes.
# In this closed parcel, 1 kg of dry air rises for 100 s at 1 m/s. Only the
# observation cadence is fixed. Each CUDA RK23 run chooses its own internal
# steps. The controller tests the embedded pressure, class liquid volume,
# vapor and temperature errors against the same relative tolerance. The
# scale floors are 10⁻⁸ for normalized pressure and liquid/dry volume ratio,
# and 10⁻¹² kg/kg dry air for vapor. A stability check also caps the step
# duration times the estimated relaxation rate at two. Both runs use the default
# minimum retry step of 10⁻⁹ s and limit of 100,000 total trials.
# SciPy's implicit Radau solver supplies an independent CPU trajectory
# sampled at the same times. A CUDA-capable GPU is required to run this example.

# %%
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
    run_parcel,
)

SPECTRA = {"Original": [45, 70, 110], "Wider": [30, 70, 150]}
TOLERANCES = {"Loose": 1e-3, "Tight": 1e-5}
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
# The same trapezoidal quadrature is applied to Radau and RK23.


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
# For each tolerance, differentiate terminal liquid water at 100 s with
# the baseline accepted CUDA step sizes held fixed. Check the derivative by running the ascent again at 0.999 and
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


def simulate():
    """Compare accepted steps, trajectories, inventories and sensitivities."""
    cases = {}
    for name, dry in SPECTRA.items():
        initial = initial_parcel(dry)
        reference = integrate(initial, "cpu")
        reference_diagnostics, activation = diagnostics(reference, initial)
        radau_fd = finite_difference(initial, "cpu")
        runs = {}
        for label, tolerance in TOLERANCES.items():
            control = ParcelRK23(rtol=tolerance)
            result = integrate(
                initial, "cuda:0", adaptive=control, gradient=True
            )
            summary, active = diagnostics(result, initial)
            adaptive_fd = finite_difference(initial, "cuda:0", control)
            stats = result.statistics
            summary.update(
                rtol=tolerance,
                accepted=stats.accepted_steps,
                rejected=stats.rejected_steps,
                evaluated=stats.evaluated_trials,
                min_step_s=stats.minimum_step,
                max_step_s=stats.maximum_step,
                max_activation_fraction_error=float(
                    np.max(np.abs(active - activation))
                ),
                peak_error_percentage_points=summary["peak_percent"]
                - reference_diagnostics["peak_percent"],
                radius_error_nm=(
                    1e9
                    * np.max(
                        np.abs(result.wet_radius - reference.wet_radius), axis=1
                    )
                ).tolist(),
                gradient=result.updraft_gradient,
                adaptive_fd=adaptive_fd,
                gradient_relative_error_radau=float(
                    abs(result.updraft_gradient / radau_fd - 1)
                ),
                gradient_relative_error_adaptive_fd=float(
                    abs(result.updraft_gradient / adaptive_fd - 1)
                ),
            )
            runs[label] = summary
        cases[name] = {
            "dry_radius_nm": dry,
            "reference": reference_diagnostics,
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
        "cases": cases,
    }


# %% [markdown]
# ## Compare droplet-size error with the work of each run
#
# At each observation the left panel shows the largest absolute wet-radius
# difference from Radau across the three classes. The bars count accepted
# steps, rejected trials and discarded speculative trials. The full bar
# height shows all trial evaluations, including speculative work beyond a
# rejected trial or an observation boundary, for the nominal forward ascent.
# Reverse replay and the two perturbed runs add separate work. Compare bar heights and errors
# to see what a tighter tolerance buys.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Show radius error and solver decisions on one common time grid."""
    fig, (error, work) = plotting.subplots(1, 2)
    labels, accepted, rejected, discarded = [], [], [], []
    for (name, case), color in zip(
        report["cases"].items(), ("#087f78", "#b44b24"), strict=True
    ):
        for label, run in case["runs"].items():
            # The common initial condition has zero error.
            error.plot(
                report["time_s"][1:],
                np.ma.masked_equal(run["radius_error_nm"][1:], 0),
                color=color,
                linestyle="-" if label == "Tight" else "--",
                label=f"{name}, {label.lower()}",
            )
            labels.append(f"{name}\n{label.lower()}")
            accepted.append(run["accepted"])
            rejected.append(run["rejected"])
            discarded.append(
                run["evaluated"] - run["accepted"] - run["rejected"]
            )
    error.set(
        xlabel="Elapsed time (s)",
        ylabel="Maximum wet-radius error (nm)",
        yscale="log",
    )
    plotting.legend(error)
    positions = np.arange(len(labels))
    work.bar(positions, accepted, color="#087f78", label="Accepted")
    work.bar(
        positions,
        rejected,
        bottom=accepted,
        color="#c59034",
        hatch="//",
        label="Rejected",
    )
    work.bar(
        positions,
        discarded,
        bottom=np.add(accepted, rejected),
        color="#898a94",
        hatch="..",
        label="Discarded",
    )
    work.set_xticks(positions, labels)
    work.set_ylabel("Evaluated RK23 trials")
    plotting.legend(work)
    plotting.fit(fig)
    return fig


# %% [markdown]
# The printed supersaturation peak is the largest value on the shared 0.5 s
# observation grid. Use it and the activated fractions to assess cloud response.
# Compare the tape derivative with both finite differences in
# (kg liquid water/kg dry air)/(m/s). The budget residuals are maxima over
# the saved times. The accepted/rejected counts describe controller decisions.
# The evaluated count includes speculative trials discarded by the batch.


# %%
def run_example():
    """Print the measured accuracy, adaptive work and gradient checks."""
    report = simulate()
    baseline = report["cases"]["Original"]["reference"][
        "max_restoring_slope_per_radius_m2"
    ]
    for name, case in report["cases"].items():
        ref = case["reference"]
        relative_rate = ref["max_restoring_slope_per_radius_m2"] / baseline
        print(
            f"{name} — initial fastest-class restoring rate {relative_rate:.2f}× baseline. Radau peak {ref['peak_percent']:.6f}% at {ref['peak_height_m']:.1f} m, "
            f"activation at 50/100 s {ref['fraction_at_50_s']:.0%}/{ref['fraction_at_100_s']:.0%}, "
            f"energy residual {ref['energy_residual_j_kg']:.2e} J/kg dry air, "
            f"water drift {ref['water_drift_kg_kg']:.2e} kg/kg, "
            f"Radau slope {case['reference_gradient_fd']:.8g}."
        )
        for label, run in case["runs"].items():
            print(
                f"  {label} (rtol {run['rtol']:g}): {run['accepted']} accepted, "
                f"{run['rejected']} rejected, {run['evaluated']} evaluated; accepted steps {run['min_step_s']:.3g}"
                f"–{run['max_step_s']:.3g} s. Peak {run['peak_percent']:.6f}% "
                f"at {run['peak_height_m']:.1f} m, activation at 50/100 s "
                f"{run['fraction_at_50_s']:.0%}/{run['fraction_at_100_s']:.0%}.\n"
                f"    Water drift {run['water_drift_kg_kg']:.2e} kg/kg, "
                f"energy residual {run['energy_residual_j_kg']:.2e} J/kg dry air. "
                f"Tape {run['gradient']:.8g}, adaptive FD {run['adaptive_fd']:.8g}. "
                f"Relative slope errors: {run['gradient_relative_error_adaptive_fd']:.2e} "
                f"against adaptive FD and {run['gradient_relative_error_radau']:.2e} against Radau."
            )
    original = report["cases"]["Original"]["runs"]["Tight"]
    wide = report["cases"]["Wider"]["runs"]
    original_trials = original["evaluated"]
    loose_trials = wide["Loose"]["evaluated"]
    tight_trials = wide["Tight"]["evaluated"]
    print(
        f"At tight tolerance, widening the spectrum changes evaluated trials from "
        f"{original_trials:,} to {tight_trials:,} ({tight_trials / original_trials:.2f}×). "
        f"For the wider spectrum, tightening the tolerance changes the largest "
        f"class wet-radius error from {max(wide['Loose']['radius_error_nm']):.3g} to "
        f"{max(wide['Tight']['radius_error_nm']):.3g} nm, and trial count from "
        f"{loose_trials:,} to {tight_trials:,}."
    )
    plot_results(report)
    plt.show()
    return report


if __name__ == "__main__":
    run_example()

# %% [markdown]
# Widening the spectrum increases the initial restoring rate of the fastest
# class by a factor of 2.81. The wider spectrum has 50% of particles above their
# critical radius at 50 s, compared with 75% in the original spectrum. Both
# reach 100% by 100 s. RK23 gives the Radau activated fraction at every saved
# time in these two cases.
#
# The evaluated trial counts measure how often RK23 computed a trial while
# following the faster relaxation
# explicitly. To test whether implicit stepping pays off for a wider spectrum,
# give a linearly implicit Rosenbrock solver (ROS3)
# the same initial states and observation schedule. Compare its linear solves
# and runtime at matched peak, wet-radius and gradient errors. Trial counts
# alone cannot establish which method would be faster.
