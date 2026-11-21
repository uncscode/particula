# %% [markdown]
# # Fit a growth trajectory from light alone
#
# Can bulk extinction and absorption measured during condensation recover
# two starting radii when the populations draw from the same vapor reservoir?
# This synthetic example fits both radii using CUDA. The measurements supply
# neither radius nor vapor concentration during growth. An independent radius
# equation generates the observations and provides those withheld histories
# for checking the fit.
#
# ## Prescribe the closed box
#
# Two monodisperse populations have known number densities of 300 and
# 700 cm⁻³. A vapor starts at 10 mg m⁻³ and condenses with unit accommodation
# onto spherical particles with a material density of 1000 kg m⁻³. The
# isothermal model holds diffusivity constant, sets equilibrium vapor at
# the surface to zero, and conserves gas plus particle mass. Each sphere has
# a prescribed absolute refractive index of 1.52 + 0.03i across the three
# vacuum wavelengths. The fit integrates 224 equal steps over 1.4 s.
#
# The calculation requires a CUDA device for both optimizers.
# [Condensation rate and transition correction](../../../../theory/dynamics/condensation-equations.md)
# · [Sphere optics conventions](../../../../guide/optics/index.md)

# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import warp as wp
from matplotlib import style as mplstyle
from particula import plotting
from particula.execution.inverse_condensation import terminal_radius
from particula.execution.inverse_optics import condensation_optics
from particula.optics import get_mie_efficiencies
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

DEVICE = "cuda:0"
TRUTH = np.array([0.845, 1.146])  # µm
NUMBER = np.array([[3e8, 7e8]])  # m⁻³, box by population
VAPOR = np.array([1e-5])  # kg m⁻³
WAVELENGTH = np.array([450e-9, 550e-9, 700e-9])
INDEX = 1.52 + 0.03j
DENSITY = 1000.0
DIFFUSION = 2e-5
TEMPERATURE = 298.15
VISCOSITY = (
    1.716e-5
    * (TEMPERATURE / 273.15) ** 1.5
    * (273.15 + 110.4)
    / (TEMPERATURE + 110.4)
)
MEAN_PATH = (
    2
    * VISCOSITY
    / 101325.0
    / np.sqrt(8 * 0.018 / (np.pi * 8.31446261815324 * TEMPERATURE))
)
DURATION = 1.4
STEPS = 224
REPLAY_REFINEMENT = 64
SEGMENT_STEPS = 16
OBSERVATION_STEPS = (32, 64, 112, 168, 224)
BOUNDS = [(0.78, 0.93), (1.05, 1.23)]  # µm
STARTS = ((0.80, 1.20), (0.91, 1.08), (0.90, 1.22))
TRANSPORT = dict(
    density=DENSITY, mean_path=MEAN_PATH, diffusivity=DIFFUSION, device=DEVICE
)


# %% [markdown]
# ## Generate optical data from a radius equation
#
# The reference solver evolves radius directly. It evaluates remaining gas
# from the volume added to both particle populations, so a larger uptake
# by one population leaves less vapor for the other. Radius is in µm
# internally and converted to metres before calling the CPU Mie solver.
# Each optical datum is a number-density-weighted sum of optical cross
# sections. The synthetic observations have no added noise. The objective
# scales each extinction and absorption residual by 1% of its observation.
# Scattering is available in the model output and is excluded from the loss.


# %%
def gas_inventory(initial, radius):
    """Return remaining vapor [kg m⁻³] from the two radii [µm]."""
    particle_change = (radius**3 - initial**3) @ NUMBER[0]
    return VAPOR[0] - (4 * np.pi / 3 * DENSITY * 1e-18) * particle_change


def solve_reference(initial, times):
    """Integrate the continuous radius equation and recover the gas history."""

    def rate(_time, radius):
        kn = MEAN_PATH / (radius * 1e-6)
        transition = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
        return (
            DIFFUSION
            * gas_inventory(initial, radius)
            * transition
            / (DENSITY * radius * 1e-12)
        )

    result = solve_ivp(
        rate,
        (0, DURATION),
        initial,
        method="DOP853",
        t_eval=times,
        atol=1e-13,
        rtol=1e-11,
    )
    if not result.success:
        raise ArithmeticError(result.message)
    return result.y.T, gas_inventory(initial, result.y.T)


def cpu_spectra(radius_um):
    """Return extinction, scattering and absorption [m⁻¹] at each time."""
    radius = radius_um[:, None, :] * 1e-6
    q = get_mie_efficiencies(radius, WAVELENGTH[None, :, None], INDEX)
    population_area = np.pi * radius**2 * NUMBER
    return np.stack(
        [
            np.sum(field * population_area, axis=-1)
            for field in (q.extinction, q.scattering, q.absorption)
        ],
        axis=-1,
    )


# %% [markdown]
# ## Fit the same loss with two derivative methods
#
# Both bounded optimizers use the same three starting points. The autodiff
# fit seeds the reverse pass with the derivative of the scalar loss with
# respect to each optical output. The seed uses the array layout (time,
# box, wavelength, channel). Its scattering entries are zero. The reverse
# pass replays the trajectory in 16-step segments from mass checkpoints.
# The finite-difference fit uses centered, three-point differences in the
# interior and a one-sided stencil near a bound. Each count below covers
# all starts and includes the forward pass inside each adjoint call.


# %%
def spectrum(initial, seed=None):
    """Compute the CUDA trajectory and optionally differentiate its loss."""
    optics, gradient = condensation_optics(
        initial[None, :],
        NUMBER,
        VAPOR,
        wavelength=WAVELENGTH,
        refractive_index=INDEX,
        observation_steps=OBSERVATION_STEPS,
        duration=DURATION,
        steps=STEPS,
        adjoint=seed,
        segment_steps=SEGMENT_STEPS,
        **TRANSPORT,
    )
    return optics, None if gradient is None else gradient[0]


def objective(optics, observations, inverse_sigma):
    """Form the half squared residual and its optical adjoint seed."""
    error = (optics - observations) * inverse_sigma
    return float(np.sum(error**2) / 2), error * inverse_sigma


def fit(method, observations, inverse_sigma):
    """Fit all starts and keep statistics for every trajectory evaluation."""
    counters = {"forward_calls": 0, "adjoint_calls": 0}

    def loss(initial):
        prediction, _ = spectrum(initial)
        counters["forward_calls"] += 1
        value, seed = objective(prediction, observations, inverse_sigma)
        if method == "autodiff":
            _, gradient = spectrum(initial, seed)
            counters["forward_calls"] += 1
            counters["adjoint_calls"] += 1
            return value, gradient
        return value

    wp.synchronize_device(DEVICE)
    start = perf_counter()
    solutions = [
        minimize(
            loss,
            initial,
            bounds=BOUNDS,
            method="L-BFGS-B",
            jac=True if method == "autodiff" else "3-point",
            options={"gtol": 1e-9, "ftol": 1e-13},
        )
        for initial in STARTS
    ]
    wp.synchronize_device(DEVICE)
    elapsed = perf_counter() - start
    if not all(result.success for result in solutions):
        raise ArithmeticError(f"{method} failed to converge from all starts")
    best = min(solutions, key=lambda result: result.fun)
    return dict(
        method=method,
        initial_um=best.x.tolist(),
        loss=float(best.fun),
        elapsed_s=elapsed,
        start_losses=[float(r.fun) for r in solutions],
        worst_initial_error_nm=1e3
        * max(float(np.max(np.abs(result.x - TRUTH))) for result in solutions),
        **counters,
    )


def state_payload_bytes(checkpointed):
    """Estimate mass and gas trajectory payload, excluding optics scratch."""
    mass = TRUTH.nbytes
    gas = VAPOR.nbytes
    if not checkpointed:
        return 3 * mass + gas  # initial, current and next masses, current gas
    checkpoints = (STEPS + SEGMENT_STEPS - 1) // SEGMENT_STEPS + 1
    # Retained checkpoints and replayed per-step masses and gas values,
    # with matching fp64 adjoints. Count the active segment and one carry.
    return 2 * (checkpoints * mass + SEGMENT_STEPS * (mass + gas) + mass)


# %% [markdown]
# ## Check the gradient and the withheld physical histories
#
# Test a centered directional difference at a starting point, before fitting.
# Replay each optimized radius pair with a separate radius observation to
# check the radius and vapor histories against the continuous solution at
# the true initial radii. Reconstruct vapor from the replayed radii by mass
# balance to check sensitivity to depletion. Also replay on a 64-times finer mesh with the
# inferred initial radii held fixed. This separates timestep error in vapor
# depletion from the error in the recovered radii. The checks allow 2 nm of
# initial-radius error, 1 nm of trajectory-radius error on the fit mesh, and
# 0.1 µg m⁻³ of vapor error on the refined mesh.
# Timings cover warm, synchronized fits. The mass and gas payload estimate
# is 1,280 B for checkpointed autodiff and 56 B for finite differences.
# It counts trajectory arrays and their adjoints. The measured CUDA pool
# high-water mark covers the combined workload, including optics scratch
# allocated through the pool.


# %%
def physical_replay(initial, *, refinement=1):
    """Observe radius and vapor on the fit mesh or a finer mesh."""
    radii = np.stack(
        [
            terminal_radius(
                initial[None, :],
                NUMBER,
                VAPOR,
                duration=DURATION * step / STEPS,
                steps=step * refinement,
                **TRANSPORT,
            )[0][0]
            for step in OBSERVATION_STEPS
        ]
    )
    return radii, gas_inventory(initial, radii)


def simulate():
    """Run the two CUDA fits and check independent physical trajectories."""
    wp.init()
    if not wp.get_device(DEVICE).is_cuda:
        raise RuntimeError("A CUDA device is required for this example")
    times = DURATION * np.asarray(OBSERVATION_STEPS) / STEPS
    radius, gas = solve_reference(TRUTH, times)
    observations = cpu_spectra(radius)[:, None, ...]
    inverse_sigma = 100 / observations
    inverse_sigma[..., 1] = 0

    x = np.array(STARTS[0])
    direction = np.array([0.6, -0.8])
    values, _ = spectrum(x)
    _, seed = objective(values, observations, inverse_sigma)
    _, gradient = spectrum(x, seed)
    for epsilon in (2e-6, 1e-6):
        high, _ = objective(
            spectrum(x + epsilon * direction)[0], observations, inverse_sigma
        )
        low, _ = objective(
            spectrum(x - epsilon * direction)[0], observations, inverse_sigma
        )
        np.testing.assert_allclose(
            gradient @ direction, (high - low) / (2 * epsilon), rtol=2e-6
        )

    fits = [
        fit(method, observations, inverse_sigma)
        for method in ("autodiff", "finite_difference")
    ]
    for result in fits:
        initial = np.array(result["initial_um"])
        inferred_radius, inferred_gas = physical_replay(initial)
        _, refined_gas = physical_replay(initial, refinement=REPLAY_REFINEMENT)
        result.update(
            radius_um=inferred_radius.tolist(),
            gas_mg_m3=(1e6 * inferred_gas).tolist(),
            refined_gas_error_ug_m3=float(
                1e9 * np.max(np.abs(refined_gas - gas))
            ),
            radius_error_nm=float(
                1e3 * np.max(np.abs(inferred_radius - radius))
            ),
            gas_error_ug_m3=float(1e9 * np.max(np.abs(inferred_gas - gas))),
            state_payload_bytes=state_payload_bytes(
                result["method"] == "autodiff"
            ),
        )
        if (
            result["worst_initial_error_nm"] > 2.0
            or result["radius_error_nm"] > 1.0
            or result["refined_gas_error_ug_m3"] > 0.1
            or result["refined_gas_error_ug_m3"] >= result["gas_error_ug_m3"]
        ):
            raise ArithmeticError(
                f"Held-out physical check failed for {result['method']}"
            )
    np.testing.assert_allclose(
        fits[0]["initial_um"], fits[1]["initial_um"], rtol=0, atol=2e-5
    )
    fitted_spectra, _ = spectrum(np.array(fits[0]["initial_um"]))
    device = wp.get_device(DEVICE)
    wp.synchronize_device(device)
    peak = (
        int(wp.get_mempool_used_mem_high(device))
        if device.is_mempool_supported
        else None
    )
    if peak is not None:
        print(f"Combined CUDA pool high-water mark {peak:,} bytes")
    return dict(
        cuda_pool_high_bytes=peak,
        time_s=times.tolist(),
        wavelength_nm=(1e9 * WAVELENGTH).tolist(),
        optical_observations=observations[:, 0].tolist(),
        optical_fit=fitted_spectra[:, 0].tolist(),
        reference_radius_um=radius.tolist(),
        reference_gas_mg_m3=(1e6 * gas).tolist(),
        true_initial_um=TRUTH.tolist(),
        fits=fits,
        device=wp.get_device(DEVICE).name,
    )


# %% [markdown]
# ## Read the fit alongside the physical checks
#
# The upper-left panel overlays the optical fit and its training targets at
# 550 nm, with lines for the fit and open markers for synthetic targets.
# The next two panels show whether the fitted radii reproduce the
# independent growth and vapor depletion. The final panel compares the total
# wall time across starts and labels the forward and adjoint evaluations.
# Agreement at all three wavelengths is tested through the full loss.


# %%
@mplstyle.context("particula.default")
def plot_results(report):
    """Compare the optical fit, withheld state and timed optimizations."""
    fig, axes = plotting.subplots(2, 2)
    times = report["time_s"]
    optical = np.asarray(report["optical_observations"])
    prediction = np.asarray(report["optical_fit"])
    adjoint, difference = report["fits"]
    for channel, marker, label, color in (
        (0, "o", "Extinction", "C0"),
        (2, "s", "Absorption", "C1"),
    ):
        axes[0, 0].plot(times, 1e3 * prediction[:, 1, channel], c=color)
        axes[0, 0].plot(
            times,
            1e3 * optical[:, 1, channel],
            linestyle="none",
            marker=marker,
            markerfacecolor="none",
            c=color,
            label=label,
        )
    axes[0, 0].set(ylabel="Coefficient (km⁻¹)", title="550 nm optical fit")
    axes[0, 0].legend()

    ref_radius = np.asarray(report["reference_radius_um"])
    fit_radius = np.asarray(adjoint["radius_um"])
    for i, color in enumerate(("C0", "C1")):
        axes[0, 1].plot(
            times, ref_radius[:, i], c=color, label=f"Mode {i + 1} ODE"
        )
        axes[0, 1].plot(
            times, fit_radius[:, i], "x", c=color, label=f"Mode {i + 1} fit"
        )
    axes[0, 1].set(ylabel="Radius (µm)", title="Withheld particle radii")
    axes[0, 1].legend(ncols=2, fontsize=8)

    axes[1, 0].plot(times, report["reference_gas_mg_m3"], label="Radius ODE")
    axes[1, 0].plot(times, adjoint["gas_mg_m3"], "x", label="Optical fit")
    axes[1, 0].set(ylabel="Vapor (mg m⁻³)", title="Withheld vapor history")
    axes[1, 0].legend()
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("Time (s)")

    bars = axes[1, 1].bar(
        ["Autodiff", "Finite diff."],
        [r["elapsed_s"] for r in report["fits"]],
        color=["C0", "C1"],
    )
    labels = [
        f"{r['forward_calls']} forward, {r['adjoint_calls']} adjoint"
        for r in (adjoint, difference)
    ]
    axes[1, 1].bar_label(bars, labels, padding=3, fontsize=8)
    axes[1, 1].margins(y=0.22)
    axes[1, 1].set(ylabel="Fit time (s)", title="Fit time over three starts")
    plotting.fit(fig)
    return fig


if __name__ == "__main__":
    result = simulate()
    figure = plot_results(result)
    plt.show()

# %% [markdown]
# Both derivative methods can find the initial radii when the optical
# response, number densities, refractive index, and growth law are prescribed.
# Across six fits the largest initial-radius error is 0.0412 nm. The
# withheld radius error reaches 0.0419 nm and the vapor error reaches
# 1.26 µg m⁻³ on the fitted mesh. Refinement reduces the latter to
# 0.043 µg m⁻³. An adjoint call
# adds checkpoint replay to its forward evaluation, so a lower count of
# forward calls need not imply a shorter fit. Broader size distributions or
# unknown composition could change the inference. A follow-up could withhold
# one wavelength and compare its prediction while fitting refractive index as
# a nuisance parameter.
