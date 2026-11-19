"""Recover initial mode radii across boxes sharing vapor within each box."""

from time import perf_counter

import numpy as np
from particula.execution.inverse_condensation import terminal_radius
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares, minimize
from scripts import assess_native_condensation as case
from scripts.assess_native_adaptive import VISCOSITY

MEAN_PATH = (
    2
    * VISCOSITY
    / case.PRESSURE
    / np.sqrt(
        8 * case.MOLAR_MASS / (np.pi * case.GAS_CONSTANT * case.TEMPERATURE)
    )
)
OPTIONS = dict(
    density=case.DENSITY,
    mean_path=MEAN_PATH,
    diffusivity=case.DIFFUSION,
    duration=1.4,
)
TRUTH = np.array([[0.845, 1.146], [0.866, 1.123]])
NUMBER = np.array([[3e8, 7e8], [6e8, 4e8]])
VAPOR = np.array([case.VAPOR, 0.8 * case.VAPOR])
CHANNELS = np.linspace(0.95, 1.46, 12)


def reference(initial, numbers, vapor):
    """Independent radius ODE with a common vapor budget (radii in µm)."""

    def rate(_t, radius):
        gas = vapor - case.MASS * np.dot(numbers, radius**3 - initial**3)
        kn = MEAN_PATH / (radius * 1e-6)
        correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
        return (
            case.DIFFUSION * gas * correction / (case.DENSITY * radius * 1e-12)
        )

    results = []
    for method in ("DOP853", "Radau"):
        solution = solve_ivp(
            rate, (0.0, 1.4), initial, method=method, rtol=1e-11, atol=1e-13
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        results.append(solution.y[:, -1])
    np.testing.assert_allclose(*results, rtol=1e-10)
    return results[0]


def observe(radius, width):
    """Count-normalized Gaussian efficiencies and their slopes per µm."""
    offset = CHANNELS[None, :, None] - radius[..., None, :]
    response = (
        np.exp(-0.5 * (offset / width) ** 2)
        * NUMBER[:, None, :]
        / NUMBER.sum(axis=1)[:, None, None]
    )
    return response.sum(axis=-1), response * offset / width**2


def assess_fit(
    device, method, width=0.04, steps=32, start=None, noise_seed=None
):
    """Fit four radii with fixed counts, chemistry and transfer coefficients."""
    final = np.array(
        [
            reference(r, n, g)
            for r, n, g in zip(TRUTH, NUMBER, VAPOR, strict=True)
        ]
    )
    observation, _ = observe(final, width)
    if noise_seed is not None:
        observation += np.random.default_rng(noise_seed).normal(
            0, 0.01, observation.shape
        )
    counters = dict(forward=0, backward=0)

    def run(x, seed=None):
        counters["forward"] += 1
        counters["backward"] += int(seed is not None)
        return terminal_radius(
            x.reshape(TRUTH.shape),
            NUMBER,
            VAPOR,
            steps=steps,
            device=device,
            adjoint=seed,
            **OPTIONS,
        )

    def residual(x):
        radii, _ = run(x)
        return ((observe(radii, width)[0] - observation) / 0.01).ravel()

    def objective(x):
        radii, _ = run(x)
        values, slopes = observe(radii, width)
        error = (values - observation) / 0.01
        loss = float(np.sum(error**2) / 2)
        if method == "ad":
            seed = np.einsum("bc,bcj->bj", error, slopes) / 0.01
            _, gradient = run(x, seed)
            return loss, gradient.ravel()
        return loss

    if start is None:
        start = np.tile([0.85, 1.15], 2)
    bounds = [(0.8, 0.999), (1.0, 1.2)] * 2
    before = perf_counter()
    if method == "global":

        def batch_cost(candidates):
            population = np.atleast_2d(candidates.T)
            count = len(population)
            radii, _ = terminal_radius(
                population.reshape(-1, 2),
                np.tile(NUMBER, (count, 1)),
                np.tile(VAPOR, count),
                steps=steps,
                device=device,
                **OPTIONS,
            )
            counters["forward"] += count
            r = radii.reshape(count, *TRUTH.shape)
            values, _ = observe(r, width)
            return np.sum(((values - observation) / 0.01) ** 2, axis=(1, 2)) / 2

        result = differential_evolution(
            batch_cost,
            bounds,
            seed=802,
            popsize=5,
            tol=1e-7,
            atol=1e-10,
            maxiter=280,
            updating="deferred",
            vectorized=True,
            polish=False,
        )
    elif method == "fd":
        result = least_squares(
            residual, start, bounds=np.transpose(bounds), gtol=1e-9
        )
    else:
        result = minimize(
            objective,
            start,
            jac=method == "ad",
            method="L-BFGS-B" if method == "ad" else "Powell",
            bounds=bounds,
            options={"ftol": 1e-12},
        )
    elapsed = perf_counter() - before
    fit = result.x.reshape(TRUTH.shape)
    singular = _sensitivity_spectrum(fit, width, steps, device)
    return dict(
        method=method,
        width_um=width,
        steps=steps,
        device=device,
        noise_seed=noise_seed,
        success=bool(result.success),
        elapsed_s=elapsed,
        **counters,
        fit_um=fit.tolist(),
        max_error_nm=float(np.max(abs(fit - TRUTH)) * 1000),
        singular_per_um=singular.tolist(),
        empty_band_fraction=(
            np.sum(NUMBER * ((fit >= 0.95) & (fit <= 1.05)), axis=1)
            / NUMBER.sum(axis=1)
        ).tolist(),
    )


def _sensitivity_spectrum(fit, width, steps, device):
    # Compute the local channel Jacobian independently of the optimizer.
    columns = []
    for idx in range(fit.size):
        direction = np.zeros_like(fit)
        direction.flat[idx] = 1e-5

        def signals(x):
            r, _ = terminal_radius(
                x, NUMBER, VAPOR, steps=steps, device=device, **OPTIONS
            )
            return observe(r, width)[0].ravel() / 0.01

        columns.append(
            (signals(fit + direction) - signals(fit - direction)) / 2e-5
        )
    return np.linalg.svd(np.column_stack(columns), compute_uv=False)
