"""Compare AD and finite differences on independently generated chamber data."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp
from coupled_reference import chamber, reference
from particula.dynamics.chemistry.inference import (
    ExperimentLoss,
    fit_experiments,
)
from particula.dynamics.chemistry.observations import ObservationSeries
from particula.execution.inverse_chemistry import simulate_chamber


def experiment(number, dose, seed_number):
    """Generate an independent noisy time series for one chamber loading."""
    times = np.array([0.0, 0.4, 1.0, 2.0, 4.0])
    controls = dict(
        dose=[dose, 1e-9],
        rates=[0.5, 0.2],
        saturation=[2e-8],
        seed=[seed_number, 1e-7, 1.4, 0.04],
    )
    truth, _ = reference(times, **controls)
    sigma = truth.max(axis=0) * 0.002
    data = (
        truth + np.random.default_rng(number).normal(size=truth.shape) * sigma
    )
    data[1, 1] = np.nan
    return (
        controls,
        truth,
        sigma,
        ObservationSeries.create(times, data, np.diag(sigma**2)),
    )


def recover(device):
    """Fit rate and volatility and predict a withheld seed and dose."""
    runs = [
        experiment(31, 1e-7, 1e10),
        experiment(32, 0.6e-7, 2e10),
        experiment(33, 1.5e-7, 1.3e10),
    ]
    model = chamber()

    traffic = {}

    def simulate(x, run, gradient):
        controls, _, _, obs = run
        inputs = {**controls, "rates": [x[0], 0.2], "saturation": [x[1]]}
        trajectory = simulate_chamber(
            model,
            obs.times,
            **inputs,
            max_step=0.05,
            observations=obs,
            gradient=gradient,
            device=device,
        )

        for k, size in trajectory.transfer_bytes.items():
            traffic[k] = traffic.get(k, 0) + size
        return trajectory

    def evaluator(run):
        def evaluate(x, gradient):
            pred = simulate(x, run, gradient)
            g = pred.gradients
            return pred.loss, np.array(
                [g["rates"][0], g["saturation"][0]]
            ) if gradient else None

        return ExperimentLoss((0, 1), evaluate)

    experiments = [evaluator(run) for run in runs[:2]]
    bounds = [(0.2, 0.85), (0.0, 5e-8)]
    starts = [np.array([0.32, 4e-8]), np.array([0.7, 0.7e-8])]
    results = {}
    for label, gradient in (("AD", True), ("finite_difference", False)):
        traffic.clear()
        start = perf_counter()
        fit = fit_experiments(
            experiments, bounds, starts, gradient=gradient, max_iterations=45
        )
        runtime = perf_counter() - start
        calibration_traffic = traffic.copy()
        trajectory = simulate(fit.parameters, runs[2], False)
        truth, sigma = runs[2][1:3]
        error = np.sqrt(
            np.mean(((trajectory.predictions - truth) / sigma) ** 2, axis=0)
        )
        np.testing.assert_array_less(error, 2.0)
        np.testing.assert_array_less(abs(fit.parameters[0] - 0.5), 0.015)
        results[label] = dict(
            parameters=fit.parameters.tolist(),
            loss=fit.loss,
            evaluations=fit.evaluations,
            converged=fit.converged,
            wall_seconds=runtime,
            calibration_transfer_bytes=calibration_traffic,
            device_pool_high_water_bytes=wp.get_mempool_used_mem_high(device),
            holdout_rmse_in_noise_units=error.tolist(),
            holdout_predictions=trajectory.predictions.tolist(),
        )
    np.testing.assert_allclose(
        results["AD"]["parameters"],
        results["finite_difference"]["parameters"],
        rtol=1e-3,
        atol=1e-14,
    )
    masked = []
    for controls, truth, sigma, series in runs[:2]:
        values = series.values.copy()
        values[:, 1:] = np.nan
        masked.append(
            evaluator(
                (
                    controls,
                    truth,
                    sigma,
                    ObservationSeries.create(
                        series.times, values, np.diag(sigma**2)
                    ),
                )
            )
        )
    profiles = {}
    for channels, training in (
        ("precursor_only", masked),
        ("joint", experiments),
    ):
        losses = []
        for fixed in (0.0, 1e-8, 2e-8, 3e-8, 5e-8):
            fit = fit_experiments(
                training,
                [bounds[0], (fixed, fixed)],
                [np.array([0.5, fixed])],
                gradient=True,
            )
            losses.append(fit.loss)
        profiles[channels] = (np.array(losses) - min(losses)).tolist()
    np.testing.assert_allclose(profiles["precursor_only"], 0, atol=1e-9)
    np.testing.assert_array_less(2.0, max(profiles["joint"]))
    results["volatility_profile"] = {
        "saturation_mol_m3": [0.0, 1e-8, 2e-8, 3e-8, 5e-8],
        "delta_loss": profiles,
    }
    return results


if __name__ == "__main__":
    Path(sys.argv[2]).write_text(
        json.dumps(recover(sys.argv[1]), indent=2) + "\n"
    )
