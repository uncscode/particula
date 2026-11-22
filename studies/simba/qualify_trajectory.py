"""Qualify coupled trajectories and derivatives on CPU and CUDA."""

import json
import sys
from pathlib import Path

import numpy as np
import warp as wp
from coupled_reference import chamber, reference
from particula.dynamics.chemistry.observations import ObservationSeries
from particula.execution.inverse_chemistry import simulate_chamber


def qualify(device, vapor):
    """Compare each observable and gradient with a Radau reference."""
    times = np.array([0.0, 0.8, 2.0])
    controls = dict(
        dose=np.array([1e-7, vapor]),
        rates=np.array([0.5, 0.2]),
        seed=np.array([1e10, 1e-7, 1.4, 0.04]),
        saturation=np.array([2e-8]),
    )
    ref, hard = reference(times, **controls)
    scales = np.max(ref, axis=0)
    obs = ObservationSeries.create(times, 0.9 * ref, np.diag(scales**2))
    model = chamber()
    errors = []
    for dt in (0.1, 0.05, 0.025):
        output = simulate_chamber(
            model,
            times,
            **controls,
            device=device,
            max_step=dt,
            observations=obs,
            gradient=dt == 0.025,
        )
        errors.append(np.max(abs((output.predictions - ref) / scales), axis=0))
    np.testing.assert_array_less(errors[-1], 2e-5)
    np.testing.assert_array_less(12 * np.max(errors[-1]), np.max(errors[0]))
    # The reference integrates amounts per particle.
    adj = obs.loss(ref)[1]
    grad_errors = {}
    for group, values in controls.items():
        for j, val in enumerate(values):
            scale = val if val else 1e-7
            discrepancies = []
            for eps in (1e-3, 3e-4, 1e-4):
                left, right = values.copy(), values.copy()
                right[j] += eps * scale
                left[j] = max(0, val - eps * scale)
                high = reference(times, **{**controls, group: right})[0]
                low = reference(times, **{**controls, group: left})[0]
                fd = (high - low) / (right[j] - left[j])
                discrepancies.append(
                    float(
                        abs(output.gradients[group][j] - np.sum(adj * fd))
                        * scale
                    )
                )
            np.testing.assert_array_less(
                discrepancies, 2e-4, err_msg=f"{group}[{j}]"
            )
            grad_errors[f"{group}[{j}]"] = discrepancies
    # Perturb all continuous controls together and isolate each detector family.
    high, low = {}, {}
    for group, values in controls.items():
        direction = np.where(values > 0, values, 0.0) * np.linspace(
            0.3, -0.2, len(values)
        )
        high[group] = values + 1e-4 * direction
        low[group] = values - 1e-4 * direction
    slopes = (reference(times, **high)[0] - reference(times, **low)[0]) / 2e-4
    masked_errors = {}
    for label, channels in (
        ("vapor", [0, 1]),
        ("aerosol_mass", [2]),
        ("size", [3, 4, 5]),
    ):
        target = np.full_like(ref, np.nan)
        target[:, channels] = 0.9 * ref[:, channels]
        series = ObservationSeries.create(times, target, np.diag(scales**2))
        result = simulate_chamber(
            model,
            times,
            **controls,
            device=device,
            max_step=0.025,
            observations=series,
            gradient=True,
        )
        measured = sum(
            result.gradients[g] @ ((high[g] - low[g]) / 2e-4) for g in controls
        )
        expected = np.sum(series.loss(ref)[1] * slopes)
        np.testing.assert_allclose(measured, expected, rtol=0.002, atol=1e-5)
        masked_errors[label] = float(abs(measured - expected))
    return {
        "initial_vapor_mol_m3": vapor,
        "isolated_channel_mixed_derivative_errors": masked_errors,
        "steps_s": [0.1, 0.05, 0.025],
        "channel_peak_normalized_errors": np.asarray(errors).tolist(),
        "gradient_absolute_errors_per_parameter_scale": grad_errors,
        "reference": ref.tolist(),
        "simulation": output.predictions.tolist(),
        "hard_bin_counts_m3": hard.tolist(),
    }


if __name__ == "__main__":
    if not wp.is_cuda_available():
        raise RuntimeError("Actual CUDA required")
    results = {
        d: [qualify(d, v) for v in (0.0, 1e-7)] for d in ("cpu", "cuda:0")
    }
    results["warp_version"] = wp.__version__
    results["gpu"] = wp.get_device("cuda:0").name
    Path(sys.argv[1]).write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
