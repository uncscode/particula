"""Fit and challenge empirical N2O5 uptake laws against Gaston Table 1."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
CLASSES = ["PEG", "azelaic acid", "succinic acid", "citric acid"]
# Frozen before optimization: withhold the closest interior composition
# to one-third organic in each series at 50 % RH.
RESERVED = [0.340, 0.350, 0.340, 0.360]


@wp.kernel
def weighted_loss(
    design: wp.array2d(dtype=wp.float64),
    observed: wp.array(dtype=wp.float64),
    halfwidth: wp.array(dtype=wp.float64),
    parameters: wp.array(dtype=wp.float64),
    loss: wp.array(dtype=wp.float64),
):
    """Accumulate squared normalized uptake residuals in float64."""
    i = wp.tid()
    log_ratio = wp.float64(0.0)
    for j in range(5):
        log_ratio += design[i, j] * parameters[j]
    error = (wp.exp(log_ratio) - observed[i]) / halfwidth[i]
    wp.atomic_add(loss, 0, error * error)


def fit_law(design, y, width):
    """Optimize continuous coefficients and verify a nonoptimal AD gradient."""
    device = "cpu"
    padded = np.pad(design, ((0, 0), (0, 5 - design.shape[1])))
    matrix, target, weights = [
        wp.array(a, dtype=wp.float64, device=device) for a in (padded, y, width)
    ]

    def objective(p):
        param = wp.array(
            np.pad(p, (0, 5 - len(p))),
            dtype=wp.float64,
            device=device,
            requires_grad=True,
        )
        loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch(
                weighted_loss,
                dim=len(y),
                inputs=[matrix, target, weights, param],
                outputs=[loss],
                device=device,
            )
        tape.backward(loss=loss)
        return loss.numpy()[0], param.grad.numpy()[: len(p)].copy()

    start = np.ones(design.shape[1])
    direction = np.arange(1, len(start) + 1, dtype=float)
    direction /= np.linalg.norm(direction)
    value, grad = objective(start)
    np.testing.assert_allclose(
        value, np.sum(((np.exp(design @ start) - y) / width) ** 2)
    )
    expected = np.dot(grad, direction)
    differences = []
    for h in (1e-4, 1e-5, 1e-6):
        numerical = (
            objective(start + h * direction)[0]
            - objective(start - h * direction)[0]
        ) / (2 * h)
        differences.append(abs(numerical - expected) / max(1, abs(expected)))
    if max(differences) > 1e-6:
        raise ArithmeticError(f"Gradient check failed: {differences}")
    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 8.0)] * len(start),
        options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 500},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x, differences, result.fun


def main():
    """Fit nested models and score all held-out laboratory preparations."""
    wp.config.kernel_cache_dir = str(
        Path(tempfile.gettempdir()) / "sabre-lab-warp"
    )
    table = pd.read_csv(HERE / "gaston-table1.csv")
    df = table[table.organic.isin(CLASSES)].copy()
    df["reserve"] = False
    for material, x in zip(CLASSES, RESERVED, strict=True):
        df.loc[
            (df.organic == material)
            & (df.rh_pct == 50)
            & np.isclose(df.organic_mass_fraction, x),
            "reserve",
        ] = True
    if df.reserve.sum() != len(CLASSES):
        raise ValueError("Wrong number of withheld laboratory preparations.")
    organic = df.organic_mass_fraction.to_numpy()
    rh = (df.rh_pct.to_numpy() - 50.0) / 20.0
    y, width = (
        df.gamma_normalized.to_numpy(),
        df.normalized_ci95_half.to_numpy(),
    )
    train = ~df.reserve.to_numpy()
    class_slope = [-organic * (df.organic.to_numpy() == c) for c in CLASSES]
    models = {
        "rh_only": [rh],
        "shared_organic": [rh, -organic],
        "specific_organic": [rh, *class_slope],
    }
    summary = {"backend": f"Warp {wp.__version__}, CPU, float64", "fits": {}}
    for model, columns in models.items():
        design = np.column_stack(columns)
        params, errors, loss = fit_law(design[train], y[train], width[train])
        prediction = np.exp(design @ params)
        diff = prediction - y
        summary["fits"][model] = {
            "parameters": params.tolist(),
            "finite_difference_relative_errors": errors,
            "training_rows": int(sum(train)),
            "training_loss": float(loss),
            "training_rmse_ratio": float(np.sqrt(np.mean(diff[train] ** 2))),
            "heldout_rows": int(sum(~train)),
            "heldout_rmse_ratio": float(np.sqrt(np.mean(diff[~train] ** 2))),
            "heldout_within_reported_ci": int(
                sum(abs(diff[~train]) <= width[~train])
            ),
        }
        df[model] = prediction
    df[
        [
            "organic",
            "rh_pct",
            "organic_mass_fraction",
            "gamma_normalized",
            "normalized_ci95_half",
            "reserve",
            *models,
        ]
    ].to_csv(
        HERE / "gaston-fit.csv",
        index=False,
        float_format="%.7g",
        quoting=csv.QUOTE_MINIMAL,
    )
    (HERE / "gaston-fit.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
