"""Replay the four frozen sulfur and potassium reactor fits and all 85 readings.

Run from the repository root with Python 3.12+, NumPy, SciPy and warp-lang.
    python scripts/straw_combustion/replay.py --device cpu --output results
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import warp as wp

if __package__:
    from .forward import dry, kernels, ladders, readout, thermal
else:
    from forward import dry, kernels, ladders, readout, thermal


INPUT = Path(__file__).resolve().with_name("inputs")
FITS = ("all_combined", "split_combined", "all_parent", "split_parent")
REFERENCE_DT = 0.0001875
TARGETS = (
    "count",
    "number_gmd",
    "lpi_mass",
    "smps_mass",
    "lpi_mass_gmd",
    "smps_mass_gmd",
    "sulfate_pct",
)


def _read(name):
    with (INPUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _score(errors, families, mask):
    errors, families = errors[mask], families[mask]
    cost = 0.5 * sum(
        np.mean(errors[families == f] ** 2) for f in np.unique(families)
    )
    return {
        "count": len(errors),
        "in_band": int(np.sum(abs(errors) <= 1)),
        "cost": float(cost),
    }


def _map_predictions(observations, feed_rows, aerosol, gas):
    feed_idx = {r["case"]: i for i, r in enumerate(feed_rows)}
    gas_idx = {
        (r["oxygen_percent"], r["temperature_K"]): i
        for i, r in enumerate(dry.G)
    }
    columns = []
    for r in observations:
        if r["dataset"] == "gas":
            column = gas[
                :,
                gas_idx[
                    (
                        float(r["case"].removeprefix("O2_")),
                        float(r["temperature_K"]),
                    )
                ],
            ]
        else:
            column = aerosol[:, feed_idx[r["case"]], TARGETS.index(r["target"])]
        columns.append(column)
    return np.stack(columns, axis=1)


def _selected_fits(fits, annuli):
    if isinstance(fits, str):
        fits = (fits,)
    fits = tuple(fits)
    if (
        not fits
        or len(set(fits)) != len(fits)
        or any(f not in FITS for f in fits)
    ):
        raise ValueError(f"Choose one or more distinct fits from {FITS}")
    if not isinstance(annuli, int):
        raise ValueError("annuli must be an integer")
    return fits


def run_replay(device="cpu", fits=FITS, dt=REFERENCE_DT, annuli=2):
    """Integrate the frozen models and return aligned rows and diagnostics.

    Device is a Warp device identifier. dt is the maximum aerosol step (s).
    The dry reactor uses 256 Rosenbrock steps, independent of this aerosol mesh.
    The reference values are used only after evaluating all model predictions.
    """
    fits = _selected_fits(fits, annuli)
    dev = wp.get_device(device)
    wp.config.use_precompiled_headers = False

    def arr(values, dtype):
        return wp.array(np.ascontiguousarray(values), dtype=dtype, device=dev)

    parameters = json.loads((INPUT / "parameters.json").read_text())
    x = np.array([parameters[f]["x"] for f in fits])
    u = np.array([parameters[f]["u"] for f in fits])
    feed_rows = _read("aerosol_feeds.csv")
    feeds = np.array(
        [
            [float(r[k]) for k in ("kcl_ppm", "so2_ppm", "oxygen_percent")]
            for r in feed_rows
        ]
    )
    series = np.array([float(r["sulfur_free_series"]) for r in feed_rows])
    prop, ticks, weights = thermal.temperature_history(dt, annuli)
    shape = (len(fits), len(feeds), annuli)
    states = wp.empty(shape, dtype=kernels.State, device=dev)
    ledger = wp.empty(shape, dtype=ladders.Ledger, device=dev)
    wp.launch(
        ladders.parcels,
        shape,
        [
            arr(x[:, :7], kernels.Params),
            arr(x[:, 9:29], ladders.Vec),
            arr(x[:, 29:], ladders.Net),
            arr(u, ladders.Pars),
            arr(feeds, kernels.Feed),
            arr(prop, kernels.Props),
            arr(ticks, kernels.F),
            arr(series, kernels.F),
            4,
            states,
            ledger,
        ],
        device=dev,
    )
    y, audit = states.numpy(), ledger.numpy()
    if not (np.isfinite(y).all() and np.isfinite(audit).all()):
        raise ArithmeticError("Nonfinite reactor endpoint or budget")
    if (
        np.max(audit[..., 7]) >= 2e-6
        or min(audit[..., 8].min(), y.min()) < -1e-7
    ):
        raise ArithmeticError(
            "Elemental balance or concentration positivity failure"
        )
    aerosol = readout.observation(y, weights, x[:, 7], x[:, 8])
    gas = dry.predict_dry(x[:, 29:], 12, 0, steps=256, device=dev)
    observations = _read("observations.csv")
    prediction = _map_predictions(observations, feed_rows, aerosol, gas)
    if not np.isfinite(prediction).all():
        raise ArithmeticError("Nonfinite detector prediction")
    references = _read("reference_predictions.csv")
    if [int(r["row"]) for r in observations] != list(range(85)) or [
        int(r["row"]) for r in references
    ] != list(range(85)):
        raise ValueError("Expected aligned rows 0–84 in both tables")
    measured = np.array([float(r["measured"]) for r in observations])
    tolerance = np.array([float(r["tolerance"]) for r in observations])
    train = np.array([r["in_fit"] == "True" for r in observations])
    family = np.array([int(r["family"]) for r in observations])
    rows, diagnostics = [], []
    for i, fit in enumerate(fits):
        expected = np.array([float(r[fit]) for r in references])
        errors = (prediction[i] - measured) / tolerance
        max_diff = float(np.max(abs(prediction[i] - expected) / tolerance))
        if annuli == 2 and dt == REFERENCE_DT and max_diff > 2e-6:
            raise ArithmeticError(
                f"{fit} differs from the archive by {max_diff:g} bands"
            )
        for j, r in enumerate(observations):
            rows.append(
                dict(
                    fit=fit,
                    row=j,
                    dataset=r["dataset"],
                    case=r["case"],
                    target=r["target"],
                    temperature_K=r["temperature_K"],
                    measured=measured[j],
                    tolerance=tolerance[j],
                    in_original_train=bool(train[j]),
                    predicted=float(prediction[i, j]),
                    residual_bands=float(errors[j]),
                    reference_prediction=expected[j],
                )
            )
        diagnostics.append(
            dict(
                fit=fit,
                device=str(dev),
                max_step_s=dt,
                annuli=annuli,
                steps_per_annulus=prop.shape[1],
                max_element_error_ppm=float(np.max(audit[i, ..., 7])),
                min_concentration=float(
                    min(audit[i, ..., 8].min(), y[i].min())
                ),
                max_reference_difference_bands=max_diff,
                all=_score(errors, family, np.ones(85, dtype=bool)),
                training=_score(errors, family, train),
                withheld=_score(errors, family, ~train),
            )
        )
    return rows, diagnostics


def main():
    """Write measurement-level predictions and replay diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cpu", help="Warp device, e.g. cpu or cuda:0"
    )
    parser.add_argument("--fits", choices=FITS, nargs="+", default=FITS)
    parser.add_argument("--dt", type=float, default=REFERENCE_DT)
    parser.add_argument("--annuli", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, diagnostics = run_replay(args.device, args.fits, args.dt, args.annuli)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    for r in diagnostics:
        print(
            f"{r['fit']}  {r['all']['in_band']}/85 inside bands, "
            f"max. difference {r['max_reference_difference_bands']:.3g} bands"
        )


if __name__ == "__main__":
    main()
