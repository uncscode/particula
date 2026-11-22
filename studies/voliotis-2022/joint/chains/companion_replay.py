"""Frozen CUDA replay for particle-wall-corrected yields and water uptake."""

from pathlib import Path

import numpy as np
import pandas as pd
import warp as wp
from bank import Channel
from compare import load

from gpu import Model

DEST = Path(__file__).resolve().parents[1] / "companions"


def isolated_parents(model, x, record, dose, objective):
    """Match each parent disappearance history with the partners removed."""
    original = model.data[0]
    source_array = original.numpy()
    pure = []
    try:
        for parent in range(3):
            isolated = np.zeros_like(source_array)
            isolated[:, :, parent] = (
                source_array[:, :, parent] + source_array[:, :, parent + 3]
            )
            model.data[0] = wp.array(
                isolated, dtype=original.dtype, device=original.device
            )
            control = model.run(
                x[:16],
                x[16:],
                record["active"],
                dose,
                objective,
                sub=32,
                diagnostics=4,
            )[3]
            pure.append(
                control[0]
                .reshape(len(model.runs), 300, 3)[:, :, :2]
                .sum(axis=2)
            )
    finally:
        model.data[0] = original
    return pure


def main():
    """Replay fixed doses and rates at two resolutions."""
    archive = load("resolution_checked_mechanism.json")
    fields = Channel.__dataclass_fields__
    model = Model(
        [Channel(**{k: r[k] for k in fields}) for r in archive["reactions"]]
    )
    rows = []
    for objective, record in archive["fits"].items():
        fit = record["fit"]
        x = np.asarray(fit["x"])
        dose = np.asarray(fit["dose"])[:, None]
        coarse = []
        for sub in (16, 32):
            score, errors, _, history = model.run(
                x[:16],
                x[16:],
                record["active"],
                dose,
                objective,
                sub=sub,
                diagnostics=4,
            )
            pure = []
            if sub == 32:
                pure = isolated_parents(model, x, record, dose, objective)
            for j, (run, steps, _, _, source) in enumerate(model.runs):
                soa, wall, seed = history[0, j].reshape(-1, 3).T
                if (
                    np.diff(wall).min() < -1e-10
                    or seed.min() <= 0
                    or np.diff(seed).max() > 1e-10
                ):
                    raise ArithmeticError(
                        f"Invalid wall or seed history in run {run}"
                    )
                voc = (
                    np.cumsum(source[:, :3] + source[:, 3:], axis=0)
                    * dose[j, 0]
                )
                series = pd.DataFrame(
                    dict(
                        run=run,
                        seconds=steps,
                        dose=dose[j, 0],
                        soa=np.r_[0, soa],
                        wall=np.r_[0, wall],
                        seed=np.r_[model.initial[j, 0], seed],
                    )
                )
                series[["alpha", "cresol", "isoprene"]] = np.vstack(
                    (np.zeros(3), voc)
                )
                if sub == 32:
                    for key in ("soa", "wall", "seed"):
                        np.testing.assert_allclose(
                            series[key], coarse[j][key], atol=0.25, rtol=0.01
                        )
                    for name, control in zip(
                        ("alpha", "cresol", "isoprene"), pure, strict=True
                    ):
                        series[f"{name}_only_soa_corrected"] = np.r_[
                            0, control[j]
                        ]
                    if np.count_nonzero(voc[-1]) == 1:
                        np.testing.assert_allclose(
                            sum(a[j] for a in pure), soa + wall, atol=1e-9
                        )
                    series["fit"] = objective
                    rows.append(series)
                else:
                    coarse.append(series)
            print(
                objective,
                60 / sub,
                "frozen score",
                score[0],
                "ledger",
                errors[0, :, 2].max(),
            )
    pd.concat(rows, ignore_index=True).to_csv(
        DEST / "replay.csv.gz",
        index=False,
        float_format="%.9g",
        compression={"method": "gzip", "mtime": 0},
    )


if __name__ == "__main__":
    main()
