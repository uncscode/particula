"""Vary sodium coefficients away from their binary-only optima before NaCl scoring."""

import numpy as np
import pandas as pd
from calibrate import fit_binary, HERE
from trial import binary_prediction, prepare
from refine import scan


def main():
    data = pd.read_csv(HERE / "binary-water.csv")
    y = np.log(data.a_w.to_numpy())
    scale = data.a_w_meter_accuracy.fillna(0.01).to_numpy() / data.a_w.to_numpy()
    base = binary_prediction(1.0, [0, 0, 0], data)
    A = (
        np.array(
            [binary_prediction(1.0, p, data) - base for p in [[1, 0, 0], [0, 1, 0]]]
        ).T
        / scale[:, None]
    )
    _, s, vh = np.linalg.svd(A, full_matrices=False)
    asdata = pd.read_csv(HERE / "phase-data.csv").query("salt=='AS'")
    delta = np.linspace(0.15, 0.75, 61)
    rows = []
    for tail in [1.3, 1.5, 1.7, 1.9]:
        for c2 in [0.006081, 0.010, 0.013, 0.016]:
            p, _ = fit_binary(tail, c2)
            for direction in [-2.0, -1.0, 1.0, 2.0]:
                for k in [0, 1]:
                    b1, b2 = p[:2] + direction * vh[k] / s[k]
                    if b1 < 0 or abs(b2) > 5:
                        continue
                    pp = [b1, b2, c2]
                    chi = np.sum(((binary_prediction(tail, pp, data) - y) / scale) ** 2)
                    m0, m1 = [prepare("AS", tail, pp, d) for d in [0.0, 1.0]]
                    curves = np.stack(
                        [scan(m0, m1, "AS", v, delta, 2) for v in asdata.ovf_pct],
                        axis=1,
                    )
                    score = chi + np.sum(
                        (
                            (curves - asdata.srh_pct.to_numpy())
                            / asdata.srh_sd_pct.to_numpy()
                        )
                        ** 2,
                        axis=1,
                    )
                    rows.extend(
                        [
                            [tail, *pp, d, chi, loss, *v]
                            for d, loss, v in zip(delta, score, curves)
                        ]
                    )
    pd.DataFrame(
        rows,
        columns=[
            "tail",
            "B1",
            "B2",
            "C2",
            "NH4_delta",
            "binary_loss",
            "training_loss",
            "AS50",
            "AS70",
            "AS90",
        ],
    ).to_csv(HERE / "joint-calibration.csv", index=False)


if __name__ == "__main__":
    main()
