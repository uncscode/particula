"""Evaluate every calibration profile on refined phase grids using GPU tangent planes."""

import argparse
import numpy as np
import pandas as pd
import warp as wp
from calibrate import HERE, TEMPERATURE, fit_binary, NH4_GRID
from stability import grid
from trial import prepare


@wp.kernel
def cross(
    a: wp.array2d(dtype=wp.float64),
    b: wp.array2d(dtype=wp.float64),
    increments: wp.array(dtype=wp.float64),
    first: wp.array(dtype=wp.float64),
):
    d, k = wp.tid()
    prev = wp.float64(0.0)
    for j in range(a.shape[0]):
        val = a[j, k] + increments[d] * b[j, k]
        if val < wp.float64(-1e-5):
            pos = wp.float64(0.0)
            if j > 0:
                pos = wp.float64(j - 1) + wp.clamp(
                    (prev + wp.float64(1e-5)) / (prev - val),
                    wp.float64(0.0),
                    wp.float64(1.0),
                )
            wp.atomic_min(first, d, pos)
            break
        prev = val


def scan(
    model0,
    model1,
    salt,
    v,
    deltas,
    refine,
    temp=TEMPERATURE,
    cap=160.0,
    rho=1.266,
):
    g = grid(salt, v, temp, model0, refine=refine, maxmol=cap, density=rho)
    h = grid(salt, v, temp, model1, refine=refine, maxmol=cap, density=rho)
    increments = wp.array(deltas, dtype=wp.float64, device="cuda:0")
    first = wp.array(
        np.full(len(deltas), float(len(g.molality))),
        dtype=wp.float64,
        device="cuda:0",
    )
    chunk = 18000
    for start in range(0, len(g.energies), chunk):
        sl = slice(start, start + chunk)
        counts = g.trial_coordinates[sl]
        a = np.ascontiguousarray(g.energies[sl][None, :] - g.feed_mu @ counts.T)
        b = np.ascontiguousarray(
            (h.energies[sl] - g.energies[sl])[None, :]
            - (h.feed_mu - g.feed_mu) @ counts.T
        )
        ga = wp.array(a, dtype=wp.float64, device="cuda:0")
        gb = wp.array(b, dtype=wp.float64, device="cuda:0")
        wp.launch(
            cross,
            (len(deltas), len(counts)),
            inputs=[ga, gb, increments, first],
            device="cuda:0",
        )
        wp.synchronize_device("cuda:0")
    out = []
    for position, delta in zip(first.numpy(), deltas):
        if position <= 0 or position >= len(g.molality) - 1:
            out.append(np.nan)
            continue
        logaw = g.feed_mu[:, 0] + delta * (h.feed_mu[:, 0] - g.feed_mu[:, 0])
        out.append(100 * np.exp(np.interp(position, np.arange(len(logaw)), logaw)))
    return np.asarray(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine", type=int, default=2)
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    targets = pd.read_csv(HERE / "phase-data.csv").query("salt=='AS'")
    profiles = pd.read_csv(HERE / "calibration.csv").drop_duplicates(["tail", "C2"])
    deltas = np.unique(np.r_[NH4_GRID, np.linspace(0.24, 0.45, 15)])
    if args.extend:
        # Widen the boundaries reached by the fit to AS, before evaluating its NaCl transfer.
        extra = []
        for tail in [1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5]:
            for c2 in [0.006081, 0.010, 0.013, 0.016, 0.020, 0.026]:
                p, chi = fit_binary(tail, c2)
                extra.append([tail, *p, 0.0, chi, 0.0, 0.0, 0.0, 0.0])
        profiles = pd.DataFrame(extra, columns=profiles.columns)
        deltas = np.unique(np.r_[deltas, np.linspace(0.4, 1.0, 31), 1.2, 1.5, 2.0])
    rows = []
    for _, r in profiles.iterrows():
        p = (r.B1, r.B2, r.C2)
        m0 = prepare("AS", r["tail"], p, 0)
        m1 = prepare("AS", r["tail"], p, 1)
        curves = np.stack(
            [scan(m0, m1, "AS", v, deltas, args.refine) for v in targets.ovf_pct],
            axis=1,
        )
        loss = r.binary_loss + np.sum(
            ((curves - targets.srh_pct.to_numpy()) / targets.srh_sd_pct.to_numpy())
            ** 2,
            axis=1,
        )
        for delta, vals, score in zip(deltas, curves, loss):
            rows.append([r["tail"], *p, delta, r.binary_loss, score, *vals])
    result = pd.DataFrame(rows, columns=profiles.columns)
    result.to_csv(
        HERE
        / ("extended-calibration.csv" if args.extend else "refined-calibration.csv"),
        index=False,
    )
    print(result.nsmallest(5, "training_loss").to_string(index=False))


if __name__ == "__main__":
    main()
