"""Locate the dry sizes responsible for the held-out activation excursions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from campaign import read
from metrics import EDGES, compress, critical
from model import SS, init, step


def compare(ref, rep):
    """Resolve the largest CDF jump and its signed contribution by dry bin."""
    resolved, reduced = critical(ref), critical(rep)
    refbins = np.searchsorted(EDGES[10], np.cbrt(ref.sum(1)), side="right")
    repbins = np.searchsorted(EDGES[10], np.cbrt(rep.sum(1)), side="right")
    levels = np.unique(
        np.r_[SS[[0, -1]], reduced[(reduced > SS[0]) & (reduced <= SS[-1])]]
    )
    best = -1
    for side in ["left", "right"]:
        x = levels[levels > SS[0]] if side == "left" else levels
        curve = np.searchsorted(np.sort(reduced), x, side) / len(
            rep
        ) - np.searchsorted(np.sort(resolved), x, side) / len(ref)
        k = np.argmax(abs(curve))
        if abs(curve[k]) > best:
            best = abs(curve[k])
            compare = np.less if side == "left" else np.less_equal
            contribution = 100 * (
                np.bincount(
                    repbins, weights=compare(reduced, x[k]), minlength=32
                )
                / len(rep)
                - np.bincount(
                    refbins, weights=compare(resolved, x[k]), minlength=32
                )
                / len(ref)
            )
            j = np.argmax(abs(contribution))
            result = (
                x[k],
                side,
                100 * best,
                *EDGES[10][j - 1 : j + 1],
                contribution[j],
            )
    return result


def locate(base):
    """Replay the hour-two snapshot and the held-out hour-three forecasts."""
    seed = 1320
    f = base / str(seed) / "c0b0n0" / "07200.npz"
    boxes, _, _ = read(f)
    b = boxes[0]
    full = b.dry.copy()
    compress(b, 10, 16, 2)
    rows = [[seed, "c0b0n0", 7200, 7200, 0, 16, *compare(full, b.dry)]]
    for regime in ["c0b0n1", "c0b1n1"]:
        folder = base / str(seed) / regime
        boxes, rng, t = read(folder / "10800.npz")
        n, dt, _ = map(int, (folder / "done").read_text().split())
        _, _, params = init(seed, n)
        for b in boxes:
            compress(b, 10, 7, 2)
        for s in range(t, 14400, dt):
            step(boxes, rng, params, regime, s, dt)
        truth, _, _ = read(folder / "14400.npz")
        rows.append(
            [seed, regime, t, 14400, 1, 7, *compare(truth[1].dry, boxes[1].dry)]
        )
    return pd.DataFrame(
        rows,
        columns=(
            "seed regime start end parcel cap supersaturation_pct "
            "side maximum_pp low_nm high_nm signed_bin_pp"
        ).split(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    locate(a.base).to_csv(a.out, index=False, float_format="%.8g")
