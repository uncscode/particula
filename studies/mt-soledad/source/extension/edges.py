"""Estimate a branch-specific plateau and its half-height mobility diameter."""

# %%
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression

EPOCH = np.datetime64("2023-04-01").astype(int)


def extract_edge(d, x, y, window):
    """Measure a large-size count ratio, normalize to it and invert the rising edge."""
    plateau = (d >= window[0]) & (d <= window[1]) & (x > 0)
    if plateau.sum() < 5 or y[plateau].sum() <= 0:
        return np.nan, np.nan
    level = y[plateau].sum() / x[plateau].sum()
    valid = (d >= 30) & (d <= 120) & (x > 0) & np.isfinite(y)
    if valid.sum() < 8:
        return level, np.nan
    ratio = isotonic_regression(y[valid] / x[valid] / level, weights=x[valid]).x
    if not ratio[0] < 0.5 < ratio[-1]:
        return level, np.nan
    return level, np.exp(np.interp(0.5, ratio, np.log(d[valid])))


def extract(inp, out):
    with np.load(inp) as a:
        hours = int(a["hours"]) if "hours" in a else 6
        keys, sums, env, n, d = [a[k] for k in ("keys", "sums", "env", "n", "diam")]
    rows = []
    for block, S in sorted(set(map(tuple, keys[:, :2]))):
        selected = (keys[:, 0] == block) & (keys[:, 1] == S)
        for phase in (-1, 0, 1):
            ix = selected if phase == -1 else selected & (keys[:, 2] == phase)
            if n[ix].sum() < (
                max(2, hours * 2 // 3) if phase == -1 else max(1, hours // 3)
            ):
                continue
            y = sums[ix].sum(axis=0)
            RH, T, P, ss = np.average(env[ix], axis=0, weights=n[ix])
            for offset, kind in ((0, "signal"), (2, "inversion")):
                for win in ((120, 250), (150, 220), (180, 290)):
                    eb = [
                        extract_edge(d, y[b, offset], y[b, offset + 1], win)
                        for b in range(2)
                    ]
                    (pb, db), (pd_, dd) = eb
                    rows.append(
                        (
                            block * hours * 3600,
                            block * hours // 24,
                            (block * hours // 24 - EPOCH) % 3 == 0,
                            S / 10,
                            ss,
                            RH,
                            T,
                            P,
                            phase,
                            kind,
                            win[0],
                            n[ix].sum(),
                            pb,
                            pd_,
                            pd_ / pb,
                            db,
                            dd,
                            dd - db,
                        )
                    )
    cols = "utc day hold S calibrated_S RH T pressure_kPa phase kind window pairs plateau_b plateau_d factor D_b D_d shift".split()
    frame = pd.DataFrame(rows, columns=cols)
    frame.to_csv(out, index=False, float_format="%.9g")
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    extract(args.input, args.out)
