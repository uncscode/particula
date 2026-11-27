"""Find the best possible four-marker error for a uniform particle with fixed D."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "radial"))
from radial import inputs, run


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    weights, t, target = inputs(HERE.parent)
    rows = []
    for n in (64, 160):

        def evaluate(logd):
            time, curve, _ = run(weights, t, 10.0**logd, n)
            values = np.interp(t, time, curve)
            err = values - target
            worst = abs(err).max()
            for ti, yi, pred, delta in zip(t, target, values, err):
                rows.append((n, logd, ti, yi, pred, delta, worst))
            return worst

        grid = np.linspace(-22, -18, 17)
        score = list(map(evaluate, grid))
        index = np.argmin(score)
        assert 0 < index < len(grid) - 1
        result = minimize_scalar(
            evaluate,
            bounds=(grid[index - 1], grid[index + 1]),
            method="bounded",
            options={"xatol": 1e-5},
        )
        assert result.success
    cols = "shells log10_D time_min measured predicted residual max_absolute_error".split()
    df = pd.DataFrame(rows, columns=cols)
    # Keep each refined solution and the prespecified sweep, omit optimizer iterations.
    best = df.groupby("shells").max_absolute_error.transform("min")
    df = df[np.isin(df.log10_D, grid) | (df.max_absolute_error == best)].copy()
    df["minimax"] = df.max_absolute_error == best.loc[df.index]
    df.to_csv(out / "constant_D_limit.csv", index=False, float_format="%.11g")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--out", type=Path, default=HERE / "results")
    main(cli.parse_args().out)
