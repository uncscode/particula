"""A null direction in size assignment between measured activation diameters."""

import json
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from calibration import KNOTS, basis
from observations import SET, P, cohort, recipes, volume_spectra
from scipy.special import expit
from study import PRIMARY, Population


def main():
    fits = (
        pd.read_csv(P / "fits.csv").query("model==@PRIMARY").set_index("date")
    )
    date = (fits.rmse - fits.rmse.median()).abs().idxmin()
    scans, _, _ = cohort()
    spec = volume_spectra([date])[date]
    centers = spec[0] + np.log(80.0)
    scan = scans[scans.date == date].sort_values(["j", "Dp"])
    obs = np.log(scan.Dp.to_numpy())
    original = np.unique(np.r_[obs, KNOTS])
    pop = Population(
        np.load(P / "thresholds.npz"), recipes().loc[date], spec, PRIMARY
    )
    weights = (
        pd.read_csv(P / "shares.csv.gz")
        .query("date==@date")
        .share.to_numpy()
        .reshape(-1, pop.N)
    )
    mesh = np.unique(np.r_[original, centers])
    wt = basis(mesh, KNOTS) @ weights
    idx = np.searchsorted(mesh, centers)
    oi = np.searchsorted(mesh, obs)
    candidates = []
    for i, k in enumerate(idx):
        if (
            not 45 < np.exp(mesh[k]) < 160
            or min(abs(original - mesh[k])) < 1e-6
        ):
            continue
        # Bound the added local slope to 3 per unit log diameter.
        candidates.append(
            (i, k, 3 * min(mesh[k] - mesh[k - 1], mesh[k + 1] - mesh[k]))
        )
    best = None
    for (i, k, lim), (j, l, lim2) in combinations(candidates, 2):
        ratio = spec[1][i] / spec[1][j]
        for a, b in permutations(range(pop.N), 2):
            step = min(lim, lim2 / ratio, wt[k, a], wt[l, b] / ratio)
            diff = step * abs(pop.c[a, 0] - pop.c[b, 0])
            if best is None or diff > best[0]:
                best = (diff, i, j, k, l, a, b, step, ratio)
    diff, i, j, k, l, a, b, step, ratio = best
    alt = wt.copy()
    alt[k, a] -= step
    alt[k, b] += step
    alt[l, b] -= ratio * step
    alt[l, a] += ratio * step

    def fitted(w):
        pr = expit((obs[:, None] - pop.ld[:, scan.j.to_numpy()].T) / pop.width)
        return (w[oi] * pr).sum(axis=1)

    obsdelta = max(abs(fitted(alt) - fitted(wt)))
    histdelta = max(abs(spec[1] @ (alt[idx] - wt[idx])))
    assert (
        alt.min() > -1e-8
        and obsdelta < 1e-13
        and histdelta < 1e-12
        and diff > 0.01
    )
    records = []
    for tag, w in [("original", wt), ("redistributed", alt)]:
        for name, col in zip(["salt", "sorbitol", "squalane"], (w @ pop.c).T):
            records.extend(
                zip([tag] * len(mesh), np.exp(mesh), [name] * len(mesh), col)
            )
    pd.DataFrame(
        records, columns=["label", "D", "component", "fraction"]
    ).to_csv(P / "between-curves.csv", index=False)
    predictions = []
    for key in [k, l]:
        response = expit((mesh[key] - pop.ld) / pop.width)
        for setting, first, second in zip(
            SET, wt[key] @ response, alt[key] @ response
        ):
            predictions.append(
                dict(
                    D=np.exp(mesh[key]),
                    SS=setting,
                    original=first,
                    redistributed=second,
                )
            )
    pd.DataFrame(predictions).to_csv(P / "new-size-test.csv", index=False)
    info = dict(
        date=date,
        diameters_nm=np.exp(mesh[[k, l]]).tolist(),
        bracketing_scan_nm=[
            [
                float(np.exp(max(obs[obs < mesh[t]]))),
                float(np.exp(min(obs[obs > mesh[t]]))),
            ]
            for t in [k, l]
        ],
        fitted_salt=(wt @ pop.c)[[k, l], 0].tolist(),
        redistributed_salt=(alt @ pop.c)[[k, l], 0].tolist(),
        max_fraction_change_at_observations=obsdelta,
        max_histogram_change=histdelta,
        chi=fits.loc[date, "chi"],
        rmse=fits.loc[date, "rmse"],
    )
    (P / "between.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info)


if __name__ == "__main__":
    main()
