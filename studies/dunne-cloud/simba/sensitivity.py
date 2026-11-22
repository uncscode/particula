"""Matched fixed-physics and longer training checks on the identical search population."""

import csv
import numpy as np
from search import write
from cloud_networks import HERE
from replay import measurements, simulate


def main():
    settings = list(csv.DictReader((HERE / "search.csv").open()))
    keys = ["gas", "wall", "extended", "gamma_ozone", "gamma_peroxide"]
    cfg = [tuple(float(r[k]) for k in keys) + (-1.0,) for r in settings]
    t, *_, obs, hp, valid, so = measurements()
    hours = t - t[0]
    train = hours[:181] < 3.8
    errs = []
    for i in range(0, len(cfg), 96):
        arr, _ = simulate(cfg[i : i + 96])
        sq = (arr[:, :, 1] - obs[None, :181, 1]) ** 2
        errs.extend(
            np.column_stack(
                [
                    sq[:, train].mean(1) ** 0.5,
                    sq[:, ~train].mean(1) ** 0.5,
                    sq.mean(1) ** 0.5,
                ]
            )
        )
    errs = np.asarray(errs)
    chosen = []
    for window, idx in [("early", 0), ("six_hours", 2)]:
        for lock in ["fitted", "fixed"]:
            for name in ["physics", "ozone", "peroxide", "both"]:
                ix = np.array(
                    [
                        i
                        for i, (g, w, e, o, p, _) in enumerate(cfg)
                        if e == 0
                        and (lock == "fitted" or (g == 1.15 and w == 1))
                        and (name in ["both", "ozone"] or not o)
                        and (name in ["both", "peroxide"] or not p)
                    ]
                )
                j = ix[np.argmin(errs[ix, idx])]
                near = ix[errs[ix, idx] <= errs[j, idx] * 1.05]
                chosen.append((window, lock, name, j, near))
    unique = sorted(set(v[3] for v in chosen))
    out, _ = simulate([cfg[i] for i in unique], count=len(t))
    rows = []
    for window, lock, name, j, near in chosen:
        curve = out[unique.index(j), :, 1]
        later = hours > 6.0001
        rows.append(
            dict(
                fit_window=window,
                physics=lock,
                family=name,
                **dict(zip(keys, cfg[j][:5])),
                train_rmse=errs[j, 0],
                peak_block_rmse=errs[j, 1],
                six_hour_rmse=errs[j, 2],
                later_rmse=np.mean((curve[later] - obs[later, 1]) ** 2) ** 0.5,
                early_peak=curve[:181].max(),
                near_count=len(near),
                near_min_peakblock_rmse=errs[near, 1].min(),
                near_max_peakblock_rmse=errs[near, 1].max(),
            )
        )
    write("sensitivity.csv", rows)


if __name__ == "__main__":
    main()
