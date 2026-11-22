"""Profile the binary fraction and rank prospective low-ammonia conditions."""

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy.optimize import least_squares
from fit import setup, thermal, log_rates, run, FITDIR
from reference import SPLITS, J, T, Q, NH3, OUT, WINDOW, dump_csv


def read_best(file):
    runs = json.loads(file.read_text())["runs"]
    converged = [r for r in runs if r["status"] > 0]
    return sorted(converged or runs, key=lambda r: r["cost"])


def profile_one(spec):
    split, line, fraction = spec
    test, train, data, w = setup(split, "nominal", "neutral", "group")
    # Eliminate the binary intercept so that the desired fraction is attained
    # exactly at the chosen held-out temperature and gas concentrations.
    anchor = data[line - 2 : line - 1]
    starts = read_best(
        FITDIR / f"{split}_full_linear_nominal_neutral_group.json"
    )

    def recover(p):
        ref = anchor.expand(len(p), -1)
        dummy = torch.cat([torch.zeros_like(p[:, :1]), p], 1)
        ternary, _ = log_rates(dummy[:, 4:], ref, "assisted", "linear")
        b_no_inter = (
            thermal(dummy, ref[:, 0], "linear") + dummy[:, 3] * ref[:, 1]
        )
        b_inter = ternary + np.log(fraction / (1 - fraction)) - b_no_inter
        return torch.cat([b_inter[:, None], p], 1)

    lo = np.array([-4, 0.005, 0.3, -100, -4, 0.005, 0.3, -20, 0.2])
    hi = np.array([4.5, 4, 9, 35, 4.5, 4, 9, 8, 12])
    best = None
    for r in starts[::2]:

        def evaluate(v):
            p = torch.tensor(
                np.broadcast_to(v, (len(w), len(v))).copy(), requires_grad=True
            )
            restored = recover(p)
            lj, _ = log_rates(restored, data[train], "full", "linear")
            # The eliminated intercept retains the bounds of the original fit.
            intercept = restored[0, 0]
            penalty = 1e4 * (
                torch.relu(intercept - 35.0) + torch.relu(-100.0 - intercept)
            )
            penjac = (
                torch.autograd.grad(penalty, p, retain_graph=True)[0][0]
                .detach()
                .numpy()
            )
            jac = torch.autograd.grad(lj.sum(), p)[0].detach().numpy()
            return np.r_[
                (lj.detach().numpy() - np.log(J[train])) * w,
                float(penalty.detach()),
            ], np.vstack([jac * w[:, None], penjac])

        res = least_squares(
            lambda p: evaluate(p)[0],
            np.clip(r["parameters"][1:], lo + 1e-8, hi - 1e-8),
            jac=lambda p: evaluate(p)[1],
            bounds=(lo, hi),
            max_nfev=350,
            ftol=1e-8,
            xtol=1e-8,
        )
        intercept = float(recover(torch.tensor(res.x)[None, :])[0, 0])
        if (
            res.status > 0
            and -100.000001 <= intercept <= 35.000001
            and (best is None or res.cost < best.cost)
        ):
            best = res
    if best is None:
        raise RuntimeError(f"No converged feasible start for {spec}")
    restored = recover(torch.tensor(best.x)[None, :])
    lj, f = log_rates(
        restored.expand(int(test.sum()), -1), data[test], "full", "linear"
    )
    j = lj.exp().detach().numpy()
    return dict(
        split=split,
        line=line,
        prescribed_fraction=fraction,
        achieved=float(f[np.flatnonzero(np.where(test)[0] == line - 2)[0]]),
        cost=best.cost,
        training_factor_rmse=np.exp(np.sqrt(2 * best.cost / train.sum())),
        heldout_median_factor=np.exp(np.median(abs(np.log(j / J[test])))),
        status=best.status,
    )


def prospective():
    SPLITS["complete"] = np.zeros_like(WINDOW, dtype=bool)
    for shape, topo in itertools.product(
        ["linear", "curved"], ["full", "assisted"]
    ):
        run(("complete", topo, shape, "nominal", "neutral", "group", 12))
    design = []
    # Compare laws where measured conditions supply temperature and acid support.
    for temp in [223.0, 248.3, 278.5]:
        # Rank points across an acid scan common to the competing predictions.
        for a, nh in itertools.product(
            np.geomspace(3, 40, 8), [0.03, 0.06, 0.1, 0.2, 0.38, 0.63, 1.0, 2.0]
        ):
            air = 1.005 * 101325 / (1.380649e-23 * temp) / 1e6
            ranges = {}
            for topo in ["full", "assisted"]:
                pred = []
                for shape in ["linear", "curved"]:
                    runs = read_best(
                        FITDIR
                        / f"complete_{topo}_{shape}_nominal_neutral_group.json"
                    )
                    for r in runs:
                        if r["cost"] > runs[0]["cost"] * 1.02:
                            continue
                        for acid_factor, nh_factor in itertools.product(
                            [2 / 3, 1.0, 1.5], [0.8, 1.0, 1.2]
                        ):
                            inp = torch.tensor(
                                [
                                    [
                                        (temp - 248) / 50,
                                        np.log(a * acid_factor),
                                        np.log(nh * nh_factor * air * 1e-18),
                                        -690.0,
                                    ]
                                ]
                            )
                            logj, _ = log_rates(
                                torch.tensor(r["parameters"])[None, :],
                                inp,
                                topo,
                                shape,
                            )
                            pred.append(float(logj.exp()[0]))
                ranges[topo] = (min(pred), max(pred))
            l, h = ranges["full"]
            u, v = ranges["assisted"]
            gap = max(l / v, u / h)
            design.append(
                dict(
                    T_K=temp,
                    acid_cm3=a * 1e6,
                    NH3_pptv=nh,
                    full_min=l,
                    full_max=h,
                    assisted_min=u,
                    assisted_max=v,
                    separation=gap,
                    both_above_floor=min(l, u) >= 0.001,
                    either_above_floor=max(l, u) >= 0.001,
                )
            )
    dump_csv(OUT / "experiment_design.csv", design)


if __name__ == "__main__":
    rows = []
    specs = [
        (s, l, f)
        for s, l in [("248_low", 336), ("223_low", 366)]
        for f in [
            0.0001,
            0.001,
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.99,
            0.9999,
        ]
    ]
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(profile_one, specs))
    dump_csv(OUT / "profile.csv", rows)
    prospective()
