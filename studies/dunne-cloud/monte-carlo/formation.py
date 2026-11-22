"""Joint calibration and condition-weight draws for neutral and ion-induced J1.7."""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product

import numpy as np
from design import HERE, draws
from scipy.optimize import least_squares
from scipy.special import expit

sys.path.insert(1, str(HERE.parent / "analysis"))
from reference import (
    ACID,
    AIR,
    AMMONIA,
    BAND,
    NH3,
    OUT,
    SPLITS,
    WINDOW,
    J,
    Q,
    T,
    dump_csv,
)

GROUP = np.unique(
    np.column_stack(
        (BAND, np.rint(np.log10(NH3) / 0.15), np.rint(np.log10(1 + Q) / 0.35))
    ),
    axis=0,
    return_inverse=True,
)[1]
BRACKET = np.select([NH3 < 1, NH3 <= 10, NH3 < 100], [0, 1, 2], 3)
DRAWS = draws()
SPLITS = {**SPLITS, "complete": np.zeros_like(WINDOW)}
ALPHA = 6e-8 * np.sqrt(300 / T) + 6e-26 * AIR * (300 / T) ** 4
BASE_BOUNDS = [(-100, 35), (-4, 4.5), (0.005, 4), (0.3, 9)]
BASE_EXTRA = [(-20, 8), (0.2, 12)]


def inputs(row):
    bases = np.array([row[f"base{i}"] for i in range(4)])
    return ACID * row["acid"], AMMONIA * bases[BRACKET]


def weight(row, mask):
    ids, inv, counts = np.unique(
        GROUP[mask], return_inverse=True, return_counts=True
    )
    cluster = np.random.default_rng(213001 + row["id"]).exponential(
        size=GROUP.max() + 1
    )
    w = cluster[ids][inv] / counts[inv]
    return np.sqrt(w / w.mean())


def kernel(p, data, topo, shape):
    x, a, n = data[:3]

    def term(v, tern):
        s = np.exp(v[1])
        if shape == "curved":
            c = v[2]
            z = np.expm1(c * x) / c
            dc = -s * (np.exp(c * x) * x - z) / c
        else:
            z, dc = x, np.zeros_like(x)
        y = v[0] - s * z + v[3] * a
        der = [np.ones_like(x), -s * z, dc, a]
        if tern:
            d = v[3] * a - v[5] * n
            f = expit(v[4] - d)
            y = y + n - np.logaddexp(v[4], d)
            der[3] = a * f
            der.extend([-f, (1 - f) * n])
        return y, np.column_stack(der)

    t, jt = term(p[-6:], True)
    if topo == "assisted":
        return t, jt, np.zeros_like(t)
    b, jb = term(p[:4], False)
    y = np.logaddexp(b, t)
    fb = expit(b - t)
    return y, np.column_stack((jb * fb[:, None], jt * (1 - fb[:, None]))), fb


def law(p, data, topo, shape, ion):
    k = 6 if topo == "assisted" else 10
    n, jn, fb = kernel(p[:k], data, topo, shape)
    if not ion:
        return n, jn, fb, np.zeros_like(n)
    q, alpha = data[3:]
    li, ji, _ = kernel(p[k:], data, "full", shape)
    beta = np.exp(li)
    b = beta + 1 / 480
    root = np.sqrt(b * b + 4 * alpha * q)
    ij = (2 * q / (b + root)) * beta
    with np.errstate(divide="ignore"):
        lj = np.log(ij)
    y = np.logaddexp(n, lj)
    fi = np.exp(lj - y)
    jac = np.column_stack(
        (jn * np.exp(n - y)[:, None], ji * (fi * (1 - beta / root))[:, None])
    )
    return y, jac, fb * np.exp(n - y), fi


@lru_cache(None)
def anchors(split, topo, shape, ion):
    stems = (
        ["nominal"]
        if ion or not split.endswith("low")
        else ["nominal", "base_low", "base_high"]
    )
    result = []
    for cal in stems:
        path = (
            OUT / "joint" / f"{split}_{topo}_{shape}.json"
            if ion
            else OUT
            / "fits"
            / f"{split}_{topo}_{shape}_{cal}_neutral_group.json"
        )
        runs = json.loads(path.read_text())["runs"]
        runs = sorted(
            (r for r in runs if r["status"] > 0), key=lambda r: r["cost"]
        )
        result.append(np.array(runs[0]["params" if ion else "parameters"]))
    return result


@lru_cache(None)
def secondary_start(split, topo, shape, ion):
    path = (
        OUT / "joint" / f"{split}_{topo}_{shape}.json"
        if ion
        else OUT / "fits" / f"{split}_{topo}_{shape}_nominal_neutral_group.json"
    )
    runs = sorted(
        (r for r in json.loads(path.read_text())["runs"] if r["status"] > 0),
        key=lambda r: r["cost"],
    )
    # Contrast a second archived basin when an alternate stationary solution exists.
    r = next(
        (r for r in runs[1:] if r["cost"] > runs[0]["cost"] * 1.005), runs[-1]
    )
    return np.array(r["params" if ion else "parameters"])


def task(args):
    idx, split, topo, shape, ion = args
    row = DRAWS[idx]
    ac, base = inputs(row)
    data = np.array([(T - 248) / 50, np.log(ac), np.log(base), Q, ALPHA])
    mask = WINDOW & ~SPLITS[split] & (True if ion else Q == 0)
    test = WINDOW & SPLITS[split] & (True if ion else Q == 0)
    train = data[:, mask]
    w = weight(row, mask)
    target = np.log(J[mask] * row["jscale"])
    bounds = (BASE_BOUNDS * (2 if topo == "full" else 1)) + BASE_EXTRA
    if ion:
        bounds = [
            (
                -120 if l == -100 else -25 if l == -20 else l,
                40 if h == 35 else h,
            )
            for l, h in bounds + BASE_BOUNDS * 2 + BASE_EXTRA
        ]
    lo, hi = np.array(bounds).T
    active = np.ones(len(lo), bool)
    if shape == "linear":
        active[np.array([2, 6] if topo == "full" else [2])] = False
        if ion:
            active[np.array([2, 6]) + (10 if topo == "full" else 6)] = False
    starts = anchors(split, topo, shape, ion)
    selected = [starts[0]]
    if len(starts) > 1:
        selected += [starts[1 if row["base0"] < 1 else 2]]
    selected.append(secondary_start(split, topo, shape, ion))
    runs = []
    for p0 in selected:
        initial = p0.copy()

        def unpack(v):
            p = initial.copy()
            p[active] = v
            return p

        def fun(v):
            y, jac, _, _ = law(unpack(v), train, topo, shape, ion)
            return (y - target) * w, jac[:, active] * w[:, None]

        cache = {}

        def ev(v):
            if "x" not in cache or not np.array_equal(v, cache["x"]):
                cache["x"] = v.copy()
                cache["f"] = fun(v)
            return cache["f"]

        res = least_squares(
            lambda v: ev(v)[0],
            initial[active],
            jac=lambda v: ev(v)[1],
            bounds=(lo[active], hi[active]),
            max_nfev=250,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        if res.status == 0:
            res = least_squares(
                lambda v: ev(v)[0],
                res.x,
                jac=lambda v: ev(v)[1],
                bounds=(lo[active], hi[active]),
                max_nfev=1250,
                ftol=2e-7,
                xtol=2e-7,
                gtol=2e-7,
            )
        if not res.success:
            # Flat, coupled ion channels can require more trust-region steps.
            res = least_squares(
                lambda v: ev(v)[0],
                res.x,
                jac=lambda v: ev(v)[1],
                bounds=(lo[active], hi[active]),
                max_nfev=5000,
                x_scale="jac",
                ftol=2e-7,
                xtol=2e-7,
                gtol=2e-7,
            )
        runs.append((res, unpack(res.x)))
    res, p = min(
        (r for r in runs if r[0].success),
        key=lambda r: r[0].cost,
        default=min(runs, key=lambda r: r[0].cost),
    )
    prefix = dict(
        id=idx,
        block=row["block"],
        split=split,
        topology=topo,
        shape=shape,
        sample="all_ions" if ion else "neutral",
    )
    stats = dict(
        **prefix,
        status=res.status,
        cost=res.cost,
        ntrain=int(sum(mask)),
        starts=len(runs),
        previous_cost=min(r.cost for r, _ in runs[:-1]),
        parameters=json.dumps(p.tolist()),
    )
    y, _, fb, fi = law(p, data[:, test], topo, shape, ion)
    predicted = np.exp(y)
    assert np.all(np.isfinite(predicted)) and np.all(
        fi * predicted <= Q[test] + 1e-8
    )
    out = [
        dict(
            **prefix,
            line=i + 2,
            predicted_J=est,
            binary_fraction=f,
            ion_fraction=fion,
            observed_J=J[i],
            adjusted_J=J[i] * row["jscale"],
        )
        for i, est, f, fion in zip(np.flatnonzero(test), predicted, fb, fi)
    ]
    scan = []
    if not ion and split in ("223_low", "complete"):
        for scenario, x, n in [
            (
                "archived",
                np.array([6.64, 13.05]) * row["acid"],
                0.06 * row["base0"],
            ),
            (
                "new_common",
                np.array([6.64, 13.05]) * (0.9 + 0.2 * row["u6"]),
                0.06 * (0.9 + 0.2 * row["u7"]),
            ),
            (
                "new_differential",
                np.array(
                    [
                        6.64 * (0.9 + 0.2 * row["u6"]),
                        13.05 * (0.9 + 0.2 * row["u8"]),
                    ]
                ),
                0.06 * (0.9 + 0.2 * row["u7"]),
            ),
        ]:
            d = np.array(
                [
                    np.full(2, -0.5),
                    np.log(x),
                    np.full(
                        2,
                        np.log(
                            n * 1.005 * 101325 / (1.380649e-23 * 223) * 1e-24
                        ),
                    ),
                ]
            )
            lp, _, _ = kernel(p, d, topo, shape)
            scan.append(
                dict(
                    **prefix,
                    scenario=scenario,
                    low_J=np.exp(lp[0]),
                    high_J=np.exp(lp[1]),
                    ratio=np.exp(lp[1] - lp[0]),
                )
            )
    return out, stats, scan


def run(phase, workers):
    if phase == "primary":
        specs = product(
            ("248_low", "223_low"), ("full", "assisted"), ["linear"], [False]
        )
    elif phase == "extension":
        specs = list(
            product(
                ("248_low", "223_low"),
                ("full", "assisted"),
                ["curved"],
                [False],
            )
        ) + list(
            product(
                (
                    "223_temperature",
                    "248_temperature",
                    "278_temperature",
                    "complete",
                ),
                ("full", "assisted"),
                ("linear", "curved"),
                [False],
            )
        )
    elif phase == "ions":
        specs = product(
            ("248_low", "223_low"),
            ("full", "assisted"),
            ["linear", "curved"],
            [True],
        )
    specs = list(specs)
    jobs = [(r["id"], *s) for s in specs for r in DRAWS]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(task, jobs, chunksize=2))
    for name, part in (("formation", 0), ("solutions", 1), ("scan", 2)):
        rows = [
            r for v in results for r in ([v[part]] if part == 1 else v[part])
        ]
        if rows:
            dump_csv(
                HERE
                / (name + ("" if phase == "primary" else "-" + phase) + ".csv"),
                rows,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["primary", "extension", "ions"])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    run(args.phase, args.workers)
