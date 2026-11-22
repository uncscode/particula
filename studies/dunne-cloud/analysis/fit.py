"""Bounded blind multistart fits to condition-blocked observations, with PyTorch AD."""

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import least_squares
import torch
from reference import (
    ACID,
    AMMONIA,
    BAND,
    J,
    NH3,
    OUT,
    Q,
    RH,
    SPLITS,
    T,
    WINDOW,
    channels,
    dump_csv,
)

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
FITDIR = OUT / "fits"
GROUP = np.stack(
    (BAND, np.rint(np.log10(NH3) / 0.15), np.rint(np.log10(1 + Q) / 0.35)),
    axis=1,
)


# Normalize temperature so linear and double-exponential slopes have similar scale.
def thermal(param, x, shape):
    offset, log_slope, curve = param[:, 0], param[:, 1], param[:, 2]
    z = torch.expm1(curve * x) / curve if shape == "curved" else x
    return offset - torch.exp(log_slope) * z


def log_rates(theta, data, topology, shape):
    x, a, n, fixed_ion = data.T
    j = 0
    if topology == "full":
        binary = thermal(theta, x, shape) + theta[:, 3] * a
        j = 4
    t = theta[:, j:]
    acid_power = t[:, 3] * a
    ternary = (
        thermal(t, x, shape)
        + n
        + acid_power
        - torch.logaddexp(t[:, 4], acid_power - t[:, 5] * n)
    )
    neutral = (
        torch.logaddexp(binary, ternary) if topology == "full" else ternary
    )
    # fixed_ion is a documented nuisance contribution in the joint sensitivity.
    total = torch.logaddexp(neutral, fixed_ion)
    return total, (
        torch.exp(binary - total)
        if topology == "full"
        else torch.zeros_like(neutral)
    )


def condition_weights(mask, weighting):
    _, inverse, count = np.unique(
        GROUP[mask], axis=0, return_inverse=True, return_counts=True
    )
    values = (
        1 / count[inverse] if weighting == "group" else np.ones(len(inverse))
    )
    return np.sqrt(values / np.mean(values))


def setup(split, calibration, sample, weighting):
    test = WINDOW & SPLITS[split] & (Q == 0)
    train = WINDOW & ~SPLITS[split]
    if sample == "neutral":
        train &= Q == 0
    acid = ACID.copy()
    base = AMMONIA.copy()
    if calibration.startswith("acid_"):
        acid *= 2 / 3 if calibration == "acid_low" else 1.5
    if calibration.startswith("base_"):
        # Conservative shared scenario for each concentration bracket, not row noise.
        factor = np.where(
            NH3 < 1,
            10.0,
            np.where(NH3 <= 10, 2.0, np.where(NH3 >= 100, 1.2, 1.5)),
        )
        base *= factor if calibration == "base_high" else 1 / factor
    if sample == "joint":
        ion = channels("S1_precise", acid=acid, ammonia=base)[2:].sum(axis=0)
    else:
        ion = np.zeros_like(T)
    data = np.stack(
        (
            (T - 248) / 50,
            np.log(acid),
            np.log(base),
            np.log(np.maximum(ion, 1e-300)),
        ),
        axis=1,
    )
    return test, train, torch.tensor(data), condition_weights(train, weighting)


def run(spec):
    split, topology, shape, calibration, sample, weighting, starts = spec
    key = "_".join(spec[:-1])
    path = FITDIR / (key + ".json")
    prior = json.loads(path.read_text()) if path.exists() else None
    if prior:
        finished = [r["cost"] for r in prior["runs"] if r["status"] > 0]
        if finished and min(finished) <= min(r["cost"] for r in prior["runs"]):
            return key
    test, train, data, weights = setup(split, calibration, sample, weighting)
    train_data = data[train]
    log_y = np.log(J[train])
    rng = np.random.default_rng(41582)
    bounds = np.array([[-100, 35], [-4, 4.5], [0.005, 4.0], [0.3, 9.0]])
    if topology == "full":
        bounds = np.vstack([bounds, bounds, [[-20, 8], [0.2, 12]]])
    else:
        bounds = np.vstack([bounds, [[-20, 8], [0.2, 12]]])
    # Neutral rates use effective parameters. The centering reduces the numerical
    # cancellation of the original u,v,w when refitting from scratch.
    runs = []
    if prior:
        grad_err = prior["gradient_error"]
    for i in range(starts):
        previous = prior["runs"][i] if prior else None
        if previous and (
            previous["status"] > 0 or previous["cost"] >= min(finished)
        ):
            runs.append(previous)
            continue
        t = np.array([-17.0, 1.7, 0.6, 3.0, -8.0, 3.0])
        p = np.r_[[-19.0, 2.0, 0.6, 3.0], t] if topology == "full" else t
        p = p + rng.normal(0, 1, len(p)) * np.resize(
            [4, 0.7, 0.45, 1.0, 2.0, 1.0], len(p)
        )
        p = np.clip(p, bounds[:, 0] + 1e-4, bounds[:, 1] - 1e-4)
        if shape == "linear":
            p[2] = 0.6
            if topology == "full":
                p[6] = 0.6
        cache = {}

        def evaluate(v):
            if "p" in cache and np.array_equal(cache["p"], v):
                return cache["r"], cache["jac"]
            per_row = torch.tensor(
                np.broadcast_to(v, (len(train_data), len(v))).copy(),
                requires_grad=True,
            )
            logj, _ = log_rates(per_row, train_data, topology, shape)
            jac = torch.autograd.grad(logj.sum(), per_row)[0].detach().numpy()
            residual = (logj.detach().numpy() - log_y) * weights
            cache.update(p=v.copy(), r=residual, jac=jac * weights[:, None])
            return cache["r"], cache["jac"]

        initial = (
            p.copy() if previous is None else np.array(previous["initial"])
        )
        if previous:
            p = np.array(previous["parameters"])
        if i == 0 and previous is None:
            direction = rng.normal(size=len(p))
            direction /= np.linalg.norm(direction)
            r, jac = evaluate(p)
            fd = (
                evaluate(p + 1e-5 * direction)[0]
                - evaluate(p - 1e-5 * direction)[0]
            ) / 2e-5
            grad_err = np.linalg.norm(fd - jac @ direction) / max(
                1e-12, np.linalg.norm(fd)
            )
            if not grad_err < 1e-6:
                raise ValueError(("gradient", grad_err))
        result = least_squares(
            lambda v: evaluate(v)[0],
            p,
            jac=lambda v: evaluate(v)[1],
            bounds=bounds.T,
            max_nfev=4000 if previous else 450,
            ftol=3e-8,
            xtol=3e-8,
            gtol=3e-8,
        )
        pred, fraction = log_rates(
            torch.tensor(result.x).expand(len(T), -1), data, topology, shape
        )
        runs.append(
            dict(
                initial=initial.tolist(),
                parameters=result.x.tolist(),
                cost=float(result.cost),
                status=int(result.status),
                nfev=int(result.nfev),
                prediction=np.exp(pred.detach().numpy()[test]).tolist(),
                binary_fraction=fraction.detach().numpy()[test].tolist(),
            )
        )
        if previous:
            runs[-1]["first_attempt"] = dict(
                status=previous["status"],
                cost=previous["cost"],
                nfev=previous["nfev"],
            )
    solution = dict(
        spec=list(spec),
        test_lines=(np.where(test)[0] + 2).tolist(),
        ntrain=int(train.sum()),
        gradient_error=grad_err,
        runs=runs,
    )
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(solution, separators=(",", ":")))
    tmp.replace(path)
    return key


def campaign(stage):
    if stage == "pilot":
        return [
            (s, t, "linear", "nominal", "neutral", "group", 4)
            for s in ["248_low", "223_low"]
            for t in ["full", "assisted"]
        ]
    specs = []
    for split in SPLITS:
        for topo, shape in itertools.product(
            ["full", "assisted"], ["linear", "curved"]
        ):
            for cal in (
                ["nominal", "base_low", "base_high", "acid_low", "acid_high"]
                if split.endswith("low")
                else ["nominal"]
            ):
                specs.append((split, topo, shape, cal, "neutral", "group", 8))
            if split.endswith("low"):
                for sample, weight in [("joint", "group"), ("neutral", "row")]:
                    specs.append(
                        (split, topo, shape, "nominal", sample, weight, 8)
                    )
    return specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "full"], default="full")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    FITDIR.mkdir(exist_ok=True)
    if args.stage == "pilot":
        FITDIR = OUT / "pilot"
        FITDIR.mkdir(exist_ok=True)
    jobs = campaign(args.stage)
    with ProcessPoolExecutor(args.workers) as pool:
        for i, done in enumerate(pool.map(run, jobs)):
            print(f"{i + 1}/{len(jobs)} {done}", flush=True)
