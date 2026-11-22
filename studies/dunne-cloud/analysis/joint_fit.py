"""Refit neutral and ion-induced channels jointly after removing the whole holdout."""

import json
import itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import least_squares
import torch
from fit import FITDIR, condition_weights, log_rates, setup, thermal
from reference import AIR, J, OUT, Q, SPLITS, T, WINDOW, dump_csv


def joint(spec):
    split, topo, shape = spec
    test, train, data, weights = setup(split, "nominal", "joint", "group")
    size = 10 if topo == "full" else 6
    # A fully refitted negative-ion balance replaces fixed nuisance predictions.
    q = torch.tensor(Q)
    alpha = torch.tensor(6e-8 * np.sqrt(300 / T) + 6e-26 * AIR * (300 / T) ** 4)
    data = data.clone()
    data[:, 3] = -690.0

    def total(v, data, q, alpha):
        neutral, f = log_rates(v[:, :size], data, topo, shape)
        ion, _ = log_rates(v[:, size:], data, "full", shape)
        log_b = torch.logaddexp(ion, torch.full_like(ion, -np.log(480.0)))
        lnq = torch.log(torch.clamp(q, min=1.0e-280))
        log_root = 0.5 * torch.logaddexp(2 * log_b, torch.log(4 * alpha) + lnq)
        lnni = np.log(2.0) + lnq - torch.logaddexp(log_b, log_root)
        ion_j = ion + lnni
        logj = torch.logaddexp(neutral, ion_j)
        return logj, f * torch.exp(neutral - logj), torch.exp(ion_j - logj)

    bounds4 = np.array([[-120, 40], [-4, 4.5], [0.005, 4], [0.3, 9.0]])
    bfull = np.vstack([bounds4, bounds4, [[-25, 8], [0.2, 12]]])
    bounds = np.vstack([bfull if topo == "full" else bfull[4:], bfull])
    start = json.loads(
        (
            FITDIR / f"{split}_{topo}_{shape}_nominal_neutral_group.json"
        ).read_text()
    )
    starters = sorted(start["runs"], key=lambda r: r["cost"])
    rng = np.random.default_rng(18763)
    results = []
    for k in range(12):
        v = np.r_[
            starters[k % len(starters)]["parameters"],
            [-18.0, 2.0, 0.4, 3.0, -29.0, 2.0, 0.6, 3.0, -5.0, 3.0]
            + rng.normal(0, [4, 0.7, 0.3, 1, 5, 0.6, 0.3, 1, 2, 1]),
        ]
        v = np.clip(v, bounds[:, 0] + 1e-7, bounds[:, 1] - 1e-7)
        cache = {}

        def evaluate(x):
            if "v" in cache and np.array_equal(x, cache["v"]):
                return cache["r"], cache["jac"]
            vr = torch.tensor(
                np.repeat(x[None, :], train.sum(), axis=0), requires_grad=True
            )
            pred, _, _ = total(vr, data[train], q[train], alpha[train])
            r = (pred.detach().numpy() - np.log(J[train])) * weights
            jac = (
                torch.autograd.grad(pred.sum(), vr)[0].detach().numpy()
                * weights[:, None]
            )
            cache.update(v=x.copy(), r=r, jac=jac)
            return r, jac

        if k == 0:
            D = rng.normal(size=len(v))
            D /= np.linalg.norm(D)
            d = evaluate(v)[1] @ D
            fd = (evaluate(v + 1e-5 * D)[0] - evaluate(v - 1e-5 * D)[0]) / 2e-5
            err = np.linalg.norm(d - fd) / np.linalg.norm(fd)
            assert err < 1e-6
        fit = least_squares(
            lambda v: evaluate(v)[0],
            v,
            jac=lambda v: evaluate(v)[1],
            bounds=bounds.T,
            max_nfev=500,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        logj, binary, ionf = total(
            torch.tensor(fit.x).expand(len(T), -1), data, q, alpha
        )
        jr = logj.exp().numpy()
        assert np.all(jr * ionf.numpy() <= Q + 1e-7)
        results.append(
            dict(
                cost=fit.cost,
                status=fit.status,
                params=fit.x.tolist(),
                neutral_predictions=jr[test].tolist(),
                binary_fraction=binary.numpy()[test].tolist(),
                all_removed_predictions=jr[WINDOW & SPLITS[split]].tolist(),
            )
        )
    out = dict(
        train=int(train.sum()),
        gradient_error=err,
        test_lines=(np.where(test)[0] + 2).tolist(),
        removed_lines=(np.where(WINDOW & SPLITS[split])[0] + 2).tolist(),
        runs=results,
    )
    (OUT / "joint" / ("_".join(spec) + ".json")).write_text(json.dumps(out))


if __name__ == "__main__":
    (OUT / "joint").mkdir(exist_ok=True)
    jobs = list(
        itertools.product(
            ["248_low", "223_low"], ["assisted", "full"], ["linear", "curved"]
        )
    )
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(joint, jobs))
