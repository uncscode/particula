"""Fit all admitted rates, shared chemistry and continuous run doses jointly."""

import json

import numpy as np
from finish import read, sized
from fit import OBJECTIVES, UPPER, encode, save
from scipy.optimize import minimize
from smoke import DOSES, OUT

from gpu import Model


def optimize(m, obj, ids, state):
    """Solve the minimax problem and reprofile on the shared dose grid."""
    p = np.asarray(state["p"])
    base = list(range(9)) + ([9] if obj == "joint" else []) + [12]
    sizes = len(base)
    low = np.r_[[1 / 12] * 3, [0] * 6, 2.818, 0, 0, 0, 0, 0, 0][base]
    high = np.r_[[4] * 3, [2] * 6, 16.5515, 0, 0, 100, 0, 0, 0][base]
    init = sized(state["k"])
    dose = np.array(state["dose"])

    def make(x):
        a = np.atleast_2d(x)
        par = np.tile(p, (len(a), 1))
        par[:, base] = low + a[:, :sizes] * (high - low)
        k = np.zeros((len(a), len(UPPER)))
        k[:, ids] = UPPER[ids] * np.expm1(4.0 * a[:, sizes:]) / np.expm1(4.0)
        return par, k

    x0 = np.r_[
        (p[base] - low) / (high - low),
        np.log1p(init[ids] / UPPER[ids] * np.expm1(4.0)) / 4.0,
    ]
    nc = len(x0)

    def errors(x, z):
        par, rates = make(x)
        d = (0.85 + 0.3 * z)[:, None]
        _, err, _, _ = m.run(par, rates, ids, d, obj, sub=2)
        return err[:, :, :2] if obj == "joint" else err[:, :, :1]

    def constraints(x):
        return x[-1] - errors(x[:nc], x[nc:-1])[0].ravel()

    def jac(x):
        active = x[:nc]
        z = x[nc:-1]
        # Forward or backward finite differences stay within normalized bounds.
        delta = np.where(active > 0.99995, -2e-5, 2e-5)
        batch = np.tile(active, (nc + 1, 1))
        batch[1:] += np.diag(delta)
        values = errors(batch, z)
        ref = values[0]
        grad = (values[1:] - ref).reshape(nc, -1).T / delta
        dz = np.where(z > 0.99995, -2e-5, 2e-5)
        dose_grad = (errors(active, z + dz)[0] - ref) / dz[:, None]
        run = np.repeat(np.arange(len(z)), ref.shape[1])
        block = np.zeros((len(run), len(z)))
        block[np.arange(len(run)), run] = dose_grad.ravel()
        return np.column_stack((-grad, -block, np.ones(len(run))))

    start = np.r_[x0, np.clip((dose - 0.85) / 0.3, 0.0, 1.0), state["score"]]
    history = []

    def record(x):
        if not history or x[-1] < history[-1][-1] - 1e-5:
            history.append(x.copy())

    result = minimize(
        lambda x: x[-1],
        start,
        jac=lambda x: np.r_[np.zeros(len(x) - 1), 1.0],
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (len(start) - 1) + [(0.0, 3.0)],
        constraints=[dict(type="ineq", fun=constraints, jac=jac)],
        callback=record,
        options=dict(ftol=2e-8, maxiter=65),
    )
    trials = np.vstack([start, *history[::3], result.x])
    par, k = make(trials[:, :nc])
    out = m.run(par, k, ids, DOSES, obj, sub=2)
    i = out[0].argmin()
    return encode(par[i], k[i], out, i), dict(
        status=int(result.status), iterations=int(result.nit)
    )


def main():
    """Restart the full mechanism from the best nested alternatives."""
    m = Model()
    polished = read("polish.json")
    paired = read("paired_controls.json")
    records = [
        json.loads(f.read_text()) for f in sorted(OUT.glob("round_*.json"))
    ]
    ids = records[-1]["active_indices"]
    results = []
    for n in (0, 20):
        active = ids[:n]
        for obj in OBJECTIVES:
            pool = [
                r["fit"]
                for r in polished
                if r["size"] == n and r["objective"] == obj
            ]
            if n:
                pool.extend(r["fit"] for r in paired if r["objective"] == obj)
                pool.extend(r["fits"][obj] for r in records)
                pool.extend(
                    r["fit"]
                    for r in results
                    if r["objective"] == obj and r["size"] == 0
                )
            # Two physically distinct starts; keep the expanded solution with
            # the other reaction coefficients set to zero as an exact control.
            native = min(pool, key=lambda r: r["score"])
            alt = min(
                (
                    r
                    for r in polished
                    if r["size"] == n and r["objective"] != obj
                ),
                key=lambda r: r["fit"]["score"],
            )["fit"]
            p = np.array(alt["p"])
            if obj == "mass":
                p[9] = native["p"][9]
            k = sized(alt["k"])
            o = m.run(p, k, active, DOSES, obj, sub=2)
            for j, st in enumerate((native, encode(p, k, o))):
                fit, solver = optimize(m, obj, active, st)
                results.append(
                    dict(size=n, objective=obj, start=j, fit=fit, solver=solver)
                )
                save(OUT / "continuous_fits.json", results)
                print("Continuous", n, obj, j, fit["score"], solver, flush=True)


if __name__ == "__main__":
    main()
