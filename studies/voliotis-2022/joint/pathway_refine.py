# %%
"""Polish nested minimax fits with explicit per-experiment error constraints."""

import json

import numpy as np
from pathway_partners import MODELS as PARTNERS
from pathways import (
    DOSES,
    LIMITS,
    MODELS,
    OUT,
    Campaign,
    anchors,
    physical,
    transformed,
)
from scipy.optimize import minimize

MODES = MODELS | {
    k: (v, [16, 17, 18] if v == 5 else [12]) for k, v in PARTNERS.items()
}


def refine(c, p, mode, obj, free, extra=None, limits=None):
    """Minimize the worst run error using batched finite differences."""
    if extra is None:
        extra = next(extra for m, extra in MODES.values() if m == mode)
    idx = list(range(9)) + ([9] if free else []) + extra
    lim = np.vstack((LIMITS, [(0, 2)] * 3)) if mode == 5 else LIMITS
    if limits is not None:
        lim = limits
    lo, hi = transformed(lim.T)[:, idx]
    original = transformed(p)[0]

    def unpack(x):
        z = np.tile(original, (len(np.atleast_2d(x)), 1))
        z[:, idx] = lo + np.atleast_2d(x) * (hi - lo)
        return physical(z)

    x0 = np.clip((original[idx] - lo) / (hi - lo), 0, 1)
    fine = DOSES[::3]

    def errors(x):
        err = c.evaluate(unpack(x), True, obj, fine, sub=1, pathway=mode)[1]
        return err.max(2) if obj == "joint" else err[:, :, 0]

    def constraints(x):
        return x[-1] - errors(x[:-1])[0]

    def derivative(x):
        n = len(x) - 1
        eps = np.where(x[:-1] > 0.9999, -2e-5, 2e-5)
        batch = np.tile(x[:-1], (n + 1, 1))
        batch[1:] += np.diag(eps)
        values = errors(batch)
        return np.column_stack(
            (-(values[1:] - values[0]).T / eps, np.ones(values.shape[1]))
        )

    start = np.r_[x0, errors(x0).max()]
    ans = minimize(
        lambda x: x[-1],
        start,
        jac=lambda x: np.r_[np.zeros(len(x) - 1), 1.0],
        method="SLSQP",
        bounds=[(0, 1)] * len(x0) + [(0, 3)],
        constraints=[dict(type="ineq", fun=constraints, jac=derivative)],
        options=dict(maxiter=48, ftol=1e-8),
    )
    candidates = np.vstack((p, unpack(ans.x[:-1])))
    scores = c.evaluate(candidates, True, obj, DOSES, sub=2, pathway=mode)[0]
    return candidates[scores.argmin()], scores.min(), ans.status, int(ans.nit)


def convert(source, model):
    """Transfer old coordinates, preserving each target's nested control."""
    p = np.array(source["physical"][:16])
    source_mode = MODES[source["model"]][0]
    if model == "partner_yields":
        return np.r_[p, source["physical"][16:] if source_mode == 5 else p[6:9]]
    if model == "volatile_diversion" and source_mode != 6:
        p[12] = 0.0
    return p


def main():
    """Cross-score chemistry across families before fitting each model."""
    records = sum(
        [
            json.loads((OUT / f).read_text())
            for f in ("restarts.json", "partner_restarts.json")
        ],
        [],
    )
    c = Campaign()
    base, *_ = anchors()
    refined = []
    for obj, free in [("mass", False), ("joint", False), ("joint", True)]:
        for model, (mode, _) in MODES.items():
            # Reuse solutions from the other models as candidate starts.
            pool = []
            records_now = records + refined
            for r in records_now:
                p = convert(r, model)
                if not free:
                    p[9] = base[9]
                pool.append(p)
            pool = np.unique(pool, axis=0)
            val = c.evaluate(pool, True, obj, DOSES, sub=2, pathway=mode)[0]
            own = [
                r
                for r in records
                if r["model"] == model
                and r["wall_free"] == free
                and r["objective"] == obj
            ]
            starts = [
                pool[val.argmin()],
                convert(min(own, key=lambda r: r["score"]), model),
            ]
            if mode in (1, 2, 3, 4, 6):
                nonzero = starts[0].copy()
                if mode in (1, 2, 4):
                    nonzero[10:13] = [0.06, 30.0, 2.0]
                if mode in (3, 4):
                    nonzero[13:16] = [0.06, 0.001, 1.0]
                if mode == 6:
                    nonzero[12] = 1.0
                starts.append(nonzero)
            fits = []
            for p in np.unique(starts, axis=0):
                vector, score, status, nit = refine(c, p, mode, obj, free)
                row = dict(
                    model=model,
                    objective=obj,
                    wall_free=free,
                    score=float(score),
                    physical=vector.tolist(),
                    solver_status=int(status),
                    iterations=nit,
                )
                fits.append(row)
            refined.extend(fits)
            print(
                model,
                obj,
                free,
                "refined",
                round(min(r["score"] for r in fits), 5),
                flush=True,
            )
            (OUT / "refinement.json").write_text(
                json.dumps(refined, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
