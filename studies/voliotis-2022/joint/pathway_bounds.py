# %%
"""Probe the upper bound on isoprene-dependent volatile diversion."""

import json

import numpy as np
from pathway_refine import refine
from pathways import LIMITS, OUT, SEEDS, Campaign, anchors, fit


def main():
    """Repeat the best-found diversion fits with wider competition bounds."""
    c = Campaign()
    base, *_ = anchors()
    original = json.loads((OUT / "partner_restarts.json").read_text())
    rows = []
    for bound in [30.0, 100.0]:
        lim = LIMITS.copy()
        lim[12, 1] = bound
        for obj, free in [("mass", False), ("joint", True)]:
            starts = [
                np.array(r["physical"]).copy()
                for r in original + rows
                if r["model"] == "volatile_diversion"
            ]
            for p in starts:
                if not free:
                    p[9] = base[9]
            active = list(range(9)) + ([9] if free else []) + [12]
            for seed in SEEDS:
                p, _ = fit(
                    c, 6, active, obj, np.unique(starts, axis=0), seed, lim
                )
                p, score, status, nit = refine(c, p, 6, obj, free, [12], lim)
                rows.append(
                    dict(
                        model="volatile_diversion",
                        objective=obj,
                        wall_free=free,
                        upper_bound=bound,
                        seed=seed,
                        score=float(score),
                        physical=p.tolist(),
                        solver_status=int(status),
                        iterations=nit,
                    )
                )
                print(
                    bound,
                    obj,
                    seed,
                    round(score, 5),
                    round(p[12], 2),
                    flush=True,
                )
            (OUT / "wider_bounds.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
