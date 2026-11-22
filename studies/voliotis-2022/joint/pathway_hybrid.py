# %%
"""Ask whether reactive uptake can coexist with a volatile-product branch."""

import json

import numpy as np
from pathway_refine import refine
from pathways import DOSES, OUT, SEEDS, Campaign, anchors, fit

HYBRID = {
    "diversion_uptake": (7, [12, 13, 14, 15]),
    "specified_uptake": (7, [12]),
}
# Declared illustrative values, independently prescribed for the positive test.
FIXED = [0.10, 0.003, 1.0]


def main():
    """Run a nested uptake model and a stress test prescribing active uptake."""
    c = Campaign()
    base, *_ = anchors()
    prior = json.loads((OUT / "partner_restarts.json").read_text())
    null = np.r_[base, [0.0, 1.0, 0.0, 0.0, 0.001, 1.0]]
    np.testing.assert_allclose(
        c.evaluate(null, True, "joint", DOSES, sub=2, pathway=7)[1],
        c.evaluate(base, True, "joint", DOSES, sub=2)[1],
        atol=1e-12,
    )
    history = []
    for obj, free in [("mass", False), ("joint", False), ("joint", True)]:
        for name, (_, extra) in HYBRID.items():
            starts = [
                np.array(r["physical"]).copy()
                for r in prior + history
                if r["model"] in ("volatile_diversion", *HYBRID)
            ]
            for p in starts:
                p[9] = p[9] if free else base[9]
                if name == "specified_uptake":
                    p[13:16] = FIXED
            active = list(range(9)) + ([9] if free else []) + extra
            seeds = np.unique(starts, axis=0)
            for seed in SEEDS:
                p, _ = fit(c, 7, active, obj, seeds, seed)
                v, s, status, nit = refine(c, p, 7, obj, free, extra)
                history.append(
                    dict(
                        model=name,
                        objective=obj,
                        wall_free=free,
                        seed=seed,
                        score=float(s),
                        physical=v.tolist(),
                        solver_status=int(status),
                        iterations=nit,
                    )
                )
                print(name, obj, free, seed, round(s, 5), flush=True)
            (OUT / "hybrid.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
