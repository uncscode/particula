# %%
"""Test whether mixed-VOC product formation depends on the reaction partner."""

import json

import numpy as np
from pathways import DEFAULT, DOSES, LIMITS, OUT, SEEDS, Campaign, anchors, fit

MODELS = {"partner_yields": 5, "volatile_diversion": 6}
PARTNER_NAMES = [
    "alpha_with_isoprene_g_g",
    "cresol_with_isoprene_g_g",
    "isoprene_with_cresol_g_g",
]


def main():
    """Fit nested alternatives initialized with matched reference solutions."""
    c = Campaign()
    base, *_ = anchors()
    primary = json.loads((OUT / "restarts.json").read_text())
    history = []
    original = c.evaluate(base, True, "joint", DOSES, sub=2)
    for model, mode in MODELS.items():
        x = np.r_[base, DEFAULT]
        if mode == 5:
            x = np.r_[x, base[6:9]]
        else:
            x[12] = 0.0
        null = c.evaluate(x, True, "joint", DOSES, sub=2, pathway=mode)
        np.testing.assert_allclose(null[1], original[1], atol=1e-12)
        limits = np.vstack((LIMITS, [(0, 2)] * 3)) if mode == 5 else LIMITS
        for obj, free in [("mass", False), ("joint", False), ("joint", True)]:
            starts = []
            for row in primary + history:
                if row["model"] not in ("reference", model):
                    continue
                p = np.array(row["physical"]).copy()
                if row["model"] == "reference":
                    p = np.r_[p, p[6:9]] if mode == 5 else p
                    p[12] = 0.0
                if not free:
                    p[9] = base[9]
                starts.append(p)
            active = (
                list(range(9))
                + ([9] if free else [])
                + ([16, 17, 18] if mode == 5 else [12])
            )
            for seed in SEEDS:
                p, v = fit(
                    c,
                    mode,
                    active,
                    obj,
                    np.unique(starts, axis=0),
                    seed,
                    limits,
                )
                history.append(
                    dict(
                        model=model,
                        objective=obj,
                        wall_free=free,
                        seed=seed,
                        score=float(v),
                        physical=p.tolist(),
                    )
                )
                print(model, obj, free, seed, round(v, 5), flush=True)
            (OUT / "partner_restarts.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
