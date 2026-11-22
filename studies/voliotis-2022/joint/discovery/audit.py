"""Verify saved rounds and profile rates in the final network."""

import json

import numpy as np
import pandas as pd
from fit import OBJECTIVES
from reactions import CANDIDATES
from smoke import DOSES, OUT

from gpu import Model


def main():
    """Rescore each snapshot and vary final rates one at a time."""
    m = Model()
    hand = []
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("round_*.json"))]
    for row in rows:
        for obj in OBJECTIVES:
            state = row["fits"][obj]
            k = np.pad(state["k"], (0, len(CANDIDATES) - len(state["k"])))
            args = (state["p"], k, row["active_indices"], DOSES, obj)
            original = m.run(*args, sub=2)[0][0]
            fine = m.run(*args, sub=4)[0][0]
            np.testing.assert_allclose(
                original, state["score"], atol=1e-10, rtol=0
            )
            hand.append(
                dict(
                    round=row["round"],
                    objective=obj,
                    archived=state["score"],
                    replayed=original,
                    fine=fine,
                )
            )
    pd.DataFrame(hand).to_csv(OUT / "round_replays.csv", index=False)
    spec = json.loads((OUT / "mechanism.json").read_text())
    cards = spec["reactions"]
    h = [
        next(i for i, r in enumerate(CANDIDATES) if r.key == c["key"])
        for c in cards
    ]
    work = []
    for obj in OBJECTIVES:
        par = spec["baseline_parameters"][obj]
        k = np.zeros(len(CANDIDATES))
        k[h] = [r[f"{obj}_fit_rate"] for r in cards]
        for j, (c, index) in enumerate(zip(cards, h, strict=True)):
            grid = np.unique(
                np.r_[
                    0.0,
                    k[index],
                    0.25 * k[index],
                    0.5 * k[index],
                    np.geomspace(0.001, 1.0, 8) * c["upper"],
                ]
            )
            r = np.tile(k, (len(grid), 1))
            r[:, index] = grid
            score, obs, _, _ = m.run(par, r, h, DOSES, obj, sub=4)
            for g, ss, ee in zip(grid, score, obs, strict=True):
                work.append(
                    dict(
                        objective=obj,
                        round=j + 1,
                        key=c["key"],
                        rate=g,
                        score=ss,
                        worst_mass=ee[:, 0].max(),
                        worst_count=ee[:, 1].max(),
                    )
                )
    pd.DataFrame(work).to_csv(OUT / "final_rate_profiles.csv", index=False)


if __name__ == "__main__":
    main()
