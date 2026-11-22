"""Reoptimize each accepted prefix with the same continuous-dose solver."""

import json

import numpy as np
from continuous import optimize
from finish import read, sized
from fit import OBJECTIVES, best, encode, save
from smoke import DOSES, OUT

from gpu import Model


def main():
    """Retain the previous best as an exact zero-channel alternative."""
    m = Model()
    prior = read("continuous_fits.json")
    current = {
        obj: min(
            (
                r["fit"]
                for r in prior
                if r["size"] == 0 and r["objective"] == obj
            ),
            key=lambda r: r["score"],
        )
        for obj in OBJECTIVES
    }
    history = []
    for f in sorted(OUT.glob("round_*.json")):
        row = json.loads(f.read_text())
        if not row["accepted"]:
            continue
        ids = row["active_indices"]
        h = ids[-1]
        for obj in OBJECTIVES:
            old = current[obj]
            raw = row["fits"][obj]
            p = np.array([old["p"], old["p"], raw["p"]])
            k = np.array([sized(old["k"]), sized(old["k"]), sized(raw["k"])])
            k[1, h] = raw["k"][h]
            output = m.run(p, k, ids, DOSES, obj, sub=2)
            i = best(output)
            st = encode(p[i], k[i], output, i)
            s, solver = optimize(m, obj, ids, st)
            current[obj] = s
            history.append(
                dict(
                    size=len(ids),
                    objective=obj,
                    key=row["key"],
                    fit=s,
                    solver=solver,
                )
            )
        save(OUT / "prefix_fits.json", history)
        print(
            len(ids),
            current["mass"]["score"],
            current["joint"]["score"],
            flush=True,
        )


if __name__ == "__main__":
    main()
