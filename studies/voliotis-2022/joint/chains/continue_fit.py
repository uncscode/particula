"""Check optimization headroom with one further matched start per network."""

import numpy as np
from bank import BANK, CAP
from compare import load
from continuous import optimize
from search import OBJECTIVES, fit_state, save

from gpu import Model


def main():
    """Extend each final fit and preserve the better nested calibration."""
    model = Model(BANK)
    stages = load("comparisons.json")
    for name, cases in stages.items():
        for obj in OBJECTIVES:
            entry = cases[obj]
            start = entry["fit"]
            active = entry["active"]
            # Bring improved narrower solutions forward at zero extra rates.
            narrower = [
                v[obj]["fit"]["x"]
                for v in stages.values()
                if set(v[obj]["active"]) < set(active)
            ]
            seed = fit_state(model, obj, [start["x"], *narrower])
            x = np.array(seed["x"])
            state = dict(
                p=x[:16], k=x[16:], score=seed["score"], dose=seed["dose"]
            )
            fit, log = optimize(
                model, obj, active, state, caps=CAP, sub=4, maxiter=60
            )
            best = fit_state(model, obj, [seed["x"], np.r_[fit["p"], fit["k"]]])
            entry["fit"] = best
            entry["solvers"].append(log)
            print(name, obj, best["score"], flush=True)
    save("continued.json", stages)


if __name__ == "__main__":
    main()
