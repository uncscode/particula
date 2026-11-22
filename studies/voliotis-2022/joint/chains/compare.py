"""Compare independently refitted submechanisms and the full product network."""

import csv
import json

import numpy as np
from bank import BANK, CAP, OUT, TRIALS
from continuous import optimize
from search import OBJECTIVES, fit_state, save

from gpu import Model


def load(name):
    """Read one archived round or fitted-network comparison."""
    return json.loads((OUT / name).read_text())


def compare():
    """Give each subnetwork two bounded optimization starts and retain nulls."""
    m = Model(BANK)
    base = load("baseline.json")
    records = [load(f"iteration_{n:02}.json") for n in range(1, 41)]
    evidence = list(
        csv.DictReader((OUT.parents[1] / "chains/evidence.csv").open())
    )
    supported = [
        t
        for t, e in zip(TRIALS, evidence, strict=True)
        if e["evidence_grade"] == "family"
    ]
    alpha = [t for t in TRIALS if t["family"] == "AA"]
    networks = {
        "backbone": [],
        "alpha_dimer": alpha,
        "family_evidence": supported,
        "all": TRIALS,
    }
    summary = {}
    for name, motifs in networks.items():
        active = sorted(set(range(20)).union(*(t["ids"] for t in motifs)))
        disabled = sorted(set(range(len(BANK))) - set(active))
        summary[name] = {}
        for obj in OBJECTIVES:
            starts = [r["fits"][obj]["x"] for r in records] + [
                records[-1]["static"][obj]["x"]
            ]
            starts += [v[obj]["fit"]["x"] for v in summary.values() if obj in v]
            x = np.array(starts)
            x[:, np.array(disabled, dtype=int) + 16] = 0
            a = fit_state(m, obj, x)
            parent = (
                summary.get("backbone", {}).get(obj, {}).get("fit", base[obj])
            )
            warm = np.array(parent["x"])
            last = np.array(a["x"])
            extra = np.array(active[20:], dtype=int) + 16
            warm[extra] = np.maximum(last[extra], 0.10)
            b = fit_state(m, obj, warm)
            best = a
            logs = []
            for st in (a, b):
                x = np.array(st["x"])
                state = dict(
                    p=x[:16], k=x[16:], score=st["score"], dose=st["dose"]
                )
                fit, log = optimize(
                    m, obj, active, state, caps=CAP, sub=4, maxiter=45
                )
                logs.append(log)
                test = fit_state(m, obj, np.r_[fit["p"], fit["k"]])
                if test["score"] < best["score"]:
                    best = test
            summary[name][obj] = dict(active=active, fit=best, solvers=logs)
            print(name, obj, best["score"], flush=True)
            save("comparisons.json", summary)


if __name__ == "__main__":
    compare()
