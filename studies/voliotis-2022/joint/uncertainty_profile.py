# %%
"""Refit shared campaign chemistry at evidence-anchored wall coefficients."""

import json

import numpy as np
import pandas as pd
from wall_campaign import HERE, OUT, Campaign, search

DEST = OUT.parent / "uncertainty"
NAMES = (
    "lag_pinene_h",
    "lag_cresol_h",
    "lag_isoprene_h",
    "own_pinene_yield",
    "own_cresol_yield",
    "own_isoprene_yield",
    "mixture_pinene_yield",
    "mixture_cresol_yield",
    "mixture_isoprene_yield",
    "wall_eddy_s",
)


def anchors():
    """Collect previously fitted chemistry and the observed wall contrasts."""
    formation = json.loads((OUT / "campaign_restarts.json").read_text())
    joint = [
        r
        for r in formation
        if r["model"] == "eddy_brownian" and r["objective"] == "joint"
    ]
    mass = [
        r
        for r in formation
        if r["model"] == "eddy_brownian" and r["objective"] == "mass"
    ]
    dark = json.loads((OUT / "dark_restarts.json").read_text())["campaign"]
    history = pd.read_csv(OUT / "conditional_profiles.csv").query(
        "model=='eddy_brownian'"
    )
    individual = history.loc[
        history.groupby("run").count_nrmse.idxmin(), "eddy_s"
    ]
    seed = json.loads((HERE / "wall_data/wall_fit.json").read_text())["fits"]
    base = np.array(min(joint, key=lambda r: r["score"])["physical"])
    levels = sorted(
        set(
            [
                individual.min(),
                base[-1],
                individual.max(),
                4.0,
                6.5,
                seed["all_readable_absolute"]["eddy_s-1"],
                seed["all_readable_log"]["eddy_s-1"],
                seed["isolated_log"]["eddy_s-1"],
            ]
        )
    )
    return base, levels, joint, dark, mass


def main():
    """Search the full kinetic bounds at each fixed wall coefficient."""
    DEST.mkdir(exist_ok=True)
    c = Campaign()
    base, levels, joint, dark, _ = anchors()
    records = []
    for e in sorted(levels, key=lambda value: abs(np.log(value / base[-1]))):
        if e == base[-1]:
            # Rescore the alternative free-wall restart at this fixed level.
            fits = [dict(r) for r in joint]
        elif e == dark[0]["physical"][-1] or np.isclose(
            e, dark[0]["physical"][-1], rtol=1e-12
        ):
            fits = [dict(r) for r in dark]
        else:
            start = min(
                [base] + [np.array(r["physical"]) for r in records],
                key=lambda x: abs(np.log(x[-1] / e)),
            )
            fits = search(c, True, "joint", start, fixed_eddy=e)
        for fit in fits:
            p = list(fit["physical"])
            p[-1] = e
            v = c.evaluate(
                p, True, "joint", np.linspace(0.85, 1.15, 61), sub=2
            )[0].item()
            records.append(
                dict(eddy=e, seed=fit["seed"], joint_score=v, physical=p)
            )
        (DEST / "profile_vectors.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        print("Completed wall level", e, flush=True)


if __name__ == "__main__":
    main()
