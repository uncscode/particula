"""Bounded joint perturbations of the Manchester particle-resolved refit."""

import json

import numpy as np
import pandas as pd
from refit import RESULTS, parameter_bounds, parameter_steps
from scipy.stats import qmc
from scoring import Targets
from simulate import Chamber

GROUPS = {
    "precursor_conversion": [*range(9), 12],
    "primary_rates": list(range(16, 36)),
    "aging_rates": list(range(36, 86)),
    "injected_doses": list(range(86, 101)),
    "seed_width": [101],
    "condensation_allocation": [102],
    "wall_eddy": [9],
    "nucleation": [103],
    "organic_kappa": [104, 105],
}
OUTPUT = RESULTS / "perturbations"


def design():
    """Draw independent coordinates over the truncated local bounds."""
    center = np.array(
        json.loads((RESULTS / "selected.json").read_text())["vector"]
    )
    params = pd.read_csv(RESULTS / "parameters.csv")
    lower, upper = parameter_bounds()
    steps = parameter_steps(center, upper[16:86])
    low = np.maximum(lower, center - steps)
    high = np.minimum(upper, center + steps)
    active = np.flatnonzero(steps > 0)
    grouped = sorted(i for group in GROUPS.values() for i in group)
    if grouped != active.tolist():
        raise ValueError("Group definition differs from active parameters")
    unit = qmc.LatinHypercube(d=len(active), seed=27182).random(48)
    vectors = np.tile(center, (len(unit), 1))
    vectors[:, active] = low[active] + unit * (high[active] - low[active])
    params.assign(low=low, high=high, step=steps).to_csv(
        OUTPUT / "ranges.csv", index=False
    )
    pd.DataFrame(vectors, columns=params.parameter).rename_axis("draw").to_csv(
        OUTPUT / "draws.csv"
    )
    return center, params.initial.to_numpy(), vectors


def replay(chamber, targets, cases, n, dt, seed, scores, curves):
    """Use common random numbers and the final instrument windows."""
    for first in range(0, len(cases), 4):
        batch = cases[first : first + 4]
        vectors = np.stack([entry[2] for entry in batch])
        values = chamber.run(
            vectors[:, :-2],
            chamber.ids,
            n=n,
            dt=dt,
            rng=seed,
            collision_sweeps=12,
        )
        history, dist, events, _, air = values
        count = len(chamber.ids)
        for i, (name, draw, _) in enumerate(batch):
            subset = slice(i * count, (i + 1) * count)
            one = (
                history[i : i + 1],
                dist[subset],
                events[subset],
                None,
                air[subset],
            )
            sc, records = targets.evaluate(
                one, vectors[i : i + 1], chamber.ids, details=True
            )
            label = dict(case=name, draw=draw, particles=n, dt_s=dt, seed=seed)
            spec = next(table for k, table in records if k == "psd")
            modes = pd.concat(table for k, table in records if k == "modes")
            sc["spectrum_rmse"] = np.sqrt(
                np.mean((spec.simulated - spec.dndlog10d_cm3) ** 2)
            )
            sc["mode_mae_nm"] = (
                (modes.predicted_nm - modes.mode_nm).abs().mean()
            )
            scores.append(
                sc.assign(max_pair_probability=events[subset, 4].max(), **label)
            )
            if name in {"joint", "center", "frozen"} and seed == 311:
                for key, table in records:
                    if key in {"surface", "psd", "runs", "water", "modes"}:
                        curves.setdefault(key, []).append(table.assign(**label))
        print(seed, n, first + len(batch), "/", len(cases), flush=True)


def main():
    """Compare joint draws, matched group controls and finer replays."""
    OUTPUT.mkdir(exist_ok=True)
    center, frozen, vectors = design()
    chamber = Chamber()
    targets = Targets(chamber, surface_window=True)
    controls = [("center", -1, center), ("frozen", -1, frozen)]
    joint = [("joint", i, vector) for i, vector in enumerate(vectors)]
    pairs = []
    for i, vector in enumerate(vectors[:12]):
        for name, indices in GROUPS.items():
            other = vector.copy()
            other[indices] = center[indices]
            pairs.append((name, i, other))
    scores, curves = [], {}
    for n, dt, seed, cases in [
        (768, 30, 311, controls + joint),
        (768, 30, 312, controls + joint),
        (1536, 15, 313, controls + joint[:12]),
        (768, 30, 311, pairs),
    ]:
        replay(chamber, targets, cases, n, dt, seed, scores, curves)
    result = pd.concat(scores, ignore_index=True)
    if not np.isfinite(result.select_dtypes("number")).all().all():
        raise ValueError("Nonfinite perturbation score")
    result.to_csv(OUTPUT / "scores.csv", index=False)
    for key, tables in curves.items():
        pd.concat(tables, ignore_index=True).to_csv(
            OUTPUT / f"{key}.csv.gz",
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )


if __name__ == "__main__":
    main()
