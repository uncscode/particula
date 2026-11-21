# %%
"""Check sensitivity to the processing of measured precursor decay."""

import json

import numpy as np
import pandas as pd
from pathway_hybrid import HYBRID
from pathway_refine import MODES
from pathways import DOSES, OUT, Campaign


def main():
    """Rescore selected chemistry on 10-, 20- and 40-minute precursor knots."""
    benchmark = pd.read_csv(OUT / "comparison.csv")
    cases = []
    for row in benchmark.itertuples():
        if row.objective == "mass" or row.wall_free:
            cases.append(
                (
                    row.model,
                    row.objective,
                    10.0,
                    np.array(json.loads(row.physical)),
                )
            )
    wide = pd.DataFrame(json.loads((OUT / "wider_bounds.json").read_text()))
    for r in wide.loc[
        wide.groupby(["objective", "upper_bound"]).score.idxmin()
    ].itertuples():
        cases.append(
            (r.model, r.objective, r.upper_bound, np.array(r.physical))
        )
    results = []
    for minutes in (10, 20, 40):
        c = Campaign(source_knot_seconds=minutes * 60)
        ids = [r[0] for r in c.runs]
        for model, obj, bound, p in cases:
            mode = (MODES | HYBRID)[model][0]
            _, error, _, out = c.evaluate(
                p, True, obj, DOSES, sub=4, pathway=mode, outcomes=True
            )
            results.append(
                dict(
                    knot_minutes=minutes,
                    model=model,
                    objective=obj,
                    upper_bound=bound,
                    strength=p[12] if mode in (2, 4, 6, 7) else np.nan,
                    eddy_s_inv=p[9],
                    worst_mass=error[0, :, 0].max(),
                    worst_count=error[0, :, 1].max(),
                    joint=error.max(),
                    mass_run=ids[error[0, :, 0].argmax()],
                    count_run=ids[error[0, :, 1].argmax()],
                    peak24=out[0, ids.index(24), 0],
                )
            )
    pd.DataFrame(results).to_csv(OUT / "forcing_and_bounds.csv", index=False)


if __name__ == "__main__":
    main()
