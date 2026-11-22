"""Replay all 40 admitted motif trials with explicitly nonzero rate probes."""

import numpy as np
import pandas as pd
from bank import BANK, OUT, TRIALS
from compare import load
from search import OBJECTIVES, evaluate

from gpu import Model


def main():
    """Hold each entering fit fixed and scan all steps in its proposed motif."""
    model = Model(BANK)
    rows = []
    audited = []
    for n, motif in enumerate(TRIALS, 1):
        rec = load(f"iteration_{n:02}.json")
        for obj in OBJECTIVES:
            kept = rec["fits"][obj]
            dose_kept = np.array(kept["dose"])[:, None]
            result = evaluate(model, obj, kept["x"], dose_kept)
            np.testing.assert_allclose(result[0][0], kept["score"], atol=1e-10)
            audited.append(
                dict(
                    iteration=n,
                    objective=obj,
                    archived=kept["score"],
                    replayed=result[0][0],
                )
            )
            prev = rec["before"][obj]
            dose = np.array(prev["dose"])[:, None]
            levels = np.array([0.0, 0.03, 0.3, 1.5])
            x = np.tile(prev["x"], (len(levels), 1))
            x[:, np.array(motif["ids"]) + 16] = levels[:, None]
            score, _, _, flux = evaluate(model, obj, x, dose, flux=True)
            for i, k in enumerate(levels):
                rows.append(
                    dict(
                        iteration=n,
                        objective=obj,
                        family=motif["family"],
                        motif=motif["motif"],
                        rate_h_inv=k,
                        score=score[i],
                        max_gross_last_step_ug_m3=flux[
                            i, :, motif["ids"][-1]
                        ].max(),
                    )
                )
    pd.DataFrame(audited).to_csv(OUT / "round_replays.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "motif_profiles.csv", index=False)


if __name__ == "__main__":
    main()
