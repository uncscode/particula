"""Give the older reaction network an extra local fit for each round."""

import json
import time

from fit import OBJECTIVES, refine, save
from reactions import CANDIDATES
from smoke import OUT

from gpu import Model


def main():
    """Record matched-budget alternatives with the new channel disabled."""
    m = Model()
    rows = []
    ids = []
    for number in range(1, len(CANDIDATES) + 1):
        path = OUT / f"round_{number:02}.json"
        while not path.exists():
            time.sleep(2)
        rec = json.loads(path.read_text())
        prop = json.loads((OUT / f"proposal_{number:02}.json").read_text())
        if rec["accepted"]:
            for obj in OBJECTIVES:
                null = prop["null"][obj]
                null["k"] += [0.0] * (len(CANDIDATES) - len(null["k"]))
                fit = refine(m, obj, null, ids, 9673 + number)
                rows.append(
                    dict(
                        round=number,
                        objective=obj,
                        key=rec["key"],
                        null_score=fit["score"],
                        expanded_score=rec["fits"][obj]["score"],
                        fit=fit,
                    )
                )
            ids = rec["active_indices"]
            save(OUT / "paired_controls.json", rows)
            if len(ids) == 20:
                break


if __name__ == "__main__":
    main()
