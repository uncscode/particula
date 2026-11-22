"""Separate time-step and particle-count effects in the reserved size check."""

import hashlib
import json

import pandas as pd
from size_replay import HERE, replay, summarize


def main():
    """Use one common seed and vary one numerical setting at a time."""
    source = (HERE / "size_inputs.json").read_bytes()
    checksum = hashlib.sha256(source).hexdigest()
    cases = json.loads(source)["cases"]
    rows = []
    # Compare with the saved 1,024/30 s and 2,048/15 s runs with seed 201.
    for particles, step_s in [(2048, 30), (4096, 15), (4096, 10)]:
        for case, parameters in cases.items():
            scores = summarize(replay(parameters, particles, step_s, 201))
            rows.append(
                dict(
                    case=case,
                    particles=particles,
                    step_s=step_s,
                    seed=201,
                    inputs_sha256=checksum,
                    **scores,
                )
            )
    pd.DataFrame(rows).to_csv(
        HERE / "size_results/resolution_check.csv", index=False
    )


if __name__ == "__main__":
    main()
