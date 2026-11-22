"""Export the calibration that best reproduces both fine-order replays."""

import json

import pandas as pd
from bank import OUT
from compare import load
from replay_chain import replay
from search import OBJECTIVES, save


def main():
    """Choose among the four refitted parameter sets at 1.875 seconds."""
    all_fits = load("continued.json")
    spec = load("mechanism.json")
    data = pd.read_csv(OUT / "sensitivity.csv")
    fine = data[data.test.isin(["1.875s_forward", "1.875s_reverse"])]
    spec["selection"] = {}
    for obj in OBJECTIVES:
        group = fine[fine.objective == obj]
        network = group.groupby("network").score.max().idxmin()
        row = group[group.network == network].set_index("test").score
        spec["fits"][obj] = all_fits[network][obj]
        spec["selection"][obj] = dict(
            network=network,
            seconds=1.875,
            forward=float(row["1.875s_forward"]),
            reverse=float(row["1.875s_reverse"]),
        )
    save("resolution_checked_mechanism.json", spec)
    print(
        json.dumps(replay(OUT / "resolution_checked_mechanism.json"), indent=2)
    )


if __name__ == "__main__":
    main()
