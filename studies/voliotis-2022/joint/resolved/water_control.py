"""Compare size-selected and bulk water response on identical particle paths."""

import json

import numpy as np
import pandas as pd
from refit import RESULTS
from scoring import read_water
from simulate import GRID, Chamber


def main():
    """Replay both water runs and compare the same measured points."""
    c = Chamber()
    v = np.array(json.loads((RESULTS / "selected.json").read_text())["vector"])
    ids = [6, 15]
    rows = []
    time = np.arange(61) * 300
    for seed in (201, 202):
        h, d, _, _, _ = c.run(v[None, :-2], ids, n=2048, dt=15, rng=seed)
        water = read_water()
        for j, run in enumerate(ids):
            s = h[0, j]
            for row in water[water.run == run].itertuples():
                if row.channel == "CCN":
                    keep = (
                        np.abs(
                            np.log(
                                2 * 1.12 ** (np.arange(60) + 0.5) / row.diameter
                            )
                        )
                        < 0.12
                    )
                    count = d[j, :, :60][:, keep].sum(axis=1)
                    organics = d[j, :, 60:][:, keep].sum(axis=1)
                    ko = v[-1]
                else:
                    k = int(np.flatnonzero(GRID == row.diameter)[0])
                    count = s[:, 4 + k]
                    organics = s[:, 7 + k]
                    ko = v[-2]
                number = np.interp(row.seconds, time, count)
                if number < 3:
                    continue
                phi = np.interp(row.seconds, time, organics) / number
                oa = np.interp(row.seconds, time, s[:, 0]) / 1400
                sulfate = np.interp(row.seconds, time, s[:, 1]) / 1770
                mixed = oa / (oa + sulfate)
                rows.append(
                    dict(
                        seed=seed,
                        run=run,
                        channel=row.channel,
                        seconds=row.seconds,
                        kappa=row.observed,
                        diameter_nm=row.diameter,
                        number=number,
                        selected_volume_fraction=phi,
                        bulk_volume_fraction=mixed,
                        selected=(1 - phi) * row.seed_k + phi * ko,
                        bulk=(1 - mixed) * row.seed_k + mixed * ko,
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "water_control.csv", index=False)
    summary = []
    for channel, group in frame.groupby("channel"):
        summary.append(
            dict(
                channel=channel,
                n_observations=len(group),
                selected_rmse=np.sqrt(
                    np.mean((group.selected - group.kappa) ** 2)
                ),
                bulk_rmse=np.sqrt(np.mean((group.bulk - group.kappa) ** 2)),
            )
        )
    pd.DataFrame(summary).to_csv(
        RESULTS / "water_control_scores.csv", index=False
    )


if __name__ == "__main__":
    main()
