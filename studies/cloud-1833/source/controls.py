"""Follow-up controls for the rate at particle entry and flexibility of UVH fits."""

import itertools
import json
import time

import numpy as np
import pandas as pd
from study import OUT, score, simulate, write_json


def controls(data, training, test):
    start = time.monotonic()
    t, d, e, y, gas, stage, _ = data
    summary = json.loads((OUT / "campaign.json").read_text())
    g, w = summary["common_growth"], summary["common_wall"]
    configs = [
        (0, a, b, k, 1)
        for k, a, b in itertools.product(
            [0, 1],
            np.round(np.arange(0.7, 2.71, 0.1), 2),
            [0.15, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0],
        )
    ]
    # Bounded chunks keep the complete fitting screen small in GPU memory.
    rows = []
    logs = []
    for chunk in range(0, len(configs), 35):
        cs = configs[chunk : chunk + 35]
        pred, log = simulate(data, training, cs, 96, 24)
        rows.extend(
            [
                (*c, err)
                for c, err in zip(cs, score(pred[:, 1:], y[training[1:]], e))
            ]
        )
        logs.append(log)
    train = pd.DataFrame(
        rows, columns=["forcing", "growth", "wall", "law", "coag", "score"]
    )
    train.to_csv(OUT / "extended_training.csv", index=False)
    best = train.loc[train.groupby("law").score.idxmin()]
    testmask = stage[test] < 18
    testmask[0] = False
    pairs = [(f, g, w, l, 1) for f, l in itertools.product([0, 1], [0, 1])]
    free = [
        (f, row.growth, row.wall, int(row.law), 1)
        for f in (0, 1)
        for row in best.itertuples()
    ]
    narrow = [
        (0, row.growth, row.wall, int(row.law), 1)
        for row in train.itertuples()
        if row.score <= best.iloc[int(row.law)].score + 0.025
    ]
    rows = []
    result = {}
    for label, group, flux, bins, dt in [
        ("common_entry", pairs, g, 192, 12),
        ("common_entry_fine", pairs, g, 288, 6),
        ("free", free, None, 192, 12),
        ("free_fine", free, None, 288, 6),
        ("free_near", narrow, None, 192, 12),
    ]:
        pred, log = simulate(data, test, group, bins, dt, common_entry=flux)
        logs.append(log)
        for c, p in zip(group, pred):
            for st in [0, *range(9, 19)]:
                mask = testmask.copy() if st == 0 else stage[test] == st
                mask[0] = False
                obs = y[test[mask]]
                counts = p[mask, 4:] @ np.diff(e)[4:]
                measured = obs[:, 4:] @ np.diff(e)[4:]
                rows.append(
                    (
                        label,
                        *c,
                        st,
                        mask.sum(),
                        float(score(p[mask], obs, e)),
                        float(counts.mean()),
                        float(measured.mean()),
                    )
                )
        if label in ("common_entry", "free_fine"):
            result[label] = pred
    np.savez_compressed(OUT / "controls.npz", **result)
    pd.DataFrame(
        rows,
        columns=[
            "label",
            "forcing",
            "growth",
            "wall",
            "law",
            "coag",
            "stage",
            "n",
            "score",
            "count_pred",
            "count_obs",
        ],
    ).to_csv(OUT / "controls.csv", index=False)
    write_json(
        "controls.json",
        dict(
            seconds=time.monotonic() - start,
            cases=sum(v["cases"] for v in logs),
            runs=logs,
        ),
    )
