# %% Isolate finite molecular collision geometry from van der Waals capture
import json
import time

import numpy as np
import pandas as pd
from study import OUT, inputs, score, simulate, write_json


def main():
    start = time.monotonic()
    data = inputs()
    t, d, e, y, gas, stage, missing = data
    cs = []
    train = np.flatnonzero(stage <= 7)
    test = np.flatnonzero((stage >= 9) & (stage <= 18) & ~missing)
    sel = stage[test] < 18
    sel[0] = False
    base = json.loads((OUT / "campaign.json").read_text())
    g, w = base["common_growth"], base["common_wall"]
    rows = []
    logs = []
    saved = {}
    cases = [(f, g, w, l, 1) for f in [0, 1] for l in [0, 2, 1]]
    for label, entry, n, dt in [
        ("density", None, 288, 6),
        ("fixed_flux", g, 288, 6),
    ]:
        pred, log = simulate(data, test, cases, n, dt, common_entry=entry)
        saved[label] = pred
        logs.append(log)
        for c, p in zip(cases, pred):
            for s in [0, *range(9, 19)]:
                valid = sel.copy() if s == 0 else stage[test] == s
                valid[0] = False
                o = y[test[valid]]
                vp = p[valid]
                num = vp[:, 4:] * np.diff(e)[4:]
                observ = o[:, 4:] * np.diff(e)[4:]
                tv = 0.5 * np.sum(
                    abs(
                        num / num.sum(1)[:, None]
                        - observ / observ.sum(1)[:, None]
                    ),
                    axis=1,
                )
                rows.append(
                    (
                        label,
                        c[0],
                        c[3],
                        s,
                        score(vp, o, e),
                        tv.mean(),
                        num.sum(1).mean(),
                        observ.sum(1).mean(),
                        np.median(d[4:][np.argmax(vp[:, 4:], axis=1)]),
                    )
                )
    pd.DataFrame(
        rows,
        columns=[
            "entry",
            "forcing",
            "law",
            "stage",
            "score",
            "shape_tv",
            "pred_count",
            "obs_count",
            "pred_mode",
        ],
    ).to_csv(OUT / "mechanism.csv", index=False)
    np.savez_compressed(OUT / "mechanism.npz", **saved)
    write_json(
        "mechanism.json",
        dict(
            cases=sum(i["cases"] for i in logs),
            runs=logs,
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    main()
