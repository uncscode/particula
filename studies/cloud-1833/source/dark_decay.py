# %% Continue the held-out dark interval where the small-size instrument is absent.
import json
import time

import numpy as np
import pandas as pd
from study import OUT, inputs, simulate, write_json


def main():
    start = time.monotonic()
    data = inputs()
    t, d, e, y, gas, stage, missing = data
    idx = np.flatnonzero((stage >= 9) & (stage <= 18))
    dark = stage[idx] == 18
    lost = missing[idx]
    prev = (stage == 18) & ~missing
    source = y[prev, 3]
    summary = json.loads((OUT / "campaign.json").read_text())
    g, w = summary["common_growth"], summary["common_wall"]
    cs = [(f, g, w, k, 1) for f in [0, 1] for k in [0, 1]]
    result = []
    runs = []
    edges = 10**e
    width = np.maximum(0, e[1:] - np.maximum(e[:-1], np.log10(9.7)))
    for label, inlet in [
        ("zero", 0.0),
        ("last", source[-1]),
        ("early_dark_max", source.max()),
    ]:
        yp = y.copy()
        yp[missing, 3] = inlet
        pred, log = simulate((*data[:3], yp, *data[4:]), idx, cs, 192, 12)
        runs.append(log)
        for conf, p in zip(cs, pred):
            obs = y[idx] @ width
            sim = p @ width
            for i in np.flatnonzero(dark):
                result.append(
                    dict(
                        boundary=label,
                        inlet_if_missing=inlet,
                        forcing=conf[0],
                        law=conf[3],
                        archive_h=t[idx[i]],
                        no_small_data=bool(lost[i]),
                        N_ge9p7_observed=obs[i],
                        N_ge9p7_predicted=sim[i],
                    )
                )
    pd.DataFrame(result).to_csv(OUT / "dark_decay.csv", index=False)
    write_json(
        "dark_decay.json",
        dict(
            cases=12,
            runs=runs,
            seconds=time.monotonic() - start,
            unavailable=int(lost.sum()),
            comparison="9.7 nm to measured upper edge",
            upper_edge_nm=float(edges[-1]),
        ),
    )


if __name__ == "__main__":
    main()
