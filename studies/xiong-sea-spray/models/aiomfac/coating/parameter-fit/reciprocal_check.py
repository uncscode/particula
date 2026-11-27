"""Verify a missing predicted onset and quantify its actual humidity coverage."""

import numpy as np
import pandas as pd
from transfer import HERE
from trial import prepare
from boundary import state_at_rh
from stability import grid, onset


def main():
    rows = []
    for _, r in pd.read_csv(HERE / "reciprocal.csv").iterrows():
        m = prepare("NaCl", r["tail"], [r.B1, r.B2, r.C2])
        v = r.withheld_OVF
        for ref, cap in [(2, 160), (3, 400), (4, 400)]:
            g = grid("NaCl", v, 275.15, m, refine=ref, maxmol=cap)
            rh, mt, left = onset(g)
            _, d = state_at_rh(m, "NaCl", v)
            rows.append(
                [
                    r.calibration_OVF,
                    v,
                    ref,
                    cap,
                    rh,
                    d,
                    100 * np.exp(g.feed_mu[:, 0]).min(),
                    100 * np.exp(g.feed_mu[:, 0]).max(),
                ]
            )
    pd.DataFrame(
        rows,
        columns=[
            "fit_OVF",
            "test_OVF",
            "grid_refinement",
            "trial_max_m",
            "onset_RH",
            "tpd_at85",
            "min_feed_RH",
            "max_feed_RH",
        ],
    ).to_csv(HERE / "reciprocal-check.csv", index=False)


if __name__ == "__main__":
    main()
