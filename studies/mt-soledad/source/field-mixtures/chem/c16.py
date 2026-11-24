"""Upper alkyl-carrier mass compatible with the independent filter constraints."""

import pandas as pd
import numpy as np
from scipy.optimize import linprog
from recipe import HERE, GROUPS, MATRIX


def bounds():
    f = pd.read_csv(HERE / "filter-fits.csv").query("recipe=='low'").copy()
    result = []
    for _, a in f.iterrows():
        y = a[["obs_" + g for g in GROUPS]].to_numpy(float)
        sd = a[["sd_" + g for g in GROUPS]].to_numpy(float)

        def maximum(z):
            lo = np.maximum(0, y - z * sd)
            upper = np.where(y > 0, y + z * sd, 3 * sd)
            fit = linprog(
                [0, 0, 0, 0, -1],
                A_ub=np.r_[MATRIX, -MATRIX, np.ones((1, 5))],
                b_ub=np.r_[upper, -lo, max(0, a.ams_om_mean - a.amine)],
                bounds=(0, None),
            )
            return -fit.fun / a.ams_om_mean if fit.success else np.nan

        result.append((a.filter_id, a["mask"], maximum(2), maximum(3)))
    pd.DataFrame(
        result,
        columns=[
            "filter_id",
            "mask",
            "c16_upper_fraction_2sd",
            "c16_upper_fraction_3sd",
        ],
    ).to_csv(HERE / "film-bounds.csv", index=False)


if __name__ == "__main__":
    bounds()
