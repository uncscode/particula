# %% Recover the effective size grid using archived moments on 14 September only.
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from study import ROOT, LWC, ED, channels, inputs

MVD = "median volume diameter (µm)"


def grid_from_water(n, water, spacing):
    template = 10 ** (spacing * np.arange(n.shape[1]))
    moment = n @ (template**3) * (np.pi / 6 * 1e-6)
    scale = np.cbrt(water / moment)
    return template * np.median(scale)


def scores(n, df, d):
    water = n @ d**3 * (np.pi / 6 * 1e-6)
    eff = (n @ d**3) / (n @ d**2)
    cumulative = np.cumsum(n * d**3, axis=1)
    idx = (cumulative < cumulative[:, -1, None] / 2).sum(axis=1)
    mvd = df[MVD].to_numpy()
    return dict(
        lwc_med=np.median(water / df[LWC]),
        lwc_p95_error=np.quantile(abs(water / df[LWC] - 1), 0.95),
        ed_med=np.median(eff / df[ED]),
        ed_p95_error=np.quantile(abs(eff / df[ED] - 1), 0.95),
        mvd_median_abs_um=np.median(abs(d[idx] - mvd)),
        mvd_inside_fraction=np.mean(
            (mvd >= d[idx] / np.sqrt(d[1] / d[0]))
            & (mvd <= d[idx] * np.sqrt(d[1] / d[0]))
        ),
    )


def calibrate(df):
    n = channels(df) * 0.05
    water = df[LWC].to_numpy()

    def objective(spacing):
        d = grid_from_water(n, water, spacing)
        pred = n @ d**3 * (np.pi / 6 * 1e-6)
        return np.median(abs(np.log(pred / water)))

    opt = minimize_scalar(
        objective,
        bounds=(0.015, 0.08),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return [
        (spacing, grid_from_water(n, water, spacing))
        for spacing in [0.05, opt.x]
    ]


if __name__ == "__main__":
    f, _ = inputs()
    df = f.loc[f.index.day == 14]
    result = []
    for spacing, d in calibrate(df):
        row = dict(
            spacing=spacing,
            first=d[0],
            last=d[-1],
            **scores(channels(df) * 0.05, df, d),
        )
        result.append(row)
    pd.DataFrame(result).to_csv(ROOT / "calibration.csv", index=False)
    print(pd.DataFrame(result).to_string(index=False))
