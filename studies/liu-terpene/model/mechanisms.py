# %%
"""Three one-parameter changes to a frozen cyclic fit, tested on late samples."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from chamber import CYCLIC, HERE, SCENARIOS, SIZE, perturb, predict
from scipy.optimize import minimize_scalar


def run(out):
    out.mkdir(exist_ok=True, parents=True)
    obs = pd.read_csv(HERE.parent / "sources" / "observations.csv")
    fits = pd.read_csv(out / "fits.csv")
    terms = {
        "constant_rate": (-6.0, 1.0),
        "low_volatility_share": (0.0, 0.999),
        "progressive_resistance": (0.0, 35.0),
    }
    scores = []
    for variant in [
        "primary",
        "low_accommodation",
        "humid_water",
        "opposed_reading_up",
        "opposed_reading_down",
        "shape_after30",
    ]:
        scenario = SCENARIOS[variant]
        for name in ["ocimene", "farnesene"]:
            cyclic = CYCLIC[name]
            for train_rh in ("dry", "wet"):

                def fraction(rh):
                    selected = fits[
                        (fits.model == variant)
                        & (fits.precursor == cyclic)
                        & (fits.rh == rh)
                    ]
                    return selected.sort_values(
                        "cstar_ug_m3"
                    ).fitted_fraction.to_numpy()

                def simulate(term, x, rh, t):
                    w = fraction(rh)
                    speed = 1.0
                    hardening = 0.0
                    if term == "constant_rate":
                        speed = np.exp(x)
                    if term == "low_volatility_share":
                        w = w * (1 - x)
                        w[0] += x
                    if term == "progressive_resistance":
                        hardening = x
                    return predict(
                        w,
                        name,
                        rh,
                        SIZE[name],
                        t,
                        scenario,
                        speed=speed,
                        hardening=hardening,
                    )

                train = perturb(
                    obs[
                        (obs.precursor == name)
                        & (obs.rh == train_rh)
                        & (obs.time_min <= 300)
                    ],
                    name,
                    scenario,
                )
                for term, bounds in terms.items():

                    def mse(x):
                        return np.mean(
                            (
                                simulate(term, x, train_rh, train.time_min)
                                - train.vfr
                            )
                            ** 2
                        )

                    opt = minimize_scalar(mse, bounds=bounds, method="bounded")
                    param = np.exp(opt.x) if term == "constant_rate" else opt.x
                    for test_rh in ("dry", "wet"):
                        held = obs[
                            (obs.precursor == name)
                            & (obs.rh == test_rh)
                            & (obs.time_min > 300)
                        ]
                        pred = simulate(term, opt.x, test_rh, held.time_min)
                        for t, y, p in zip(held.time_min, held.vfr, pred):
                            scores.append(
                                (
                                    variant,
                                    name,
                                    train_rh,
                                    test_rh,
                                    term,
                                    param,
                                    len(train),
                                    opt.fun**0.5,
                                    t,
                                    y,
                                    p,
                                    p - y,
                                )
                            )
    pd.DataFrame(
        scores,
        columns="model precursor train_rh test_rh term parameter early_points early_rmse time_min observed_vfr predicted_vfr signed_error".split(),
    ).to_csv(out / "mechanisms.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "results")
    run(p.parse_args().out)
