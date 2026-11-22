"""Isolate the shape of the gas-phase acid forcing at matched training exposure."""

from balance import write
from sectional import simulate
from replay import measurements
import numpy as np


def main():
    t, *_, obs, _, _, _ = measurements()
    h = t - t[0]
    rows = []
    for scale in (1.15, 1.25):
        for name in ["effective", "gas_equal_training_mean"]:
            y = simulate(name, scale, 0.0, 0.0, True, 140, len(t), 15)[:, 1]
            for key, m in [
                ("train", h < 3.8),
                ("peak", (h >= 3.8) & (h <= 6.0001)),
                ("transfer", h >= 6.36),
            ]:
                rows.append(
                    dict(
                        model=name,
                        gas=scale,
                        period=key,
                        rmse=np.mean((y[m] - obs[m, 1]) ** 2) ** 0.5,
                        peak=y[m].max(),
                    )
                )
    write("matched_acid.csv", rows)


if __name__ == "__main__":
    main()
