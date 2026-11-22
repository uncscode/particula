"""Numerical and clearing-field transfer checks for the training-selected models."""

import numpy as np
from balance import HERE, write
from sectional import simulate
from replay import measurements


def main():
    data = np.genfromtxt(
        HERE / "sectional_scores.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf8",
    )
    t, *_, obs, _, _, _ = measurements()
    h = t - t[0]
    rows = []
    for name in ["parent", "ozone", "peroxide", "gas_oh"]:
        s = next(r for r in data if r["family"] == name)
        model = name if name == "gas_oh" else "effective"
        for bins, step, field in [
            (140, 15, False),
            (140, 15, True),
            (220, 10, False),
        ]:
            n = len(t) if bins == 140 else 181
            curve = simulate(
                model,
                s["gas"],
                s["a"],
                s["b"],
                True,
                bins,
                n,
                step,
                clear=field,
            )[:, 1]
            for block, m in [
                ("train", h[:n] < 3.8),
                ("peak", (h[:n] >= 3.8) & (h[:n] <= 6.0001)),
                ("late", h[:n] >= 6.36),
                ("after11", h[:n] >= 11),
            ]:
                if not m.any():
                    continue
                rows.append(
                    dict(
                        family=name,
                        bins=bins,
                        max_step=step,
                        precharge_removal=field,
                        period=block,
                        rmse=np.mean((curve[m] - obs[:n, 1][m]) ** 2) ** 0.5,
                        peak=curve[m].max(),
                    )
                )
    write("refinement.csv", rows)


if __name__ == "__main__":
    main()
