# %% Empirical response by measured droplet size
import numpy as np
import pandas as pd
from study import ROOT, inputs, native, window

GRIDS = [[6.0, 45.0], [6.0, 13.07, 45.0], [6.0, 9.0, 13.07, 20.0, 30.0, 45.0]]


def run():
    f, i = inputs()
    ei, ni, _ = native(i, "I", "center")
    days = f.index.day.to_numpy()
    rows = []
    ratios = []
    for geom in ["upper", "center"]:
        ef, nf, _ = native(f, "F", geom)
        for cutoffs in GRIDS:
            intervals = list(zip(cutoffs[:-1], cutoffs[1:]))
            x = np.column_stack(
                [nf @ window(ef, low, high) for low, high in intervals]
            )
            y = np.column_stack(
                [ni @ window(ei, low, high) for low, high in intervals]
            )
            for test in sorted(set(days) - {14}):
                for fold, train in [
                    ("first_date", [14]),
                    ("other_dates", sorted(set(days) - {test})),
                ]:
                    # Equal-weight daily means prevent the longest date dominating calibration.
                    a = np.mean([x[days == t].mean(0) for t in train], axis=0)
                    b = np.mean([y[days == t].mean(0) for t in train], axis=0)
                    gain = b / a
                    sel = days == test
                    predicted = x[sel] * gain
                    observed = y[sel]
                    nominal = x[sel]
                    err = (predicted - observed).sum(1)
                    rows.append(
                        dict(
                            geometry=geom,
                            day=test,
                            minutes=sel.sum(),
                            fold=fold,
                            bands=len(intervals),
                            mae=abs(err).mean(),
                            bias=err.mean(),
                            raw_mae=abs((nominal - observed).sum(1)).mean(),
                            sum_abs_band_error=abs(predicted - observed)
                            .sum(1)
                            .mean(),
                            raw_sum_abs_band_error=abs(nominal - observed)
                            .sum(1)
                            .mean(),
                        )
                    )
                    for j, (low, high) in enumerate(intervals):
                        ratios.append(
                            dict(
                                geometry=geom,
                                day=test,
                                fold=fold,
                                bands=len(intervals),
                                low_um=low,
                                high_um=high,
                                train_f=a[j],
                                train_i=b[j],
                                gain=gain[j],
                                test_f=nominal[:, j].mean(),
                                test_i=observed[:, j].mean(),
                                test_ratio=observed[:, j].mean()
                                / nominal[:, j].mean(),
                            )
                        )
    pd.DataFrame(rows).to_csv(ROOT / "band-transfer.csv", index=False)
    pd.DataFrame(ratios).to_csv(ROOT / "band-support.csv", index=False)


if __name__ == "__main__":
    run()
