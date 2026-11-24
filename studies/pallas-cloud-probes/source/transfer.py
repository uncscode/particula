# %% Best achievable two-date transfer for a scalar counting efficiency
import numpy as np
import pandas as pd
from study import ROOT, inputs, native, window, COUNT, FILES, LWC, ED
from scipy.io import netcdf_file


def run():
    f, i = inputs()
    sizes = pd.read_csv(ROOT / "sizing.csv")
    ei, ni, _ = native(i, "I", "center")
    out = []
    cutoff = []
    for conv in ["upper", "center"]:
        ef, nf, _ = native(f, "F", conv)
        # Direct CPU integrals for the surviving fraction across cutoffs.
        for day in [14, 15, 16, 17, 18, 22, 23]:
            sel = f.index.day == day
            x, y = nf[sel].mean(0), ni[sel].mean(0)
            for lo in [6.0, 8.0, 10.0, 12.0, 13.07, 15.0, 20.0, 25.0, 30.0]:
                a = x @ window(ef, lo, 45)
                b = y @ window(ei, lo, 45)
                cutoff.append(
                    dict(
                        geometry=conv,
                        day=day,
                        lower_um=lo,
                        f_count=a,
                        i_count=b,
                        gap=a - b,
                        i_retained_fraction=b / (y @ window(ei, 6, 45)),
                    )
                )
        # For two positive mean counts F1,F2 and targets I1,I2 the best
        # nonnegative common gain in maximum absolute error is (I1+I2)/(F1+F2).
        # Add a deliberately symmetric ±one ICEMET channel stress to its diameter labels.
        for idelta in [-1.0, 0.0, 1.0]:
            targets = [
                ni[i.index.day == day].mean(0) @ window(ei + idelta, 6, 45)
                for day in (15, 16)
            ]
            for gridstep in [4, 2, 1]:
                a = sizes[(sizes.geometry == conv) & (sizes.day == 15)].iloc[
                    ::gridstep
                ]
                b = sizes[(sizes.geometry == conv) & (sizes.day == 16)].iloc[
                    ::gridstep
                ]
                np.testing.assert_array_equal(a.scale, b.scale)
                x, y = a.f_common.to_numpy(), b.f_common.to_numpy()
                gain = sum(targets) / (x + y)
                miss = np.maximum(
                    abs(gain * x - targets[0]), abs(gain * y - targets[1])
                )
                k = miss.argmin()
                out.append(
                    dict(
                        geometry=conv,
                        imager_offset_um=idelta,
                        size_grid_points=len(x),
                        minimax_cm3=miss[k],
                        best_size=a.scale.iloc[k],
                        best_gain=gain[k],
                        day15_gain_min=min(targets[0] / x),
                        day15_gain_max=max(targets[0] / x),
                        day16_gain_min=min(targets[1] / y),
                        day16_gain_max=max(targets[1] / y),
                    )
                )
    pd.DataFrame(out).to_csv(ROOT / "transfer.csv", index=False)
    pd.DataFrame(cutoff).to_csv(ROOT / "cutoff.csv", index=False)
    # Cross-format scalar fields as well as the spectral arrays.
    for probe in ["F", "I"]:
        name = FILES[probe][0]
        df = pd.read_csv(ROOT / "input" / name, encoding="latin1")
        with netcdf_file(
            ROOT / "input" / name.replace(".csv", ".nc"), mmap=False
        ) as ds:
            for key, col in [
                ("Nd", COUNT),
                ("LWC", LWC),
                ("ED", ED),
                ("MVD", "median volume diameter (µm)"),
            ]:
                np.testing.assert_allclose(
                    ds.variables[key][:],
                    df[col],
                    atol=1e-10,
                    rtol=0,
                    equal_nan=True,
                )


if __name__ == "__main__":
    run()
