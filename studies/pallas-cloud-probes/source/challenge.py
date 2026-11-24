# %% Independent channels, interval bounds, alignment and sampling support
import itertools
import json
import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from study import (
    ROOT,
    COUNT,
    LWC,
    ED,
    WIND,
    FILES,
    inputs,
    channels,
    window,
    native,
    read_source,
)
from calibration import calibrate, scores


def verify_archives():
    checks = {}
    from hashlib import md5

    for probe, digest in [
        ("F", "44da7eede3b6de634b1f4f1df941d40c"),
        ("I", "6f602177b203219ad9c9342f1eaade2f"),
    ]:
        name = FILES[probe][0]
        csv = pd.read_csv(ROOT / "input" / name, encoding="latin1")
        path = ROOT / "input" / name.replace(".csv", ".nc")
        assert md5(path.read_bytes()).hexdigest() == digest
        with netcdf_file(path, mmap=False) as ds:
            v = ds.variables
            np.testing.assert_array_equal(
                v["conc_bin"][:], csv.filter(regex="^dN/").to_numpy()
            )
            coord = v["bin"][:].copy()
            np.testing.assert_array_equal(
                coord,
                np.r_[np.arange(3, 46, 1.5), 47.0]
                if probe == "F"
                else np.arange(5.5, 200),
            )
            checks[probe] = dict(
                channels_csv_exactly_equal_netcdf=True,
                coordinate=v["bin"].long_name.decode(),
                rows=len(csv),
            )
    return checks


def structural(f, i):
    rows = []
    ei, ni, _ = native(i, "I", "center")
    c_i = ni @ window(ei, 6, 45)
    reference = calibrate(f[f.index.day == 14])
    for conv in ["upper", "center"]:
        ef, n, _ = native(f, "F", conv)
        reassigned = channels(f) * np.diff(np.log10(ef))
        reassigned *= (
            f[COUNT].to_numpy()[:, None] / reassigned.sum(axis=1)[:, None]
        )
        for day in sorted(set(f.index.day)):
            idx = f.index.day == day
            ff = f.loc[idx]
            for rule, nn in [
                ("empirical", n[idx]),
                ("renormalized_geometric", reassigned[idx]),
            ]:
                water_min = nn @ (ef[:-1] ** 3) * (np.pi / 6 * 1e-6)
                water_max = nn @ (ef[1:] ** 3) * (np.pi / 6 * 1e-6)
                archived = ff[LWC].to_numpy()
                x = nn @ window(ef, 6, 45)
                rows.append(
                    dict(
                        day=day,
                        geometry=conv,
                        rule=rule,
                        minutes=len(nn),
                        gap=(x - c_i[idx]).mean(),
                        mae=abs(x - c_i[idx]).mean(),
                        water_min_median_ratio=np.median(water_min / archived),
                        fraction_below_min=np.mean(archived < water_min),
                        fraction_above_max=np.mean(archived > water_max),
                        fraction_below_stress_min=np.mean(
                            archived * 1.4 < water_min * 0.8**3 * 0.84
                        ),
                        linear_gap=(
                            nn @ window(ef, 6, 45, linear=True) - c_i[idx]
                        ).mean(),
                        native_max_error=np.max(
                            abs(nn.sum(axis=1) - ff[COUNT])
                        ),
                        **{
                            "loggrid_" + k: v
                            for k, v in scores(
                                n[idx], ff, reference[0][1]
                            ).items()
                        },
                    )
                )
    pd.DataFrame(rows).to_csv(ROOT / "structure.csv", index=False)


def timing(f, i):
    # Compare published pairing with the alternative weather-aligned pairing
    # on exactly the same FSSP indices. Shift I back 120 min -> I(t+120) at t.
    wide = read_source("I")
    rows = []
    for day in sorted(set(f.index.day)):
        base = f.index[f.index.day == day]
        future = base + pd.Timedelta(minutes=120)
        ok = future.isin(wide.index)
        base = base[ok]
        future = future[ok]
        ok = (
            (future.normalize() == base.normalize())
            & (wide.loc[future, "flag"].to_numpy() == 0)
            & (wide.loc[future, COUNT].to_numpy() > 0)
        )
        base = base[ok]
        future = future[ok]
        if len(base) < 5:
            continue
        for lag, ii in [(0, base), (-120, future)]:
            a, b = f.loc[base], wide.loc[ii]
            adjacent = np.diff(base.to_numpy()) == np.timedelta64(1, "m")
            for col in [
                ED,
                COUNT,
                WIND,
                "FD12P visibility (m)",
                "Temp 570m (C)",
            ]:
                x = a[col].to_numpy()
                y = b[col].to_numpy()

                def corr(x, y):
                    return (
                        np.corrcoef(x, y)[0, 1]
                        if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0
                        else np.nan
                    )

                rows.append(
                    dict(
                        day=day,
                        shift_I_minutes=lag,
                        field=col,
                        minutes=len(x),
                        adjacent=int(adjacent.sum()),
                        level_r=corr(x, y),
                        increment_r=corr(
                            np.diff(x)[adjacent], np.diff(y)[adjacent]
                        ),
                        matching_fraction=np.mean(x == y),
                    )
                )
    pd.DataFrame(rows).to_csv(ROOT / "clock.csv", index=False)


def summarize():
    d = pd.read_csv(ROOT / "dates.csv")
    rows = []
    for conv, df in d.groupby("geometry"):
        for group, ds in [
            ("held", df[df.day != 14]),
            ("held_at_least_60", df[(df.day != 14) & (df.minutes >= 60)]),
        ]:
            # Enumerate all ordered date-bootstrap replicates exactly (6^6).
            ids = np.asarray(
                list(itertools.product(range(len(ds)), repeat=len(ds)))
            )
            differences = (ds.gap_mae - ds.native_mae).to_numpy()
            limits = np.quantile(differences[ids].mean(axis=1), [0.025, 0.975])
            loo = (differences.sum() - differences) / (len(ds) - 1)
            rows.append(
                dict(
                    geometry=conv,
                    group=group,
                    dates=len(ds),
                    minutes=ds.minutes.sum(),
                    native_mae=ds.native_mae.mean(),
                    common_mae=ds.gap_mae.mean(),
                    date_mean_change=differences.mean(),
                    bootstrap_low=limits[0],
                    bootstrap_high=limits[1],
                    loo_low=loo.min(),
                    loo_high=loo.max(),
                    minute_mean_change=np.average(
                        differences, weights=ds.minutes
                    ),
                    native_gap=ds.native.mean(),
                    common_gap=ds.gap.mean(),
                    dates_improved=(differences < 0).sum(),
                )
            )
    pd.DataFrame(rows).to_csv(ROOT / "held.csv", index=False)


if __name__ == "__main__":
    f, i = inputs()
    checks = verify_archives()
    structural(f, i)
    timing(f, i)
    summarize()
    c = read_source("C")
    ix = c.index.intersection(f.index)
    ix = ix[
        (c.loc[ix, "flag"] == 0)
        & c.loc[ix, WIND].between(190, 250)
        & (c.loc[ix, COUNT] > 0)
    ]
    df = pd.DataFrame(
        dict(
            f_native=f.loc[ix, COUNT],
            i_native=i.loc[ix, COUNT],
            cas_native=c.loc[ix, COUNT],
        )
    )
    df.to_csv(ROOT / "third-probe.csv", index_label="utc")
    (ROOT / "source-check.json").write_text(json.dumps(checks, indent=2) + "\n")
