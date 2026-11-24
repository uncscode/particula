# %% Compare every counterfactual using the identical number in each true-diameter cell.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from study import ROOT as RUN_DIR, inputs, edges, channels, weights, window, COUNT


def measurements():
    f, i = inputs()
    c = pd.read_csv(
        RUN_DIR / "input/FMI.CDA.b1.20220914.csv", encoding="latin1"
    )
    dt = c.iloc[:, :5].copy()
    dt.columns = ["year", "month", "day", "hour", "minute"]
    c.index = pd.to_datetime(dt)
    c = c.loc[~c.index.duplicated(keep=False)]
    # Reuse the independently specified (primary-data) author upper edges.
    ec = np.array(
        [
            0.15,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.9,
            1.1,
            1.3,
            1.5,
            1.8,
            2.1,
            2.4,
            2.8,
            3.3,
            3.8,
            4.5,
            5.2,
            6,
            7,
            8.1,
            9.4,
            10.9,
            12.6,
            14.6,
            16.9,
            19.5,
            22.6,
            26.1,
            30.1,
            34.8,
            40.3,
            46.5,
            53.7,
            62.1,
            71.8,
            82.9,
            95.8,
        ]
    )
    ef = edges("F", "upper")
    ei = edges("I", "center")
    records = []
    for day in [14, 15, 16]:
        ix = f.index[f.index.day == day].intersection(c.index)
        ix = ix[c.loc[ix, COUNT] > 0]
        counts = {}
        for name, e, n in [
            ("CDA", ec, channels(c.loc[ix]) * np.diff(np.log10(ec))),
            ("FSSP", ef, channels(f.loc[ix]) * weights("F", ef)),
            ("ICEMET", ei, channels(i.loc[ix]) * weights("I", ei)),
        ]:
            counts[name] = (e, n.mean(axis=0))
        records.append((day, len(ix), counts))
    return records


def map_to_grid(e, n, g):
    # Constant dN/dlogD inside each measured source channel.
    left = np.maximum(g[:-1, None], e[:-1])
    right = np.maximum(left, np.minimum(g[1:, None], e[1:]))
    return (np.log(right / left) / np.log(e[1:] / e[:-1])) @ n


def run(out):
    records = measurements()
    results = []
    input_rows = []
    decomp = []
    for stage in ["pilot", "main", "refined", "convergence"]:
        if not (out / f"{stage}-curves.csv.gz").exists():
            continue
        df = pd.read_csv(out / f"{stage}-curves.csv.gz")
        x = np.sort(df.true_um.unique())
        step = x[1] - x[0]
        g = np.r_[x - step / 2, x[-1] + step / 2]
        populations = []
        for day, minutes, counts in records:
            for name, (e, n) in counts.items():
                b = map_to_grid(e, n, g)
                chk = n @ window(e, g[0], g[-1])
                np.testing.assert_allclose(b.sum(), chk)
                for lower, high in [(6.0, 13.0), (6.0, 13.07), (0.0, 13.0)]:
                    ob = [
                        nn @ window(ee, lower, high)
                        for ee, nn in [counts["FSSP"], counts["ICEMET"]]
                    ]
                    populations.append(
                        (str(day), name, minutes, b, lower, high, *ob)
                    )
                if stage == "refined":
                    for d, w in zip(x, b):
                        input_rows.append(
                            dict(day=day, source=name, true_um=d, number_cm3=w)
                        )
                    results.append(
                        dict(
                            day=day,
                            source=name,
                            minutes=minutes,
                            total_cm3=n.sum(),
                            within_model_cm3=b.sum(),
                            below_1_cm3=n @ window(e, 0, 1),
                            above_40_cm3=n @ window(e, 40, np.inf),
                        )
                    )
        synthetic = 100 * np.diff(ndtr(np.log(g / 9.0) / np.log(1.35)))
        populations.extend(
            [
                ("synthetic", "lognormal", 0, synthetic, l, h, np.nan, np.nan)
                for l, h in [(6.0, 13.0), (6.0, 13.07), (0.0, 13.0)]
            ]
        )
        for (case, cal), group in df.groupby(["case", "calibration"]):
            curves = {
                (probe, nm): v.sort_values("true_um")
                for (probe, nm), v in group.groupby(["probe", "nm"])
            }
            for day, name, minutes, pop, low, h, of, oi in populations:
                band = (
                    ("band13" if h == 13.0 else "band1307")
                    if low
                    else "nativebelow13"
                )
                rawband = (
                    ("unbinned13" if h == 13.0 else "unbinned1307")
                    if low
                    else "below13"
                )
                pred = {k: np.dot(pop, v[band]) for k, v in curves.items()}
                true_mask = window(g, low, h)
                detected = {
                    k: np.dot(pop * true_mask, v.detect)
                    for k, v in curves.items()
                }
                f1, f2, h1, h2 = [
                    pred[k]
                    for k in [("F", 633), ("F", 660), ("I", 633), ("I", 660)]
                ]
                wav = ((f1 - f2) + (h1 - h2)) / 2
                method = ((f1 - h1) + (f2 - h2)) / 2
                gap = f1 - h2
                np.testing.assert_allclose(wav + method, gap, atol=1e-10)
                detgap = detected["F", 633] - detected["I", 660]
                triggergap = np.dot(
                    pop * true_mask,
                    curves["F", 633].trigger.to_numpy()
                    - curves["I", 660].trigger.to_numpy(),
                )
                rawgap = pop @ (
                    curves["F", 633][rawband].to_numpy()
                    - curves["I", 660][rawband].to_numpy()
                )
                decomp.append(
                    dict(
                        stage=stage,
                        case=case,
                        calibration=cal,
                        day=day,
                        source=name,
                        minutes=minutes,
                        low_um=low,
                        high_um=h,
                        true_band=pop @ true_mask,
                        f633=f1,
                        f660=f2,
                        i633=h1,
                        i660=h2,
                        native_gap=gap,
                        wavelength=wav,
                        method=method,
                        interaction=(f1 - f2) - (h1 - h2),
                        true_band_trigger_gap=triggergap,
                        native_range_censor_gap=detgap - triggergap,
                        true_band_detection_gap=detgap,
                        net_migration_gap=gap - detgap,
                        continuous_size_migration=rawgap - detgap,
                        channel_reporting=gap - rawgap,
                        observed_f=of,
                        observed_i=oi,
                        observed_gap=of - oi,
                        residual=of - oi - gap,
                    )
                )
    pd.DataFrame(decomp).to_csv(out / "attribution.csv", index=False)
    pd.DataFrame(results).to_csv(out / "inputs-summary.csv", index=False)
    pd.DataFrame(input_rows).to_csv(
        out / "input-spectra.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
