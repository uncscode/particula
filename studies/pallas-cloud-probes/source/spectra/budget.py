"""Propagate spectrum-wide calibration offsets and finite-sample counts in native channels."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from score import measurements, map_to_grid
from study import ROOT as RUN_DIR, edges

ROOT = Path(__file__).resolve().parent
OPTICS = RUN_DIR / "optics"
GRID = np.array([6, 7.5, 9, 10.5, 12, 13, 15, 18, 21, 25.5, 30, 36, 40.0])
BAND = slice(0, 5)
N = 16384
SCENARIOS = [
    "total",
    "counting",
    "number_scale",
    "sizing",
    "narrow_icemet",
    "wide_icemet",
    "shared_scales",
    "fssp_center",
    "fssp_as_sd",
    "repeat",
    "small_drop_losses",
]


def symmetric(f, i):
    f, i = np.broadcast_arrays(f, i)
    return np.divide(
        200 * (f - i), f + i, where=(f + i) > 0, out=np.full(f.shape, np.nan)
    )


def remap_draws(edges_in, bins, boundary):
    # Each draw has one calibrated axis. Evaluate the native cumulative spectrum
    # at the inverse-shifted common boundaries, then difference adjacent values.
    u = np.clip(np.log(boundary), np.log(edges_in[0]), np.log(edges_in[-1]))
    loge = np.log(edges_in)
    j = np.clip(np.searchsorted(loge, u, side="right") - 1, 0, len(loge) - 2)
    w = (u - loge[j]) / (loge[j + 1] - loge[j])
    cum = np.pad(np.cumsum(bins, axis=1), ((0, 0), (1, 0)))
    cdf = np.take_along_axis(cum, j, axis=1) + w * np.take_along_axis(
        bins, j, axis=1
    )
    return np.maximum(np.diff(cdf, axis=1), 0)


def field_draws(day, minutes, counts, case):
    rng = np.random.default_rng((12087 if case == "repeat" else 2947) + day)
    size = rng.uniform(-1, 1, (N, 2))
    gain = rng.uniform(-1, 1, (N, 2))
    if case == "shared_scales":
        size[:, 1], gain[:, 1] = size[:, 0], gain[:, 0]
    spectra = []
    for j, (name, volume) in enumerate(
        [("FSSP", 0.534 * 36 * 60), ("ICEMET", 7.5)]
    ):
        e, n = counts[name]
        if j == 0 and case == "fssp_center":
            e = edges("F", "center")
        vsamp = volume * minutes
        native = rng.poisson(n * vsamp, size=(N, len(n))) / vsamp
        if case in ("number_scale", "sizing"):
            native = np.broadcast_to(n, native.shape)
        amp = 0.16
        delta = 0.20 if j == 0 else 1.0
        if case == "fssp_as_sd" and j == 0:
            amp *= np.sqrt(3)
            delta *= np.sqrt(3)
        if j == 1 and case in ("narrow_icemet", "wide_icemet"):
            factor = 0.5 if case == "narrow_icemet" else 2.0
            amp *= factor
            delta *= factor
        if case in ("counting", "sizing"):
            amp = 0.0
        if case in ("counting", "number_scale"):
            delta = 0.0
        if j == 1 and case == "small_drop_losses":
            # Stress test an additional 0–30% loss in the two 5–7 um image channels.
            missed = rng.uniform(0, 0.30, (N, 1))
            native[:, e[1:] <= 7] /= 1 - missed
        perturb = delta * size[:, j, None]
        boundary = GRID / (1 + perturb) if j == 0 else GRID - perturb
        frame = remap_draws(e, native, boundary) * (1 + amp * gain[:, j, None])
        spectra.append(frame)
    return spectra


def summaries(f, i):
    metrics = {"FSSP": f, "ICEMET": i, "gap": f - i, "percent": symmetric(f, i)}
    return {
        k: (
            np.nanquantile(x, 0.025, axis=0),
            np.nanquantile(x, 0.975, axis=0),
            np.nanstd(x, axis=0, ddof=1),
        )
        for k, x in metrics.items()
    }


def run(out):
    response = pd.read_csv(out / "transfer.csv.gz")
    prior = pd.read_csv(OPTICS / "attribution.csv")
    inputs = pd.read_csv(OPTICS / "input-spectra.csv.gz")
    curves = pd.read_csv(OPTICS / "refined-curves.csv.gz").query(
        'case=="nominal" and calibration=="frozen"'
    )
    rows, budgets, signs = [], [], []
    for day, minutes, counts in measurements():
        if day == 14:
            continue
        field = {k: map_to_grid(e, n, GRID) for k, (e, n) in counts.items()}
        model = {}
        for probe, name, nm in [
            ("F", "FSSP", 633),
            ("I", "ICEMET", 660),
            ("F", "FSSP", 660),
            ("I", "ICEMET", 633),
        ]:
            g = inputs.query('day==@day and source=="CDA"').sort_values(
                "true_um"
            )
            e = counts[name][0]
            matrix = (
                response.query("probe==@probe and nm==@nm")
                .pivot(index="true_um", columns="channel", values="fraction")
                .reindex(index=g.true_um, columns=range(len(e) - 1))
                .fillna(0)
                .to_numpy()
            )
            np.testing.assert_allclose(
                matrix.sum(axis=1),
                curves.query("probe==@probe and nm==@nm")
                .sort_values("true_um")
                .detect,
                atol=6e-10,
            )
            model[
                name
                if (probe, nm) in [("F", 633), ("I", 660)]
                else name + "_swap"
            ] = map_to_grid(e, g.number_cm3.to_numpy() @ matrix, GRID)
        prior_day = prior.query(
            'source=="CDA" and stage=="refined" and case=="nominal" and calibration=="frozen" and low_um==6 and high_um==13 and day==@daystr',
            local_dict={"daystr": str(day)},
        ).iloc[0]
        np.testing.assert_allclose(
            [model["FSSP_swap"][BAND].sum(), model["ICEMET_swap"][BAND].sum()],
            prior_day[["f660", "i633"]].to_numpy(float),
            atol=1e-7,
        )
        for name, fkey, mkey in [
            ("FSSP", "observed_f", "f633"),
            ("ICEMET", "observed_i", "i660"),
        ]:
            np.testing.assert_allclose(
                [field[name][BAND].sum(), model[name][BAND].sum()],
                prior_day[[fkey, mkey]].to_numpy(float),
                atol=1e-7,
            )
        for case in SCENARIOS:
            f, i = field_draws(day, minutes, counts, case)
            s = summaries(f[:, BAND].sum(axis=1), i[:, BAND].sum(axis=1))
            native_f = field["FSSP"][BAND].sum()
            if case == "fssp_center":
                native_f = map_to_grid(
                    edges("F", "center"), counts["FSSP"][1], GRID
                )[BAND].sum()
            native_i = field["ICEMET"][BAND].sum()
            for k, v in [
                ("FSSP", native_f),
                ("ICEMET", native_i),
                ("gap", native_f - native_i),
                ("percent", symmetric(native_f, native_i)),
            ]:
                lo, hi, sd = s[k]
                budgets.append((day, minutes, case, k, float(v), lo, hi, sd))
            band_gap = f[:, BAND].sum(axis=1) - i[:, BAND].sum(axis=1)
            signs.append(
                (
                    day,
                    case,
                    N,
                    int((band_gap >= 0).sum()),
                    band_gap.min(),
                    band_gap.max(),
                )
            )
            if case != "total":
                continue
            per = summaries(f, i)
            for j, (low, high) in enumerate(zip(GRID[:-1], GRID[1:])):
                row = dict(
                    day=day,
                    minutes=minutes,
                    low_um=low,
                    high_um=high,
                    CDA=field["CDA"][j],
                )
                for probe in ["FSSP", "ICEMET"]:
                    row[probe] = field[probe][j]
                    row["model_" + probe] = model[probe][j]
                    row["model_" + probe + "_swap"] = model[probe + "_swap"][j]
                row["percent"] = symmetric(row["FSSP"], row["ICEMET"])
                row["model_percent"] = symmetric(
                    row["model_FSSP"], row["model_ICEMET"]
                )
                for k, (lo, hi, sd) in per.items():
                    row[k + "_p025"], row[k + "_p975"], row[k + "_sd"] = (
                        lo[j],
                        hi[j],
                        sd[j],
                    )
                rows.append(row)
    pd.DataFrame(
        signs,
        columns=[
            "day",
            "scenario",
            "draws",
            "nonnegative",
            "sample_min_cm3",
            "sample_max_cm3",
        ],
    ).to_csv(out / "sign-check.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "distributions.csv", index=False)
    pd.DataFrame(
        budgets,
        columns=[
            "day",
            "minutes",
            "scenario",
            "quantity",
            "nominal",
            "p025",
            "p975",
            "standard_uncertainty",
        ],
    ).to_csv(out / "uncertainty-budget.csv", index=False)


if __name__ == "__main__":
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT)
