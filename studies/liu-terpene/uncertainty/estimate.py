"""Propagate finite plotted diameter resolution and common starting-size uncertainty."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Native tick centers (pixels) and their vacuum aerodynamic diameters (nm), Figure S4.
AXES = {
    "pinene": (97, 686, 100, 300),
    "ocimene": (820, 1410, 150, 400),
    "caryophyllene": (96, 685, 100, 300),
    "farnesene": (820, 1410, 100, 250),
}
READ_ERROR = 2 / 498.5  # Two vertical Figure 4 pixels in volume fraction.


def estimate(out, data_root=ROOT):
    peaks = pd.read_csv(data_root / "uncertainty/peaks.csv")
    obs = pd.read_csv(data_root / "sources/observations.csv").query("rh=='wet'")
    match = pd.read_csv(data_root / "sources/time_verification.csv").query(
        "rh=='wet'"
    )
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, spectra in peaks.groupby("precursor"):
        x0, x1, d0, d1 = AXES[name]
        scale = (d1 - d0) / (x1 - x0)
        diameter = lambda x: d0 + (x - x0) * scale
        start = spectra.query("spectrum_min==0").iloc[0]
        ref = diameter(start.peak_x_px)
        for pt in obs.query("precursor==@name").itertuples():
            matching = match[
                (match.precursor == name)
                & (match.figure4_time_min == pt.time_min)
            ].iloc[0]
            age = (
                int(matching.compatible_legend_times_min)
                if matching.status == "unique"
                else -1
            )
            sample = spectra[spectra.spectrum_min == age]
            reason = (
                "paired"
                if len(sample)
                else ("peak obscured" if age >= 0 else matching.status)
            )
            for label, multiple in [
                ("pixels_only", 0.0),
                ("half_bin", 0.5),
                ("full_bin", 1.0),
            ]:
                # One native image pixel for the peak position, plus the selected
                # fraction of the local distance between adjacent spectral samples.
                def width(row):
                    return scale * (
                        1.0 + multiple * abs(row.peak_x_px - row.adjacent_x_px)
                    )

                a, b = ref - width(start), ref + width(start)
                vlow = max(0.0, pt.vfr - READ_ERROR)
                vhigh = min(1.0, pt.vfr + READ_ERROR)
                sample_d = lo = hi = slo = shi = np.nan
                if len(sample):
                    row = sample.iloc[0]
                    sample_d = diameter(row.peak_x_px)
                    lo, hi = sample_d - width(row), sample_d + width(row)
                    slo, shi = (lo / b) ** 3, (hi / a) ** 3
                results.append(
                    (
                        name,
                        pt.initial_diameter_nm,
                        pt.time_min,
                        age,
                        reason,
                        label,
                        pt.vfr,
                        vlow,
                        vhigh,
                        ref,
                        a,
                        b,
                        sample_d,
                        lo,
                        hi,
                        slo,
                        shi,
                        np.fmin(slo, vlow),
                        np.fmax(shi, vhigh),
                    )
                )
    cols = "precursor diameter_nm time_min spectrum_min status resolution observed figure_low figure_high d0_nm d0_low d0_high dt_nm dt_low dt_high spectrum_low spectrum_high read_low read_high".split()
    df = pd.DataFrame(results, columns=cols)
    df.to_csv(out / "read_bounds.csv", index=False, float_format="%.12g")
    overnight = df.query("time_min>900 and resolution=='half_bin'")
    assert len(overnight) == 5 and (overnight.status == "paired").all()
    assert (overnight.read_low < overnight.observed).all() and (
        overnight.read_high > overnight.observed
    ).all()
    # Original wet spectra were displayed independently on a second, lower-resolution rendering.
    reread = pd.read_csv(data_root / "sources/peak_check.csv").query(
        "rh=='wet'"
    )
    for row in reread.itertuples():
        rr = (
            overnight.query("precursor==@row.precursor")
            .sort_values("time_min")
            .iloc[-1]
        )
        assert rr.read_low <= row.volume_ratio <= rr.read_high


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--out", type=Path, default=HERE / "results")
    estimate(cli.parse_args().out)
