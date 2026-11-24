"""Predict the six salt-size calibration growth factors and inspect plot anchoring."""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import brentq
from particula_aiomfac import AIOMFACModel, MixtureSpec
from particula.particles import get_kelvin_radius

BASE = Path(__file__).resolve().parent


def salt_check():
    # Both dry crystal densities are in g/cm³. The mobility-to-volume shape
    # alternative assumes D_volume/D_mobility=0.96, only as a perturbation.
    settings = {"AS": ({204: 2, 261: 1}, 1.77), "NaCl": ({202: 1, 242: 1}, 2.165)}
    t = 295.0
    rho_w = 0.997
    length_nm = 2e9 * get_kelvin_radius(0.072, rho_w * 1000, 0.01801528, t)
    data = []
    inverted = []
    standards = pd.read_csv(BASE / "standards.csv").set_index(["salt", "size_nm"])
    for name, (groups, rho_s) in settings.items():
        backend = AIOMFACModel.prepare(MixtureSpec({"water": {16: 1}, name: groups}))
        for size in (30, 60, 90):
            observed = standards.loc[name, size]
            for diameter_scale in (0.96, 1.0):
                for statistic in (
                    "median_gf",
                    "median_scan_median",
                    "median_scan_mode",
                    "mean_scan_mean",
                ):
                    obs_gf = observed[statistic]
                    ratio = (obs_gf / diameter_scale) ** 3 - 1
                    aw = backend.evaluate([rho_w * ratio, rho_s], t).neutral_activities[
                        0
                    ]
                    effective_rh = aw * np.exp(length_nm / (size * obs_gf))
                    inverted.append(
                        dict(
                            salt=name,
                            size_nm=size,
                            volume_to_mobility_ratio=diameter_scale,
                            statistic=statistic,
                            observed_gf=obs_gf,
                            effective_rh=effective_rh,
                        )
                    )
            for rh in (0.83, 0.84, 0.85, 0.86, 0.87):
                for diameter_scale in (0.96, 1.0):
                    volume_diameter = size * diameter_scale

                    def difference(q):
                        water_activity = backend.evaluate(
                            [rho_w * q, rho_s], t
                        ).neutral_activities[0]
                        return (
                            water_activity
                            * np.exp(length_nm / (volume_diameter * np.cbrt(1 + q)))
                            - rh
                        )

                    water_ratio = brentq(difference, 0.5, 40)
                    gf = diameter_scale * np.cbrt(1 + water_ratio)
                    data.append(
                        dict(
                            salt=name,
                            size_nm=size,
                            rh=rh,
                            volume_to_mobility_ratio=diameter_scale,
                            model_gf=gf,
                            observed_median=observed.median_gf,
                            residual_gf=gf - observed.median_gf,
                        )
                    )
    pd.DataFrame(inverted).to_csv(BASE / "salt-effective-rh.csv", index=False)
    pd.DataFrame(data).to_csv(BASE / "salt-reference.csv", index=False)


def check_clock_image():
    image = np.asarray(Image.open(BASE / "author-summary.png").convert("RGB"))
    assert image.shape == (3760, 4424, 3)
    # Bottom-panel ticks (y = 3400 px) carry labels at 10-day intervals from May 1.
    ix = np.flatnonzero(
        (image[3400].max(axis=1) < 20) & (np.arange(image.shape[1]) > 1000)
    )
    marks = np.array([k.mean() for k in np.split(ix, np.where(np.diff(ix) > 1)[0] + 1)])
    assert len(marks) == 5
    slope, intercept = np.polyfit(marks, np.arange(5) * 10, 1)
    # Top-panel 90 nm NaCl calibration at GF about 1.9. Independent colors
    # and three manually bounded x intervals distinguish it from ambient.
    top = image[300:510]
    red = (top[:, :, 0] > 220) & (top[:, :, 1] < 50) & (top[:, :, 2] < 50)
    mayfirst = pd.Timestamp("2023-05-01", tz="UTC")
    scans = pd.read_csv(BASE / "scan-qc.csv.gz")
    rows = []
    for date, xbounds in zip(
        ["2023-04-26", "2023-05-16", "2023-05-30"],
        [(810, 940), (1970, 2120), (2790, 2920)],
    ):
        xmin, xmax = xbounds
        _, xx = np.where(red[:, xmin:xmax])
        raw = scans[
            (scans.calibration_day == date)
            & (scans.size_nm == 90)
            & (scans.scan_gf_mean > 1.8)
        ]
        img = mayfirst + pd.to_timedelta(
            slope * (xmin + xx.mean()) + intercept, unit="D"
        )
        center = pd.to_datetime(raw.raw_time.mean(), unit="s", utc=True)
        rows.append(
            dict(
                date=date,
                raw_center=str(center),
                plot_center_assuming_midnight_ticks=str(img),
                raw_minus_plot_h=(center - img).total_seconds() / 3600,
                first_red_x=xmin + xx.min(),
                last_red_x=xmin + xx.max(),
                pixels_per_day=1 / slope,
            )
        )
    pd.DataFrame(rows).to_csv(BASE / "clock-figure-check.csv", index=False)


if __name__ == "__main__":
    salt_check()
    check_clock_image()
