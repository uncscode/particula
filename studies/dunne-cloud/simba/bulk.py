"""Replay sulfate production from published dilute-water kinetics in the wet volume."""

from zipfile import ZipFile
import numpy as np
import warp as wp
from replay import measurements, GRID, F, arrays, advance, AIR, PRESSURE, TEMP
from extract_run1833 import median_samples, curves, ARCHIVE
from search import write


def water_fraction(d):
    x = 0.58 * np.exp(
        -4 * 0.079 * 0.01801528 / (1000 * 8.314462618 * TEMP * d * 1e-9)
    )
    water = 18.01528 * 3 * x / (1 - x)
    return 1.35 * water / (132.14004 + water)


def aqueous(pH):
    t, _, _, _, _, _, _, _, hp, _, so = measurements()
    with ZipFile(ARCHIVE) as z:
        oz = median_samples(curves(z, "04a")[0], t)
    pr = PRESSURE / 101325.0
    soaq = 1.23 * so / AIR * pr
    ozaq = 0.011 * oz * 1e-9 * pr
    hpaq = 7.45e4 * hp / AIR * pr
    h = 10.0 ** (-pH)
    bis = soaq * 0.013 / h
    sulfite = bis * 6.6e-8 / h
    return (
        2.4e4 * soaq + 3.7e5 * bis + 1.5e9 * sulfite
    ) * ozaq, 7.45e7 * h * bis * hpaq / (1 + 13 * h)


def main():
    wp.init()
    dev = "cuda:0"
    (g, f, i, o), hi = arrays(181, 30, dev)
    geo = g.numpy()
    liters = np.pi / 6 * (GRID * 1e-9) ** 3 * 1000 * water_fraction(GRID)
    geo[:, 3] *= liters / geo[:, 5]
    geo[:, 5] = liters
    grid = wp.array(geo, dtype=F, device=dev)
    base = f.numpy()
    mid = base[:, 0].cumsum() - 0.5 * base[:, 0]
    t, *_, obs, hp, valid, so = measurements()
    first = (t - t[0]) * 3600
    config = [
        (1.25, 0.5, 0, x, y, -1.0) for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]
    ]
    cfg = wp.array(np.asarray(config), dtype=F, device=dev)
    mask = first[:181] >= 3.8 * 3600
    rows = []
    for ph in (2, 4, 6):
        forc = base.copy()
        for col, reaction in enumerate(aqueous(ph), start=2):
            forc[:, col] = np.interp(mid, first, 6.02214076e23 * reaction)
        ff = wp.array(forc, dtype=F, device=dev)
        out = wp.zeros((4, 181, 3), dtype=F, device=dev)
        budget = wp.zeros((4, 2), dtype=F, device=dev)
        wp.launch(
            advance,
            dim=(4, len(i) + len(ff)),
            inputs=[cfg, grid, ff, i, o, hi, 2.9, out, budget],
            device=dev,
        )
        c = out.numpy()[:, :, 1]
        for k, label in enumerate(["none", "ozone", "peroxide", "both"]):
            rows.append(
                dict(
                    pH=ph,
                    pathways=label,
                    peak_cm3=c[k].max(),
                    peak_difference_cm3=c[k].max() - c[0].max(),
                    max_point_difference_cm3=np.max(abs(c[k] - c[0])),
                    withheld_rmse=np.mean(
                        (c[k, mask] - obs[:181, 1][mask]) ** 2
                    )
                    ** 0.5,
                )
            )
    write("bulk_replay.csv", rows)


if __name__ == "__main__":
    main()
