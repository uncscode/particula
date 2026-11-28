"""Replay a bounded four-layer cloud regression from Bai optical moments.

With --source, extract the matched clean 0.25 m/s dark- and sunlight-aged
columns from the original Bai study packet's sunlight/optics directory.
Otherwise run from the compact fixture stored beside this script.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from disort import HERE, write_csv
from particula.optics.column import SpectralColumn
from particula.optics.radiative_transfer import solve_shortwave
from particula.optics.solar import SolarInput
from particula.optics.surface import lambertian_surface


def extract(source):
    """Derive four layers from archived height-resolved Mie moments."""
    with (
        np.load(source / "cloud_extinction.npz") as data,
        np.load(source / "cloud_phase.npz") as scatter,
    ):
        tau_src = data["beta"]
        heights = data["z_m"]
        names = list(data["cases"])
        wl = data["wave_nm"]
        samples = [
            names.index("clean_w0.25_ranked_" + x) for x in ("dark", "sun")
        ]
        z_bound = np.linspace(0, len(heights) - 1, 5).astype(int)[::-1]
        out = []
        for hi, lo in zip(z_bound[:-1], z_bound[1:], strict=True):
            sl = slice(lo, hi + 1)
            beta = tau_src[:, samples, sl]
            moment = scatter["moments"][:, samples, sl, :]
            out.append(
                np.trapezoid(beta[..., None] * moment, x=heights[sl], axis=2)
            )
        totals = np.array(out).transpose(2, 0, 1, 3)
        with (source / "cloud_reflectance.csv").open() as stream:
            obs = [
                r
                for r in csv.DictReader(stream)
                if r["model"] == "mie"
                and r["case"] in [names[i] for i in samples]
            ]
        rows = []
        for i in samples:
            for w in wl:
                r = next(
                    row
                    for row in obs
                    if row["case"] == names[i]
                    and float(row["wavelength_nm"]) == w
                    and float(row["cos_zenith"]) == 0.5
                )
                rows.append(float(r["reflectance"]))
        np.savez_compressed(
            HERE / "bai_visible.npz",
            wavelength_m=wl * 1e-9,
            z_m=heights[z_bound],
            tau=totals[..., 0],
            moments=totals / totals[..., 0, None],
            legacy_reflectance=np.array(rows).reshape(2, -1),
        )


def replay():
    """Compare hemispheric reflectance with the original visible solution."""
    with np.load(HERE / "bai_visible.npz") as d:
        w = d["wavelength_m"]
        rows = []
        for i, case in enumerate(("dark", "sun")):
            col = SpectralColumn(
                w,
                d["z_m"],
                d["tau"][i],
                np.ones_like(d["tau"][i]),
                d["moments"][i],
            )
            solar = SolarInput(w, 0.5, np.ones_like(w), np.zeros_like(w))
            for n in [4, 8, 16, 32]:
                result = solve_shortwave(
                    col, solar, lambertian_surface(w, 0.0), streams=n
                )
                for j, wave in enumerate(w):
                    rows.append(
                        dict(
                            case=case,
                            streams=n,
                            wavelength_nm=round(wave * 1e9),
                            legacy_R=d["legacy_reflectance"][i, j],
                            native_R=result.up[0, j] / 0.5,
                        )
                    )
    write_csv(HERE / "bai_visible.csv", rows)
    for n in [4, 8, 16, 32]:
        diff = np.array(
            [r["native_R"] - r["legacy_R"] for r in rows if r["streams"] == n]
        ).reshape(2, 3)
        print(
            n,
            "max absolute and paired R error",
            np.max(abs(diff)),
            np.max(abs(np.diff(diff, axis=0))),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    src = parser.parse_args().source
    if src:
        extract(src)
    replay()
