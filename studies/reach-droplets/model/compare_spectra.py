"""Project a replay onto the original sampling times and published diameter classes."""

import argparse
from zipfile import ZipFile
import numpy as np
import pandas as pd
from inverse import interpolator, project
from observations import load
from workspace import ARCHIVE, WORK


def compare(run, series):
    t, observed = load(run)
    if series == "Hologram":
        spectrum = observed[:, 1:-1]
    elif series == "Forward physics":
        file = WORK / "physical/runs" / f"{run}-full-n768-s2/spectrum.npz"
        with np.load(file) as replay:
            t, d = replay["t_s"], replay["D_um"]
            spectrum = replay["cm3"].mean(axis=1)[:, (d >= 11) & (d < 79)]
    else:
        root = WORK / "inversion"
        if series == "Spatial update":
            file = root / "state-check" / f"{run}-576-redistribute-1"
        else:
            split = "early" if series == "Fit before 5 s" else "hold"
            file = root / "native" / f"{run}-18-1-0.05-joint-0.5-{split}-native-0.05-576-65536"
        with np.load(file / "prediction.npz") as replay:
            d = replay["d"]
            spectrum = (interpolator(t, replay["t"]) @ replay["spec"]) @ project(
                d, d[1] - d[0], np.arange(11, 80)
            )
    quarters = np.arange(3.0, 7.0, 0.25)
    result = np.stack(
        [spectrum[(t >= start) & (t < start + 0.25)].mean(axis=0) for start in quarters]
    )
    with ZipFile(ARCHIVE.with_name("figure-data.zip")) as store, store.open("spectra.csv") as f:
        records = pd.read_csv(f)
    target = records[(records.run == run) & (records.series == series)].pivot(
        index="t_mid_s", columns="diameter_um", values="density_cm-3_um-1"
    )
    np.testing.assert_allclose(result, target, rtol=2e-5, atol=1e-7)
    tail = target.columns.to_numpy()
    print(
        f"{run}, {series}: {result[(quarters >= 5) & (quarters < 5.5)][:, (tail >= 50) & (tail < 65)].sum(1).mean():.4f} droplets/cm³, 5–5.5 s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=["forced", "unforced"])
    parser.add_argument(
        "series",
        choices=["Hologram", "Forward physics", "Fit before 5 s", "Blocked fit", "Spatial update"],
    )
    args = parser.parse_args()
    compare(args.run, args.series)
