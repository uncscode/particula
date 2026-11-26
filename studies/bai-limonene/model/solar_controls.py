"""Repeat the clean July source-age, station UV and clear-sky comparisons."""

from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SEASON = ROOT / "seasonal"
SCRATCH = Path(os.environ["SEASONAL_TMP"])
MESHES = {
    432: ["--bins", 192, "--subbins", 8, "--probes", 72],
    1728: ["--bins", 768, "--subbins", 32, "--probes", 36],
}


def call(script, *args):
    subprocess.run(
        [sys.executable, str(SEASON / script), *map(str, args)], check=True
    )


def cloud(label, mesh, w, month, mode, stride=3, *args):
    call(
        "experiment.py",
        "--label",
        label,
        "--columns",
        "--midpoint",
        "--merge-physical",
        *MESHES[mesh],
        "--sub",
        60,
        "--dt",
        0.025,
        "--levels",
        100,
        "--months",
        month,
        "--profiles",
        "clean",
        "--wind",
        w,
        "--modes",
        mode,
        "--stride",
        stride,
        *args,
    )


def flux(label, w, boundary="observed"):
    tag = f"{label}-{boundary}"
    call(
        "radiation/seasonal_flux.py",
        "--tag",
        tag,
        "--prefix",
        label,
        "--months",
        7,
        "--profiles",
        "clean",
        "--updrafts",
        w,
        "--models",
        "ranked",
        "--boundary",
        boundary,
        "--redo",
    )
    df = pd.read_csv(
        SCRATCH / "radiation" / tag / f"{label}_07_clean_{w:g}_ranked.csv.gz"
    )
    if len(df) != 168 or df.hour_in_week.nunique() != 168:
        raise ValueError("Incomplete July radiative comparison")
    return df.delta_up_site_wm2.mean()


def main():
    records = []
    for age in [6, 24, 72]:
        label = f"ageflux{age}"
        for m, mode in [(0, "dark"), (7, "ranked")]:
            cloud(label, 432, 0.25, m, mode, 3, "--age_hours", age)
        records.append(
            dict(
                test=f"{age} h age",
                mesh=432,
                updraft=0.25,
                delta_up_wm2=flux(label, 0.25),
            )
        )
    # Each UV history uses the same full-resolution, dark-aged control.
    label = "resolved-uv-control"
    cloud(label, 1728, 0.25, 0, "dark")
    dark = pd.read_csv(
        SEASON / f"results/{label}/m00_clean_dark.csv"
    ).median_um.iloc[0]
    for source in ["uvi_gray", "uvi_shape"]:
        label = f"resolved-{source}"
        cloud(label, 1728, 0.25, 7, "ranked", 6, "--forcing", source)
        df = pd.read_csv(
            SEASON / f"results/{label}/m07_clean_ranked.csv"
        ).sort_values("hour_in_week")
        median = np.interp(np.arange(168), df.hour_in_week, df.median_um).mean()
        records.append(
            dict(
                test=source,
                mesh=1728,
                updraft=0.25,
                delta_median_um=median - dark,
            )
        )
    label = "resolved-clear-boundary"
    for m, mode in [(0, "dark"), (7, "ranked")]:
        cloud(label, 1728, 2.0, m, mode)
    for boundary in ["observed", "clear_native"]:
        records.append(
            dict(
                test=boundary,
                mesh=1728,
                updraft=2.0,
                delta_up_wm2=flux(label, 2.0, boundary),
            )
        )
    results = pd.DataFrame(records)
    expected_flux = {
        "6 h age": -0.0074720494,
        "24 h age": -0.3518152,
        "72 h age": -1.011279,
        "observed": -1.0279035,
        "clear_native": -2.114305,
    }
    indexed = results.set_index("test")
    for name, expected in expected_flux.items():
        np.testing.assert_allclose(
            indexed.loc[name, "delta_up_wm2"], expected, rtol=0, atol=1e-5
        )
    np.testing.assert_allclose(
        indexed.loc[["uvi_gray", "uvi_shape"], "delta_median_um"],
        [0.8704464, 1.0790708],
        rtol=0,
        atol=1e-5,
    )
    destination = SEASON / "july-solar-controls.csv"
    results.to_csv(destination, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
