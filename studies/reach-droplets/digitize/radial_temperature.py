"""Read five array positions from Fig S2."""

import argparse
import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SENSORS = {"unforced": [192, 285, 384, 483, 577], "forced": [594, 688, 787, 885, 979]}


def save(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == "17d97f9931f27e354f846f66fa3f31443b00985576a0f3e192d91001c28bbf97"
    )
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "s2"
        subprocess.run(
            ["pdftoppm", "-f", "5", "-l", "5", "-r", "144", "-singlefile", "-png", source, dest],
            check=True,
        )
        a = np.asarray(Image.open(dest.with_suffix(".png")).convert("RGB")).astype(float)
    # The axes and temperature bar share horizontal boundaries y=254 and 542 in this 144 dpi image.
    yp = np.arange(255, 542)
    lut = a[yp, 1003]
    lut_T = 270 + (542 - yp) * 25 / 288.0
    rows = []
    with (args.output / "figure6.csv").open() as f:
        records = list(csv.DictReader(f))
    for run, xs in SENSORS.items():
        for row in [r for r in records if r["run"] == run]:
            t = float(row["time_s"])
            # Select all raster time samples that fall inside the 50 ms time bin.
            trow = (542 - yp) * 9 / 288
            which = yp[abs(trow - t) < 0.025 + 1e-7]
            if not len(which):
                which = yp[[abs(trow - t).argmin()]]
            for j, x in enumerate(xs):
                # Avoid the black line and plot border. Interior positions are sampled on both sides.
                offsets = [2, 3] if j == 0 else [-3, -2] if j == 4 else [-3, -2, 2, 3]
                pixels = a[which[:, None], x + np.asarray(offsets)]
                distances = ((pixels[:, :, None, :] - lut[None, None, :, :]) ** 2).sum(axis=-1)
                sample = lut_T[distances.argmin(axis=-1)]
                rows.append(
                    dict(
                        run=run,
                        time_s=t,
                        sensor_from_center=j,
                        temperature_K=round(float(sample.mean()), 3),
                        reading_range_K=round(float(np.ptp(sample)), 3),
                        low_clip=bool(np.any(sample <= 270.13)),
                    )
                )
    save(args.output / "figureS2.csv", rows)


if __name__ == "__main__":
    main()
