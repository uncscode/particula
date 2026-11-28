"""Read Sagan et al. Figure 6 from the versioned paper's PDF; write a 50 ms table."""

import argparse
import csv
import hashlib
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
X0, X_PER_S = 363.084991, (525.090988 - 363.084991) / 8
# Published figure tick locations in PDF points, down from top.
AXES = {
    "temperature_K": (273.788010, 222.931992, 265, 295),
    "vapor_mol_mol": (337.357986, 286.500992, 0.014, 0.016),
    "count_cm3": (464.497993, 413.641968, 0, 15),
    "mean_diameter_um": (528.067993, 477.211990, 10, 60),
}
SERIES = {"forced": (31, 119, 180), "unforced": (255, 127, 14)}
DISPLAYED = {"forced": "12.156", "unforced": "100%"}
FILL = {"forced": (203, 214, 232), "unforced": (246, 223, 200)}
OVERLAP = (207, 195, 188)
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
TIMES = np.arange(0.0005, 9, 0.001)


def read_vectors(paths, run, quantity):
    bottom, top, low, high = AXES[quantity]
    values = np.full(TIMES.shape, np.nan)
    for path in paths:
        if not path.get("stroke", "").startswith("rgb(" + DISPLAYED[run]):
            continue
        match = re.fullmatch(
            r"matrix\(1, 0, 0, -1, (" + NUM + r"), (" + NUM + r")\)", path.get("transform", "")
        )
        if not match:
            continue
        xoff, yoff = map(float, match.groups())
        if not (X0 <= xoff < X0 + X_PER_S * 9 and top - 0.2 <= yoff <= bottom + 0.2):
            continue
        raw = path.get("d")
        if not re.fullmatch(r"(?:[ML] " + NUM + " " + NUM + r"\s*)+", raw):
            raise ValueError("Unexpected line geometry")
        points = np.array(re.findall(r"[ML] (" + NUM + ") (" + NUM + ")", raw), dtype=float)
        t = (xoff + points[:, 0] - X0) / X_PER_S
        y = low + (bottom - yoff + points[:, 1]) * (high - low) / (bottom - top)
        keep = (TIMES >= t[0]) & (TIMES <= t[-1])
        values[keep] = np.interp(TIMES[keep], t, y)
    if quantity == "vapor_mol_mol" and run == "unforced":
        values -= 0.004
    return values


def render_panel(source, temp, name, scale=12):
    bottom, top, low, high = AXES[name]
    xcut, ycut = int(X0 * scale), int(top * scale)
    out = temp / name
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            "12",
            "-l",
            "12",
            "-singlefile",
            "-r",
            str(72 * scale),
            "-x",
            str(xcut),
            "-y",
            str(ycut),
            "-W",
            str(int(9 * X_PER_S * scale) + 1),
            "-H",
            str(int((bottom - top) * scale) + 1),
            "-png",
            source,
            out,
        ],
        check=True,
    )
    img = np.array(Image.open(out.with_suffix(".png")).convert("RGB"))
    t = ((xcut + np.arange(img.shape[1]) + 0.5) / scale - X0) / X_PER_S
    y = low + (bottom - (ycut + np.arange(img.shape[0]) + 0.5) / scale) * (high - low) / (
        bottom - top
    )
    return img, t, y


def reduce_color(img, t, y, colors, statistic):
    visible = np.zeros(img.shape[:2], dtype=bool)
    for color in colors:
        visible |= np.all(img == color, axis=2)
    cols = np.flatnonzero(visible.any(axis=0))
    return t[cols], np.array([statistic(y[visible[:, col]]) for col in cols])


def bin_average(t, y, left):
    selected = y[(t >= left) & (t < left + 0.05)]
    selected = selected[np.isfinite(selected)]
    return float(np.mean(selected)) if len(selected) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert (
        hashlib.sha256(args.source.read_bytes()).hexdigest()
        == "633e98ea54c11917217d3d3f404a9beeff7d9002caadec71de258a4ef1338f2e"
    )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        svg = tmp / "page12.svg"
        subprocess.run(["pdftocairo", "-f", "12", "-l", "12", "-svg", args.source, svg], check=True)
        paths = list(ET.parse(svg).iter("{http://www.w3.org/2000/svg}path"))
        temp = render_panel(args.source, tmp, "temperature_K")
        count = render_panel(args.source, tmp, "count_cm3")
        records = []
        for run in SERIES:
            data = {k: (TIMES, read_vectors(paths, run, k)) for k in AXES if k != "count_cm3"}
            for label, stat in [("visible_cold_K", np.min), ("visible_warm_K", np.max)]:
                data[label] = reduce_color(*temp, [FILL[run], OVERLAP, SERIES[run]], stat)
            data["count_cm3"] = reduce_color(*count, [SERIES[run]], np.median)
            count_t = data["count_cm3"][0]
            for k in range(180):
                left = k / 20
                row = {"run": run, "time_s": round(left + 0.025, 3)}
                for name, (t, y) in data.items():
                    value = bin_average(t, y, left)
                    row[name] = "" if value is None else f"{value:.9g}"
                all_cols = np.count_nonzero((count[1] >= left) & (count[1] < left + 0.05))
                row["count_visible_fraction"] = (
                    f"{np.count_nonzero((count_t >= left) & (count_t < left + 0.05)) / max(all_cols, 1):.3f}"
                )
                records.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        writer = csv.DictWriter(out, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
