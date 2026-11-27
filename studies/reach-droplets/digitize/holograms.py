"""Recover the plotted Fig. 7 cell lattice and both independent pooled curves."""

from pathlib import Path
import argparse
import hashlib
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
SCALE = 30  # pixels per PDF point


def run(pdf, out):
    if (
        hashlib.sha256(pdf.read_bytes()).hexdigest()
        != "633e98ea54c11917217d3d3f404a9beeff7d9002caadec71de258a4ef1338f2e"
    ):
        raise ValueError("Figure coordinates require the archived NOAA article PDF")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "fig.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", "13", "-l", "13", pdf, svg], check=True)
        paths = list(ET.parse(svg).iter("{http://www.w3.org/2000/svg}path"))
        # Render the page crop with vector antialiasing off, retaining gradients.
        crop = Path(tmp) / "fig"
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "13",
                "-l",
                "13",
                "-singlefile",
                "-png",
                "-aaVector",
                "no",
                "-r",
                str(SCALE * 72),
                "-x",
                str(85 * SCALE),
                "-y",
                str(107 * SCALE),
                "-W",
                str(196 * SCALE),
                "-H",
                str(154 * SCALE),
                pdf,
                crop,
            ],
            check=True,
        )
        rgb = np.asarray(Image.open(crop.with_suffix(".png")).convert("RGB"))

    def pixels(x, y):
        return rgb[np.floor((y - 107) * SCALE).astype(int), np.floor((x - 85) * SCALE).astype(int)]

    curves = []
    samples = []
    for name, base, xref, step, color, bar_y in [
        ("unforced", 168.342, 119.598, 0.2771, "100%", 114.5),
        ("forced", 257.927, 101.502, 0.2936, "12.156", 204.0),
    ]:
        xs = []
        for p in paths:
            v = p.get("d", "").split(" Z")[0].split()
            if len(v) != 12 or v[0] != "M" or v[3] != "L":
                continue
            xy = np.array(v).reshape(4, 3)[:, 1:].astype(float)
            x, y = xy.mean(axis=0)
            w, h = np.ptp(xy, axis=0)
            if base - 46.2 < y < base and 0.24 < w < 0.32 and 0.54 < h < 0.72 and 89 < x < 278:
                xs.append(x)
        xs = np.array(xs)
        index = np.rint((xs - xref) / step)
        dx, intercept = np.polyfit(index, xs, 1)
        assert np.quantile(abs(xs - (intercept + dx * index)), 0.99) < 0.009
        x = intercept + dx * np.arange(
            np.floor((89.122 - intercept) / dx) + 1, np.ceil((277.84 - intercept) / dx)
        )
        t = (x - 89.122) / 20.9685
        d = np.arange(10.5, 80, 1.0)
        cols = pixels(x[:, None], base - (d[None, :] - 10) * 23.119 / 35).astype(float)
        # The bar's 0.1 and 1 decadal ticks are at x=96.30 and 197.92 pt.
        bx = np.arange(96.6, 269.85, 0.05)
        lut = pixels(bx, np.full_like(bx, bar_y)).astype(float)
        dist, idx = cKDTree(lut).query(cols)
        chroma = np.ptp(cols, axis=2)
        valid = (chroma > 8) & (dist < 15)
        value = 10 ** (-1 + (bx[idx] - 96.3) / (197.92 - 96.3))
        floor = valid & (value < 0.104)
        # Drop axis and dashed line columns. Retain every other time sample,
        # including empty ones, so mean concentration has its proper denominator.
        ok = np.min(abs(t[:, None] - [0, 4, 7, 9]), axis=1) > 0.019
        times, diams = np.broadcast_arrays(t[:, None], d[None, :])
        use = valid & ok[:, None]
        samples.append(
            pd.DataFrame(
                dict(
                    run=name,
                    t_s=times[use],
                    D_um=diams[use],
                    density=value[use],
                    floor=floor[use].astype(int),
                )
            )
        )
        pd.DataFrame({"t_s": t[ok]}).to_csv(
            out / f"times-{name}.csv", index=False, float_format="%.7f"
        )
        for p in paths:
            if p.get("stroke", "").startswith("rgb(" + color) and len(p.get("d", "")) > 400:
                xoff, yoff = map(float, re.findall(NUM, p.get("transform"))[-2:])
                xy = np.array(
                    re.findall(r"[ML] (" + NUM + r") (" + NUM + r")", p.get("d")), dtype=float
                )
                dia = 10 ** (1 + (xy[:, 0] + xoff - 123.387) / 154.455)
                density = 10 ** (-2 + (322.395 - yoff + xy[:, 1]) / 25.1175)
                curves.append(pd.DataFrame(dict(run=name, D_um=dia, density=density)))
    pd.concat(samples).to_csv(
        out / "cells.csv.gz",
        index=False,
        float_format="%.7f",
        compression={"method": "gzip", "mtime": 0},
    )
    pd.concat(curves).to_csv(out / "pooled.csv", index=False, float_format="%.7f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
