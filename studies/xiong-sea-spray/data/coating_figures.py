"""Recover Figure 6a,b markers and Table S2 from Xiong et al. (2026)."""

# %%
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FOLDER = Path.cwd()
MAIN = Path(sys.argv[1])
with tempfile.TemporaryDirectory() as directory:
    stem = Path(directory) / "fig6"
    subprocess.run(
        ["pdfimages", "-f", "11", "-l", "11", "-png", MAIN, stem], check=True
    )
    canvas = Image.open(stem.with_name("fig6-000.png")).convert("RGB")
assert canvas.size == (1066, 396)
pixels = np.array(canvas, dtype=float)
draw = ImageDraw.Draw(canvas)


def save(name, columns, records):
    with (FOLDER / name).open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(columns.split(","))
        writer.writerows(records)


# %%
# Left vertical axis: bottom tick 0 M, y346. Top tick 4.8 M, y29.
# Within each cluster the bars are in the order of the legend.
# Read the fill transition away from the centered black error bar.
ovfs = (5, 10, 18, 33, 50, 70)
scale_a = 4.8 / (346 - 29)
sulfate = []
for hour, centers in [
    (2, (122, 148, 175, 201, 228, 254)),
    (3, (319, 346, 373, 399, 426, 452)),
]:
    for ovf, x in zip(ovfs, centers):
        fill = pixels[334, x + 7]
        axis = 255.0 - fill
        column = pixels[94:344, x + 7]
        saturation = np.dot(255.0 - column, axis) / np.dot(axis, axis)
        first = int(np.flatnonzero(saturation >= 0.5)[0])
        # Midpoint of the transition relative to a 0.5-pixel raster cell.
        edge = 94 + first - 0.5
        # Caps are near the edge. The vertical stem is only 1–3 px wide.
        black_rows = (pixels[:, x - 5 : x + 6].max(axis=2) < 60).sum(
            axis=1
        ) >= 5
        ys = np.flatnonzero(
            black_rows & (np.abs(np.arange(len(black_rows)) - edge) < 23)
        )
        ys = ys[(ys >= 94) & (ys < 341)]
        groups = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)
        assert len(groups) == 2
        top, bottom = (float(g.mean()) for g in groups)
        assert abs((top + bottom) / 2 - edge) < 1.5
        sulfate.append(
            [
                ovf,
                hour,
                round((346 - edge) * scale_a, 4),
                round((bottom - top) / 2 * scale_a, 4),
                round(2 * scale_a, 4),
                x,
                edge,
                top,
                bottom,
            ]
        )
        draw.line((x + 5, edge, x + 11, edge), fill="white", width=2)
        for y in (top, bottom):
            draw.ellipse(
                (x - 6, y - 2, x + 6, y + 2), outline="#8f2d72", width=1
            )
save(
    "figure6a.csv",
    "ovf_pct,time_h,sulfate_mol_per_L,bar_halfspan_mol_per_L,reading_halfwidth_mol_per_L,x_px,y_px,bar_top_px,bar_bottom_px",
    sulfate,
)

# %%
# Right axis: y29=0.7/h; y317=0.2/h; tick interval 57.6 px per 0.1/h.
# Dot coordinates read on the native PDF image. The 70% marker is cut
# off at the frame; its upper arc and widest rows place the center at y342.
rates = []
for ovf, x, y in [
    (0, 693.6, 56.5),
    (5, 718.8, 89.5),
    (10, 743.2, 103.7),
    (18, 782.6, 114),
    (33, 856.7, 147.4),
    (50, 940.4, 224.6),
    (70, 1039.5, 342),
]:
    k = 0.7 - (y - 29) * 0.5 / (317 - 29)
    rates.append([ovf, round(k, 5), round(2 * 0.5 / 288, 5), x, y])
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline="#273e80", width=1)
save("figure6b.csv", "ovf_pct,k_per_h,reading_halfwidth_per_h,x_px,y_px", rates)
canvas.save(FOLDER / "figure6_marker_check.png")
