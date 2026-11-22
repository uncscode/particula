"""Recover two published diagnostic curves using the archived PDFs and Poppler."""

import csv
from pathlib import Path
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent


def extract_s3(stage):
    svg = stage / "s3.svg"
    subprocess.run(
        ["pdftocairo", "-f", "34", "-l", "34", "-svg",
         ROOT.parent / "raw/aaf2649-Dunne-SM.pdf", svg], check=True
    )
    tree = ET.parse(svg)
    centers = []
    colors = []
    parent = {child: element for element in tree.iter() for child in element}
    for marker in tree.findall(".//{*}path"):
        path = marker.get("d", "")
        if marker.get("fill-rule") == "nonzero" and path.count(" L ") == 2:
            vertices = np.array(re.findall(r"-?\d+(?:\.\d+)?", path), float)[:6].reshape(3, 2)
            centers.append(vertices.mean(axis=0))
            colors.append(marker.get("fill"))
    for marker in tree.findall(".//{*}use"):
        if marker.get("{http://www.w3.org/1999/xlink}href") == "#glyph-15-0":
            centers.append((float(marker.get("x")) + 1.078125, float(marker.get("y")) - 0.9375))
            colors.append(parent[marker].get("fill"))
    assert len(centers) == 355, "The PDF or SVG marker vocabulary changed"
    x, y = np.array(centers).T
    # Dotted grid lines at 10^-4 and 10^2, read in the SVG frame of the plot.
    plot_x = (x - 148.535) / 0.3609
    plot_y = (489.738 - y) / 0.3609
    log_obs = -4 + 6 * (plot_x - 118.237826) / (755.555123 - 118.237826)
    log_ref = -4 + 6 * (plot_y - 82.659324) / (534.665160 - 82.659324)
    with (ROOT / "figure_s3_plotted_pairs.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["marker_index", "graphical_J_obs_cm-3_s-1", "graphical_J_param_cm-3_s-1"])
        for index, (obs, pred) in enumerate(zip(10**log_obs, 10**log_ref), 1):
            writer.writerow([index, f"{obs:.6g}", f"{pred:.6g}"])
    error = log_ref - log_obs
    print("Fig S3:", len(centers), "R² (log)", 1 - sum(error**2)/sum((log_obs-log_obs.mean())**2))
    print("Fraction within ×3", np.mean(abs(error) <= np.log10(3)))
    recover_figure_membership(np.array(colors), 10**log_obs)


def within_roundoff(graph, archive):
    tolerance = np.where(archive < 0.005, 0.005 * archive, 0.005 + 0.001 * graph)
    return np.abs(graph - archive) <= tolerance


def prefix_feasible(match):
    """Can the first i markers be matched in order in the first j archive rows?"""
    n, m = match.shape
    result = np.zeros((n + 1, m + 1), bool)
    result[0, :] = True
    for i in range(n):
        result[i + 1, 1:] = np.logical_or.accumulate(match[i] & result[i, :-1])
    return result


def recover_figure_membership(colors, observed):
    src = ROOT.parent / "raw/cloud_data_table.csv"
    table = np.loadtxt(src, delimiter=",", skiprows=1)
    cold = [(200, 215, "rgb(100%, 0%, 100%)"), (215, 235, "rgb(0%, 0%, 100%)"),
            (235, 255, "rgb(0%, 100%, 100%)"), (255, 265, "rgb(56.471252%, 93.333435%, 56.471252%)"),
            (265, 275, "rgb(0.392151%, 100%, 0%)"), (275, 285, "rgb(100%, 100%, 0%)")]
    status = {}
    for low, high, color in cold:
        rows = np.flatnonzero((low < table[:, 0]) & (table[:, 0] < high))
        plotted = np.sort(observed[colors == color])
        assert len(plotted) == len(rows)
        assert within_roundoff(plotted, np.sort(table[rows, -1])).all()
        for row in rows:
            status[row] = ("included_cold_group", "")
    # Red and orange markers are emitted in the archive's relative order.
    # Retain every feasible monotone assignment and expose the one tie.
    warm_marker = np.flatnonzero(np.isin(colors, ["rgb(100%, 0%, 0%)", "rgb(100%, 64.706421%, 0%)"]))
    rows = np.flatnonzero(table[:, 0] >= 290)
    match = within_roundoff(observed[warm_marker, None], table[rows, -1][None, :])
    prefix = prefix_feasible(match)
    assert prefix[-1, -1]
    suffix = prefix_feasible(match[::-1, ::-1])[::-1, ::-1]
    possible = match & prefix[:-1, :-1] & suffix[1:, 1:]
    for j, row in enumerate(rows):
        indices = np.flatnonzero(possible[:, j])
        label = "excluded_by_alignment"
        if indices.size:
            label = "included_warm" if np.any(possible[indices].sum(axis=1) == 1) else "ambiguous_warm"
        status[row] = (label, "/".join(str(warm_marker[i] + 1) for i in indices))
    assert len(status) == len(table)
    with (ROOT / "figure_s3_archive_membership.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source_csv_line", "status", "compatible_warm_marker_index"])
        for row in sorted(status):
            writer.writerow([row + 2, *status[row]])


def extract_collision_limit(stage):
    prefix = stage / "fig7"
    subprocess.run(
        ["pdfimages", "-f", "10", "-l", "10", "-png",
         ROOT.parent / "companion/kuerten2015-correction.pdf", prefix], check=True
    )
    pixels = np.asarray(Image.open(stage / "fig7-000.png").convert("RGB"), dtype=int)
    assert pixels.shape == (1685, 2067, 3)
    green = (pixels[:, :, 1] - pixels[:, :, 0] > 75) & (pixels[:, :, 1] - pixels[:, :, 2] > 65)
    # This is the uppermost GREEN curve in Fig 7, where all cluster evaporation
    # is zero; the green curve to its right refers to a different scenario.
    with (ROOT / "kurten_fig7_kinetic_limit.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["H2SO4_cm-3", "graphical_J1.7_kinetic_cm-3_s-1", "source_x_px", "source_y_px"])
        for x in [375, *range(385, 1116, 10), 1117]:
            ys = np.flatnonzero(green[:650, x])
            runs = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1)
            assert len(runs[0]) >= 2, (x, runs[0])
            y = np.median(runs[0])
            # Tick centers: 10^6..10^9 on x, 10^5..10^-6 on y.
            acid = 10 ** (6 + (x - 426) * 3 / (2021 - 426))
            rate = 10 ** (5 - (y - 33.5) * 11 / (1503.5 - 33.5))
            writer.writerow([f"{acid:.6g}", f"{rate:.6g}", x, y])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        extract_s3(Path(temporary))
        extract_collision_limit(Path(temporary))
