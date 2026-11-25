"""Recover film observations and model lines from Bai et al., 2026.

Pass the article and supplement PDFs. Coordinate origin is top left in PDF
points after conversion with Poppler pdftocairo. The public source files are
fixed by SHA-256. All derived tables use the output directory specified.
"""

import csv
import hashlib
import re
import subprocess
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from particula.particles.properties.convert_kappa_volumes import (
    get_water_volume_from_kappa,
)

NUM = re.compile(r"[-+]?(?:\d*\.)?\d+(?:e[-+]?\d+)?", re.I)
COLORS = {
    "0%, 44": "blue",
    "85.09": "orange",
    "92.89": "yellow",
    "49.39": "purple",
    "50.19": "gray",
    "46.69": "green",
}


def numbers(text):
    """Read all numeric coordinates in one SVG attribute."""
    return np.array([float(v) for v in NUM.findall(text)])


def paths(pdf, page, tmp):
    """Convert one page, selecting the figure colors used for extraction."""
    svg = tmp / f"{pdf.stem}-{page}.svg"
    subprocess.run(  # noqa: S603 -- checksum-qualified source PDF
        [
            "/usr/bin/pdftocairo",
            "-f",
            str(page),
            "-l",
            str(page),
            "-svg",
            str(pdf),
            str(svg),
        ],
        check=True,
    )
    for el in ET.parse(svg).findall(".//{*}path"):  # noqa: S314 -- local SVG
        paint = el.get("stroke", el.get("fill", ""))
        color = next(
            (
                color
                for prefix, color in COLORS.items()
                if paint.startswith("rgb(" + prefix)
            ),
            None,
        )
        if color is None:
            continue
        d = el.attrib["d"]
        pts = numbers(d).reshape(-1, 2)
        if el.get("transform"):
            a, b, c, d1, dx, dy = numbers(el.attrib["transform"])
            pts = pts @ np.array([[a, b], [c, d1]]) + [dx, dy]
        yield color, pts, d


def marks_and_bars(shapes, sizes):
    """Recover colored marker centers with their attached vertical half-bars."""
    markers, bars = [], []
    for color, pts, _ in shapes:
        w, h = np.ptp(pts, axis=0)
        center = (pts.min(0) + pts.max(0)) / 2
        if abs(w - h) < 0.015 and any(abs(w - size) < 0.015 for size in sizes):
            markers.append((color, *center, w))
        if len(pts) == 2 and w < 0.005 and h > 0:
            bars.append((color, *pts[0], h))
    for color, x, y, w in markers:
        spans = [
            h
            for c, px, py, h in bars
            if c == color and abs(x - px) < 0.015 and abs(y - py) < 0.015
        ]
        yield color, x, y, w, max(spans, default=0.0)


def replicate_rows(shapes):
    """Recover Figure S15 trials and calculate film water at the stated RH."""
    rows = []
    for color, x, y, _, half in marks_and_bars(shapes, [4, 4.5]):
        if color not in ("blue", "orange", "yellow", "purple"):
            continue
        # One legend marker at fixed x in each panel, drawn inside the axes.
        if abs(x - 174.77) < 0.015 or abs(x - 410.77) < 0.015:
            continue
        lim = x > 300
        trial_no = (4 if lim else 3) + [
            "blue",
            "orange",
            "yellow",
            "purple",
        ].index(color)
        mass = ((457.7695 if lim else 292.7695) - x) / 135.2383
        rows.append(
            dict(
                precursor="limonene" if lim else "isoprene",
                trial=f"S{trial_no}",
                mass_fraction=mass,
                kappa=(499.52734 - y) * 0.3 / 135,
                kappa_bar=half * 0.3 / 135,
                initial_mass_ug=(
                    [21.2, 18.3, 18.0, 15.8]
                    if lim
                    else [22.9, 19.3, 22.1, 14.9]
                )[trial_no - (4 if lim else 3)],
                rh=0.8 if lim else 0.6,
                constant_rh=not (lim and trial_no == 6),
            )
        )
    for row in rows:
        group = [
            r
            for r in rows
            if r["precursor"] == row["precursor"] and r["trial"] == row["trial"]
        ]
        fresh = max(group, key=lambda r: r["mass_fraction"])
        np.testing.assert_allclose(fresh["mass_fraction"], 1, atol=0.001)
        if not row["constant_rh"]:
            row.update(
                rh=None,
                water_ratio=None,
                water_fresh_ug=None,
                water_coupled_ug=None,
                water_kappa_only_ug=None,
            )
            continue
        # In cm3, water density 1 g/cm3 gives micrograms after multiplication.
        dry_v = (
            row["initial_mass_ug"]
            * 1e-6
            / (1.25 if row["precursor"] == "limonene" else 1.40)
        )
        aw = row["rh"]
        v0 = get_water_volume_from_kappa(dry_v, fresh["kappa"], aw)
        va = get_water_volume_from_kappa(
            dry_v * row["mass_fraction"], row["kappa"], aw
        )
        vk = get_water_volume_from_kappa(dry_v, row["kappa"], aw)
        row.update(
            water_ratio=va / v0,
            water_fresh_ug=v0 * 1e6,
            water_coupled_ug=va * 1e6,
            water_kappa_only_ug=vk * 1e6,
        )
        np.testing.assert_allclose(
            row["water_ratio"],
            row["mass_fraction"] * row["kappa"] / fresh["kappa"],
            rtol=0.0,
            atol=1e-10,
        )
    return sorted(
        rows, key=lambda r: (r["precursor"], r["trial"], -r["mass_fraction"])
    )


def rh_rows(shapes):
    """Recover fresh and aged hygroscopicity from separate Figure S16 scans."""
    treatments = {
        "gray": "fresh",
        "blue": "dark humid",
        "orange": "300 nm humid",
        "yellow": "345 nm humid",
        "purple": "300 nm dry",
    }
    rows = []
    for color, x, y, w, half in marks_and_bars(shapes, [2, 3.5]):
        if not 121 < x < 279 or color not in treatments:
            continue
        lim = y > 300
        rows.append(
            dict(
                precursor="limonene" if lim else "isoprene",
                treatment=treatments[color],
                symbol="square" if w < 3 else "circle",
                rh_percent=(x - 121.39844) / 157 * 100,
                kappa=((482.10156 if lim else 272.10156) - y) * 0.3 / 168.5,
                kappa_bar=half * 0.3 / 168.5,
            )
        )
    return sorted(
        rows, key=lambda r: (r["precursor"], r["treatment"], r["rh_percent"])
    )


def model_centerline(pts):
    """Intersect a filled stroke with vertical slices to recover its midline."""
    # Vector stroke outlines contain both edges. Intersect vertical slices
    # with the closed polygon and average the two edge ordinates.
    a, b = pts[:-1], pts[1:]
    for x in np.linspace(pts[:, 0].min() + 0.18, pts[:, 0].max() - 0.18, 151):
        mask = (a[:, 0] > x) != (b[:, 0] > x)
        y = a[mask, 1] + (x - a[mask, 0]) * (b[mask, 1] - a[mask, 1]) / (
            b[mask, 0] - a[mask, 0]
        )
        if len(y):
            yield x, (y.min() + y.max()) / 2


def main_figure(shapes):
    """Recover the measurements and model curves in main Figure 3."""
    names = {
        "blue": "dark air",
        "orange": "300 nm air",
        "yellow": "345 nm air",
        "green": "300 nm N2",
    }
    rows = []
    for color, pts, d in shapes:
        w, h = np.ptp(pts, axis=0)
        x, y = (pts.min(0) + pts.max(0)) / 2
        if color not in names or not (120 < x < 210 and 148 < y < 339):
            continue
        lim = y > 246
        if abs(w - h) < 0.015 and any(
            abs(w - i) < 0.015 for i in (3.37, 2.375)
        ):
            kind = "observation"
            centers = [(x, y)]
        elif (
            w > 10
            and 0.5 < h < 30
            and d.count("Z") == 1
            and len(pts) > 65
            and color != "green"
        ):
            kind = "published model"
            centers = model_centerline(pts)
        else:
            continue
        for xc, yc in centers:
            rows.append(
                dict(
                    precursor="limonene" if lim else "isoprene",
                    treatment=names[color],
                    kind=kind,
                    mass_fraction=(210.171875 - xc) / (210.171875 - 125.496094),
                    kappa=((338.480469 if lim else 239.078125) - yc)
                    * 0.3
                    / (89.914063 if lim else 90.412109),
                )
            )
    return sorted(
        rows,
        key=lambda r: (
            r["precursor"],
            r["treatment"],
            r["kind"],
            -r["mass_fraction"],
        ),
    )


def matched_clock(shapes, fig3):
    """Join the condition-level mass and hygroscopicity curves at equal time."""
    rows = []
    seen = {}
    for color, p, d in shapes:
        if color not in ("blue", "orange"):
            continue
        x, y = (p.min(0) + p.max(0)) / 2
        w, h = np.ptp(p, axis=0)
        if not (120 < x < 145 and 290 < y < 330 and w > 80 and h > 15):
            continue
        # A dark dashed line is dry. Only the two solid humid profiles qualify.
        if d.count("M") > 2:
            continue
        condition = "dark air" if color == "blue" else "300 nm air"
        seen[condition] = seen.get(condition, 0) + 1
        comparison = [
            r
            for r in fig3
            if r["precursor"] == "limonene"
            and r["kind"] == "observation"
            and r["treatment"] == condition
        ]
        fg = np.array([r["mass_fraction"] for r in comparison])[::-1]
        kg = np.array([r["kappa"] for r in comparison])[::-1]
        curve = np.array(list(model_centerline(p)))
        for time in (2, 4, 8, 12, 24, 32):
            xp = 85.390625 + 2.43515625 * time
            f = (367.714844 - np.interp(xp, curve[:, 0], curve[:, 1])) / (
                367.714844 - 291.425781
            )
            if not fg[0] <= f <= fg[-1]:
                # Retain the elapsed time even outside measured κ coverage.
                k = water = None
            else:
                k = np.interp(f, fg, kg)
                water = f * k / kg[-1]
            # Sensitivity to a half-record time shift (15 min), quarter PDF
            # point in both mass and κ ordinates. These are reading/record
            # allowances. They do not model source measurement covariance.
            xends = xp + np.array([-0.25, 0.25]) * 2.43515625
            fends = (
                367.714844 - np.interp(xends, curve[:, 0], curve[:, 1])
            ) / 76.289063
            flow = max(fg[0], float(fends.min()) - 0.25 / 76.289063)
            fhigh = min(fg[-1], float(fends.max()) + 0.25 / 76.289063)
            kr = 0.3 * 0.25 / 89.914063
            if flow <= fhigh:
                fcheck = np.unique(
                    np.r_[flow, fg[(fg > flow) & (fg < fhigh)], fhigh]
                )
                kval = np.interp(fcheck, fg, kg)
                lower = float(np.min(fcheck * (kval - kr) / (kg[-1] + kr)))
                upper = float(np.max(fcheck * (kval + kr) / (kg[-1] - kr)))
            else:
                lower = upper = None
            rows.append(
                dict(
                    treatment=condition,
                    trace=seen[condition],
                    time_h=time,
                    mass_fraction=f,
                    kappa_from_fig3=k,
                    water_ratio=water,
                    read_low=lower,
                    read_high=upper,
                )
            )
    if seen != {"dark air": 2, "300 nm air": 5}:
        raise ValueError(f"Unexpected number of humid curves: {seen}")
    return rows


def write(path, rows):
    """Write a compact numerical CSV with reproducible precision."""
    with path.open("w") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: round(v, 7) if isinstance(v, float) else v
                    for k, v in row.items()
                }
            )


def main():
    """Check primary PDF versions and generate the four source tables."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("article", type=Path)
    parser.add_argument("supplement", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    for p, expected in [
        (
            args.article,
            "1c58775829762acab04e4391dcd38da77ec16f2bee0b753c66021397cf763ed5",
        ),
        (
            args.supplement,
            "c0a06cbe2e8ceb9b5654f0579e8bcfdf830bd425254b48acda5465bfeb44d8e3",
        ),
    ]:
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Different PDF version at {p}. Check axes before extracting."
            )
    args.out.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory() as temp:
        tmp = Path(temp)
        write(
            args.out / "replicates.csv",
            replicate_rows(paths(args.supplement, 34, tmp)),
        )
        write(
            args.out / "rh_scans.csv", rh_rows(paths(args.supplement, 35, tmp))
        )
        comparison = main_figure(paths(args.article, 6, tmp))
        write(args.out / "main_figure.csv", comparison)
        write(
            args.out / "matched_clock.csv",
            matched_clock(paths(args.article, 4, tmp), comparison),
        )


if __name__ == "__main__":
    main()
