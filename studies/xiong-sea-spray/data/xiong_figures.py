"""Recover Xiong Tables and axis-calibrated readings from publisher PDF images."""

import argparse
import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path.cwd()


def save(name, fields, records):
    with (HERE / name).open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(fields.split(","))
        writer.writerows(records)


def figure_points():
    # Native image 002 = Fig 2. Printed GF=1.0 at row 636.5, GF=2.0 at 119.
    # Blue marker and cap coordinates. 0-based pixels, y increases down.
    rh = np.array([40, 60, 70, 76, 80, 86])
    nacl_xy = [
        (423, 615.5, 602, 628),
        (601, 626, 612, 640),
        (690, 610, 600, 620),
        (743.5, 605.5, 593, 618),
        (779, 130, 120, 140),
        (832, 62, 51, 73),
    ]
    # S21 native image plus soft mask, proportionally resized to exactly
    # 1000 pixels wide. GF=1.0 at 613.5 and GF=2.0 at 175.3.
    bis_xy = [
        (443, 538.5, 528, 549),
        (594, 473.5, 462, 485),
        (670, 406.5, 399, 414),
        (715.5, 353.5, 344.5, 362.5),
        (746, 325, 315, 335),
        (791, 238.5, 231, 246),
    ]
    nacl, bis = [], []
    coords = []
    for series, calibration, values, parsed in [
        ("NaCl", (636.5, 517.5), nacl_xy, nacl),
        ("NaHSO4", (613.5, 438.2), bis_xy, bis),
    ]:
        offset, scale = calibration
        for humidity, (x, y, upper, lower) in zip(rh, values):
            coords.append([series, humidity, x, y, upper, lower, offset, scale])
            parsed.append(
                [1 + (offset - y) / scale, (lower - upper) / 2 / scale]
            )
    save(
        "endmember-pixels.csv",
        "salt,rh_pct,x_px,y_px,top_px,bottom_px,gf1_y_px,px_per_gf",
        coords,
    )
    rows = []
    for target in [40, 60, 70, 75, 80, 85]:
        row = [target]
        for obs in [nacl, bis]:
            row.extend(
                [
                    round(np.interp(target, rh, np.array(obs)[:, j]), 5)
                    for j in range(2)
                ]
            )
        rows.append(row)
    save(
        "endmembers.csv",
        "rh_pct,nacl_gf,nacl_spread,bisulfate_gf,bisulfate_spread",
        rows,
    )
    # Rows use the printed slope labels in Fig 5c/d. Error bars in panel e
    # are read to ±2 pixels, with no confidence convention claimed.
    rates = [0.4879, 0.6523, 1.0018, 1.1686, 1.0796, 0.9644, 0.7927, 0.7247]
    half_px = np.array([17, 13.5, 13, 10, 13, 4.5, 15, 7.5])
    save(
        "size-rates.csv",
        "diameter_nm,k_per_h,fig5e_errorbar_halfspan_per_h",
        [
            [d, k, round(h * 0.3 / 73, 4)]
            for d, k, h in zip(range(50, 401, 50), rates, half_px)
        ],
    )


def reference_series():
    # Marker centers as read on 2x nearest-axis crops of Fig 5a/b.
    # Each crop starts at native image y=18. Pixels are specified in
    # the enlarged image; uncertainty is approximately +/- 4 crop pixels.
    # Panel b has overlapping markers. Hollow centers are still traceable.
    readings = {
        50: [626, 539, 504, 367, 241, 165, 124],
        100: [473, 415, 325, 248, 185, 129, 81],
        150: [436, 390, 278, 187, 124, 100, 60],
        200: [348, 217, 153, 82, 57, 50, 46],
        250: [419, 329, 174, 126, 74, 50, 34],
        300: [482, 310, 187, 136, 86, 70, 57],
        350: [513, 441, 299, 167, 123, 84, 63],
        400: [536, 411, 243, 185, 149, 120, 85],
    }
    # Both panels have y=25.3 px at 3.5 mol/L, y=342.5 px at 0.5 mol/L.
    rows = []
    for d, yy in readings.items():
        for k, y in enumerate(yy, 1):
            y_native = 18 + (y - 0.5) / 2
            c = 3.5 - (y_native - 25.3) * 3 / (342.5 - 25.3)
            rows.append([d, k / 2, round(c, 4), 0.025, round(y_native, 2)])
    save(
        "fig5-sulfate.csv",
        "diameter_nm,time_h,sulfate_mol_per_L,reading_halfwidth_mol_per_L,y_px",
        rows,
    )

    # Fig 5c/d circles and triangles share the sulfate sampling times.
    # Enlargement to 642 rows spans 321 native rows; offset in source
    # image is 465 px in c and 475 px in d. y=0 is at 784.7/794.8 px.
    log_pixel = {
        50: [615, 595, 584, 540, 473, 413, 362],
        100: [577, 557, 520, 479, 431, 370, 276],
        150: [546, 518, 457, 367, 291, 215, 115],
        200: [535, 459, 409, 290, 222, 165, 68],
        250: [553, 524, 427, 356, 263, 148, 109],
        300: [578, 518, 443, 370, 305, 227, 157],
        350: [589, 563, 514, 425, 378, 295, 212],
        400: [594, 557, 487, 440, 405, 334, 262],
    }
    logs = []
    for d, series in log_pixel.items():
        offset, zero = (465, 784.7) if d <= 200 else (475, 794.8)
        for k, ypix in enumerate(series, 1):
            native_y = offset + (ypix - 0.5) / 2
            logs.append(
                [
                    d,
                    k / 2,
                    round((zero - native_y) * 4.5 / 317.0, 4),
                    0.03,
                    round(native_y, 2),
                ]
            )
    save(
        "fig5-log.csv",
        "diameter_nm,time_h,log_c0_c,reading_halfwidth_log,y_px",
        logs,
    )

    # Fig 4a, linear y=354.3 for 0.5 mol/L; 62.0 for 3.5.
    sulfate_y = [273, 246.5, 204.5, 169.5, 140, 115, 92, 82.5, 71]
    sulfate_bar = [7, 5.5, 13, 14.5, 12, 9, 9, 8, 7]
    times = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]
    save(
        "fig4a-100nm.csv",
        "time_h,sulfate_mol_per_L,spread_mol_per_L,y_px",
        [
            [
                t,
                round(0.5 + (354.3 - y) * 3 / 292.3, 4),
                round(s * 3 / 292.3, 4),
                y,
            ]
            for t, y, s in zip(times, sulfate_y, sulfate_bar)
        ],
    )
    # Fig 4b: full IC markers. Some hollow GF markers partly obscured.
    # Use matched Fig5c for separate GF marker readings beyond 0h.
    ic_y = [330.5, 302.6, 298.9, 280.6, 259.7, 239.3, 211.9]
    gf5c_y = [784.7, 752.7, 744.1, 725.8, 704.1, 680.8, 649.3, 602.6]
    rows = []
    for i, y in enumerate(gf5c_y):
        gf = (784.7 - y) * 4.5 / (784.7 - 467.7)
        ic = (
            round((330.5 - ic_y[i]) / (330.5 - 272.0), 4)
            if i < len(ic_y)
            else ""
        )
        rows.append(
            [i / 2, round(gf, 4), ic, y, ic_y[i] if i < len(ic_y) else ""]
        )
    save(
        "fig4b5c-100nm.csv",
        "time_h,log_c0_c_gf,log_c0_c_ic,y_fig5c_gf_px,y_fig4b_ic_px",
        rows,
    )

    # Fig3: high-contrast 1h and 2h points and 5h center. Gray/teal/red
    # markers at 3 and 4h overlap the 5h point. For the two channels for
    # which Sec 3.1.1 reports every relative change, retain those as text.
    obs3 = [
        (60, 1, 528),
        (60, 2, 493),
        (60, 5, 456),
        (80, 1, 217.5),
        (80, 2, 255),
        (80, 5, 284),
    ]
    save(
        "fig3-100nm.csv",
        "rh_pct,time_h,gf,reading_halfwidth_gf,y_px",
        [
            [rh, t, round(1 + (636.5 - y) / 517.5, 4), 0.01, y]
            for rh, t, y in obs3
        ],
    )


def chromatograms(folder):
    # Composite S19 is a black-only image with an 8-bit soft mask. Set
    # axis calibration on a 743 x 900 preview, then integrate native rows.
    # Use each ion window's endpoints to remove the local baseline.
    alpha = np.asarray(Image.open(folder / "supp-005.png"))
    assert alpha.shape == (2048, 1691)
    sx, sy = alpha.shape[1] / 743.0, alpha.shape[0] / 900.0
    # (time, first and second peak x, top and baseline y)
    panels = [
        (0.5, 120, 280, 55, 245),
        (1, 470, 628, 113, 242),
        (1.5, 122, 281, 403, 538),
        (2, 469, 628, 392, 538),
        (2.5, 121, 280, 654, 828),
        (3, 469, 628, 719, 832),
    ]
    records = []
    for time, x1, x2, ymin, yzero in panels:
        ratios = []
        raster_bounds = []
        horizontal_ratios = []
        for threshold in (24, 64, 96):
            for width_change in (-1, 0, 1):
                areas = []
                area_bounds = []
                horizontal_areas = []
                for xpos, left, right in [(x1, 8, 9), (x2, 15, 15)]:
                    x = np.arange(
                        round((xpos - left - width_change) * sx),
                        round((xpos + right + width_change) * sx) + 1,
                    )
                    ytop = round(ymin * sy)
                    piece = alpha[ytop : round((yzero + 4) * sy), x]
                    valid = piece > threshold
                    # The source represents a thin single trace. Steep
                    # edges occupy several pixels vertically per column.
                    center = []
                    bottom = []
                    top = []
                    for col in valid.T:
                        yy = np.flatnonzero(col)
                        center.append(
                            np.nan if not len(yy) else (yy.min() + yy.max()) / 2
                        )
                        bottom.append(np.nan if not len(yy) else yy.max())
                        top.append(np.nan if not len(yy) else yy.min())
                    y = np.array(center)
                    good = np.isfinite(y)
                    assert good.sum() > 0.9 * len(y)
                    y = np.interp(x, x[good], y[good])
                    line = np.linspace(
                        np.median(y[:3]), np.median(y[-3:]), len(y)
                    )
                    areas.append(np.trapezoid(line - y, x))
                    # Second estimator integrates peak widths horizontally.
                    # At a level with two discernible flanks use their
                    # centers; for a merged top, half the occupied width.
                    widths = []
                    for yi, mask in enumerate(valid):
                        indices = np.flatnonzero(mask & (yi < line - 1))
                        if not len(indices):
                            widths.append(0)
                            continue
                        groups = np.split(
                            indices, np.flatnonzero(np.diff(indices) > 1) + 1
                        )
                        if len(groups) > 1:
                            widths.append(
                                np.mean(groups[-1]) - np.mean(groups[0])
                            )
                        else:
                            widths.append((indices[-1] - indices[0] + 1) / 2)
                    horizontal_areas.append(sum(widths))
                    y_bottom = np.interp(x, x[good], np.array(bottom)[good])
                    y_top = np.interp(x, x[good], np.array(top)[good])
                    # Bound even a steep, unresolved rise by assigning
                    # every column coherently to either visible line edge.
                    area_bounds.append(
                        [
                            np.trapezoid(line - y_bottom, x),
                            np.trapezoid(line - y_top, x),
                        ]
                    )
                ratios.append(areas[1] / areas[0])
                horizontal_ratios.append(
                    horizontal_areas[1] / horizontal_areas[0]
                )
                raster_bounds.append(
                    [
                        area_bounds[1][0] / area_bounds[0][1],
                        area_bounds[1][1] / area_bounds[0][0],
                    ]
                )
        records.append(
            [
                time,
                round(ratios[4], 3),
                round(min(ratios), 3),
                round(max(ratios), 3),
                round(np.min(raster_bounds), 3),
                round(np.max(raster_bounds), 3),
                round(horizontal_ratios[4], 3),
            ]
        )
    save(
        "figs19-ic-peaks.csv",
        "time_h,late_to_early_peak_area_ratio,reading_min,reading_max,raster_edge_min,raster_edge_max,horizontal_widths_ratio",
        records,
    )


def check_overlays(folder):
    # Circles enclose the source markers at the recorded coordinates.
    fig5 = Image.open(folder / "main-005.png").convert("RGB")
    draw = ImageDraw.Draw(fig5)
    for name in ("fig5-sulfate.csv", "fig5-log.csv"):
        with (HERE / name).open() as source:
            for row in csv.DictReader(source):
                d = int(row["diameter_nm"])
                t = float(row["time_h"])
                if name == "fig5-sulfate.csv":
                    x = (
                        (71.5 + 123.4 * (t - 0.5))
                        if d <= 200
                        else (667.5 + 117.8 * (t - 0.5))
                    )
                else:
                    x = (59.5 + 109.8 * t) if d <= 200 else (638.5 + 109.4 * t)
                y = float(row["y_px"])
                draw.ellipse(
                    (x - 6, y - 6, x + 6, y + 6), outline="#9d00ff", width=1
                )
    fig5.crop((0, 0, 1034, 840)).save(HERE / "fig5-marker-check.png")
    alpha = Image.open(folder / "supp-005.png")
    ic = Image.new("RGB", alpha.size, "white")
    ic.paste(Image.open(folder / "supp-004.png"), mask=alpha)
    ic = ic.resize((743, 900), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(ic)
    for a, b, top, y in [
        (120, 280, 55, 245),
        (470, 628, 113, 242),
        (122, 281, 403, 538),
        (469, 628, 392, 538),
        (121, 280, 654, 828),
        (469, 628, 719, 832),
    ]:
        draw.rectangle((a - 8, top, a + 9, y + 4), outline="#1565c0")
        draw.rectangle((b - 15, top, b + 15, y + 4), outline="#cd6318")
    ic.save(HERE / "figs19-window-check.png")


def extract(main, supp):
    expected = [
        "8263b19ed0689c2df9f2c3ab33beadadc6636ff5f5b0dfbc60ad1a6bd648eb79",
        "1d675ad6f61d2348533e97d0e083003f5d45787e0da56b49ac0a9f9d3f23b4fb",
    ]
    assert [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (main, supp)
    ] == expected
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        subprocess.run(["pdfimages", "-png", main, folder / "main"], check=True)
        subprocess.run(
            [
                "pdfimages",
                "-f",
                "15",
                "-l",
                "18",
                "-png",
                supp,
                folder / "supp",
            ],
            check=True,
        )
        figure_points()
        reference_series()
        chromatograms(folder)
        check_overlays(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("main_pdf", type=Path)
    parser.add_argument("supplement_pdf", type=Path)
    args = parser.parse_args()
    extract(args.main_pdf, args.supplement_pdf)
