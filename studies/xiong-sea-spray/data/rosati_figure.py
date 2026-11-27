"""Extract Rosati et al. (2021) Figure 2b number spectra from a verified raster."""

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path.cwd()
FIG = Path(sys.argv[1])
DIGEST = "1dcb67d7740bed54faf7d8e4cbf9be4ed166ab1de1d0330f882eddb9e932a5d8"
X10 = 1091.4
X_PER_DECADE = (1528.4 - X10) / 2
Y0 = 576.3
Y_PER_NUMBER = (576.3 - 87.9) / 12000

# Visual centerline guides separate the three co-plotted red traces. These
# guides extend beyond the reported domain. Only red pixels enter the result.
GUIDES = {
    "mean": [
        (1237, 343),
        (1247, 333),
        (1257, 326),
        (1267, 320),
        (1277, 331),
        (1287, 341),
        (1297, 357),
        (1307, 375),
        (1317, 395),
        (1327, 416),
        (1337, 438),
        (1347, 456),
        (1357, 475),
        (1367, 487),
        (1377, 494),
        (1387, 506),
        (1394, 512),
        (1402, 508),
        (1412, 520),
        (1422, 531),
        (1432, 539),
        (1442, 548),
        (1448, 551),
    ],
    "initial": [
        (1237, 177),
        (1247, 186),
        (1257, 189),
        (1267, 201),
        (1277, 231),
        (1287, 253),
        (1297, 289),
        (1307, 320),
        (1317, 344),
        (1327, 370),
        (1332, 393),
        (1342, 414),
        (1352, 440),
        (1362, 456),
        (1372, 465),
        (1382, 479),
        (1387, 487),
        (1397, 489),
        (1402, 485),
        (1412, 501),
        (1422, 519),
        (1432, 524),
        (1437, 526),
        (1442, 530),
        (1448, 534),
    ],
    "five_hour": [
        (1237, 504),
        (1247, 491),
        (1257, 479),
        (1267, 465),
        (1277, 456),
        (1282, 450),
        (1287, 455),
        (1297, 456),
        (1307, 453),
        (1312, 464),
        (1322, 470),
        (1332, 479),
        (1342, 492),
        (1352, 504),
        (1362, 515),
        (1367, 514),
        (1377, 522),
        (1387, 530),
        (1397, 533),
        (1402, 531),
        (1412, 543),
        (1422, 547),
        (1432, 553),
        (1442, 556),
        (1448, 560),
    ],
}


def trace(mask, columns, points):
    xs, ys = zip(*points)
    estimate = np.interp(columns, xs, ys)
    sampled = np.full(columns.size, np.nan)
    for index, (x, center) in enumerate(zip(columns, estimate)):
        pixels = np.flatnonzero(mask[:, x])
        groups = np.split(pixels, np.flatnonzero(np.diff(pixels) > 1) + 1)
        centers = [np.median(group) for group in groups if group.size >= 2]
        nearest = min(centers, key=lambda y: abs(y - center), default=np.nan)
        if abs(nearest - center) <= 8:
            sampled[index] = nearest
    valid = np.isfinite(sampled)
    filled = np.interp(columns, columns[valid], sampled[valid])
    return filled, valid


def main():
    assert hashlib.sha256(FIG.read_bytes()).hexdigest() == DIGEST
    image = Image.open(FIG).convert("RGB")
    rgb = np.asarray(image)
    # The printed curves bifurcate near 340 nm. Retain the bright red
    # small-diameter branch through 349 nm and the dark red branch from
    # 350 nm. Each joins an unambiguous instrument range in the source.
    bright = (rgb[..., 0] > 232) & (rgb[..., 1] < 72) & (rgb[..., 2] < 72)
    dark = (
        (rgb[..., 0] > 135)
        & (rgb[..., 0] < 202)
        & (rgb[..., 1] < 40)
        & (rgb[..., 2] < 40)
    )
    optical_guides = {
        "initial": [(1425, 520), (1435, 525), (1447, 534), (1453, 539)],
        "mean": [(1425, 534), (1435, 538), (1445, 543), (1453, 549)],
        "five_hour": [(1425, 549), (1435, 552), (1445, 555), (1453, 560)],
    }
    columns = np.arange(1237, 1449)
    curves = {
        key: trace(bright, columns, points) for key, points in GUIDES.items()
    }
    optical_x = np.arange(1425, 1454)
    optical = {
        key: trace(dark, optical_x, points)
        for key, points in optical_guides.items()
    }
    diameters = np.arange(50, 401)
    x_values = X10 + X_PER_DECADE * np.log10(diameters / 10)
    output = {"diameter_nm": diameters}
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    colors = {"mean": "black", "initial": "green", "five_hour": "purple"}
    for key, (curve, valid) in curves.items():
        small_branch = np.interp(x_values, columns, curve)
        large_curve, large_valid = optical[key]
        large_branch = np.interp(x_values, optical_x, large_curve)
        heights = np.where(diameters >= 350, large_branch, small_branch)
        number = (Y0 - heights) / Y_PER_NUMBER
        # Physical axis/line reading sensitivity, not an instrument standard error.
        # Allow +/-1.5 horizontal pixels and +/-3.5 vertical pixels (tick, stroke
        # edges, and guide choice). Include the bounding visible centerlines in a
        # dash gap or where a blue trace occludes the red one.
        sensitivity = []
        for d, x, y, small_y, large_y in zip(
            diameters, x_values, heights, small_branch, large_branch
        ):
            sample_x, sample_y, sample_valid = (
                (optical_x, large_curve, large_valid)
                if d >= 350
                else (columns, curve, valid)
            )
            visible_x = sample_x[sample_valid]
            left = visible_x[visible_x <= x]
            right = visible_x[visible_x >= x]
            low = min(x - 1.5, left[-1] if left.size else x)
            high = max(x + 1.5, right[0] if right.size else x)
            candidates = np.interp(
                np.r_[
                    low, high, sample_x[(sample_x > low) & (sample_x < high)]
                ],
                sample_x,
                sample_y,
            )
            # Add the local branch split in the SMPS/OPS overlap. For the
            # obscured end of the initial SMPS dashes, bound the lower branch
            # between the last identified initial dash and the visible SMPS
            # mean (their published ordering over this domain).
            branch_diff = abs(small_y - large_y) if x >= 1425 else 0
            if key == "initial" and x > 1430:
                small_mean = np.interp(x, columns, curves["mean"][0])
                branch_diff = max(branch_diff, abs(small_mean - y))
            sensitivity.append(
                (3.5 + np.max(abs(candidates - y)) + branch_diff) / Y_PER_NUMBER
            )
        output[key + "_cm3_decade"] = np.round(number, 1)
        output[key + "_read_sensitivity"] = np.ceil(sensitivity).astype(int)
        for grid, center, sampling, is_large in (
            (columns, curve, valid, False),
            (optical_x, large_curve, large_valid, True),
        ):
            for x, y, seen in zip(grid, center, sampling):
                if seen and x % 2 == 0 and (x >= 1429) == is_large:
                    draw.point((int(x), int(round(y))), fill=colors[key])
        print(
            key,
            "visible columns",
            np.sum(valid),
            "longest span without direct center (px)",
            np.max(np.diff(columns[valid])) - 1,
        )
    assert np.all(output["initial_cm3_decade"] > output["mean_cm3_decade"])
    assert np.all(output["mean_cm3_decade"] > output["five_hour_cm3_decade"])
    with (HERE / "spectra.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(output)
        writer.writerows(zip(*output.values()))
    overlay.crop((1225, 154, 1455, 580)).resize((690, 1278)).save(
        HERE / "trace-check.png"
    )


if __name__ == "__main__":
    main()
