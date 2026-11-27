"""Recreate numeric Figure 4 marker readings and SI tables from retained transcriptions."""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Manual marker centers in the embedded 1513 x 651 pixel Figure 4 bitmap.
# Origin is the upper-left pixel. Triangle centers are at their geometric centroid.
# Overplotted symbols whose centers could not be located separately are omitted.
# No connecting line values are used as observations.
markers = {
    ("pinene", "dry"): [
        (107, 75.5),
        (128, 237),
        (150, 288.5),
        (182, 320.5),
        (202.5, 333.5),
        (593, 407.5),
    ],
    ("pinene", "wet"): [
        (116, 193),
        (128, 263.5),
        (154, 325),
        (181.5, 353.5),
        (561.5, 429.5),
    ],
    ("ocimene", "dry"): [
        (117.5, 144),
        (150.5, 210),
        (168.5, 222.5),
        (201.5, 242.5),
        (287.5, 272.5),
        (615.5, 310.5),
        (675, 312),
    ],
    ("ocimene", "wet"): [
        (106.5, 54),
        (117, 139.5),
        (137.5, 201.5),
        (161.5, 238),
        (190.5, 257.5),
        (280.5, 303.5),
        (602.5, 345),
        (659, 348),
    ],
    ("caryophyllene", "dry"): [
        (883.5, 195),
        (894, 206),
        (922, 225),
        (946.5, 232),
        (1354, 281),
    ],
    ("caryophyllene", "wet"): [
        (854, 67),
        (867, 171),
        (881.5, 201),
        (893.5, 214),
        (920.5, 235),
        (944.5, 249),
        (1353, 308.5),
    ],
    ("farnesene", "dry"): [
        (880, 130.5),
        (910, 154.5),
        (944, 165.5),
        (1380.5, 215.5),
    ],
    ("farnesene", "wet"): [
        (853.5, 54),
        (866, 124.5),
        (879.5, 140.5),
        (909, 161),
        (942.5, 173),
        (1371, 236),
    ],
}
# 0 and 1400 min x ticks: (106,718.5) and (852.5,1465).
# 0 and 1 VFR y ticks in both panels: (552.5,54).
diameters = dict(pinene=165, ocimene=220, caryophyllene=165, farnesene=140)
densities = dict(pinene=1.24, ocimene=1.35, caryophyllene=1.17, farnesene=1.20)
# These are Table S4(dry) and S5(wet) from Liu et al. Rounded values as printed.
vbs = {
    ("pinene", "dry"): [
        0.252,
        0.000157,
        0.164,
        0.360,
        0.221,
        0.00142,
        0.000158,
    ],
    ("ocimene", "dry"): [0.212, 0.265, 0.114, 0.253, 0.100, 0.0336, 0.0223],
    ("caryophyllene", "dry"): [
        0.516,
        0.00195,
        0.0971,
        0.201,
        0.145,
        0.0158,
        0.0228,
    ],
    ("farnesene", "dry"): [
        0.636,
        0.00781,
        0.113,
        0.158,
        0.0551,
        0.0229,
        0.00712,
    ],
    ("pinene", "wet"): [0.220, 0.00444, 0.105, 0.425, 0.237, 0.00805, 0.000751],
    ("ocimene", "wet"): [0.354, 0.0145, 0.183, 0.329, 0.102, 0.00803, 0.009],
    ("caryophyllene", "wet"): [
        0.254,
        0.243,
        0.0709,
        0.284,
        0.0398,
        0.104,
        0.00448,
    ],
    ("farnesene", "wet"): [0.523, 0.0752, 0.149, 0.154, 0.0706, 0.0177, 0.0102],
}


def main():
    with (HERE / "observations.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "precursor",
                "rh",
                "time_min",
                "vfr",
                "pixel_x",
                "pixel_y",
                "panel",
                "initial_diameter_nm",
                "density_g_cm3",
            ]
        )
        for (precursor, rh), series in markers.items():
            mono = precursor in ("pinene", "ocimene")
            zero_x = 106.0 if mono else 852.5
            for x, y in series:
                writer.writerow(
                    [
                        precursor,
                        rh,
                        round((x - zero_x) * 1400 / 612.5, 2),
                        round((552.5 - y) / 498.5, 5),
                        x,
                        y,
                        "a" if mono else "b",
                        diameters[precursor],
                        densities[precursor],
                    ]
                )
    with (HERE / "fractions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["precursor", "rh", "cstar_ug_m3", "mass_fraction_as_printed"]
        )
        for (precursor, rh), fractions in vbs.items():
            for cstar, fraction in zip(
                [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], fractions, strict=True
            ):
                writer.writerow([precursor, rh, cstar, fraction])


if __name__ == "__main__":
    main()
