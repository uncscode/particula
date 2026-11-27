"""Recover the lower-dose S20 markers and overlay them on the source pixels."""

# %%
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

FOLDER = Path.cwd()
SUPPLEMENT = Path(sys.argv[1])
# Physical page 17 in PDF = supplemental page S16. The first native color image is S20.
with tempfile.TemporaryDirectory() as temp:
    stem = Path(temp) / "panel"
    subprocess.run(
        ["pdfimages", "-f", "17", "-l", "17", "-png", SUPPLEMENT, stem],
        check=True,
    )
    image = Image.open(stem.with_name("panel-000.png")).convert("RGB")
    mask = Image.open(stem.with_name("panel-001.png")).convert("L")
    fig = Image.new("RGB", image.size, "white")
    fig.paste(image, mask=mask)
assert fig.size == (2048, 785)

# %%
# Native image coordinates refer to marker center dots, including the hollow
# triangles' centroids. Panel a error caps are distinct from line/marker widths.
# At 1h the caps overlap the marker. No numeric spread assigned there.
samples = [
    # hours, xa, ya, cap_top, cap_bottom, xb, yb
    (1, 241, 598.5, None, None, 1265, 635.5),
    (2, 299, 521.5, 507, 535.5, 1323.5, 609.5),
    (4, 415, 390, 376.5, 404, 1440, 532),
    (6, 531, 266.5, 250.5, 282.5, 1556, 399.5),
    (8, 647.5, 235.5, 216.5, 254.5, 1672, 270.5),
    (10, 763.5, 199.5, 187.5, 212.5, 1788.5, 247.5),
]
# The same top and bottom axes correspond to different units in the panels.
top, bottom = 95.5, 654.5
a_scale = 3.5 / (bottom - top)
b_scale = 3.0 / (bottom - top)
header = [
    "time_h",
    "sulfate_mol_per_L",
    "sulfate_graphical_halfwidth_mol_per_L",
    "sulfate_plotted_bar_halfspan_mol_per_L",
    "log_c0_c",
    "log_graphical_halfwidth",
    "xa_px",
    "ya_px",
    "cap_top_px",
    "cap_bottom_px",
    "xb_px",
    "yb_px",
]
draw = ImageDraw.Draw(fig)
with (FOLDER / "s20.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(header)
    for t, xa, ya, hi, lo, xb, yb in samples:
        spread = "" if hi is None else round((lo - hi) * a_scale / 2, 4)
        writer.writerow(
            [
                t,
                round((bottom - ya) * a_scale, 4),
                round(3 * a_scale, 4),
                spread,
                round((bottom - yb) * b_scale, 4),
                round(3 * b_scale, 4),
                xa,
                ya,
                hi,
                lo,
                xb,
                yb,
            ]
        )
        for x, y in [(xa, ya), (xb, yb)]:
            draw.ellipse(
                [x - 16, y - 16, x + 16, y + 16], outline="#a83b95", width=2
            )
            draw.point((round(x), round(y)), fill="#a83b95")
fig.resize((1024, 393), Image.Resampling.LANCZOS).save(
    FOLDER / "s20-marker-check.png"
)
