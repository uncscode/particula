"""Read Cheng et al. Figure 2a retrievals and uncertainty from the vector artwork."""
import csv
import hashlib
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from extract import HERE, cheng

# Plot frame corners in Poppler SVG points: horizontal molality 0 to 200,
# vertical a_w 1 to 0. The published paths snap to an approximately 0.274 pt grid.
X0, X200, Y1, Y0 = 150.066, 281.604, 76.957, 169.646
EXPECTED = "e8b9bd8216301b5cc9bf5418ea8f2a03efd0bd9c48d31a0da962f62985763a4f"
NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def axis_value(y):
    return (Y0 - y) / (Y0 - Y1)


def intersections(poly, x):
    points = []
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if min(x1, x2) <= x < max(x1, x2):
            points.append(y1 + (x - x1) / (x2 - x1) * (y2 - y1))
    return points


def main():
    pdf = HERE / "cheng2015.pdf"
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == EXPECTED
    with tempfile.TemporaryDirectory() as temp:
        svg = Path(temp) / "page.svg"
        subprocess.run(["pdftocairo", "-f", "4", "-l", "4", "-svg", pdf, svg], check=True)
        paths = list(ET.parse(svg).getroot().iter("{http://www.w3.org/2000/svg}path"))
    polygons = []
    centers = []
    for i, path in enumerate(paths):
        data = path.get("d", "")
        fill = path.get("fill", "")
        if fill == "rgb(93.725586%, 74.508667%, 74.508667%)":
            points = np.array(re.findall(NUMBER, data), dtype=float).reshape(-1, 2)
            if X0 - 1 < points[0, 0] < X200 and Y1 - 1 < points[0, 1] < Y0:
                polygons.append(points)
        if fill != "rgb(100%, 100%, 100%)" or " C " not in data:
            continue
        if paths[i + 1].get("stroke") != "rgb(85.881042%, 0%, 0%)":
            continue
        pts = np.array(re.findall(NUMBER, data), dtype=float).reshape(-1, 2)
        x, y = (pts.min(axis=0) + pts.max(axis=0)) / 2
        if X0 + 8 / 200 * (X200 - X0) <= x < X200 and Y1 < y < Y0:
            molality = 200 * (x - X0) / (X200 - X0)
            # The legend sample sits above 0.7 at a graphical molality above 100.
            cut = [v for poly in polygons for v in intersections(poly, x)]
            if cut and not (molality > 100 and axis_value(y) > 0.7):
                centers.append([molality, axis_value(y),
                                axis_value(max(cut)), axis_value(min(cut)), cheng(molality)])
    assert len(polygons) == 6 and len(centers) >= 20
    with (HERE / "cheng-F2a.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_kg_graphic", "aw_DKA", "aw_lower_graphic",
                         "aw_upper_graphic", "aw_table1_at_graphic_molality"])
        writer.writerows(sorted(centers))
    # Dense slices of the source polygon support comparison at chosen molalities.
    with (HERE / "cheng-F2a-band.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_kg", "aw_lower_graphic", "aw_upper_graphic"])
        for molality in np.arange(10, 100.01, 0.125):
            x = X0 + molality / 200 * (X200-X0)
            cut = [v for poly in polygons for v in intersections(poly, x)]
            assert cut
            writer.writerow([molality, axis_value(max(cut)), axis_value(min(cut))])


if __name__ == "__main__":
    main()
