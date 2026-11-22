"""Read the two single-VOC dashed curves in the native Figure 5 PDF image."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import find_objects, label


def read_curves(path):
    """Extract dashed reference curves without extrapolating their domains."""
    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape[:2] != (2613, 2067):
        raise ValueError("Expected the original 2067 by 2613 Figure 5 raster")
    scale = image.shape[1] / 720
    left, top, right, bottom = [round(p * scale) for p in (432, 696, 645, 852)]
    crop = image[top:bottom, left:right].astype(float)
    rows = []
    for parent, ink in (("alpha", [58, 85, 165]), ("cresol", [255, 0, 0])):
        mask = np.linalg.norm(crop - ink, axis=2) < 65
        labels, _ = label(mask, np.ones((3, 3)))
        for i, (ys, xs) in enumerate(find_objects(labels), 1):
            if xs.stop - xs.start < 4:
                continue
            y, x = np.nonzero(labels[ys, xs] == i)
            px = (left + xs.start + x.mean()) / scale
            py = (top + ys.start + y.mean()) / scale
            mass = 10 ** (1 + (px - 515.7) / (632.8 - 515.7))
            yield_fraction = (853.58 - py) * 0.4 / (853.58 - 621.6)
            rows.append(
                dict(
                    parent=parent,
                    x=px,
                    y=py,
                    mass=mass,
                    yield_fraction=yield_fraction,
                    consumed=mass / yield_fraction,
                )
            )
    curves = pd.DataFrame(rows).sort_values(["parent", "x"])
    if not (
        curves.groupby("parent")
        .consumed.apply(lambda c: c.is_monotonic_increasing)
        .all()
    ):
        raise ValueError("Nonmonotone single-VOC consumption curve")
    return curves


if __name__ == "__main__":
    read_curves(sys.argv[1]).to_csv(
        Path(__file__).with_name("data") / "single_curves.csv", index=False
    )
