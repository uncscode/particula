"""Read the run 24 number-size heatmap and geometric mean from Wang Fig. S1."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

DEST = Path(__file__).resolve().parent
# Axes in PDF page 5's native 1534 x 899 raster, measured in pixel centres.
X_ZERO = 349.5
X_HOUR = (1100.5 - X_ZERO) / 5
Y_100 = 262.0
Y_1000 = 72.0
BAR_TOP = 72
BAR_BOTTOM = 395


def diameter_nm(y):
    """Convert image row to mobility diameter on the displayed log axis."""
    return 100 * 10 ** ((Y_100 - np.asarray(y)) / (Y_100 - Y_1000))


def image_row(diameter):
    """Return the subpixel row coordinate for a diameter in nanometres."""
    return Y_100 - (Y_100 - Y_1000) * np.log10(diameter / 100)


def read_panel(image, width=4, threshold=38):
    """Read the unmasked colour field and the solid black GM line."""
    if image.shape != (899, 1534, 3):
        raise ValueError("Expected the native S1 figure from page 5")
    # Interior column medians suppress JPEG artefacts at the colour-bar edges.
    palette = np.median(image[BAR_TOP : BAR_BOTTOM + 1, 1271:1282], axis=1)
    colors = cKDTree(palette)
    results, means = [], []
    for hours in np.arange(0, 5.01, 0.25):
        cx = round(X_ZERO + hours * X_HOUR)
        xs = slice(cx - width, cx + width + 1)
        # Coarse line anchors exclude the two black dashed GM ±50% overlays.
        line = np.interp(hours, np.arange(6), [269, 249, 225, 209, 200, 196])
        start, stop = round(line) - 10, round(line) + 11
        black_y, _ = np.nonzero(
            image[start:stop, cx - 7 : cx + 8].max(axis=2) < 75
        )
        if len(black_y) < 2:
            raise ValueError(f"Geometric mean line obscured at {hours} h")
        gm_y = start + np.median(black_y)
        gmd = float(diameter_nm(gm_y))
        means.append(
            dict(
                run=24,
                time_s=int(hours * 3600),
                gmd_nm=gmd,
                reading_band_nm=gmd * (10 ** (2 / (Y_100 - Y_1000)) - 1),
                x_px=cx,
                gm_y_px=gm_y,
            )
        )
        for left in np.linspace(1.4, 2.6, 13):
            low, high = 10**left, 10 ** (left + 0.1)
            # Uniform image row spacing corresponds to uniform log10 diameter.
            y_lo, y_hi = (
                int(np.ceil(image_row(high))),
                int(np.ceil(image_row(low))),
            )
            patch = image[y_lo:y_hi, xs]
            distance, idx = colors.query(patch)
            valid = distance <= threshold
            # The field is discrete contour colours spaced about 5000 cm^-3,
            # whereas the labelled colour bar is a continuous gradient.
            level = 5000 * np.rint((1 - idx / (BAR_BOTTOM - BAR_TOP)) * 8)
            row_means = [
                np.median(values[mask])
                for values, mask in zip(level, valid, strict=True)
                if mask.any()
            ]
            if not row_means:
                raise ValueError(
                    f"Entire bin is obscured at {hours} h, {low} nm"
                )
            results.append(
                dict(
                    run=24,
                    time_s=int(hours * 3600),
                    low_nm=low,
                    high_nm=high,
                    dndlog10d_cm3=np.mean(row_means),
                    reading_band_cm3=5000,
                    valid_fraction=valid.mean(),
                    x_px=cx,
                    top_y_px=y_lo,
                    bottom_y_px=y_hi - 1,
                )
            )
    return pd.DataFrame(results), pd.DataFrame(means)


def main():
    """Extract coordinates and quantify two digitization sensitivities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("png", type=Path)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.png).convert("RGB"))
    if hashlib.sha256(image.tobytes()).hexdigest() != RGB_SHA256:
        raise ValueError("Input differs from published panel raster")
    base, gm = read_panel(image)
    narrow, _ = read_panel(image, width=2, threshold=32)
    wide, _ = read_panel(image, width=7, threshold=46)
    interval = np.maximum(
        abs(narrow.dndlog10d_cm3 - base.dndlog10d_cm3),
        abs(wide.dndlog10d_cm3 - base.dndlog10d_cm3),
    )
    base["mask_sensitivity_cm3"] = interval
    base.to_csv(DEST / "run24_psd.csv", index=False, float_format="%.7g")
    gm.to_csv(DEST / "run24_gmd.csv", index=False, float_format="%.7g")
    print(
        f"{len(base)} cells, usable fraction ≥{base.valid_fraction.min():.3f}, "
        f"maximum sensitivity {interval.max():.1f} cm^-3 per decade"
    )


RGB_SHA256 = "f72ccdf93d9ecde6dfdb628d61557a30f5ce9c3820091c61ec88fe5b61432163"


if __name__ == "__main__":
    main()
