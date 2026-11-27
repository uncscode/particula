"""Read Liu Figure 7 using retained marker centers and the printed log-viscosity axis."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGE = np.asarray(
    Image.open(HERE / "fig7-000.png").convert("RGB"), dtype=float
)
# Centroids of the seven labeled major ticks, measured on the left axis.
TICK_Y = [16.5, 94.5, 172.5, 251.5, 329.5, 407.5, 486.0]
SLOPE, INTERCEPT = np.polyfit(TICK_Y, [13, 11, 9, 7, 5, 3, 1], 1)


# Five and ten percent are inside the 0-15% segment and have no printed symbols.
# At 75% obtain the marker anchor from its two connecting line segments.
# This avoids assuming that the bounding-box center of a triangle is its anchor.
def segment_height(x1, x2, y1, y2, evaluate_at):
    xx, yy = [], []
    for x in range(round(x1) + 17, round(x2) - 16):
        y = round(y1 + (y2 - y1) * (x - x1) / (x2 - x1))
        indices = np.arange(y - 6, y + 7)
        center = IMAGE[indices, x, :].mean(axis=1)
        background = np.median(IMAGE[indices, x - 7 : x + 8, :], axis=1).mean(
            axis=1
        )
        dark = np.maximum(0, background - center)
        if dark.max() < 10:
            continue
        selected = np.abs(indices - indices[dark.argmax()]) <= 2
        xx.append(x)
        yy.append(np.average(indices[selected], weights=dark[selected]))
    assert len(xx) > 25
    return np.polyval(np.polyfit(xx, yy, 1), evaluate_at)


def main():
    names = ["pinene", "ocimene", "caryophyllene", "farnesene"]
    # Approximate y locations at 60,75,90% RH constrain local curve identification.
    positions = [
        (355.5, 440.3, 541),
        (150, 335.5, 509.5),
        (48, 217, 435),
        (17, 164, 421),
    ]
    rows = []
    for name, (a, b, c) in zip(names, positions, strict=True):
        left = segment_height(503.5, 588.75, a, b, 588.75)
        right = segment_height(588.75, 674, b, c, 588.75)
        assert abs(left - right) < 0.6
        for rh in (0, 5, 10, 75):
            capped = rh != 75 and name != "pinene"
            if rh == 75:
                x, y = 588.75, (left + right) / 2
                source = "plotted marker"
            else:
                x = 162 + rh * (247.5 - 162) / 15
                # Native square centers at 0% and 15%; the other three series
                # are overlaid on the chart ceiling for this entire interval.
                y = 16.5 if capped else 119.5 + (167 - 119.5) * rh / 15
                source = "plotted marker" if rh == 0 else "log-linear segment"
            log_eta = 13.0 if capped else SLOPE * y + INTERCEPT
            rows.append(
                [
                    name,
                    rh,
                    source,
                    round(x, 3),
                    round(y, 3),
                    round(log_eta, 5),
                    f"{10**log_eta:.6g}",
                    int(capped),
                    round(abs(2 * SLOPE), 5),
                ]
            )
    with (HERE / "figure7_viscosity.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "precursor",
                "rh_pct",
                "reading",
                "x_px",
                "y_px",
                "log10_displayed_eta_pa_s",
                "displayed_eta_pa_s",
                "at_plot_ceiling",
                "two_pixel_reading_dex",
            ]
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
