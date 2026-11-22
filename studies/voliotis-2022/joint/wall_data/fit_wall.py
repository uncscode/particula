"""Fit an effective neutral box parameter to Shao et al. (2022) Fig. 5."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from particula.dynamics.properties.wall_loss_coefficient import (
    get_rectangle_wall_loss_coefficient_via_system_state,
)
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
SOURCE_PDF = (
    "https://amt.copernicus.org/articles/15/539/2022/amt-15-539-2022.pdf"
)
PDF_SHA = "d66067b17792ec2b8488228be3cab931a6be73cc45ede0408a9b865a8ceab8b2"
# Centers of outer frame strokes, in embedded-image pixels. See README.
X0, X600, Y0, YMAX = 390.5, 2002.0, 1589.5, 34.0
DIMENSIONS_M = (3.0, 2.0, 3.0)
DENSITY = 1770.0
TEMPERATURE = 298.15
PRESSURE = 101325.0


def rates(diameter_nm, eddy):
    """Compute neutral loss (s⁻¹) for spherical particle diameters in nm."""
    return get_rectangle_wall_loss_coefficient_via_system_state(
        wall_eddy_diffusivity=eddy,
        particle_radius=diameter_nm * 0.5e-9,
        particle_density=DENSITY,
        temperature=TEMPERATURE,
        pressure=PRESSURE,
        chamber_dimensions=DIMENSIONS_M,
    )


def fit(diameter, target, log_residual):
    """Fit one E (s⁻¹) across diameters and positive decay rates."""

    def residual(eddy_log):
        model = rates(diameter, np.exp(eddy_log))
        return np.log(model / target) if log_residual else model - target

    fit_result = minimize_scalar(
        lambda eddy_log: np.mean(residual(eddy_log) ** 2),
        bounds=np.log([1e-4, 1e4]),
        method="bounded",
        options={"xatol": 1e-9},
    )
    if not fit_result.success:
        raise RuntimeError(fit_result.message)
    eddy = np.exp(fit_result.x)
    predicted = rates(diameter, eddy)
    return {
        "count": len(diameter),
        "eddy_s-1": eddy,
        "rms_log_rate_ratio": np.mean(np.log(predicted / target) ** 2) ** 0.5,
        "rmse_s-1": np.mean((predicted - target) ** 2) ** 0.5,
        "max_absolute_relative_error": np.max(np.abs(predicted / target - 1)),
    }


def verify_overlay(source, x, y, isolated):
    """Draw the transcribed centers over the source raster."""
    from PIL import Image, ImageDraw

    with Image.open(source) as im:
        if im.size != (2067, 1827):
            raise ValueError(
                "Use the original embedded 2067 × 1827 pixel image"
            )
        overlay = im.convert("RGB")
    drawing = ImageDraw.Draw(overlay)
    for xp, yp, clear in zip(x, y, isolated, strict=True):
        color = "#b80068" if clear else "#c73a00"
        drawing.line((xp - 8, yp, xp + 8, yp), fill=color, width=4)
        drawing.line((xp, yp - 8, xp, yp + 8), fill=color, width=4)
    overlay.thumbnail((1060, 1000), Image.Resampling.LANCZOS)
    overlay.save(HERE / "figure5_digitization.png")


def main():
    """Rebuild the calibration and optionally verify the publisher source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-image", type=Path, help="page 10 pdfimages PNG"
    )
    parser.add_argument(
        "--source-pdf", type=Path, help="verify publisher PDF hash"
    )
    args = parser.parse_args()
    if args.source_pdf:
        if hashlib.sha256(args.source_pdf.read_bytes()).hexdigest() != PDF_SHA:
            raise ValueError("PDF checksum differs from the digitized source")

    points = np.genfromtxt(
        HERE / "figure5_markers.csv", names=True, delimiter=","
    )
    x, y = points["x_px"], points["y_px"]
    isolated = points["isolated"].astype(bool)
    diameter = (x - X0) * 600 / (X600 - X0)
    measured = (Y0 - y) * 2e-4 / (Y0 - YMAX)
    results = {}
    for name, mask in (
        ("all_readable", np.ones_like(isolated)),
        ("isolated", isolated),
    ):
        for kind in ("log", "absolute"):
            results[f"{name}_{kind}"] = fit(
                diameter[mask], measured[mask], kind == "log"
            )
    model = rates(diameter, results["all_readable_log"]["eddy_s-1"])
    with (HERE / "wall_rates.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["diameter_nm", "apparent_loss_s-1", "neutral_fit_s-1", "isolated"]
        )
        for d, obs, mod, clear in zip(
            diameter, measured, model, isolated, strict=True
        ):
            writer.writerow(
                [f"{d:.3f}", f"{obs:.7g}", f"{mod:.7g}", int(clear)]
            )
    data = {
        "source_pdf": SOURCE_PDF,
        "source_pdf_sha256": PDF_SHA,
        "fits": results,
    }
    (HERE / "wall_fit.json").write_text(json.dumps(data, indent=2) + "\n")
    if args.source_image:
        verify_overlay(args.source_image, x, y, isolated)


if __name__ == "__main__":
    main()
