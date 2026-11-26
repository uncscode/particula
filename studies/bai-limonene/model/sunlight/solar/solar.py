"""Recover the Bai S4 input photon spectrum and two limonene action spectra."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np

SUP_HASH = "c0a06cbe2e8ceb9b5654f0579e8bcfdf830bd425254b48acda5465bfeb44d8e3"
PAPER_HASH = "1c58775829762acab04e4391dcd38da77ec16f2bee0b753c66021397cf763ed5"
NUMBER = re.compile(r"[-+]?(?:\d*\.)?\d+(?:e[-+]?\d+)?", re.I)
TERMS = {"mvos": (7.0e-20, 1.4e-21), "lvos": (1.3e-20, 2.4e-22)}
PUBLISHED = {"mvos": 7.8e-6, "lvos": 1.3e-6}
OUT = Path(__file__).parent


def curves(pdf, page, tint, style):
    """Read MATLAB paths exported in the PDF, after the SVG affine transform."""
    with TemporaryDirectory() as folder:
        dest = Path(folder) / "page.svg"
        subprocess.run(  # noqa: S603
            [  # noqa: S607
                "pdftocairo",
                "-f",
                str(page),
                "-l",
                str(page),
                "-svg",
                str(pdf),
                str(dest),
            ],
            check=True,
        )
        paths = ElementTree.parse(dest).findall(".//{*}path")  # noqa: S314
    for element in paths:
        color = element.get(style, "")
        if not color.startswith(tint):
            continue
        path = element.attrib["d"]
        if len(path) < 1000 or "C" in path:
            continue
        points = np.array(NUMBER.findall(path), dtype=float).reshape(-1, 2)
        if "transform" in element.attrib:
            mat = np.array(
                NUMBER.findall(element.attrib["transform"]), dtype=float
            )
            points = points @ mat[:4].reshape(2, 2) + mat[4:]
        yield points


def s4(pdf):
    """Tabulate individual vector vertices on the labeled linear axes."""
    traces = {}
    for source, color in [
        ("solar", "63.499"),
        ("lamp300", "85.099"),
        ("lamp345", "92.898"),
    ]:
        (p,) = curves(pdf, 23, "rgb(" + color, "stroke")
        # Calibrate using x ticks 280, 400 nm and y ticks 0, 12e14.
        wavelength = 280 + (p[:, 0] - 179.3203125) * 120 / 158.0
        photons = (409.5625 - p[:, 1]) * 12e14 / (409.5625 - 295.33984375)
        # PDF quantization is ±0.002 point. A line on the zero baseline is zero.
        photons = np.where(
            abs(photons) < 0.002 * 12e14 / 114.22265625, 0, photons
        )
        if np.min(photons) < 0 or np.any(np.diff(wavelength) < 0):
            raise ValueError("Invalid digitized solar or lamp curve")
        traces[source] = wavelength, photons
    return traces


def fig4(paper):
    """Recover the upper envelope of the shaded limonene LVOS curve in 4c."""
    # The source SVG flattens two overlapping translucent fills. Where the
    # limonene trace exceeds the isoprene trace, its upper edge is pink. Else
    # it is the upper boundary of their brown overlap. Read both contours.
    segments = []
    for tint in ("rgb(97.599", "rgb(85.899"):
        (points,) = curves(paper, 8, tint, "fill")
        wave = 280 + (points[:, 0] - 101.8671875) * 90 / (
            260.21484375 - 101.8671875
        )
        rate = (569.9375 - points[:, 1]) * 1.4e-7 / (569.9375 - 445.55859375)
        nodes = np.unique(wave)
        segments.append(
            (nodes, np.array([max(rate[wave == w]) for w in nodes]))
        )
    nodes = np.unique(np.concatenate([segment[0] for segment in segments]))
    rates = np.maximum.reduce(
        [np.interp(nodes, *segment, left=0, right=0) for segment in segments]
    )
    return nodes, rates


def action(wavelength, species):
    """Log-linear spectral action per incident photon per Table S12."""
    first, last = TERMS[species]
    return first * np.exp(np.log(last / first) * (wavelength - 300) / 45)


def write(path, rows):
    """Write one numerical table with its named header."""
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(paper, supplement, dest, interval=0.25):
    """Reconstruct the integrated rates from both source figures."""
    for p, digest in [(paper, PAPER_HASH), (supplement, SUP_HASH)]:
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source version mismatch: {p}")
    traces = s4(supplement)
    nm = np.arange(280.0, 370.0 + 0.5 * interval, interval)
    photons = np.interp(nm, *traces["solar"])
    rows = {
        "wavelength_nm": nm,
        "solar_photons_cm2_s_nm": photons,
        "solar_irradiance_w_m2_nm": photons
        * 1e4
        * (6.62607015e-34 * 299792458.0)
        / (nm * 1e-9),
    }
    rates = []
    for which in ("lamp300", "lamp345"):
        rows[f"{which}_photons_cm2_s_nm"] = np.interp(nm, *traces[which])
    measured_nm, measured_j = fig4(paper)
    rows["fig4_lvos_per_s_nm"] = np.interp(
        nm, measured_nm, measured_j, left=0, right=measured_j[-1]
    )
    for species in TERMS:
        a = action(nm, species)
        kernel = photons * a
        j = np.trapezoid(kernel, nm)
        rows[f"action_{species}_cm2"] = a
        rows[f"j_{species}_per_s_nm"] = kernel
        mask = (nm >= 320) & (nm <= 350)
        rates.append(
            dict(
                species=species,
                standard_j_per_s=j,
                published_j_per_s=PUBLISHED[species],
                ratio_to_published=j / PUBLISHED[species],
                standard_lifetime_h=1 / (j * 3600),
                peak_nm=nm[np.argmax(kernel)],
                band_320_350_share=np.trapezoid(kernel[mask], nm[mask]) / j,
            )
        )
    dest.mkdir(exist_ok=True, parents=True)
    write(
        dest / "spectrum.csv",
        [
            dict(zip(rows, vs, strict=True))
            for vs in zip(*rows.values(), strict=True)
        ],
    )
    write(dest / "rates.csv", rates)
    # The incident energy in the complete plotted S4 ranges includes lamp tails.
    diagnostics = {}
    for tag, (wav, flux) in traces.items():
        diagnostics[tag + "_s4_280_400_irradiance_w_m2"] = np.trapezoid(
            flux * 1e4 * 6.62607015e-34 * 299792458.0 / (wav * 1e-9), wav
        )
    published_shape = rows["fig4_lvos_per_s_nm"]
    diagnostics["fig4_lvos_integral_per_s"] = np.trapezoid(published_shape, nm)
    diagnostics["fig4_lvos_peak_nm"] = nm[np.argmax(published_shape)]
    diagnostics["fig4_lvos_peak_per_s_nm"] = max(published_shape)
    diagnostics["s4_action_lvos_peak_per_s_nm"] = max(rows["j_lvos_per_s_nm"])
    (dest / "checks.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return rates, diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", type=Path)
    parser.add_argument("supplement", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--step", type=float, default=0.25)
    args = parser.parse_args()
    print(analyze(args.paper, args.supplement, args.out, args.step))
