"""Export paired 1, 6 and 24 hour responses from the saved study."""

import csv
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["uf_cm3", "above_cm3", "volume_above_100_um3_cm3"]


def export(root):
    """Keep particle seed pairing in uncertainty estimates for each effect."""
    data = json.loads(
        (root / "website/results/mode-attraction/data.json").read_text()
    )
    path = root / "website/public/study-data/mam4xx/attraction-response.csv"
    for name in ("archived-attraction.csv", "mode-coefficients.csv"):
        copyfile(
            root / "docs/data/mam4xx-coagulation" / name, path.parent / name
        )
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "regime",
                "initial_widths",
                "coarse_present",
                "method",
                "bins_per_decade",
                "hamaker_J",
                "observable",
                "hour",
                "bare_mean",
                "attraction_mean",
                "absolute_change",
                "percent_change",
                "paired_95ci_halfwidth_points",
                "seeds",
            ]
        )
        for method in ("modal", "sectional", "monte_carlo"):
            rows = data[method]
            for r in rows:
                if r["law"] != "complete":
                    continue
                key = ("regime", "widths", "coarse", "resolution")
                (bare,) = [
                    q
                    for q in rows
                    if q["law"] == "bare" and all(q[k] == r[k] for k in key)
                ]
                mat = r.get("hamaker", data["nominal_hamaker_J"])
                for j, quantity in enumerate(COLUMNS):
                    for i, hour in enumerate(data["hours"][1:], 1):
                        uncertainty = None
                        seeds = None
                        if method == "monte_carlo":
                            if bare["seeds"] != r["seeds"]:
                                raise ValueError("Seed pairing mismatch")
                            a, b = (
                                np.asarray(q["endpoints"])[:, i, j]
                                for q in (bare, r)
                            )
                            change = 100 * (b / a - 1)
                            seeds = len(change)
                            uncertainty = (
                                student.ppf(0.975, seeds - 1)
                                * change.std(ddof=1)
                                / np.sqrt(seeds)
                            )
                            mean_a, mean_b, pct = (
                                a.mean(),
                                b.mean(),
                                change.mean(),
                            )
                        else:
                            mean_a, mean_b = bare[quantity][i], r[quantity][i]
                            pct = 100 * (mean_b / mean_a - 1)
                        writer.writerow(
                            [
                                r["regime"],
                                r["widths"],
                                r["coarse"],
                                method,
                                r["resolution"],
                                mat,
                                quantity,
                                hour,
                                mean_a,
                                mean_b,
                                mean_b - mean_a,
                                pct,
                                uncertainty,
                                seeds,
                            ]
                        )


def export_spectra(root):
    """Write finite-bin number densities and empirical bands for the regimes."""
    data = json.loads((root / "website/results/regimes/data.json").read_text())
    out = root / "website/public/study-data/mam4xx/regime-spectra.csv"
    with out.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        fields = [
            "regime",
            "lower_nm",
            "upper_nm",
            "initial_aitken",
            "initial_accumulation",
            "initial_coarse",
            "fine_brownian_mc_mean",
            "fine_brownian_mc_sd",
        ]
        for law in ("neutral", "dispersion"):
            fields.extend(
                law + "_" + k
                for k in ("sectional", "mc_mean", "mc_p16", "mc_p84")
            )
        writer.writerow(fields)
        for r in data["atmospheric_reference"]:
            reg = r["regime"]
            native = r["refinement"]
            edges = np.asarray(native["interior_edges_nm"])
            width = np.diff(np.log10(edges))
            curves = [
                number
                * np.diff(
                    ndtr(np.log(edges / (1000 * diameter)) / np.log(sigma))
                )
                for diameter, sigma, number in data["input_modes"][reg]
            ]
            n = len(native["seeds"])
            curves.extend(
                [
                    np.array(native["spectrum"]["mean"])[-1, 1:-1],
                    np.array(native["spectrum"]["ci95"])[-1, 1:-1]
                    * np.sqrt(n)
                    / student.ppf(0.975, n - 1),
                ]
            )
            for law in ("neutral", "dispersion"):
                section = next(
                    x
                    for x in data["cases"]
                    if (x["regime"], x["control"], x["mechanism"])
                    == (reg, "central", law)
                )
                particle = data["ranking_particles"][reg][law]
                curves.append(section["spectrum_cm3"][-1])
                curves.extend(
                    np.array(particle[k])[-1, 1:-1]
                    for k in ("mean", "p16", "p84")
                )
            densities = np.asarray(curves) / width
            for low, high, values in zip(
                edges[:-1], edges[1:], densities.T, strict=True
            ):
                writer.writerow([reg, low, high, *values])


if __name__ == "__main__":
    export(ROOT)
    export_spectra(ROOT)
