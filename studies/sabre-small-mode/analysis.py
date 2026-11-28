"""Fit graphical SABRE number spectra and predict area and extinction loss.

See review.md for replay commands.
All integrations retain the reported spectrum's concentration basis.
"""

import csv
import itertools
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import particula
import scipy
from particula.optics import get_cross_sections, get_mie_efficiencies
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
WAVELENGTH = 532e-9
# Limits separate a nucleation-size feature, the mode of interest and the
# larger-particle shoulder. Median diameters in nm, heights per log10 D.
LIMITS = {
    "ultrafine": ([0.01, 2, 1.10], [2000, 20, 3.00]),
    "small": ([0.01, 20, 1.12], [2000, 180, 3.00]),
    "large": ([0.01, 180, 1.10], [2000, 900, 3.00]),
}


def component(diameter, height, median, gsd):
    """Evaluate a lognormal number density per decade."""
    return height * np.exp(
        -0.5 * (np.log(diameter / median) / np.log(gsd)) ** 2
    )


def fit(diameter, density, modes, weighting):
    """Estimate bounded mode parameters from three initial conditions."""
    bounds = np.log(np.hstack([LIMITS[key] for key in modes]))
    guesses = {
        "ultrafine": (50, 8, 1.8),
        "small": (200, 65, 1.8),
        "large": (8, 400, 1.6),
    }

    def residual(par):
        predicted = sum(
            component(diameter, *row) for row in np.exp(par).reshape(-1, 3)
        )
        if weighting == "log-number":
            return np.log(predicted) - np.log(density)
        return (predicted - density) / np.sqrt(density)

    trials = []
    for factor in [0.7, 1.0, 1.4]:
        seed = np.array([guesses[key] for key in modes])
        seed[:, 1] *= factor
        trial = least_squares(
            residual, np.log(seed.ravel()), bounds=bounds, max_nfev=3000
        )
        if trial.success:
            trials.append(trial)
    best = min(trials, key=lambda result: result.cost)
    return np.exp(best.x).reshape(-1, 3)


def integrate(diameter, density, optical_area):
    """Integrate a per-decade spectrum times its particle cross section."""
    # optical_area in µm². Its product with per-cm³ number gives Mm⁻¹.
    return np.trapezoid(density * optical_area, x=np.log10(diameter))


def optical_kernel(diameter, index):
    """Return 532 nm extinction cross sections in square micrometers."""
    radius = diameter * 0.5e-9
    q = get_mie_efficiencies(radius, WAVELENGTH, index)
    return get_cross_sections(q, radius).extinction * 1e12


def write_rows(name, rows):
    """Save a table of derived study results."""
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _figure_rows(group, grid, observed, full, small, area_kernel, optics):
    curves = []
    for i in range(0, grid.size, 5):
        curves.append(
            dict(
                n2o_lo_ppbv=group,
                diameter_nm=grid[i],
                graph_number=observed[i],
                fit_number=full[i],
                small_number=small[i],
                graph_surface=observed[i] * area_kernel[i],
                fit_surface=full[i] * area_kernel[i],
                retained_surface=(full[i] - small[i]) * area_kernel[i],
                small_extinction=small[i] * optics[i],
                full_extinction=full[i] * optics[i],
            )
        )
    return curves


def analyze_group(group, diameter, density):
    """Compare modal fits and optics for one tracer group."""
    np.testing.assert_array_less(0, np.diff(diameter))
    np.testing.assert_array_less(0, density)
    grid = np.geomspace(diameter[0], diameter[-1], 1001)
    observed = np.exp(
        np.interp(np.log(grid), np.log(diameter), np.log(density))
    )
    area_kernel = np.pi * (grid * 0.001) ** 2
    optics = optical_kernel(grid, 1.50)
    empirical_area = integrate(grid, observed, area_kernel)
    empirical_ext = integrate(grid, observed, optics)
    curves, summary, parameters, sensitivities, bands, precision = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    cases = {
        "reference": (list(LIMITS), "log-number", 0),
        "number-weighted": (list(LIMITS), "sqrt-number", 0),
        "above-25-nm": (["small", "large"], "log-number", 25),
        "without-small-refit": (["ultrafine", "large"], "log-number", 0),
    }
    for label, (modes, weight, cutoff) in cases.items():
        keep = diameter >= cutoff
        estimates = fit(diameter[keep], density[keep], modes, weight)
        fitted = {
            key: component(grid, *par)
            for key, par in zip(modes, estimates, strict=True)
        }
        full = sum(fitted.values())
        predicted_knots = sum(component(diameter, *par) for par in estimates)
        error = np.sqrt(
            np.mean(np.log(predicted_knots[keep] / density[keep]) ** 2)
        )
        for key, (height, median, gsd) in zip(modes, estimates, strict=True):
            parameters.append(
                dict(
                    n2o_lo_ppbv=group,
                    case=label,
                    mode=key,
                    height_cm3=height,
                    number_median_nm=median,
                    gsd=gsd,
                    surface_peak_nm=median * np.exp(2 * np.log(gsd) ** 2),
                )
            )
        # Score the truncated fit only on its fitted support.
        use = grid >= cutoff

        def moment(spectrum, kernel, use=use):
            return integrate(grid[use], spectrum[use], kernel[use])

        area, ext = moment(full, area_kernel), moment(full, optics)
        small = fitted.get("small", np.zeros_like(grid))
        small_area, small_ext = (
            moment(small, area_kernel),
            moment(small, optics),
        )
        summary.append(
            dict(
                n2o_lo_ppbv=group,
                case=label,
                n_knots=int(keep.sum()),
                min_nm=grid[use][0],
                max_nm=grid[-1],
                rms_log_number=error,
                graphical_area=moment(observed, area_kernel),
                full_area=area,
                retained_area=area - small_area,
                small_area=small_area,
                graphical_ext=moment(observed, optics),
                full_ext=ext,
                retained_ext=ext - small_ext,
                small_ext=small_ext,
                area_fraction=small_area / area,
                ext_fraction=small_ext / ext,
            )
        )
        if label != "reference":
            continue
        for population, spectrum in [
            ("graphical total", observed),
            ("fitted small mode", small),
        ]:
            for observable, kernel in [
                ("surface", area_kernel),
                ("extinction", optics),
            ]:
                cumulative = cumulative_trapezoid(
                    spectrum * kernel, x=np.log10(grid), initial=0
                )
                bands.append(
                    dict(
                        n2o_lo_ppbv=group,
                        population=population,
                        observable=observable,
                        total=cumulative[-1],
                        median_contribution_nm=np.interp(
                            0.5 * cumulative[-1], cumulative, grid
                        ),
                        fraction_below_100nm=np.interp(100, grid, cumulative)
                        / cumulative[-1],
                        fraction_below_150nm=np.interp(150, grid, cumulative)
                        / cumulative[-1],
                        fraction_below_200nm=np.interp(200, grid, cumulative)
                        / cumulative[-1],
                    )
                )
        # Conditional on full model = measured extinction. Here u is a
        # hypothetical fractional tolerance on the measurement.
        for tolerance in [0.05, 0.10, 0.20]:
            fraction = min(1.0, tolerance * ext / small_ext)
            precision.append(
                dict(
                    n2o_lo_ppbv=group,
                    tolerance_fraction=tolerance,
                    tolerance_Mm_inverse=tolerance * ext,
                    max_small_fraction_omitted=fraction,
                    max_area_omitted_um2_cm3=fraction * small_area,
                )
            )
        curves.extend(
            _figure_rows(
                group, grid, observed, full, small, area_kernel, optics
            )
        )
        for scale, real, imag in itertools.product(
            [0.90, 1.0, 1.10], [1.40, 1.50, 1.60], [0.0, 0.01]
        ):
            # Apply a shared sizing change; only the optical index of the
            # removed mode sets the absolute change in extinction.
            kernel = optical_kernel(grid * scale, complex(real, imag))
            delta_area = moment(small, area_kernel) * scale**2
            delta_ext = moment(small, kernel)
            area_cdf = cumulative_trapezoid(
                observed * area_kernel * scale**2, x=np.log10(grid), initial=0
            )
            ext_cdf = cumulative_trapezoid(
                observed * kernel, x=np.log10(grid), initial=0
            )
            cutoff = 150 / scale
            sensitivities.append(
                dict(
                    n2o_lo_ppbv=group,
                    diameter_scale=scale,
                    real_index=real,
                    imaginary_index=imag,
                    removed_area=delta_area,
                    removed_ext=delta_ext,
                    area_per_ext=delta_area / delta_ext,
                    graphical_fraction_area_below_150=np.interp(
                        cutoff, grid, area_cdf
                    )
                    / area_cdf[-1],
                    graphical_fraction_ext_below_150=np.interp(
                        cutoff, grid, ext_cdf
                    )
                    / ext_cdf[-1],
                )
            )
        # Independent re-integration on half the nodes catches grid effects.
        for spectrum, kernel in [(small, area_kernel), (small, optics)]:
            coarse = integrate(grid[::2], spectrum[::2], kernel[::2])
            np.testing.assert_allclose(
                coarse, moment(spectrum, kernel), rtol=2e-4
            )
        np.testing.assert_allclose((full - small) + small, full)
        np.testing.assert_array_equal(full >= small, True)
        np.testing.assert_array_equal(
            np.isfinite([area, ext, empirical_area, empirical_ext]), True
        )
    return curves, summary, parameters, sensitivities, bands, precision


def main():
    """Fit both tracer groups and save the replay evidence."""
    source = np.genfromtxt(
        HERE / "jensen_spectra.csv", names=True, delimiter=","
    )
    outputs = [[], [], [], [], [], []]
    for group in [270, 300]:
        subset = source[source["n2o_lo_ppbv"] == group]
        result = analyze_group(
            group, subset["diameter_nm"], subset["dndlog10d_cm3"]
        )
        for dest, rows in zip(outputs, result, strict=True):
            dest.extend(rows)
    for name, rows in zip(
        [
            "figure-data.csv",
            "comparisons.csv",
            "fits.csv",
            "sensitivity.csv",
            "size-contributions.csv",
            "conditional-bounds.csv",
        ],
        outputs,
        strict=True,
    ):
        write_rows(name, rows)
    metadata = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,
        particula=particula.__version__,
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=HERE,
            text=True,
        ).strip(),
        wavelength_nm=532,
        integration_points=1001,
        backend="CPU",
        medium_index=1.0,
    )
    (HERE / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
