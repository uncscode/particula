"""Match modal moments to native size-bin counts and check reference closure."""

from pathlib import Path

import numpy as np
from scipy.special import ndtr


def attach_modal(report):
    """Attach common-bin reference spectra and timestep checks."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/data/mam4xx-coagulation/modal.csv"
    )
    table = np.genfromtxt(path, names=True, delimiter=",")
    edges = np.r_[0.0, report["edges_nm"], np.inf] * 1e-9
    points = []
    for case in report["cases"]:
        if case["slots"] != 3840:
            continue
        rows = table[
            (table["acc_nm"] == case["diameter_nm"])
            & (table["ratio"] == case["ratio"])
        ]
        levels = {dt: rows[rows["dt_s"] == dt] for dt in (10, 30, 60)}
        reference = levels[10]
        trajectory = reference["aitken_m3"] + reference["accum_m3"]
        spectra = []
        for row in reference:
            with np.errstate(divide="ignore"):
                cdf_a = ndtr(np.log(edges / row["aitken_dg_m"]) / np.log(1.6))
                cdf_b = ndtr(np.log(edges / row["accum_dg_m"]) / np.log(1.8))
            spectra.append(
                row["aitken_m3"] * np.diff(cdf_a)
                + row["accum_m3"] * np.diff(cdf_b)
            )
        spectra = np.asarray(spectra)
        np.testing.assert_allclose(trajectory, spectra.sum(axis=1), rtol=1e-12)
        for level in levels.values():
            np.testing.assert_array_equal(level["time_s"], report["times_s"])
            mass = level["aitken_mass_kg_m3"] + level["accum_mass_kg_m3"]
            np.testing.assert_allclose(mass / mass[0], 1.0, rtol=1e-12)
        points.append(
            dict(
                diameter_nm=case["diameter_nm"],
                ratio=case["ratio"],
                number_cm3=(trajectory / 1e6).tolist(),
                spectrum_cm3=(spectra / 1e6).tolist(),
                initial_rate_cm3_s=reference["loss_m3_s"][0] / 1e6,
                max_dt_difference_fraction=max(
                    float(
                        np.max(
                            np.abs(
                                (lev["aitken_m3"] + lev["accum_m3"])
                                / trajectory
                                - 1
                            )
                        )
                    )
                    for lev in levels.values()
                ),
            )
        )
    report["modal"] = points
    report["diagnostics"] = diagnostics(report)
    return report


def diagnostics(report):
    """Report endpoint differences, refinement shifts and sampling intervals."""
    output = []
    for modal in report["modal"]:

        def matches(case, modal=modal):
            return (case["diameter_nm"], case["ratio"]) == (
                modal["diameter_nm"],
                modal["ratio"],
            )

        series = {
            case["slots"]: case for case in report["cases"] if matches(case)
        }
        high = series[3840]
        medium = series[1920]
        reference = modal["number_cm3"][-1]
        n_high = high["number_cm3"][-1]
        final = np.asarray(high["spectrum_cm3"][-1])
        ref_final = np.asarray(modal["spectrum_cm3"][-1])
        change = [c for c in report["refinements"] if matches(c)]
        output.append(
            dict(
                diameter_nm=modal["diameter_nm"],
                ratio=modal["ratio"],
                initial_rate_percent=100
                * (
                    high["initial_rate_cm3_s"] / modal["initial_rate_cm3_s"] - 1
                ),
                final_mam_cm3=reference,
                final_gpu_cm3=n_high,
                final_percent=100 * (n_high / reference - 1),
                halfwidth_percent=100 * high["ci95_cm3"][-1] / reference,
                resolution_percent=100
                * (n_high - medium["number_cm3"][-1])
                / reference,
                refinement_percent=(
                    100
                    * (change[0]["number_cm3"][-1] - medium["number_cm3"][-1])
                    / reference
                    if change
                    else None
                ),
                spectrum_total_variation=0.5
                * np.abs(
                    final / final.sum() - ref_final / ref_final.sum()
                ).sum(),
                max_mass_error=max(
                    c["max_mass_error"] for c in series.values()
                ),
            )
        )
    return output
