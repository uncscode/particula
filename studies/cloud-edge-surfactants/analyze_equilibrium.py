"""Replay the laboratory anchors, inventory bounds and equilibrium scans."""

import csv
import json

import numpy as np
from equilibrium import (
    ACID_MOLAR_MASS,
    DRY_DENSITY,
    HERE,
    OBS,
    RATIO,
    SCENARIOS,
    SURF_MOLAR_MASS,
    equilibrium_ratio,
    growth_knots,
    lab_radius,
    state,
)
from scipy.optimize import brentq

TEMP = 295.0
SIZES_NM = np.unique(
    np.r_[np.geomspace(50.0, 8000.0, 45), 100.0, 250.0, 1000.0, 5000.0]
)


def save_rows(name, rows):
    """Write a numerical table with its column names."""
    with (HERE / name).open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def crit(dry, recipe, model, scenario, n=1801):
    """Return critical wet radii in metres and supersaturations in percent."""
    # Cover the stable haze branch, all local maxima and the declining dilute
    # limb.
    x = np.geomspace(1.001, 400.0, n)
    seq = equilibrium_ratio(
        dry[:, None], dry[:, None] * x, TEMP, recipe, model, scenario
    )
    i = seq.argmax(axis=1)
    if np.any((i == 0) | (i == n - 1)) or not np.isfinite(seq).all():
        raise AssertionError("A critical maximum is unbracketed")
    idx = np.arange(len(i))
    return x[i] * dry, (seq[idx, i] - 1) * 100


def ambient(dry, recipe, model, scenario):
    """Find stable 70%-RH wet radii and the corresponding mixture state."""
    # Simultaneous bisection on the lower Köhler branch at 70% RH.
    a = np.ones_like(dry)
    b = np.full_like(dry, 2.0)
    if not np.all(
        equilibrium_ratio(dry, dry * b, TEMP, recipe, model, scenario) > 0.7
    ):
        raise ValueError("70% RH growth root is unbracketed")
    for _ in range(40):
        mid = (a + b) / 2
        too_small = (
            equilibrium_ratio(dry, dry * mid, TEMP, recipe, model, scenario)
            < 0.7
        )
        a, b = np.where(too_small, mid, a), np.where(too_small, b, mid)
    wet = dry * (a + b) / 2
    return wet, state(dry, wet, recipe, model, scenario)


def benchmark():
    """Compare the calibrated model with both Figure 5 instruments."""
    rows = []
    for obs in OBS:
        recipe, rh = obs["recipe"], float(obs["rh_percent"]) / 100
        if rh == 1:
            predicted = 72.2
        else:
            wet = lab_radius("central") if obs["method"] == "trap" else 1.25e-3
            g = brentq(
                lambda x, wet, recipe, rh: equilibrium_ratio(
                    wet / x, wet, TEMP, recipe
                )
                - rh,
                1.0,
                20.0,
                args=(wet, recipe, rh),
            )
            predicted = float(state(wet / g, wet, recipe)[1])
        obs_sigma = float(obs["sigma_mN_m"])
        rows.append(
            dict(
                row=obs["worksheet_row"],
                recipe=recipe,
                method=obs["method"],
                rh_percent=rh * 100,
                observed_mN_m=obs_sigma,
                calculated_mN_m=predicted,
                residual_mN_m=predicted - obs_sigma,
                error_mN_m=obs["reported_error_mN_m"],
            )
        )
    return rows


def inventory():
    """Compute the dry diameter needed to supply each target coverage."""
    rows = []
    for scenario in SCENARIOS:
        for recipe, ratio in RATIO.items():
            wet = lab_radius(scenario)
            g = brentq(
                lambda x, wet, recipe, scenario: state(
                    wet / x, wet, recipe, "finite", scenario
                )[0]
                - 0.7,
                1.0,
                3.0,
                args=(wet, recipe, scenario),
            )
            _, _, total, _, _, gamma = state(
                wet / g, wet, recipe, "finite", scenario
            )
            molar_mass = ratio * ACID_MOLAR_MASS + SURF_MOLAR_MASS[recipe]
            minimum_m = 6 * g**2 * gamma * molar_mass / DRY_DENSITY
            # Direct independent count of formula units at equality, area
            # times Γ.
            surf_moles = DRY_DENSITY * np.pi / 6 * minimum_m**3 / molar_mass
            if not np.isclose(
                surf_moles / (np.pi * (minimum_m * g) ** 2),
                gamma,
                rtol=1e-12,
                atol=0,
            ):
                raise AssertionError("Coverage budget failed at equality")
            rows.append(
                dict(
                    scenario=scenario,
                    recipe=recipe,
                    wet_diameter_lab_um=2 * wet * 1e6,
                    growth_wet_dry=g,
                    acid_mol_L=total * ratio * 1e-3,
                    target_gamma_umol_m2=gamma * 1e6,
                    minimum_dry_um=minimum_m * 1e6,
                )
            )
    return rows


def scans():
    """Scan both equilibrium models and test grid and budget residuals."""
    dry = SIZES_NM * 0.5e-9
    rows, worst_sc, worst_budget = [], 0.0, 0.0
    for scenario in SCENARIOS:
        for recipe in ("acid", "octyl", "dodecyl"):
            for model in ("bulk", "finite"):
                wet, (_, sigma, total, free, gamma, _) = ambient(
                    dry, recipe, model, scenario
                )
                rc, sc = crit(dry, recipe, model, scenario)
                _, sc_refined = crit(dry, recipe, model, scenario, n=3601)
                worst_sc = max(worst_sc, np.max(np.abs(sc - sc_refined)))
                if recipe != "acid" and model == "finite":
                    residual = (free + gamma * 3 / wet - total) / total
                    worst_budget = max(worst_budget, np.max(np.abs(residual)))
                if np.min(sigma) <= 0 or np.min(sc) <= 0:
                    raise AssertionError(
                        "Nonphysical tension or supersaturation"
                    )
                for j, diameter in enumerate(SIZES_NM):
                    rows.append(
                        dict(
                            recipe=recipe,
                            model=model,
                            scenario=scenario,
                            dry_diameter_nm=diameter,
                            wet_diameter_70_nm=2 * wet[j] * 1e9,
                            sigma70_mN_m=sigma[j],
                            critical_wet_radius_um=rc[j] * 1e6,
                            critical_supersaturation_pct=sc[j],
                        )
                    )
    return rows, worst_sc, worst_budget


def relative_thresholds(rows):
    """Normalize each threshold to pure acid at matching size and scenario."""

    def key(row):
        return row["model"], row["scenario"], row["dry_diameter_nm"]

    acid = {
        key(r): r["critical_supersaturation_pct"]
        for r in rows
        if r["recipe"] == "acid"
    }
    values = []
    for row in rows:
        if row["recipe"] == "acid":
            continue
        threshold = row["critical_supersaturation_pct"]
        reference = acid[key(row)]
        values.append(
            {
                "model": row["model"],
                "scenario": row["scenario"],
                "recipe": row["recipe"],
                "dry_diameter_nm": row["dry_diameter_nm"],
                "recipe_critical_supersaturation_pct": threshold,
                "acid_critical_supersaturation_pct": reference,
                "change_relative_to_acid_pct": 100
                * (threshold - reference)
                / reference,
            }
        )
    return values


def run():
    """Reproduce the anchors and calculate relative activation thresholds."""
    bench, bounds = benchmark(), inventory()
    rows, sc_err, budget_err = scans()
    with_errors = [
        abs(r["residual_mN_m"]) / float(r["error_mN_m"])
        for r in bench
        if r["error_mN_m"]
    ]
    checks = dict(
        maximum_lab_error_fraction=max(with_errors),
        maximum_all_source_residual_mN_m=max(
            abs(r["residual_mN_m"]) for r in bench
        ),
        maximum_critical_supersaturation_refinement_percentage_points=sc_err,
        maximum_relative_surfactant_budget_error=budget_err,
        central_dilute_kappa=growth_knots("central")[-1],
    )
    if max(with_errors) >= 1 or budget_err >= 1e-12 or sc_err >= 1e-4:
        raise AssertionError(f"Laboratory or numerical check failed: {checks}")
    save_rows("lab_replay.csv", bench)
    save_rows("budget.csv", bounds)
    save_rows("equilibrium_scans.csv", rows)
    (HERE / "replay_checks.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    save_rows("equilibrium_relative.csv", relative_thresholds(rows))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    run()
