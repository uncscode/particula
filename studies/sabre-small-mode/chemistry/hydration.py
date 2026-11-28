"""Hydrate the two SABRE populations using liquid water activity."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from particula_aiomfac import AIOMFACModel, MixtureSpec
from scipy.constants import R

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
MODEL = AIOMFACModel.prepare(
    MixtureSpec(
        {
            "water": {16: 1},
            "butanediol": {142: 2, 150: 2, 153: 2},
            "sulfuric_acid": {205: 1, 248: 1},
        }
    )
)
WATER_MW = 0.01801528
WATER_RHO = 1000.0
ACID_RHO = 1830.0
ORGANIC_RHO = 1020.0
SIGMA = 0.072


class ActivityTable:
    """Single-liquid AIOMFAC speciation by temperature and water content."""

    def __init__(self, organic, temperatures, resolution, salt_fraction=0.0):
        """Tabulate activity and speciation at fixed dry composition."""
        self.organic = organic
        self.salt = salt_fraction
        self.acid = 1 - organic - salt_fraction
        self.rho = 1 / (
            organic / ORGANIC_RHO
            + self.acid / ACID_RHO
            + salt_fraction / 1780.0
        )
        self.model = (
            MODEL
            if salt_fraction == 0
            else AIOMFACModel.prepare(
                MixtureSpec(
                    {
                        "water": {16: 1},
                        "butanediol": {142: 2, 150: 2, 153: 2},
                        "sulfuric_acid": {205: 1, 248: 1},
                        "ammonium_bisulfate": {204: 1, 248: 1},
                    }
                )
            )
        )
        self.temperature = np.linspace(
            np.floor(min(temperatures)),
            np.ceil(max(temperatures)),
            resolution // 10 + 1,
        )
        # Acid-rich, nearly water-free mixtures have numerically singular
        # limits. Use the supported interval and reject out-of-grid targets.
        lower = {0.0: 0.08, 0.25: 0.002, 0.5: 0.0001, 0.75: 0.0001}.get(
            organic, 0.0001 if organic >= 0.5 else 0.001
        )
        self.w = np.geomspace(lower, 50, resolution)
        self.aw = []
        self.xw = []
        for temp in self.temperature:
            z = np.column_stack(
                (
                    self.w,
                    np.full_like(self.w, organic),
                    np.full_like(self.w, 1 - organic),
                )
            )
            if salt_fraction:
                z[:, 2] = self.acid
                z = np.column_stack((z, np.full_like(self.w, salt_fraction)))
            result = self.model.evaluate(z, temp)
            expected_warning = 16 if 288 <= temp <= 309 else 10
            if not np.all(result.warnings == expected_warning):
                raise ValueError("Unexpected AIOMFAC evaluation warning.")
            self.aw.append(result.neutral_activities[:, 0])
            self.xw.append(result.neutral_mole_fractions[:, 0])
        self.aw, self.xw = np.array(self.aw), np.array(self.xw)
        if np.any(np.diff(self.aw, axis=1) <= 0):
            raise ValueError(
                "Nonmonotone water activity on selected liquid branch."
            )

    def interpolate(self, temp):
        """Linearly interpolate log activity between bracketing temperatures."""
        if not self.temperature[0] <= temp <= self.temperature[-1]:
            raise ValueError("Temperature outside the activity table.")
        hi = np.searchsorted(self.temperature, temp, side="right")
        hi = min(hi, len(self.temperature) - 1)
        lo = hi - 1
        f = (temp - self.temperature[lo]) / (
            self.temperature[hi] - self.temperature[lo]
        )
        return [
            (1 - f) * np.log(v[lo]) + f * np.log(v[hi])
            for v in (self.aw, self.xw)
        ]


def water_equilibrium(table, temp, rh, diameter, sigma, ideal):
    """Invert water activity times Kelvin enhancement at fixed liquid RH."""
    if not 0 < rh < 1:
        raise ValueError("This equilibrium branch requires 0 < liquid RH < 1.")
    law, lx = table.interpolate(temp)
    rho = table.rho
    kelvin_nm = 4 * sigma * WATER_MW / (R * temp * WATER_RHO) * 1e9
    logw = np.log(table.w)
    logrh = np.log(rh)

    def inverse(target):
        if np.min(target) < law[0] or np.max(target) > law[-1]:
            raise ValueError(
                f"Outside activity table: {rh}, {temp}, {table.organic}"
            )
        return np.exp(np.interp(target, law, logw))

    w = np.full_like(diameter, inverse(logrh))
    for _ in range(12):
        growth = (1 + w * rho / WATER_RHO) ** (1 / 3)
        w = inverse(logrh - kelvin_nm / (diameter * growth))
    if ideal:
        # Remove the water activity coefficient while holding the electrolyte
        # species inventory at that of the corresponding nonideal solution.
        xw = np.exp(np.interp(np.log(w), logw, lx))
        solute_moles = w / WATER_MW * (1 / xw - 1)
        for _ in range(12):
            xw = rh * np.exp(
                -kelvin_nm / (diameter * (1 + w * rho / WATER_RHO) ** (1 / 3))
            )
            w = WATER_MW * solute_moles * xw / (1 - xw)
    return w, (1 + w * rho / WATER_RHO) ** (2 / 3)


def size_quadrature(edges, group, order=4):
    """Split at the physical cutoff and the support of the modal allocation."""
    cuts = np.unique(np.r_[edges, 150.0, 3.80019, 950.8064])
    left, right = np.log10(cuts[:-1]), np.log10(cuts[1:])
    nodes, weights = leggauss(order)
    d = 10 ** (((left + right)[:, None] + (right - left)[:, None] * nodes) / 2)
    dw = (right - left)[:, None] * weights / 2
    params = pd.read_csv(STUDY / "fits.csv").query(
        "n2o_lo_ppbv==@group and case=='reference'"
    )
    fractions = {}
    for row in params.itertuples():
        fractions[row.mode] = row.height_cm3 * np.exp(
            -0.5 * (np.log(d / row.number_median_nm) / np.log(row.gsd)) ** 2
        )
    small = fractions["small"] / sum(fractions.values())
    small[(d < 3.80019) | (d > 950.8064)] = 0.0
    return (
        d.ravel(),
        dw.ravel(),
        small.ravel(),
        (np.searchsorted(edges, d) - 1).ravel(),
    )


def hydrated_population(table, temp, rh, d, area, sigma, ideal, reference):
    """Calculate hydrated areas, accounting for any assumed reference water."""
    dry_d = d.copy()
    if reference:
        # The AMP inlet reports dry RH <40%, but its exact RH is unavailable.
        # A second, stipulated equilibration gives a dry core for each bin.
        amp_table, amp_temp, amp_rh = reference
        for _ in range(8):
            _, reference_area = water_equilibrium(
                amp_table, amp_temp, amp_rh, dry_d, sigma, False
            )
            dry_d = d / np.sqrt(reference_area)
    w, factor = water_equilibrium(table, temp, rh, dry_d, sigma, ideal)
    wet = area * factor * (dry_d / d) ** 2
    wet_d = dry_d * (1 + w * table.rho / WATER_RHO) ** (1 / 3)
    wet_volume = wet * wet_d / 6000
    water_volume = wet_volume * w * table.rho / (WATER_RHO + w * table.rho)
    return wet, w, wet_volume, water_volume


def run(frame, density, edges, resolution, order, checks):
    """Reconstruct area on the measured dry-diameter coordinate in each case."""
    temperatures = frame.temperature_k.to_numpy()
    tables = {
        org: ActivityTable(org, temperatures, resolution)
        for org in [0.0, 0.25, 0.5, 0.75]
    }
    palm_intervals = pd.read_csv(HERE / "palms-contributing-intervals.csv")
    palm_org_ratio = palm_intervals.org_sulf_mf.median()
    palm_nh4_ratio = palm_intervals.nh4_so4_molar.median()
    molar_dry = np.array(
        [
            96.06 * palm_org_ratio / (1 - palm_org_ratio),
            98.079 * (1 - palm_nh4_ratio),
            115.11 * palm_nh4_ratio,
        ]
    )
    palm_org, _, palm_salt = molar_dry / molar_dry.sum()
    tables[palm_org] = ActivityTable(
        palm_org, temperatures, resolution, palm_salt
    )
    inlet = {
        org: ActivityTable(org, [297.0, 299.0], resolution)
        for org in [0.0, 0.5]
    }
    sections = {g: size_quadrature(edges, g, order) for g in [270, 300]}
    cases = {
        "aiomfac": (0.5, SIGMA, False, 1.0, False),
        "palms_homogeneous": (palm_org, SIGMA, False, 1.0, False),
        "ideal": (0.5, SIGMA, True, 1.0, False),
        "no_kelvin": (0.5, 0.0, False, 1.0, False),
        "25pct_organic": (0.25, SIGMA, False, 1.0, False),
        "75pct_organic": (0.75, SIGMA, False, 1.0, False),
        "rh_minus10pct": (0.5, SIGMA, False, 0.90, False),
        "rh_plus10pct": (0.5, SIGMA, False, 1.10, False),
        "inlet_10pct_rh": (0.5, SIGMA, False, 1.0, True),
    }
    results = []
    residuals = []
    for i, row in enumerate(frame.itertuples()):
        d, width, small, bins = sections[270 if row.n2o_ppbv < 300 else 300]
        area = (
            np.pi
            * (d / 1000) ** 2
            * width
            * density[i, bins]
            * row.number_density_ambient_over_STP
        )
        if not np.isclose(
            area.sum(),
            row.full_area * row.number_density_ambient_over_STP,
            rtol=1e-7,
        ):
            raise ValueError(
                "Dry integration disagrees with archived full population."
            )
        if not np.isclose(
            area @ small,
            row.removed_mode_area * row.number_density_ambient_over_STP,
            rtol=1e-6,
        ):
            raise ValueError(
                "Frozen fitted mode allocation differs from dry replay."
            )
        for case, (org, sigma, ideal, rh_scale, has_inlet) in cases.items():
            if case == "palms_homogeneous" and not np.isfinite(row.org_sulf_mf):
                continue
            other_org = palm_org if case == "palms_homogeneous" else 0.0
            rh = row.rhw_pct / 100 * rh_scale
            if not 0 < rh < 1:
                continue
            rec = dict(
                flight=row.flight,
                start_s=row.start_s,
                stop_s=row.stop_s,
                case=case,
                temperature_k=row.temperature_k,
                rh_liquid=rh,
            )
            for label, frac, organic in [
                ("small", small, org),
                ("other", 1 - small, other_org),
            ]:
                ref = (inlet[organic], 298.0, 0.10) if has_inlet else None
                wet, w, wetvol, watervol = hydrated_population(
                    tables[organic],
                    row.temperature_k,
                    rh,
                    d,
                    area * frac,
                    sigma,
                    ideal,
                    ref,
                )
                rec[f"wet_{label}_volume_um3_cm3"] = wetvol.sum()
                rec[f"water_{label}_volume_um3_cm3"] = watervol.sum()
                rec[f"dry_{label}_area"] = area @ frac
                rec[f"wet_{label}_area"] = wet.sum()
                rec[f"wet_{label}_below150_area"] = wet[d < 150].sum()
                rec[f"{label}_h2so4_wt_pct"] = 100 * np.average(
                    tables[organic].acid / (1 + w), weights=wet
                )
                rec[f"{label}_water_wt_pct"] = 100 * np.average(
                    w / (1 + w), weights=wet
                )
                if checks and case == "aiomfac" and i % 12 == 0:
                    # Compare interpolated equilibrium with the full backend,
                    # including points in the curvature-sensitive diameter bins.
                    take = np.arange(0, len(d), 20)
                    z = np.column_stack(
                        (
                            w[take],
                            np.full(len(take), organic),
                            np.full(len(take), 1 - organic),
                        )
                    )
                    direct = MODEL.evaluate(z, row.temperature_k)
                    kelvin = (
                        4
                        * sigma
                        * WATER_MW
                        / (R * row.temperature_k * WATER_RHO)
                    )
                    ratio = (1 + w[take] * tables[organic].rho / WATER_RHO) ** (
                        1 / 3
                    )
                    check_rh = direct.neutral_activities[:, 0] * np.exp(
                        kelvin / (d[take] * 1e-9 * ratio)
                    )
                    residuals.extend(abs(check_rh / rh - 1))
            rec["dry_total_area"] = area.sum()
            rec["dry_below150_area"] = area[d < 150].sum()
            rec["wet_total_area"] = (
                rec["wet_small_area"] + rec["wet_other_area"]
            )
            rec["wet_below150_area"] = (
                rec["wet_small_below150_area"] + rec["wet_other_below150_area"]
            )
            results.append(rec)
    return pd.DataFrame(results), max(residuals, default=None)


def main():
    """Build kinetic surface inputs from matched DLH humidity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=140)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument(
        "--input", type=Path, default=HERE / "flight-chemistry.csv"
    )
    parser.add_argument("--output", type=Path, default=HERE / "hydration.csv")
    args = parser.parse_args()
    base = pd.read_csv(STUDY / "flight-windows.csv")
    water = pd.read_csv(args.input)
    frame = base.merge(
        water[["flight", "start_s", "rhw_pct"]],
        on=["flight", "start_s"],
        validate="one_to_one",
    )
    composition = pd.read_csv(HERE / "flight-composition.csv")
    frame = frame.merge(
        composition[["flight", "start_s", "org_sulf_mf"]],
        on=["flight", "start_s"],
        how="left",
        validate="one_to_one",
    )
    frame = frame.loc[frame.rhw_pct.gt(0) & frame.rhw_pct.lt(100)].reset_index(
        drop=True
    )
    joined = np.load(STUDY / "raw/joined.npz")
    indices = {
        (f, s): i
        for i, (f, s) in enumerate(
            zip(joined["flight"], joined["start_s"], strict=True)
        )
    }
    ix = [indices[k] for k in zip(frame.flight, frame.start_s, strict=True)]
    output, error = run(
        frame,
        joined["dndlog10d_cm3"][ix],
        joined["edges_nm"],
        args.resolution,
        args.order,
        True,
    )
    output.to_csv(args.output, index=False, float_format="%.10g")
    print(
        json.dumps(
            dict(
                cases=output.groupby("case").size().to_dict(),
                direct_max_rel_RH_error=error,
            )
        )
    )


if __name__ == "__main__":
    main()
