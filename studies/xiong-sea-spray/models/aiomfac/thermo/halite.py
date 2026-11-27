# %% Partition dissolved and crystalline NaCl at fixed humidity.
"""Partition dissolved and crystalline NaCl at fixed humidity."""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from solve import DENSITY, HERE, MW, RECIPE, SPECIES, AIOMFACModel, one


def add_halite(liquid):
    """Partition the liquid table and retain each halite boundary."""
    cpu = AIOMFACModel.prepare(RECIPE)
    pieces = []
    thresholds = []
    for (q, rh), group in liquid.groupby(["neutral_na_share", "rh_pct"], sort=False):
        group = group.sort_values("f").copy()
        s = group.halite_saturation.to_numpy() - 1
        crossings = np.flatnonzero(s[:-1] * s[1:] < 0)
        if len(crossings) > 1 or (len(crossings) == 0 and np.any(s > 0)):
            raise ValueError(f"Unexpected saturation crossings for {q=} and {rh=}")
        group["solid_nacl_mol"] = 0.0
        group["liquid_na_fraction"] = 1.0
        if len(crossings) == 1:
            i = crossings[0]
            a, b = group.f.iloc[[i, i + 1]]
            fstar = brentq(
                lambda x, q=q, rh=rh: one(cpu, x, q, rh).halite_saturation - 1,
                a,
                b,
                xtol=1e-11,
            )
            sat = one(cpu, fstar, q, rh)
            assert abs(sat.halite_saturation - 1) < 2e-9
            for idx in group.index[group.f < fstar]:
                f = group.at[idx, "f"]
                dissolved = f / fstar
                group.at[idx, "solid_nacl_mol"] = 1 - dissolved
                group.at[idx, "liquid_na_fraction"] = dissolved
                water = sat.water_mol * dissolved
                group.at[idx, "water_mol"] = water
                for ion in SPECIES:
                    group.at[idx, ion + "_mol"] = sat[ion + "_mol"] * dissolved
                    group.at[idx, "a_" + ion] = sat["a_" + ion] if f else np.nan
                for col in [
                    "ionic_strength",
                    "halite_saturation",
                    "aw",
                    "solver_status",
                    "warning_code",
                    "ln_ka_error",
                ]:
                    group.at[idx, col] = sat[col] if f else np.nan
                state = "brine_plus_halite" if f else "dry_halite"
                group.at[idx, "liquid_status"] = state
                group.at[idx, "gf"] = (
                    1
                    + water
                    * MW[0]
                    / DENSITY[0]
                    / (1e-6 * group.at[idx, "dry_volume_cm3"])
                ) ** (1 / 3)
            thresholds.append(
                dict(
                    neutral_na_share=q,
                    rh_pct=rh,
                    f_saturated_liquid=fstar,
                    sat_water_mol_per_liquid_na=sat.water_mol,
                    sat_a_na=sat.a_na,
                    sat_a_cl=sat.a_cl,
                )
            )
        else:
            thresholds.append(
                dict(neutral_na_share=q, rh_pct=rh, f_saturated_liquid=0.0)
            )
        group["charge_mol"] = (
            group.na_mol
            + group.h_mol
            - group.cl_mol
            - group.hso4_mol
            - 2 * group.so4_mol
        )
        group["sulfur_error_mol"] = (
            group.hso4_mol + group.so4_mol - group.f * (1 - q / 2)
        )
        group["acid_h_error_mol"] = group.hso4_mol + group.h_mol - group.f * (1 - q)
        assert np.max(abs(group.na_mol + group.solid_nacl_mol - 1)) < 5e-9
        assert np.max(abs(group.cl_mol + group.solid_nacl_mol - (1 - group.f))) < 5e-9
        assert (
            np.abs(
                group[["charge_mol", "sulfur_error_mol", "acid_h_error_mol"]].to_numpy()
            ).max()
            < 5e-9
        )
        assert np.all(group.halite_saturation.dropna() < 1 + 2e-8)
        pieces.append(group)
    return pd.concat(pieces).sort_values(
        ["neutral_na_share", "f", "rh_pct"]
    ), pd.DataFrame(thresholds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="liquid")
    parser.add_argument("--output", default="halite")
    args = parser.parse_args()
    table, roots = add_halite(pd.read_csv(HERE / (args.source + ".csv")))
    table.to_csv(HERE / (args.output + ".csv"), index=False)
    roots.to_csv(HERE / (args.output + "-thresholds.csv"), index=False)
    print(roots.to_string(index=False))
