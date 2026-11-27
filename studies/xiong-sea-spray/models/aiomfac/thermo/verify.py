# %% Verify table interpolation and check for sulfate supersaturation.
"""Verify table interpolation and check for sulfate supersaturation."""

import json

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from solve import DENSITY, HERE, MW, RECIPE, RH, AIOMFACModel, T, one


def read_pair(binary, mixed):
    """Collect both pure-bisulfate and mixed-product tables."""
    return pd.concat(
        [
            pd.read_csv(HERE / (binary + ".csv")),
            pd.read_csv(HERE / (mixed + ".csv")),
        ],
        ignore_index=True,
    )


def table_lookup(table, col):
    """Interpolate the chosen quantity on the two composition axes."""
    tab = table.pivot(index="f", columns="neutral_na_share", values=col)
    return RegularGridInterpolator((tab.index, tab.columns), tab.to_numpy())


def main():
    """Check independent scalar roots and excluded sodium sulfate phases."""
    liquid = read_pair("liquid", "neutral-liquid")
    halite = read_pair("halite", "neutral-halite")
    # USGS PHREEQC pitzer.dat analytical log10 constants use
    # unit-molality solute and pure-liquid water standard states.
    thenardite_logk = (
        57.185 + 0.086024 * T - 30.8341 * np.log10(T) - 0.000076905 * T * T
    )
    mirabilite_logk = -301.9326 - 0.16232 * T + 141.078 * np.log10(T)
    out = halite[["f", "neutral_na_share", "rh_pct", "liquid_status"]].copy()
    iap = halite.a_na**2 * halite.a_so4
    out["thenardite_saturation"] = iap / 10**thenardite_logk
    out["mirabilite_saturation"] = iap * halite.aw**10 / 10**mirabilite_logk
    out.to_csv(HERE / "sulfate-solids.csv", index=False)
    cpu = AIOMFACModel.prepare(RECIPE)

    def iap_at_logwater(log_water):
        salt = MW[1]
        r = cpu.evaluate([np.exp(log_water), salt, 0.0, 0.0], T)
        return (
            r.ion_activity_products[r.electrolyte_ids.index((202, 242))] / 10**1.57
            - 1.0
        )

    lgw = brentq(iap_at_logwater, -4, 0, xtol=1e-13)
    rh_saturated = float(
        100 * cpu.evaluate([np.exp(lgw), MW[1], 0.0, 0.0], T).neutral_activities[0]
    )
    rows = []
    # Fixed 0/.5/1 compositions check native GPU vs CPU solver. Off-grid points
    # test interpolation independently in conversion and neutral-product share.
    q_checks = [0.0, 0.125, 0.375, 0.5, 0.625, 0.875, 1.0]
    f_checks = [0.175, 0.405, 0.685, 0.895, 0.995]
    for rh in RH:
        lut = liquid[liquid.rh_pct == rh]
        mixed = halite[halite.rh_pct == rh]
        interp_g = table_lookup(lut, "gf")
        interp_a = table_lookup(lut, "a_cl")
        interp_h = table_lookup(mixed, "gf")
        for q in q_checks:
            fstar = 0
            if rh < rh_saturated:
                fstar = brentq(
                    lambda f, q=q, rh=rh: one(cpu, f, q, rh).halite_saturation - 1,
                    0.0,
                    1.0,
                    xtol=2e-12,
                )
                sat = one(cpu, fstar, q, rh)
            for f in f_checks + [0.5]:
                ref = one(cpu, f, q, rh)
                gh = ref.gf
                if f < fstar:
                    w = sat.water_mol * f / fstar
                    gh = (1 + w * MW[0] / DENSITY[0] / (ref.dry_volume_cm3 * 1e-6)) ** (
                        1 / 3
                    )
                point = np.array([[f, q]])
                rows.append(
                    dict(
                        f=f,
                        neutral_na_share=q,
                        rh_pct=rh,
                        liquid_gf_direct=ref.gf,
                        liquid_gf_interp=interp_g(point)[0],
                        halite_gf_direct=gh,
                        halite_gf_interp=interp_h(point)[0],
                        a_cl_direct=ref.a_cl,
                        a_cl_interp=interp_a(point)[0],
                    )
                )
    check = pd.DataFrame(rows)
    check.to_csv(HERE / "refinement.csv", index=False)

    def err(a, b, df):
        return float(abs(df[a] - df[b]).max())

    exact = check[check.f.eq(0.5) & check.neutral_na_share.isin([0, 0.5, 1])]
    metadata = dict(
        halite_deliquescence_rh_pct=rh_saturated,
        thenardite_log10_K=thenardite_logk,
        mirabilite_log10_K=mirabilite_logk,
        independent_root_pairs=len(check),
        liquid_grid_vertex_max_gf_diff=err(
            "liquid_gf_direct", "liquid_gf_interp", exact
        ),
        chloride_grid_vertex_max_diff=err("a_cl_direct", "a_cl_interp", exact),
        max_linear_liquid_gf_error=err("liquid_gf_direct", "liquid_gf_interp", check),
        max_linear_halite_gf_error=err("halite_gf_direct", "halite_gf_interp", check),
        max_linear_chloride_act_error_at_85=err(
            "a_cl_direct", "a_cl_interp", check[check.rh_pct.eq(85)]
        ),
    )
    (HERE / "validation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata)


if __name__ == "__main__":
    main()
