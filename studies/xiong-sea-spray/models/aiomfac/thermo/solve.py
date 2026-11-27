# %% Solve water activity and sulfur speciation at prescribed humidity.
"""Solve water activity and sulfur speciation at prescribed humidity."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from particula_aiomfac import (
    UPSTREAM_REVISION,
    AIOMFACModel,
    MixtureSpec,
    __version__,
)
from particula_aiomfac._equilibrium import log_constants
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
T = 298.15
MW = np.array([0.01801528, 0.058443, 0.120061, 0.142043])  # kg/mol, pinned tables
DENSITY = np.array([997.0474, 2165.0, 2742.0, 2660.0])  # kg/m3, additive volume model
RH = [40, 60, 70, 75, 80, 85]
SPECIES = dict(na=202, h=205, cl=242, hso4=248, so4=261)
KSP = 10**1.57
RECIPE = MixtureSpec(
    dict(
        water={16: 1},
        NaCl={202: 1, 242: 1},
        NaHSO4={202: 1, 248: 1},
        Na2SO4={202: 2, 261: 1},
    )
)


def dry(f, q):
    """Return formal salt moles on a basis of 1 mol initial sodium."""
    return np.stack([1 - f, f * (1 - q), f * q / 2], axis=-1)


def inputs(f, q, water_kg):
    """Build a native analytical mass array in kg per mol initial sodium."""
    return np.column_stack([water_kg, dry(f, q) * MW[1:]])


def solve_water(p, f, q, rh, iterations):
    """The broad initial water bracket is checked separately for every row."""
    lo = np.full(len(f), 0.015)
    hi = np.full(len(f), 0.6)

    def aw(water):
        return p.evaluate(inputs(f, q, water), T).neutral_activities[:, 0]

    assert np.all(aw(lo) < rh / 100) and np.all(aw(hi) > rh / 100)
    for _ in range(iterations):
        mid = (hi + lo) / 2
        below = aw(mid) < rh / 100
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return (hi + lo) / 2


def one(p, f, q, rh):
    """Solve independently for water mass using a scalar Brent root."""
    f, q = np.array([f]), np.array([q])

    def residual(water):
        return (
            p.evaluate(inputs(f, q, np.array([water]))[0], T).neutral_activities[0]
            - rh / 100
        )

    w = brentq(residual, 0.015, 0.6, xtol=1e-13)
    return frame(p, f, q, np.array([rh]), np.array([w])).iloc[0]


def frame(p, f, q, rh, water):
    """Calculate water, speciation, activities, growth and inventory errors."""
    amounts = inputs(f, q, water)
    res = p.evaluate(amounts, T, diagnostics=True)
    assert res.neutral_ids == ("water",)
    solvent = res.neutral_activities[:, 0]
    dry_volume = (amounts[:, 1:] / DENSITY[1:]).sum(axis=1)
    gf = (1 + water / DENSITY[0] / dry_volume) ** (1 / 3)
    d = dict(
        f=f,
        neutral_na_share=q,
        rh_pct=rh,
        gf=gf,
        water_mol=water / MW[0],
        aw=solvent,
        dry_volume_cm3=dry_volume * 1e6,
    )
    moles = res.species_moles_per_kg_input * amounts.sum(axis=1)[:, None]
    assert np.allclose(moles[:, 0], d["water_mol"], rtol=2e-11)
    for name, ion in SPECIES.items():
        i = res.ion_ids.index(ion)
        molality = res.ion_molalities[:, i]
        activity = np.zeros_like(molality)
        present = molality > 0
        assert np.all(np.isfinite(res.ion_log_coefficients[present, i]))
        activity[present] = molality[present] * np.exp(
            res.ion_log_coefficients[present, i]
        )
        d["a_" + name] = activity
        d[name + "_mol"] = moles[:, 1 + i]
    d["ionic_strength"] = res.diagnostics["ionic_strength"]
    d["halite_saturation"] = d["a_na"] * d["a_cl"] / KSP
    d["liquid_status"] = np.where(
        d["halite_saturation"] > 1.0,
        "supersaturated_halite",
        "undersaturated_halite",
    )
    d["solver_status"] = res.solver_status
    d["warning_code"] = res.warnings
    d["ln_ka_error"] = res.diagnostics["equilibrium_residuals"][:, 0]
    d["charge_mol"] = (
        d["na_mol"] + d["h_mol"] - d["cl_mol"] - d["hso4_mol"] - 2 * d["so4_mol"]
    )
    d["sulfur_error_mol"] = d["hso4_mol"] + d["so4_mol"] - f * (1 - q / 2)
    d["acid_h_error_mol"] = d["hso4_mol"] + d["h_mol"] - f * (1 - q)
    assert np.max(np.abs(d["na_mol"] - 1)) < 5e-9
    assert np.max(np.abs(d["cl_mol"] - (1 - f))) < 5e-9
    for var in ["charge_mol", "sulfur_error_mol", "acid_h_error_mol"]:
        assert np.max(np.abs(d[var])) < 5e-9, var
    # The reconstructed activities reproduce the native electrolyte product.
    nacl = res.electrolyte_ids.index((202, 242))
    assert np.allclose(
        res.ion_activity_products[:, nacl],
        d["halite_saturation"] * KSP,
        rtol=1e-10,
        atol=1e-14,
    )
    return pd.DataFrame(d)


def main():
    """Solve each composition on the selected CPU or CUDA device."""
    pa = argparse.ArgumentParser()
    pa.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    pa.add_argument("--step", type=float, default=0.1)
    pa.add_argument("--steps-water", type=int, default=36)
    pa.add_argument("--neutral", type=float, nargs="+", default=[0.0])
    pa.add_argument("--output", required=True)
    pa.add_argument("--trace", action="store_true")
    args = pa.parse_args()
    start = time.monotonic()
    cpu = AIOMFACModel.prepare(RECIPE)
    assert np.allclose(cpu.salt_mw[:3], MW[1:], rtol=2e-5), cpu.salt_mw
    model = (
        cpu
        if args.device == "cpu"
        else AIOMFACModel.prepare(RECIPE, execution="cuda", device="cuda:0")
    )
    fractions = np.arange(int(round(1 / args.step)) + 1) * args.step
    if args.trace:
        fractions = np.unique(
            np.r_[
                fractions,
                1 - np.geomspace(1e-8, 1e-3, 6),
                np.geomspace(1e-7, 1e-3, 5),
            ]
        )
    f, q, rh = [
        a.flatten() for a in np.meshgrid(fractions, args.neutral, RH, indexing="ij")
    ]
    w = solve_water(model, f, q, rh, args.steps_water)
    table = frame(model, f, q, rh, w)
    table.to_csv(HERE / (args.output + ".csv"), index=False)
    out = dict(
        seconds=time.monotonic() - start,
        rows=len(f),
        direct_rows_including_bisection=len(f) * (args.steps_water + 3),
        min_water_mol=float(table.water_mol.min()),
        max_water_mol=float(table.water_mol.max()),
        max_abs_aw_error=float(abs(table.aw - rh / 100).max()),
        max_backend_ka_residual=float(table.ln_ka_error.max()),
        max_charge_error_mol=float(abs(table.charge_mol).max()),
        max_sulfur_error_mol=float(abs(table.sulfur_error_mol).max()),
        backend=__version__,
        revision=UPSTREAM_REVISION,
        device=args.device,
        temperature_K=T,
        ka_bisulfate=float(np.exp(log_constants(T)[3])),
    )
    assert out["max_abs_aw_error"] < 1e-8
    (HERE / (args.output + ".json")).write_text(json.dumps(out, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    main()
