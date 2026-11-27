# %% Fit batch trajectories from native reaction stoichiometry to optical and ionic observations.
import argparse
import json
import os
import time
from pathlib import Path

import jax

import jax.numpy as jnp
import numpy as np
import pandas as pd
from particula.dynamics.chemistry.inference import ExperimentLoss, fit_experiments
from pathways import CHANNELS, CANDIDATES, sodium_stoichiometry

jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "cpu"))
jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parent
D = np.arange(50, 401, 50)
RH = np.array([40, 60, 70, 75, 80, 85])
PARAMS = [
    "peak_k_h",
    "small_size_power",
    "large_size_power",
    "oxidation_or_gate2_h",
    "capture_branch",
    "neutralization_h",
    "low_rh_brine_weight",
    "ic_loggain",
    "siv_optical_equivalent",
    "mixed_volume_delta",
]
VOL = np.array([58.443 / 2.165, 120.061 / 2.742, 142.043 / 2.66 / 2])
S = jnp.asarray(sodium_stoichiometry())


def load_tables(refine=1):
    xs = np.linspace(0, 1, 201)
    tables = []
    for mode in ["liquid", "halite"]:
        table = pd.concat(
            [
                pd.read_csv(ROOT / "thermo" / f"{prefix}{mode}.csv")
                for prefix in ["", "neutral-"]
            ]
        )
        curves = []
        for (q, rh), g in table.groupby(["neutral_na_share", "rh_pct"], sort=True):
            g = g.sort_values("f")
            if refine > 1:
                rows = np.unique(np.r_[np.arange(0, len(g), refine), len(g) - 1])
                g = g.iloc[rows]
            curves.append(np.interp(xs, g.f, g.gf**3))
        tables.append(np.stack(curves).reshape(5, 6, -1).transpose(0, 2, 1))
        if mode == "liquid":
            groups = table.query("rh_pct==85").groupby("neutral_na_share", sort=True)
            # Activity per remaining mole of Cl, normalized by the pure NaCl state.
            modifier = []
            for _, a in groups:
                a = a.sort_values("f")
                rate = a.a_cl.iloc[:-1] / (1 - a.f.iloc[:-1]) / a.a_cl.iloc[0]
                modifier.append(np.interp(xs, a.f.iloc[:-1], rate))
    e = pd.read_csv(ROOT.parent / "xiong/endmembers.csv")
    return (
        jnp.array(tables),
        jnp.array(modifier),
        jnp.array(e[["nacl_gf", "bisulfate_gf"]].values.T ** 3),
    )


def observations():
    s = pd.read_csv(ROOT.parent / "xiong/table-s1.csv")
    s["scale"] = s.spread + 0.005
    s["source"] = "S1"
    s["train"] = (s.time_h <= 2.5) & s.rh_pct.isin([40, 60, 70, 80])
    g = pd.read_csv(ROOT.parent / "xiong/fig3-100nm.csv")
    g["diameter_nm"] = 100
    g["source"] = "Fig3"
    # Selected Fig 3 markers do not supply digitized pointwise spreads. 0.03 GF is an analyst scale.
    g["scale"] = 0.03
    g["train"] = g.time_h <= 2
    gf = pd.concat([s, g], ignore_index=True)
    ic = pd.read_csv(ROOT.parent / "xiong/figs19-ic-peaks.csv")
    ic["scale"] = 0.5 * np.log(ic.raster_edge_max / ic.raster_edge_min)
    ic["train"] = ic.time_h <= 2
    return gf, ic


def make_model(steps=12, grid_stride=1, peak_nm=200.0):
    tables, modifier, end = load_tables(grid_stride)
    gf, ic = observations()

    def as_jax_array(value):
        return jnp.array(np.asarray(value))

    di = as_jax_array(np.searchsorted(D, gf.diameter_nm))
    ti = as_jax_array(np.rint(gf.time_h * 2).astype(int) - 1)
    hi = as_jax_array(np.searchsorted(RH, gf.rh_pct))
    ci = as_jax_array(np.rint(ic.time_h * 2).astype(int) - 1)
    logD = jnp.log(jnp.array(D) / peak_nm)
    V = jnp.array(VOL)

    def interp1(x, v):
        z = jnp.clip(x, 0, 1) * 200
        i = jnp.minimum(z.astype(int), 199)
        w = z - i
        return v[i] * (1 - w) + v[i + 1] * w

    def lookup(tab, x, q):
        u = jnp.clip(q, 0, 1) * 4
        i = jnp.minimum(u.astype(int), 3)
        a = (u - i)[..., None]
        z = jnp.clip(x, 0, 1) * 200
        j = jnp.minimum(z.astype(int), 199)
        b = (z - j)[..., None]
        return (1 - a) * ((1 - b) * tab[i, j] + b * tab[i, j + 1]) + a * (
            (1 - b) * tab[i + 1, j] + b * tab[i + 1, j + 1]
        )

    def predicted(par, flags, dose=1.0):
        # Flags are direct/capture/oxidation/access/ready/neutral channel switches, chloride law,
        # pure endpoint correction, metastable liquid, ZSR control, IC recovery of S(IV).
        (
            direct,
            cap,
            ox,
            gate,
            ready,
            neutral,
            activity,
            anchor,
            metastable,
            zsr,
            ic_recovery,
        ) = flags
        k = (
            par[0]
            * jnp.exp(-par[1] * jnp.maximum(-logD, 0) - par[2] * jnp.maximum(logD, 0))
            * dose
        )
        dt = 0.5 / steps

        def tendency(x):
            nacl, exposed_nacl, bisulfite, bisulfate, neutral_salt = x.T
            chem = jnp.clip(bisulfate + 2 * neutral_salt, 0, 1)
            q = 2 * neutral_salt / jnp.maximum(chem, 1e-12)
            factor = lookup(modifier[..., None], chem, q)[..., 0]
            rate = k * (1 + activity * (factor - 1))
            flux = jnp.stack(
                [
                    direct * (1 - par[4]) * rate * nacl,
                    cap * par[4] * rate * nacl,
                    ox * par[3] * bisulfite,
                    gate * (1 - direct + direct * par[4]) * rate * nacl,
                    ready * par[3] * exposed_nacl,
                    neutral * par[5] * bisulfate * nacl,
                ],
                axis=1,
            )
            return flux @ S

        def sample(x, _):
            def advance(_, v):
                m = tendency(v)
                m2 = tendency(v + dt * 0.5 * m)
                return v + dt * m2

            x = jax.lax.fori_loop(0, steps, advance, x)
            return x, x

        start = jnp.zeros((len(D), 5)).at[:, 0].set(1)
        _, x = jax.lax.scan(sample, start, None, length=20)
        nacl, exposed_nacl, bisulfite, bisulfate, neutral_salt = jnp.moveaxis(x, -1, 0)
        # Optically uncharacterized S(IV) is bracketed by parent-like and acid-product-like response.
        f = jnp.clip(bisulfate + 2 * neutral_salt + par[8] * bisulfite, 0, 1)
        q = 2 * neutral_salt / jnp.maximum(f, 1e-12)
        selected = (1 - metastable) * tables[1] + metastable * tables[0]
        native = lookup(selected, f, q)
        a = (1 - f) * V[0]
        b = f * (1 - q) * V[1]
        n = f * q * V[2]
        total = a + b + n
        # Distinct pure-phase endpoint observations calibrate the optical GF operator.
        adj0 = end[0] - selected[0, 0]
        adj1 = end[1] - selected[0, -1]
        corrected = (
            native
            + anchor * (a[..., None] * adj0 + b[..., None] * adj1) / total[..., None]
        )
        separated = (
            a[..., None] * (selected[0, 0] + anchor * adj0)
            + b[..., None] * (selected[0, -1] + anchor * adj1)
            + n[..., None] * selected[-1, -1]
        ) / total[..., None]
        # External/separate-phase proxy at low RH; high-RH branch stays internally mixed.
        low = (jnp.array(RH) <= 75) * (1 - par[6])
        g3 = (1 - zsr) * ((1 - low) * corrected + low * separated) + zsr * separated
        growth = jnp.cbrt(g3 / (1 + 4 * par[9] * f * (1 - f))[..., None])
        blank = par[10] if par.shape[0] > 10 else 0.0
        ratio = (
            jnp.exp(par[7])
            * (bisulfate + neutral_salt + ic_recovery * bisulfite)
            / jnp.maximum(nacl + exposed_nacl + blank, 1e-12)
        )
        return x, growth, jnp.log(jnp.maximum(ratio, 1e-10))

    yg = as_jax_array(gf.gf)
    si = as_jax_array(gf.scale)
    yi = as_jax_array(np.log(ic.late_to_early_peak_area_ratio))
    isci = as_jax_array(ic.scale)

    def residual(par, flags):
        x, g, r = predicted(par, flags)
        return (g[ti, di, hi] - yg) / si, (r[ci, 1] - yi) / isci

    def obj(par, flags, wg, wi):
        dg, dr = residual(par, flags)
        return 0.5 * (wg @ (dg**2) + wi @ (dr**2))

    return jax.jit(jax.value_and_grad(obj)), jax.jit(predicted), jax.jit(residual)


def config(name, variant="primary"):
    labels = CANDIDATES[name]
    flags = [float(c in labels) for c in CHANNELS] + [
        float(name in ("chloride_activity", "activity_neutral", "retention_activity")),
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    lim = [
        (0.05, 6.0),
        (0.0, 3.0),
        (0.0, 3.0),
        (0.05, 10.0),
        (0.0, 1.0),
        (0.0, 8.0),
        (0.0, 1.0),
        (-4.0, 3.0),
        (0.5, 0.5),
        (0.0, 0.0),
    ]
    if "capture" not in labels and not ("direct" in labels and "expose" in labels):
        lim[4] = (0, 0)
    elif "direct" not in labels:
        lim[4] = (1, 1)
    if not ("oxidize" in labels or "react_exposed" in labels):
        lim[3] = (0, 0)
    if "neutralize" not in labels:
        lim[5] = (0, 0)
    if variant in ("dry_volume", "liquid_dry"):
        lim[9] = (-0.12, 0.12)
    if variant.startswith("profile_"):
        lim[5] = (float(variant[8:]),) * 2
    if variant == "raw":
        flags[7] = 0.0
    if variant == "metastable":
        flags[7] = 0.0
        flags[8] = 1.0
        lim[6] = (1, 1)
    if variant == "separate":
        flags[9] = 1.0
        lim[6] = (0, 0)
    if variant == "native_halite":
        lim[6] = (1, 1)
    if variant == "raw_halite":
        flags[7] = 0.0
        lim[6] = (1, 1)
    if variant in ("ic_blank", "eq8_blank"):
        lim.append((0.0, 0.6))
    if variant in ("eq8", "eq8_blank"):
        lim[7] = (0, 0)
    if variant.startswith("blank_"):
        lim.append((float(variant[6:]),) * 2)
    if variant.startswith("retained_"):
        lim[4] = (float(variant[9:]),) * 2
    if variant == "siv_dark":
        lim[8] = (0, 0)
    if variant == "siv_bright":
        lim[8] = (1, 1)
    if variant == "siv_free":
        lim[8] = (0, 1)
    if variant == "ic_oxidized":
        flags[10] = 1.0
    if variant == "shape_free":
        raise ValueError("Use per-size profiles separately.")
    return np.array(flags), np.array(lim, dtype=float)


def training_weights(g, ic, variant):
    use = g.train.copy()
    if variant in ("liquid_training", "liquid_dry"):
        use = (g.time_h <= 2.5) & (g.rh_pct >= 80)
    if variant == "omit200":
        use &= g.diameter_nm != 200
    if variant == "omit300":
        use &= g.diameter_nm != 300
    if variant == "train_early":
        use &= g.time_h <= 1.5
    # Give each set of readouts at the same D and t one total weight.
    count = use.groupby([g.diameter_nm, g.time_h]).transform("sum").clip(lower=1)
    wg = np.asarray(use / count)
    wi = ic.train.to_numpy().astype(float)
    if variant == "optical_only":
        wi[:] = 0
    return wg, wi


def run(args):
    out = ROOT / args.stage
    out.mkdir(exist_ok=True)
    g, ic = observations()
    vg, predict, residual = make_model(args.steps, args.grid_stride, args.peak_nm)
    results = []
    growth = []
    peaks = []
    states = []
    starts_log = []
    jobs = [(n, v) for v in args.variants.split(",") for n in args.names.split(",")]
    for name, var in jobs:
        start = time.monotonic()
        flags, bounds = config(name, var)
        wg, wi = training_weights(g, ic, var)

        def evaluate(x, gradient):
            loss, grad = vg(x, flags, wg, wi)
            return float(loss), np.asarray(grad)

        suffix = [0.08] if len(bounds) > 10 else []
        first = np.clip(
            [1.5, 0.8, 0.8, 2, 0.5, 0.3, 0.25, -0.5, 0.5, 0.0] + suffix, *bounds.T
        )
        starts = [first]
        if args.starts >= 2:
            starts.append(
                np.clip(
                    [2.7, 1.3, 0.2, 6.5, 0.85, 2.0, 0.7, -1.5, 0.8, 0.05] + suffix,
                    *bounds.T,
                )
            )
        if args.starts >= 3:
            starts.append(
                np.clip(
                    [0.9, 0.3, 1.7, 0.7, 0.2, 4.0, 0.05, 0.3, 0.2, -0.04] + suffix,
                    *bounds.T,
                )
            )
        fit = fit_experiments(
            [ExperimentLoss(tuple(range(len(bounds))), evaluate)],
            bounds,
            starts,
            max_iterations=args.maxiter,
            max_evaluations=2 * args.maxiter,
        )
        p = fit.parameters
        loss = fit.loss
        x, gg, ii = map(np.asarray, predict(p, flags, 1.0))
        assert (
            x.min() > -1e-8 and np.max(abs(x @ np.array([1, 1, 1, 1, 2]) - 1)) < 2e-11
        )
        pred = gg[
            np.rint(g.time_h * 2).astype(int) - 1,
            np.searchsorted(D, g.diameter_nm),
            np.searchsorted(RH, g.rh_pct),
        ]
        ir = np.exp(ii[np.rint(ic.time_h * 2).astype(int) - 1, 1])
        styles = dict(network=name, variant=var)
        ggobs = g.assign(
            **styles, predicted_gf=pred, residual=pred - g.gf, fitted=wg > 0
        )
        iiobs = ic.assign(
            **styles,
            predicted_ratio=ir,
            log_residual=np.log(ir / ic.late_to_early_peak_area_ratio),
            fitted=wi > 0,
        )
        growth.append(ggobs)
        peaks.append(iiobs)
        # Compare reserved blocks independently. Their losses are descriptive since the graphical scales are not standard errors.
        row = dict(
            **styles,
            loss=loss,
            converged=any(fit.converged),
            evals=sum(fit.evaluations),
        )
        for label, mask in {
            "train": g.train,
            "held": ~g.train,
            "late_s1": (g.source == "S1") & (g.time_h >= 3),
            "held_rh": (g.source == "S1") & (g.time_h <= 2.5) & g.rh_pct.isin([75, 85]),
            "held100": (g.source == "Fig3") & (g.time_h == 5),
        }.items():
            row[label + "_gf_rmse"] = float(
                np.sqrt(np.mean(ggobs.loc[mask, "residual"] ** 2))
            )
        row["held_ic_log_rmse"] = float(
            np.sqrt(np.mean(iiobs.loc[~ic.train, "log_residual"] ** 2))
        )
        row["ic_graphical_overlap"] = int(
            ((ir >= ic.raster_edge_min) & (ir <= ic.raster_edge_max) & ~ic.train).sum()
        )
        row["train_cluster_score"] = 2 * loss / (wg.sum() + wi.sum())
        # Directional check at a feasible interior point with multiple scale lengths.
        direction = np.where(
            bounds[:, 1] > bounds[:, 0], np.cos(np.arange(len(p)) + 1), 0.0
        )
        probe = 0.79 * first + 0.21 * bounds.mean(axis=1)
        _, grad = evaluate(probe, True)
        ad = grad @ direction
        errors = []
        for h in [1e-4, 1e-5, 1e-6]:
            fd = (
                evaluate(probe + h * direction, False)[0]
                - evaluate(probe - h * direction, False)[0]
            ) / (2 * h)
            errors.append(abs(fd - ad) / max(1.0, abs(fd), abs(ad)))
        row.update(
            gradient_min_relative_error=min(errors),
            seconds=time.monotonic() - start,
            **dict(zip(PARAMS, p)),
        )
        row["ic_blank"] = float(p[10]) if len(p) > 10 else 0.0
        starts_log.append(
            dict(
                **styles,
                converged=fit.converged,
                evaluations=fit.evaluations,
                gradient_error=errors,
            )
        )
        for j, t in enumerate(np.arange(0.5, 10.1, 0.5)):
            for k, d in enumerate(D):
                nacl, exposed_nacl, bisulfite, bisulfate, neutral_salt = x[j, k]
                states.append(
                    dict(
                        **styles,
                        time_h=t,
                        diameter_nm=d,
                        nacl=nacl,
                        exposed_nacl=exposed_nacl,
                        bisulfite=bisulfite,
                        bisulfate_inventory=bisulfate,
                        neutral_salt_inventory=neutral_salt,
                        total_svi=bisulfate + neutral_salt,
                        total_particle_chloride=nacl + exposed_nacl,
                    )
                )
        results.append(row)
        print(
            name,
            var,
            round(loss, 2),
            row["held_gf_rmse"],
            row["held_ic_log_rmse"],
            fit.converged,
            flush=True,
        )
        pd.DataFrame(results).to_csv(out / "scores.csv", index=False)
    pd.concat(growth).to_csv(out / "growth.csv", index=False)
    pd.concat(peaks).to_csv(out / "ion-ratios.csv", index=False)
    pd.DataFrame(states).to_csv(out / "inventories.csv", index=False)
    (out / "checks.json").write_text(json.dumps(starts_log, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--stage", required=True)
    cli.add_argument("--names", default=",".join(CANDIDATES))
    cli.add_argument("--variants", default="primary")
    cli.add_argument("--steps", type=int, default=12)
    cli.add_argument("--peak-nm", type=float, default=200.0)
    cli.add_argument("--grid-stride", type=int, default=1)
    cli.add_argument("--starts", type=int, default=3)
    cli.add_argument("--maxiter", type=int, default=150)
    run(cli.parse_args())
