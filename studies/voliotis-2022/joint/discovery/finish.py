"""Evaluate final fits, matched controls and ablations on CUDA."""

import json

import numpy as np
import pandas as pd
from fit import OBJECTIVES, UPPER, best, encode, refine, save
from reactions import CANDIDATES, catalog
from smoke import DOSES, OUT

from gpu import Model


def read(name):
    """Read a saved search result."""
    return json.loads((OUT / name).read_text())


def sized(k):
    """Extend an early rate vector with the later candidate slots."""
    return np.pad(k, (0, len(CANDIDATES) - len(k)))


def polish(model, obj, states, ids, seed):
    """Use two independent multistart searches at both network sizes."""
    rng = np.random.default_rng(seed)
    incumbent = min(states, key=lambda state: state["score"])
    p = np.array(incumbent["p"])
    k = sized(incumbent["k"])
    params = np.tile(p, (96, 1))
    rates = np.tile(k, (96, 1))
    for j in range(1, 96):
        s = states[j % len(states)]
        params[j] = s["p"]
        rates[j] = sized(s["k"])
        params[j, :3] = np.clip(
            params[j, :3] * np.exp(rng.normal(0, 0.3, 3)), 1 / 12, 4.0
        )
        params[j, 3:9] = np.clip(
            params[j, 3:9] * np.exp(rng.normal(0, 0.15, 6)), 0, 2.0
        )
        params[j, 12] = np.clip(
            params[j, 12] * np.exp(rng.normal(0, 0.25)), 0, 100
        )
        if obj == "joint":
            params[j, 9] = np.clip(
                params[j, 9] * np.exp(rng.normal(0, 0.14)), 2.818, 16.5515
            )
        if ids:
            shift = np.where(
                rng.random(len(ids)) < 0.3,
                rng.uniform(0, 0.035, len(ids)) * UPPER[ids],
                0.0,
            )
            rates[j, ids] = np.clip(
                rates[j, ids] * np.exp(rng.normal(0, 0.35, len(ids))) + shift,
                0.0,
                UPPER[ids],
            )
    out = model.run(params, rates, ids, DOSES[::10], obj, sub=1)
    use = np.unique(np.r_[0, np.argsort(out[0])[:8]])
    out = model.run(params[use], rates[use], ids, DOSES, obj, sub=2)
    i = best(out)
    current = encode(params[use[i]], rates[use[i]], out, i)
    for iteration in range(4):
        current = refine(model, obj, current, ids, seed + iteration)
    return current


def initial_searches(m, accepted, ids):
    """Fit zero and twenty reactions with matched multistart searches."""
    initial = read("baseline.json")
    stages = {}
    polish_runs = []
    for n in (0, 20):
        stage = initial if n == 0 else accepted[-1]["fits"]
        active = ids[:n]
        for obj in OBJECTIVES:
            own = stage[obj]
            other = stage["joint" if obj == "mass" else "mass"]
            p = np.array(other["p"])
            if obj == "mass":
                p[9] = initial["mass"]["p"][9]
            others = sized(other["k"])
            o = m.run(p, others, active, DOSES, obj, sub=2)
            pool = [own, encode(p, others, o)]
            if n:
                control = stages[(0, obj)]
                pool.append(control)
            for seed in (31807, 47111):
                s = polish(m, obj, pool, active, seed)
                polish_runs.append(
                    dict(size=n, objective=obj, seed=seed, fit=s)
                )
            stages[n, obj] = min(
                [
                    r["fit"]
                    for r in polish_runs
                    if r["size"] == n and r["objective"] == obj
                ],
                key=lambda r: r["score"],
            )
            print("Polished", n, obj, stages[n, obj]["score"], flush=True)
    save(OUT / "polish.json", polish_runs)


def omissions(m, obj, active, p, k, doses, score):
    """Remove each reaction and locally refit the remaining chemistry."""
    abl = []
    best_variant = None
    best_score = score[0]
    for h in active:
        reduced = [j for j in active if j != h]
        k0 = k.copy()
        k0[h] = 0.0
        fixed = m.run(p, k0, reduced, doses[:, None], obj, sub=4)[0][0]
        fresh = m.run(p, k0, reduced, DOSES, obj, sub=2)
        null = refine(m, obj, encode(p, k0, fresh), reduced, 53311 + h)
        tuned = m.run(null["p"], sized(null["k"]), reduced, DOSES, obj, sub=4)[
            0
        ][0]
        if tuned < best_score:
            best_score = tuned
            best_variant = (CANDIDATES[h].key, null)
        abl.append(
            dict(
                key=CANDIDATES[h].key,
                objective=obj,
                rate=k[h],
                active_score=score[0],
                same_dose_off=fixed,
                refit_off=tuned,
                refit_difference=tuned - score[0],
            )
        )
    return abl, best_variant


def retained_products(m, p, k, active, doses, obj, transfers, err):
    """Verify retained channel tracers against total stabilized product."""
    retained = None
    if active:
        tagged = m.run(p, k, active, doses[:, None], obj, sub=4, diagnostics=2)
        np.testing.assert_allclose(
            tagged[3][0, :, : len(CANDIDATES)], transfers[0], atol=1e-9
        )
        retained = tagged[3][0, :, len(CANDIDATES) :]
        if (retained < -1e-8).any() or np.any(retained > transfers[0] + 1e-7):
            raise ArithmeticError("Invalid labelled product inventory")
        stable = [h for h in active if CANDIDATES[h].dest == 9]
        np.testing.assert_allclose(
            retained[:, stable].sum(1), err[:, 7], atol=1e-8
        )
    return retained


def perturbations(m, n, obj, p, k, active, err):
    """Change vapor supply, process order and dose at fixed chemistry."""
    sensitivity = []
    for label, share, reverse, ds, ss in [
        ("half_vapor", 0.05, False, DOSES, 4),
        ("double_vapor", 0.2, False, DOSES, 4),
        ("reverse_order", 0.1, True, DOSES, 4),
        ("reverse_order_fine", 0.1, True, DOSES, 8),
        ("nominal_dose", 0.1, False, np.array([1.0]), 4),
    ]:
        v, err_test, _, _ = m.run(
            p, k, active, ds, obj, sub=ss, share=share, reverse=reverse
        )
        sensitivity.append(
            dict(
                channels=n,
                objective=obj,
                test=label,
                score=v[0],
                worst_mass=err_test[0, :, 0].max(),
                worst_count=err_test[0, :, 1].max(),
                max_error_change=np.abs(err[:, :2] - err_test[0, :, :2]).max(),
            )
        )
    return sensitivity


def write_mechanism(cat, ids, evidence, stages):
    """Export the admitted kinetic network with its literature provenance."""
    network = []
    for seq, h in enumerate(ids, 1):
        key = cat[h]["key"]
        network.append(
            dict(
                order=seq,
                **cat[h],
                product_label=cat[h]["equation"].split("→")[1].strip(),
                mass_fit_rate=stages[20, "mass"]["k"][h],
                joint_fit_rate=stages[20, "joint"]["k"][h],
                literature=evidence[key],
            )
        )
    save(
        OUT / "mechanism.json",
        dict(
            reactions=network,
            baseline_parameters={
                obj: stages[20, obj]["p"] for obj in OBJECTIVES
            },
            pool_index=[
                "G_A",
                "G_C",
                "G_I",
                "V_A",
                "V_C",
                "V_I",
                "P_A",
                "P_C",
                "P_I",
                "S",
                "F",
            ],
            backbone_parameter_names=[
                "alpha_lag_h",
                "cresol_lag_h",
                "isoprene_lag_h",
                "alpha_own_yield",
                "cresol_own_yield",
                "isoprene_own_yield",
                "alpha_mixed_yield",
                "cresol_mixed_yield",
                "isoprene_mixed_yield",
                "wall_eddy_s_inv",
                "unused_alpha_branch",
                "unused_auto_time_s",
                "diversion_per_ppb_min",
                "unused_uptake_yield",
                "unused_gamma",
                "unused_vapor_loss",
            ],
            parent_order=["alpha-pinene", "o-cresol", "isoprene"],
            surrogate_molar_masses_g_mol=[184.0, 140.0, 118.0],
            secondary_vapor_share=0.1,
            secondary_vapor_background_loss_h_inv=1.0,
        ),
    )


def write_best_seen(ids, variants):
    """Export improved omission fits in the same replayable format."""
    best_seen = read("mechanism.json")
    best_seen["omission_refits_used"] = {}
    for obj, variant in variants.items():
        if variant is None:
            continue
        key, state = variant
        best_seen["omission_refits_used"][obj] = key
        best_seen["baseline_parameters"][obj] = state["p"]
        for h, row in zip(ids, best_seen["reactions"], strict=True):
            row[f"{obj}_fit_rate"] = state["k"][h]
    save(OUT / "best_seen_mechanism.json", best_seen)


def main(polish_only=False, refined=False):
    """Compare polished networks and evaluate retained reaction mass."""
    m = Model()
    cat = catalog()
    records = [
        json.loads(f.read_text()) for f in sorted(OUT.glob("round_*.json"))
    ]
    accepted = [r for r in records if r["accepted"]]
    ids = accepted[-1]["active_indices"]
    if len(accepted) != 20 or len(ids) != 20:
        raise ValueError("Twenty admitted fits required for final evaluation")
    if polish_only:
        initial_searches(m, accepted, ids)
        return
    records = read("polish.json")
    if refined:
        records += read("continuous_fits.json")
        records += [r for r in read("prefix_fits.json") if r["size"] == 20]
    stages = {
        (n, obj): min(
            (
                r["fit"]
                for r in records
                if r["size"] == n and r["objective"] == obj
            ),
            key=lambda st: st["score"],
        )
        for n in (0, 20)
        for obj in OBJECTIVES
    }
    runs = []
    metrics = []
    sensitivity = []
    flux = []
    abl = []
    evidence = {}
    variants = {}
    for f in sorted(OUT.glob("decision_*.json")):
        e = json.loads(f.read_text())
        evidence[e["key"]] = e
    for (n, obj), s in stages.items():
        active = ids[:n]
        p = np.array(s["p"])
        k = sized(s["k"])
        outputs = {}
        for sub in (2, 4, 8):
            outputs[sub] = m.run(
                p, k, active, DOSES, obj, sub=sub, diagnostics=sub == 4
            )
        score, err, doses, transfers = outputs[4]
        err = err[0]
        doses = doses[0]
        metrics.append(
            dict(
                channels=n,
                objective=obj,
                score=score[0],
                mass=err[:, 0].max(),
                count=err[:, 1].max(),
                median_mass=np.median(err[:, 0]),
                mass_below_10=int((err[:, 0] <= 0.1).sum()),
                wall_s_inv=p[9],
                integration_change=np.abs(
                    err[:, :2] - outputs[8][1][0, :, :2]
                ).max(),
                coarse_change=np.abs(
                    err[:, :2] - outputs[2][1][0, :, :2]
                ).max(),
                ledger=err[:, 2].max(),
            )
        )
        retained = retained_products(
            m, p, k, active, doses, obj, transfers, err
        )
        for j, (r, _, _t, y, _) in enumerate(m.runs):
            runs.append(
                dict(
                    channels=n,
                    objective=obj,
                    run=r,
                    dose=doses[j],
                    mass=err[j, 0],
                    count=err[j, 1],
                    peak=err[j, 3],
                    observed_peak=y.max(),
                    last=err[j, 4],
                    observed_last=y[-1],
                    final_number=err[j, 5],
                    stable_soa=err[j, 7],
                )
            )
            for h in active:
                flux.append(
                    dict(
                        objective=obj,
                        run=r,
                        key=cat[h]["key"],
                        gross_ug_m3=transfers[0, j, h],
                        endpoint_product_ug_m3=retained[j, h],
                    )
                )
        sensitivity.extend(perturbations(m, n, obj, p, k, active, err))
        drops, variant = omissions(m, obj, active, p, k, doses, score)
        abl.extend(drops)
        variants[obj] = variant
    for name, rows in [
        ("final_scores", metrics),
        ("per_run", runs),
        ("reaction_flux", flux),
        ("sensitivities", sensitivity),
        ("ablations", abl),
    ]:
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)
    write_mechanism(cat, ids, evidence, stages)
    write_best_seen(ids, variants)
    print("Final replays complete", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polish-only", action="store_true")
    parser.add_argument("--refined", action="store_true")
    args = parser.parse_args()
    main(args.polish_only, args.refined)
