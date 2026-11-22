"""Replay scores, budgets, branching sensitivity and observed post-peak mass."""

import csv

import numpy as np
import pandas as pd
from bank import BANK, DOSES, OUT
from compare import load
from search import KEYS, OBJECTIVES, evaluate, save

from gpu import Model


def rounds():
    """Join each matched-control comparison to the chemical evidence."""
    evidence = list(
        csv.DictReader((OUT.parents[1] / "chains/evidence.csv").open())
    )
    rows = []
    for n, e in enumerate(evidence, 1):
        rec = load(f"iteration_{n:02}.json")
        if not (
            e["family"] == rec["trial"]["family"]
            and e["motif"] == rec["trial"]["motif"]
        ):
            raise ValueError(f"Evidence and fit differ at round {n}")
        for obj in OBJECTIVES:
            fit = rec["fits"][obj]
            null = min(
                rec["controls"][obj]["score"], rec["static"][obj]["score"]
            )
            rows.append(
                dict(
                    iteration=n,
                    objective=obj,
                    family=e["family"],
                    motif=e["motif"],
                    evidence=e["evidence_grade"],
                    before=rec["before"][obj]["score"],
                    previous_refit=rec["controls"][obj]["score"],
                    static=rec["static"][obj]["score"],
                    expanded=fit["score"],
                    gain_pp=100 * (null - fit["score"]),
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "rounds.csv", index=False)


def diagnostics(model, name, obj, fit):
    """Replay run-level scores, gross transfers and time series."""
    x = np.array(fit["x"])
    score, errors, doses, _ = evaluate(model, obj, x)
    np.testing.assert_allclose(score[0], fit["score"], atol=1e-9)
    _, _, _, flux = evaluate(model, obj, x, doses[0, :, None], flux=True)
    _, _, _, trace = evaluate(model, obj, x, doses[0, :, None], flux=3)
    rows = []
    rates = []
    curves = []
    for j, (r, _, _t, y, _) in enumerate(model.runs):
        a = errors[0, j]
        rows.append(
            dict(
                network=name,
                objective=obj,
                run=r,
                dose=doses[0, j],
                mass=a[0],
                count=a[1],
                observed_peak=max(y),
                peak=a[3],
                observed_last=y[-1],
                last=a[4],
                observed_drop=max(y) - y[-1],
                drop=a[3] - a[4],
            )
        )
        for h, reaction in enumerate(BANK):
            rates.append(
                dict(
                    network=name,
                    objective=obj,
                    run=r,
                    key=reaction.key,
                    rate=x[16 + h],
                    gross_ug_m3=flux[0, j, h],
                )
            )
        for i, v in enumerate(trace[0, j].reshape(60, 8), 1):
            curves.append(
                dict(
                    network=name,
                    objective=obj,
                    run=r,
                    minutes=i * 5,
                    mass=v[0],
                    relative_count=v[1],
                    aa_dimer=v[2],
                    aa_oxidized=v[3],
                    aa_residue=v[4],
                    aa_vapor=v[5],
                    ac_dimer=v[6],
                    alpha_uptake=v[7],
                )
            )
    ids = np.flatnonzero(x[16:] > 0).tolist()
    checks = []
    for label, sub, residue, reverse, test_doses in [
        ("reference", 4, 0.5, False, DOSES),
        ("fine", 8, 0.5, False, DOSES),
        ("3.75s_forward", 16, 0.5, False, DOSES),
        ("3.75s_reverse", 16, 0.5, True, DOSES),
        ("1.875s_forward", 32, 0.5, False, DOSES),
        ("1.875s_reverse", 32, 0.5, True, DOSES),
        ("reversed", 4, 0.5, True, DOSES),
        ("quarter_residue", 4, 0.25, False, DOSES),
        ("three_quarter_residue", 4, 0.75, False, DOSES),
        ("nominal_dose", 4, 0.5, False, [1.0]),
    ]:
        v, err, _, _ = model.run(
            x[:16],
            x[16:],
            ids,
            test_doses,
            obj,
            sub=sub,
            reverse=reverse,
            split_residue=residue,
        )
        checks.append(
            dict(
                network=name,
                objective=obj,
                test=label,
                score=v[0],
                ledger=err[0, :, 2].max(),
                worst_mass=err[0, :, 0].max(),
                worst_count=err[0, :, 1].max(),
                wall_eddy_s_inv=x[9],
            )
        )
    return rows, rates, curves, checks


def counterfactuals(model, fits):
    """Change rate pairs on the tuned backbone and remove complete families."""
    data = []
    for obj in OBJECTIVES:
        old = fits["backbone"][obj]["fit"]
        for k in (0.0, 0.036, 0.111, 1.0):
            for pair in (
                ["AA_oxidize"],
                ["AA_oxidize", "AA_oxsplit"],
                ["AA_split"],
            ):
                x = np.array(old["x"])
                for key in pair:
                    x[16 + KEYS.index(key)] = k
                score, err, _, gross = evaluate(
                    model, obj, x, np.array(old["dose"])[:, None], flux=True
                )
                data.append(
                    dict(
                        objective=obj,
                        change="+".join(pair),
                        rate=k,
                        score=score[0],
                        max_gross_last_step=gross[
                            0, :, KEYS.index(pair[-1])
                        ].max(),
                        closure=err[0, :, 2].max(),
                    )
                )
        full = fits["all"][obj]["fit"]
        for prefix in ("AA", "AC", "AU", "CC", "II", "all"):
            x = np.array(full["x"])
            for h in range(20, len(BANK)):
                if prefix == "all" or KEYS[h].startswith(prefix + "_"):
                    x[16 + h] = 0.0
            score, err, _, _ = evaluate(
                model, obj, x, np.array(full["dose"])[:, None]
            )
            data.append(
                dict(
                    objective=obj,
                    change=f"disable_{prefix}",
                    rate=0.0,
                    score=score[0],
                    max_gross_last_step=None,
                    closure=err[0, :, 2].max(),
                )
            )
    pd.DataFrame(data).to_csv(OUT / "counterfactuals.csv", index=False)


def rate_swap(model, fits):
    """Exchange the rates of the isolated two-step dimer chain."""
    rows = []
    traces = []
    for obj in OBJECTIVES:
        st = fits["backbone"][obj]["fit"]
        cases = []
        for rates in ((0.1, 1.0), (1.0, 0.1)):
            x = np.array(st["x"])
            x[16 + KEYS.index("AA_oxidize")] = rates[0]
            x[16 + KEYS.index("AA_oxsplit")] = rates[1]
            out = evaluate(model, obj, x, np.array(st["dose"])[:, None], flux=3)
            cases.append(out[3][0].reshape(15, 60, 8))
        difference = abs(cases[0] - cases[1])
        delta = difference.max(axis=(0, 1))
        run_idx = np.argmax(difference[:, :, 3].max(axis=1))
        for case, oxidation in zip(cases, (0.1, 1.0), strict=True):
            for minute, value in enumerate(case[run_idx], 1):
                traces.append(
                    dict(
                        objective=obj,
                        run=model.runs[run_idx][0],
                        minutes=minute * 5,
                        oxidation=oxidation,
                        fragmentation=0.1 / oxidation,
                        soa=value[0],
                        unaged=value[2],
                        oxidized=value[3],
                    )
                )
        rows.append(
            dict(
                objective=obj,
                max_soa_ug_m3=delta[0],
                max_relative_count=delta[1],
                max_unaged_dimer_ug_m3=delta[2],
                max_oxidized_dimer_ug_m3=delta[3],
            )
        )
    pd.DataFrame(rows).to_csv(OUT / "rate_swap.csv", index=False)
    pd.DataFrame(traces).to_csv(OUT / "rate_swap_trajectories.csv", index=False)


def main():
    """Read saved parameters and produce all comparison tables."""
    rounds()
    fits = (
        load("continued.json")
        if (OUT / "continued.json").exists()
        else load("comparisons.json")
    )
    model = Model(BANK)
    tables = [[], [], [], []]
    for name, objs in fits.items():
        for obj, case in objs.items():
            for table, rows in zip(
                tables, diagnostics(model, name, obj, case["fit"]), strict=True
            ):
                table.extend(rows)
    for table, name in zip(
        tables,
        ["per_run.csv", "flux.csv", "trajectories.csv.gz", "sensitivity.csv"],
        strict=True,
    ):
        pd.DataFrame(table).to_csv(OUT / name, index=False)
    counterfactuals(model, fits)
    rate_swap(model, fits)
    metadata = dict(
        integration_seconds=15,
        secondary_vapor_loss_per_hour=1,
        retained_split_fraction=0.5,
        device="cuda:0",
        mass_eddy_fixed=True,
        objective_definitions=[
            "worst mass nRMSE",
            "worst mass or number nRMSE",
        ],
    )
    save(
        "mechanism.json",
        dict(
            reactions=load("manifest.json")["reactions"],
            fits=fits["all"],
            conditions=metadata,
        ),
    )
    print("All final networks replayed.", flush=True)


if __name__ == "__main__":
    main()
