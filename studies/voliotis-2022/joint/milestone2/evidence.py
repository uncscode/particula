"""Evaluate frozen networks at instrument times and remove a channel."""

import json

import numpy as np
import pandas as pd
from baseline_replay import HERE, Model, run, run_rows, save
from growth import BANK, refit


def omission(model, x, ids, incumbent_channels, dose):
    """Find the worst frozen deletion and give both models matched refits."""
    archived = {r["key"] for r in incumbent_channels}
    added = [i for i in ids if BANK[i].key not in archived]
    test_ids = added or ids
    cases = np.tile(x, (1 + len(test_ids), 1))
    cases[1 + np.arange(len(test_ids)), 16 + np.array(test_ids)] = 0
    fixed = np.maximum(
        *[
            run(model, cases, ids, 32, dose, reverse=rev)[0]
            for rev in (False, True)
        ]
    )
    increases = fixed[1:] - fixed[0]
    worst = int(np.argmax(increases))
    k = test_ids[worst]
    without = [i for i in ids if i != k]
    tests = {}
    for name, point, active in (
        ("intact", x, ids),
        ("removed", cases[worst + 1], without),
    ):
        fits = [
            refit(model, point, active, ids, restart, 5, sub=16)
            for restart in (0, 1)
        ]
        fitted, calls = min(
            fits, key=lambda f: run(model, f[0], active, 16)[0][0]
        )
        train, _, d, _ = run(model, fitted, active, 16)
        refined = [
            float(
                run(model, fitted, active, 32, d[0, :, None], reverse=rev)[0][0]
            )
            for rev in (False, True)
        ]
        tests[name] = dict(
            score=[float(train[0]), *refined],
            x=fitted.tolist(),
            dose=d[0].tolist(),
            per_start_calls=calls,
        )
    save(
        "ablation.json",
        dict(
            key=BANK[k].key,
            channel_set="added" if added else "inherited",
            fixed_intact=float(fixed[0]),
            fixed_removed=float(fixed[worst + 1]),
            fixed_omission_gain=float(increases[worst]),
            all_omissions={
                BANK[i].key: float(v)
                for i, v in zip(test_ids, increases, strict=True)
            },
            refit_score_order=[
                "3.75s forward training",
                "1.875s forward",
                "1.875s reverse",
            ],
            refits=tests,
        ),
    )


def sample_measurements(
    experiment, label, measured, hourly, consumed, n0, mass_scale, expected
):
    """Reconstruct both native RMSEs from predictions at instrument times."""
    fits = []
    times = np.arange(301) / 60
    for k, (kind, endpoints, normalization) in enumerate(
        (
            ("mass", np.r_[0, hourly[:, 0]], mass_scale),
            ("number", np.r_[1, hourly[:, 2]] * n0, n0),
        )
    ):
        records = [
            r
            for r in measured
            if r["run"] == experiment and r["observable"] == kind
        ]
        sample_times = [r["time_h"] for r in records]
        points = np.interp(sample_times, times, endpoints)
        wall = np.interp(sample_times, times, np.r_[0, hourly[:, 1]])
        precursor = np.interp(sample_times, times, np.r_[0, consumed])
        reference = np.array([r["value"] for r in records])
        reproduced = np.sqrt(
            np.mean(((points - reference) / normalization) ** 2)
        )
        np.testing.assert_allclose(reproduced, expected[k], rtol=0, atol=1e-11)
        fits.extend(
            dict(
                **r,
                network=label,
                prediction=v,
                normalization=normalization,
                deposited_ug_m3=w,
                consumed_ug_m3=c,
            )
            for r, v, w, c in zip(records, points, wall, precursor, strict=True)
        )
    return fits


def check_minute_output(model, x, ids, dose, minute, sub):
    """Compare the new sampling export to both archived diagnostic formats."""
    old_budget = run(model, x, ids, sub, dose, diagnostics=4)[3][0].reshape(
        15, 300, 3
    )
    old_counts = run(model, x, ids, sub, dose, diagnostics=3)[3][0].reshape(
        15, 60, 8
    )
    minute = minute[0].reshape(15, 300, 3)
    np.testing.assert_allclose(
        minute[:, :, :2], old_budget[:, :, :2], rtol=0, atol=1e-11
    )
    np.testing.assert_allclose(
        minute[:, 4::5, 2], old_counts[:, :, 1], rtol=0, atol=1e-12
    )
    return dict(
        max_organic_or_wall_ug_m3=float(
            abs(minute[:, :, :2] - old_budget[:, :, :2]).max()
        ),
        max_relative_count=float(
            abs(minute[:, 4::5, 2] - old_counts[:, :, 1]).max()
        ),
    )


def measurements(model):
    """Restore instrument times and dimensional values from CUDA inputs."""
    measured = []
    obs = model.data[1].numpy()
    for j, (experiment, *_) in enumerate(model.runs):
        for i, row in enumerate(obs[j]):
            for offset, label in ((0, "mass"), (2, "number")):
                if row[offset] >= 0:
                    measured.append(
                        dict(
                            run=experiment,
                            observable=label,
                            time_h=(i + row[offset]) / 60,
                            value=row[offset + 1]
                            * (model.initial[j, 1] / 1e6 if offset else 1),
                        )
                    )
    return measured


def main():
    """Collect frozen trajectories, observations, and budget checks."""
    cases = json.loads((HERE / "size_inputs.json").read_text())["cases"]
    model = Model(BANK)
    keymap = {r.key: i for i, r in enumerate(BANK)}
    summary, per_run, traces, fluxes = {}, [], [], []
    measured = measurements(model)
    pd.DataFrame(measured).to_csv(HERE / "measurements.csv.gz", index=False)
    decision = json.loads((HERE / "selection.json").read_text())
    validation_sub = round(60 / decision["qualification_step_seconds"])
    observation_fits = []
    for label, case in cases.items():
        ids = [keymap[r["key"]] for r in case["channels"]]
        x = np.zeros(16 + len(BANK))
        x[:16] = case["x"][:16]
        x[16 + np.array(ids)] = case["x"][16:]
        d = np.array(case["dose"])[:, None]
        for step, sub, reverse in (
            ("15s", 4, False),
            ("1.875s", 32, False),
            ("1.875s_reverse", 32, True),
            ("0.9375s", 64, False),
            ("0.9375s_reverse", 64, True),
            ("0.46875s", 128, False),
            ("0.46875s_reverse", 128, True),
        ):
            score, errors, _, _ = run(
                model, x, ids, sub=sub, doses=d, reverse=reverse
            )
            per_run.extend(
                dict(**row, step=step)
                for row in run_rows(model, label, errors[0], d[:, 0])
            )
        _, _, _, flux = run(
            model, x, ids, sub=validation_sub, doses=d, diagnostics=1
        )
        _, forward_errors, _, history = run(
            model, x, ids, sub=validation_sub, doses=d, diagnostics=5
        )
        for j, (experiment, _, _, _, source) in enumerate(model.runs):
            fluxes.extend(
                dict(
                    network=label,
                    run=experiment,
                    key=BANK[h].key,
                    rate=x[16 + h],
                    gross_ug_m3=flux[0, j, h],
                )
                for h in ids
            )
            hourly = history[0, j].reshape(300, 3)
            used = np.cumsum(source.sum(axis=1)) * d[j, 0]
            n0 = model.initial[j, 1] / 1e6
            rows = sample_measurements(
                experiment,
                label,
                measured,
                hourly,
                used,
                n0,
                model.initial[j, 2],
                forward_errors[0, j],
            )
            observation_fits.extend(rows)
            for idx in range(4, len(hourly), 5):
                corrected = hourly[idx, 0] + hourly[idx, 1]
                traces.append(
                    dict(
                        network=label,
                        run=experiment,
                        time_h=(idx + 1) / 60,
                        mass=hourly[idx, 0],
                        number=hourly[idx, 2] * n0,
                        deposited_mass=hourly[idx, 1],
                        consumed_ug_m3=used[idx],
                        corrected_yield=corrected / used[idx]
                        if used[idx] > 0
                        else None,
                    )
                )
        summary[label] = dict(
            trajectory_step_seconds=60 / validation_sub,
            num_channels=len(ids),
            appreciable=int((flux[0].max(axis=0) >= 1.0).sum()),
            max_closure_ug_m3=float(
                max(
                    r["closure_ug_m3"] for r in per_run if r["network"] == label
                )
            ),
        )
        if label == "incumbent":
            summary[label]["archived_export_check"] = check_minute_output(
                model, x, ids, d, history, validation_sub
            )
        if label == "selected":
            omission(model, x, ids, cases["incumbent"]["channels"], d)
    pd.DataFrame(observation_fits).to_csv(
        HERE / "instrument_fits.csv.gz", index=False
    )
    pd.DataFrame(per_run).to_csv(HERE / "scorecard.csv", index=False)
    pd.DataFrame(traces).to_csv(HERE / "trajectories.csv.gz", index=False)
    pd.DataFrame(fluxes).to_csv(HERE / "channel_flux.csv.gz", index=False)
    save("diagnostics.json", summary)


if __name__ == "__main__":
    main()
