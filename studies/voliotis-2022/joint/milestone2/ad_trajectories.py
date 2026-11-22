"""Export and independently verify the frozen dose-to-observation Jacobian."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from ad_campaign import CampaignDerivatives
from baseline_replay import HERE, Model, run, save
from growth import BANK, HI, initial


def native(model, x, ids, doses, sub):
    """Return native mass and relative number at minute boundaries."""
    diagnostics = run(model, x, ids, sub, doses[:, None], diagnostics=5)[3][0]
    return diagnostics.reshape(len(doses), 300, 3)[:, :, [0, 2]]


def native_difference(model, x, rates, doses, coordinate, sub, step):
    """Use feasible centered or one-sided changes at the campaign bounds."""
    xp, xm = x.copy(), x.copy()
    dp, dm = doses.copy(), doses.copy()
    if coordinate == -1:
        # The Jacobian in local doses is block diagonal across experiments.
        dp = np.minimum(1.15, dp + step)
        dm = np.maximum(0.85, dm - step)
        span = (dp - dm)[:, None, None]
    else:
        offset = 16 + coordinate
        xp[offset] = min(HI[offset], xp[offset] + step)
        xm[offset] = max(0.0, xm[offset] - step)
        span = xp[offset] - xm[offset]
    return (
        native(model, xp, rates, dp, sub) - native(model, xm, rates, dm, sub)
    ) / span


def rows_at_measurements(model, minute, changes, names):
    """Interpolate each derivative to the same observation times as the loss."""
    samples = []
    obs = model.data[1].numpy()
    for r, (experiment, *_) in enumerate(model.runs):
        number_cm3 = model.initial[r, 1] / 1e6
        values = np.vstack(([0, 1], minute[r])) * [1, number_cm3]
        jacobian = np.concatenate(
            [np.zeros((len(names), 1, 2)), changes[r]], axis=1
        ) * [1, number_cm3]
        for i, observed in enumerate(obs[r]):
            for j, kind in enumerate(("mass", "number")):
                weight = observed[2 * j]
                if weight < 0:
                    continue
                interp = (1 - weight) * values[i, j] + weight * values[i + 1, j]
                tangent = (1 - weight) * jacobian[:, i, j] + weight * jacobian[
                    :, i + 1, j
                ]
                samples.append(
                    dict(
                        run=experiment,
                        observable=kind,
                        time_h=(i + weight) / 60,
                        units=("ug/m3", "particles/cm3")[j],
                        measured=observed[2 * j + 1] * (number_cm3 if j else 1),
                        model=interp,
                        **dict(zip(names, tangent, strict=True)),
                    )
                )
    return samples


def aggregate_check(
    model, table, names, scores, dose_sensitivity, rate_sensitivities
):
    """Recover the loss Jacobian from the observation tangents."""
    reference = np.concatenate(
        [dose_sensitivity[:, :, None], rate_sensitivities], axis=2
    )
    reconstructed = np.zeros_like(reference)
    for i, (experiment, *_) in enumerate(model.runs):
        for j, kind in enumerate(("mass", "number")):
            group = table[
                (table.run == experiment) & (table.observable == kind)
            ]
            scale = model.initial[i, 1] / 1e6 if j else model.initial[i, 2]
            residual = (
                group.model.to_numpy() - group.measured.to_numpy()
            ) / scale
            rms = np.sqrt(np.mean(residual**2))
            np.testing.assert_allclose(rms, scores[i, j], atol=1e-11, rtol=0)
            if rms > 1e-12:
                tau = group[names].to_numpy() / scale
                reconstructed[i, j] = (
                    np.mean(residual[:, None] * tau, axis=0) / rms
                )
    np.testing.assert_allclose(reconstructed, reference, atol=2e-9, rtol=1e-9)
    return float(np.max(abs(reconstructed - reference)))


def main():
    """Record the local Jacobian to fitted physical rates and precursor dose."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True, dest="number")
    parser.add_argument("--sub", type=int, default=64)
    args = parser.parse_args()
    selection = json.loads((HERE / "selection.json").read_text())
    if args.number != selection["round"]:
        raise ValueError("Use the round of the published calibration")
    selected = selection["selected"]
    ids = selected["ids"]
    report = json.loads((HERE / "size_inputs.json").read_text())["cases"][
        "selected"
    ]
    if [BANK[h].key for h in ids] != [r["key"] for r in report["channels"]]:
        raise ValueError(
            "Frozen reaction order differs from the candidate bank"
        )
    x = np.zeros(16 + len(BANK))
    x[:16] = report["x"][:16]
    x[16 + np.array(ids)] = report["x"][16:]
    doses = np.array(report["dose"])
    _, original_ids = initial()
    added = sorted(set(ids) - set(original_ids))
    model = Model(BANK)
    engine = CampaignDerivatives(
        model, ids, added, args.sub, keep_trajectory=True
    )
    scores, local, shared = engine.evaluate(x[:16], x[16:], doses)
    changes = engine.minute_tangents.numpy()
    minute = native(model, x, ids, doses, args.sub)
    scales = np.column_stack([model.initial[:, 2], np.ones(len(doses))])[
        :, None, :
    ]
    names = [
        "d_per_unit_precursor_dose",
        *[f"d_per_unit_rate_{BANK[h].key}" for h in added],
    ]
    checks = []
    for i, coordinate in enumerate([-1, *added]):
        for step in (1e-4, 1e-5):
            diff = native_difference(
                model, x, ids, doses, coordinate, args.sub, step
            )
            disagreement = float(np.max(abs(diff - changes[:, i]) / scales))
            relative_scale = float(np.max(abs(changes[:, i]) / scales))
            if step == 1e-5 and disagreement > 5e-5 * (1 + relative_scale):
                raise ArithmeticError(
                    f"Unverified {names[i]} at step {step}: {disagreement}"
                )
            checks.append(
                dict(
                    coordinate=names[i],
                    step=step,
                    max_normalized_error=disagreement,
                    max_normalized_derivative=relative_scale,
                )
            )
    rows = pd.DataFrame(rows_at_measurements(model, minute, changes, names))
    max_reconstruction_error = aggregate_check(
        model, rows, names, scores, local, shared
    )
    rows.to_csv(
        HERE / "ad_observation_jacobian.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    digests = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("ad_observation_jacobian.csv.gz", "size_inputs.json")
    }
    save(
        "ad_observation_jacobian.json",
        dict(
            round=args.number,
            selected_fit=selected["name"],
            sha256=digests,
            channels=ids,
            added=added,
            dose=doses.tolist(),
            x=x.tolist(),
            sub=args.sub,
            order="forward",
            observations=len(rows),
            max_rmse_jacobian_reconstruction_error=max_reconstruction_error,
            checks=checks,
            normalization=(
                "Mass by each measured peak (floor 1 ug/m3), "
                "number by initial number"
            ),
            numerical_method=(
                "Native forward replays at both ends of each centered or "
                "bound-limited perturbation"
            ),
        ),
    )


if __name__ == "__main__":
    main()
