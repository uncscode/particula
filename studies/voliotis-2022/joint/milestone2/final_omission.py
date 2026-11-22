"""Remove each addition from the final network and refit the largest loss."""

import argparse
import hashlib
import json

import numpy as np
from ad_fine import fit_both_orders
from baseline_replay import HERE, Model, run, save
from growth import BANK


def main(calls):
    """Use matched starts, bounds and both orders for final ablation."""
    lineage_file = HERE / "lineage.json"
    fit = json.loads(lineage_file.read_text())[-1]
    inherited = json.loads((HERE / "selection_01.json").read_text())["archive"][
        "ids"
    ]
    additions = sorted(set(fit["ids"]) - set(inherited))
    candidates = additions or fit["ids"]
    model = Model(BANK)
    x, doses = np.array(fit["x"]), np.array(fit["dose"])
    trials = np.tile(x, (len(candidates) + 1, 1))
    trials[np.arange(1, len(trials)), 16 + np.array(candidates)] = 0
    errors = np.maximum(
        *[
            run(model, trials, fit["ids"], 64, doses[:, None], reverse=rev)[0]
            for rev in (False, True)
        ]
    )
    losses = errors[1:] - errors[0]
    removed = candidates[int(np.argmax(losses))]
    reduced = [h for h in fit["ids"] if h != removed]
    previous = json.loads(
        (HERE / "ad_ablation_adjudicated_01.json").read_text()
    )["parent_refit"]
    donors = [
        ("frozen calibration", x, doses),
        (
            "refitted 38-channel calibration",
            np.array(previous["x"]),
            np.array(previous["dose"]),
        ),
    ]
    output = dict(
        inputs_sha256=hashlib.sha256(
            (HERE / "size_inputs.json").read_bytes()
        ).hexdigest(),
        chosen_channel=BANK[removed].key,
        channel_id=removed,
        fixed_omissions={
            BANK[h].key: float(errors[j + 1]) for j, h in enumerate(candidates)
        },
        intact_fixed_error=float(errors[0]),
        validated_step_s=0.9375,
        calls_per_start=calls,
        fitted_step_s=1.875,
        best_validation_errors={},
    )
    for name, ids in (("intact", fit["ids"]), ("removed", reduced)):
        for label, shared, ds in donors:
            start = shared.copy()
            if name == "removed":
                start[16 + removed] = 0.0
            calibrated = fit_both_orders(model, start, ds, ids, 32, calls, 64)
            error = max(calibrated["fixed_scores"][-2:])
            best = output["best_validation_errors"]
            best[name] = min(error, best.get(name, error))
            print(name, label, calibrated["fixed_scores"], flush=True)
    save("final_omission.json", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=100)
    main(parser.parse_args().calls)
