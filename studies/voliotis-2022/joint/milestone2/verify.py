"""Verify saved campaign evidence, frozen size inputs and study coverage."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def read(name):
    """Read one study result."""
    return json.loads((HERE / name).read_text())


def require(valid, message):
    """Raise even when Python runs with optimization enabled."""
    if not valid:
        raise ValueError(message)


def verify_lineage(bank):
    """Verify material identities, nested lineage and both order controls."""

    def material_identity(r):
        sources = tuple(sorted((r["a"], r["b"])))
        products = tuple(sorted((r["dest"], r["split"])))
        return sources, products, r["area_scaled"]

    require(
        len({material_identity(r) for r in bank}) == len(bank),
        "Equivalent material channels appear twice",
    )
    previous = set(read("selection_01.json")["archive"]["ids"])
    for record in read("lineage.json"):
        decision = read(f"selection_{record['round']:02d}.json")
        selected = decision["selected"]
        ids = set(selected["ids"])
        require(previous <= ids and ids == set(record["ids"]), "Broken lineage")
        require(
            len(ids) <= 1000 and ids <= set(range(len(bank))),
            "Invalid channels",
        )
        require(
            bool(ids - previous) == record["admitted"], "Admission mismatch"
        )
        if record["admitted"]:
            parent = decision["best_parent"]
            gain = min(
                a - b
                for a, b in zip(
                    parent.get("qualification_scores", parent["scores"][-2:]),
                    selected.get(
                        "qualification_scores", selected["scores"][-2:]
                    ),
                    strict=True,
                )
            )
            require(
                gain > selected["threshold"],
                "Selected addition failed the matched parent control",
            )
        previous = ids
    require(
        previous == set(read("selection.json")["selected"]["ids"]),
        "Frozen selection is not the lineage endpoint",
    )


def verify_ablation(inputs_hash, expected_keys):
    """Verify final frozen omission results against the published calibration."""
    omission = read("final_omission.json")
    require(omission["inputs_sha256"] == inputs_hash, "Stale omission results")
    errors = omission["fixed_omissions"]
    require(set(errors) == expected_keys, "Missing omission controls")
    require(
        max(errors, key=errors.get) == omission["chosen_channel"],
        "Wrong omission",
    )
    best = omission["best_validation_errors"]
    require(
        set(best) == {"intact", "removed"}
        and all(np.isfinite(v) and v >= 0 for v in best.values()),
        "Missing final errors",
    )


def verify_derivatives_and_size(inputs_hash, selected, additions):
    """Check derivative and size replay provenance against frozen inputs."""
    jacobian = read("ad_observation_jacobian.json")
    for name, digest in jacobian["sha256"].items():
        require(
            hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest,
            "Jacobian provenance differs",
        )
    require(
        jacobian["observations"] == 1940
        and set(jacobian["added"]) == additions,
        "Incomplete observation derivatives",
    )
    require(set(jacobian["channels"]) == selected, "Wrong Jacobian topology")
    require(
        jacobian["max_rmse_jacobian_reconstruction_error"] < 1e-9,
        "Trajectory derivatives did not recover the fitted objective",
    )
    extra = pd.read_csv(HERE / "size_results/resolution_check.csv")
    require(
        len(extra) == 6
        and (extra.inputs_sha256 == inputs_hash).all()
        and (extra.psd_scored == 272).all()
        and (extra.mode_rows == 212).all(),
        "Size-resolution replays differ from the frozen calibration",
    )


def verify():
    """Check run-level metrics and consistency across calibrations."""
    measurements = pd.read_csv(HERE / "measurements.csv.gz")
    observed = measurements.groupby("observable").size().to_dict()
    if (
        observed != {"mass": 1258, "number": 682}
        or measurements.run.nunique() != 15
    ):
        raise ValueError("Unexpected five-hour observation coverage")
    baselines = read("baselines.json")
    best = min(
        baselines,
        key=lambda k: max(
            baselines[k][s]["score"]
            for s in ("1.875s_forward", "1.875s_reverse")
        ),
    )
    require(best == "family_evidence", "Archived starting network changed")
    if (HERE / "lineage.json").exists():
        verify_lineage(read("bank.json"))
    cases = read("size_inputs.json")["cases"]
    scores = pd.read_csv(HERE / "scorecard.csv")
    instrument = pd.read_csv(HERE / "instrument_fits.csv.gz")
    provenance = read("size_results/provenance.json")
    require(
        provenance["inputs_sha256"]
        == hashlib.sha256((HERE / "size_inputs.json").read_bytes()).hexdigest(),
        "Size replay used different frozen inputs",
    )
    summary = read("diagnostics.json")
    select = read("selection.json")
    selected_channels = cases["selected"]["channels"]
    bank = dict(zip(select["selected"]["ids"], selected_channels, strict=True))
    keys = {r["key"] for r in selected_channels}
    require(
        len(keys) == len(bank) == len(selected_channels) <= 1000,
        "Invalid network",
    )
    admitted = {
        k for r in read("search_summary.json")["rounds"] for k in r["admitted"]
    }
    inherited = {r["key"] for r in cases["incumbent"]["channels"]}
    require(
        keys == inherited | admitted and inherited.isdisjoint(admitted),
        "Invalid admissions",
    )
    for key, case in cases.items():
        rates = np.array(case["x"][16:])
        dose = np.array(case["dose"])
        require(
            len(dose) == 15 and np.all((dose >= 0.85) & (dose <= 1.15)),
            "Invalid dose scales",
        )
        require(
            len(rates) == len(case["channels"]) and np.all(rates >= 0),
            "Invalid kinetic rates",
        )
        rows = instrument.query("network==@key")
        require(len(rows) == len(measurements), "Missing fitted observations")
        step = f"{select['qualification_step_seconds']:g}s"
        uncertainties = scores.query("network==@key and step==@step").set_index(
            "run"
        )
        ledger = uncertainties.closure_ug_m3
        require(
            np.all((ledger >= 0) & (ledger <= 1e-07)),
            "Organic mass ledger failed",
        )
        for (run, kind), group in rows.groupby(["run", "observable"]):
            scale = group.normalization.iloc[0]
            require(
                scale > 0 and group.normalization.nunique() == 1,
                "Invalid normalization",
            )
            if kind == "mass":
                np.testing.assert_equal(scale, max(1.0, group.value.max()))
            rmse = np.sqrt(
                np.mean(((group.prediction - group.value) / scale) ** 2)
            )
            np.testing.assert_allclose(
                rmse,
                uncertainties.at[run, "count" if kind == "number" else "mass"],
                atol=1e-11,
                rtol=0,
            )
        flux = (
            pd.read_csv(HERE / "channel_flux.csv.gz")
            .query("network==@key")
            .groupby("key")
            .gross_ug_m3.max()
        )
        require(
            (flux >= 1.0).sum() == summary[key]["appreciable"],
            "Appreciable flux count differs",
        )
        decision = select["archive" if key == "incumbent" else "selected"]
        require(
            [bank[h]["key"] for h in decision["ids"]]
            == [c["key"] for c in case["channels"]],
            "Fitted channels differ from selection",
        )
        for setting, actual in (
            ("15s", 0),
            ("1.875s", 3),
            ("1.875s_reverse", 4),
        ):
            scored = scores[(scores.network == key) & (scores.step == setting)]
            np.testing.assert_allclose(
                scored[["mass", "count"]].to_numpy().max(),
                decision["scores"][actual],
                atol=1e-11,
                rtol=0,
            )
        for suffix, expected in zip(
            ("", "_reverse"), decision["qualification_scores"], strict=True
        ):
            selected_step = step + suffix
            frame = scores[
                (scores.network == key) & (scores.step == selected_step)
            ]
            np.testing.assert_allclose(
                frame[["mass", "count"]].to_numpy().max(),
                expected,
                rtol=0,
                atol=1e-11,
            )
    size = pd.read_csv(HERE / "size_results/summary.csv")
    require(
        len(size) == 6
        and (size.mode_runs == 7).all()
        and (size.mode_rows == 212).all(),
        "Size replay coverage differs",
    )
    require(
        (size.psd_digitized == 273).all() and (size.psd_scored == 272).all(),
        "Digitized spectrum mask differs",
    )
    np.testing.assert_allclose(size.gmd_5h_observed_nm, 222.5197)
    select_ids = set(select["selected"]["ids"])
    verify_derivatives_and_size(
        provenance["inputs_sha256"],
        select_ids,
        select_ids - set(select["archive"]["ids"]),
    )
    verify_ablation(provenance["inputs_sha256"], admitted or keys)
    print("Verified observations, fits, size data, mass ledgers and ablation.")


if __name__ == "__main__":
    verify()
