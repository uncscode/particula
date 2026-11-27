"""Freeze and replay forecasts across independent aerosol families."""

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ensemble_manifest import create as freeze_ensemble
from projection_study import digest

FORECASTS = [
    "first-full-cycle",
    "scheduled-first-s916",
    "temporal-recursive-pilot-w8-s1831",
    "continuous-flow-continuous-h1-w8-s192",
    "expanded-ode-h1-w8-s77",
    "capacity-continuous-h1-w8-s1609",
    "capacity-transformer-h1-w8-s1609",
    "capacity-residual-h1-w8-s1609",
    "capacity-recurrent-h4-w8-s1609",
    "capacity-transformer-h4-w8-s1609",
    "expanded-transformer-h16-w8-s77",
    "expanded-recurrent-h16-w8-s77",
    "expanded-transformer-h16-w32-s77",
]
STOCHASTIC = [
    "continuous-flow-continuous-h1-w8-s192",
    "capacity-residual-h1-w8-s1609",
    "expanded-recurrent-h4-w8-s77",
    "capacity-transformer-h4-w8-s1609",
    "temporal-recursive-pilot-w8-s1831",
    "first-full-cycle",
    "thermal-capacity-residual-h1-w8-s1609",
]
GROUPS = (
    "pulse",
    "multicycle",
    "tails",
    "dual",
    "async",
    "long-async",
    "covariance",
    "ternary",
    "multivapor",
    "history",
    "equal-volume",
    "stochastic",
)


def paper_trail(root, entries):
    """Bind every model, data source and shared basis to its content hash."""
    result = {}
    for file in sorted(set(entries)):
        if not file.is_file():
            raise FileNotFoundError(file)
        result[str(file.relative_to(root))] = digest(file)
    return result


def sorted_inputs(folder):
    """Reject a missing cohort before publishing a frozen manifest."""
    from dual_experiment import completed_shards

    files = completed_shards(folder)
    if not files:
        raise FileNotFoundError(folder)
    return files


def source_identity():
    """Hash the study implementation and qualification criteria."""
    study = Path(__file__).resolve().parent
    return {
        p.name: digest(p)
        for p in [*sorted(study.glob("*.py")), study / "qualification.md"]
    }


def model_files(root, config, paths, groups):
    """Identify each trained model, encoder and basis in the comparison."""
    # Trainable checkpoints in these fixed panels use one independent seed
    # across qualitative variants or all three seeds for the selected families.
    for group, names in (
        ("forecasts", FORECASTS),
        ("stochastic_models", STOCHASTIC),
    ):
        config[group] = [x + ".pt" for x in names]
        if group == "stochastic_models" and "stochastic" not in groups:
            continue
        for name in config[group]:
            cp = root / name
            state = torch.load(cp, map_location="cpu", weights_only=True)
            enc = root / state["encoder"]
            if state.get("encoder_sha256", digest(enc)) != digest(enc):
                raise ValueError("Encoder changed")
            paths.update((cp, enc))
    config["bulk"] = [
        f"tail-learning/broad/models/{enc}-{kind}-s2801.pt"
        for enc in ("learned", "fourth")
        for kind in ("ff", "gru", "mixture")
    ]
    config["hybrid"] = [
        f"tail-hybrid/broad/models/n6-{kind}-l{strength}-s2901.pt"
        for kind in ("power", "mlp", "mixture")
        for strength in ("0", "0.1")
    ]
    config["dual"] = [
        f"dual-vapor/models/{kind}-n1024-u1200-s{seed}.pt"
        for kind in ("ff", "gru", "mixture")
        for seed in (3101, 3102, 3103)
    ]
    config["dual"] += [
        f"dual-vapor/context-models/{kind}-n1024-u1200-s{seed}.pt"
        for kind in ("set", "attention", "spatiotemporal")
        for seed in (3101, 3102, 3103)
    ]
    config["async"] = [x for x in config["dual"] if x.endswith("s3101.pt")]
    config["async"] += (
        [
            f"dual-schedules/models/{kind}-n1024-u2400-s3201.pt"
            for kind in ("ff", "gru", "attention", "spatiotemporal")
        ]
        + [
            f"dual-schedules/stages/rk{order}i1-spatiotemporal-n1024-u1200-s3301.pt"
            for order in (2, 3)
        ]
        + [
            f"dual-schedules/contrast-models/contrast0.1-rk0i1-{kind}-u1800-s3401.pt"
            for kind in ("ff", "gru", "spatiotemporal")
        ]
    )
    config["async"] += [
        f"dual-schedules/capacity-4096/contrast0.1-rk0i1-{kind}-u4800-s3801{suffix}.pt"
        for kind in ("ff", "gru", "spatiotemporal")
        for suffix in ("", "-c4")
    ]
    config["long_async"] = [
        f"dual-schedules/models/{kind}-n1024-u2400-s3201.pt"
        for kind in ("ff", "gru", "spatiotemporal")
    ]
    config["ternary"] = [
        f"dual-ternary/learned-models/joint-w{w}-flux{mix}-s3701.pt"
        for w in (1, 2, 3, 4)
        for mix in (0, 0.9, 1)
    ]
    config["multivapor"] = [
        f"multiple-vapors/models/v{counts}-{kind}-u1800-s4001.pt"
        for counts in ("23", "2", "3")
        for kind in ("ff", "gru", "attention", "mixture")
    ]
    config["history"] = [
        f"history-restart/models/{kind}-h{frames}-noise0-s4201.pt"
        for kind in ("ff", "gru", "mixture")
        for frames in (2, 8)
    ]
    config["recursive_history"] = [
        f"history-restart/recursive-models/recursive-{kind}-n65536-u{updates}-s4401.pt"
        for kind, updates in (
            ("physical", 0),
            ("ff", 6000),
            ("gru", 6000),
            ("mixture", 6000),
        )
    ]
    config["history"].append(
        "history-restart/models/instant-h2-noise0-s4201.pt"
    )
    config["widths"] = [16, 20, 21, 24, 28]
    for group, keys in (
        ("tails", ("bulk", "hybrid")),
        ("multivapor", ("multivapor",)),
        ("ternary", ("ternary",)),
        ("history", ("history", "recursive_history")),
        ("equal-volume", ("history", "recursive_history")),
        ("dual", ("dual",)),
        ("async", ("async",)),
        ("long-async", ("long_async",)),
    ):
        if group in groups:
            for key in keys:
                paths.update(root / x for x in config[key])
    if set(groups) & {"pulse", "multicycle"}:
        models = root / "profile-quant/models"
        paths.update(models / f"n5-w{x}.npz" for x in config["widths"])
        paths.add(models / "train-ids.npy")


def vapor_inputs(root, groups, split, paths):
    """Bind each two-vapor forcing and native timestep."""
    if "dual" in groups:
        for phase in ("together", "opposed"):
            for n in (64, 128):
                paths.update(
                    sorted_inputs(
                        root / f"dual-vapor/{split}/{phase}/s{n}-b16-t16-cuda"
                    )
                )
    if "long-async" in groups:
        for steps in (64, 128):
            for blocks in (3, 24):
                paths.update(
                    sorted_inputs(
                        root
                        / "dual-schedules"
                        / split
                        / "conditioned"
                        / f"b{blocks}-s{steps}-k16"
                    )
                )
    if "async" in groups:
        for geometry in ("conditioned", "proportional"):
            for n in (64, 128):
                paths.update(
                    sorted_inputs(
                        root / f"dual-schedules/{split}/{geometry}/b3-s{n}-k16"
                    )
                )


def colocated_inputs(root, groups, split, paths):
    """Bind the binary and ternary native resolution pairs."""
    if "covariance" in groups:
        for mode in ("uniform", "size", "composition"):
            for s in (64, 128):
                paths.update(
                    sorted_inputs(
                        root / f"dual-covariance/{split}/{mode}/s{s}-k12-t16"
                    )
                )
    if "ternary" in groups:
        for name in (
            "cubic-uniform",
            "quartic-uniform",
            "cubic-composition",
            "quartic-composition",
        ):
            for s in (128, 256):
                paths.update(
                    sorted_inputs(root / f"dual-ternary/{split}/{name}/s{s}-k8")
                )


def species_inputs(root, groups, split, paths):
    """Bind the restart histories and variable-species native trajectories."""
    for group, directory in (
        ("history", "history-restart"),
        ("equal-volume", "history-isovolume"),
    ):
        if group in groups:
            families = 192 if split == "validation" else 96
            paths.add(root / directory / f"{split}-{families}.npz")
            for steps in (64, 128):
                paths.update(
                    sorted_inputs(root / directory / f"native/{split}/s{steps}")
                )
    if "multivapor" in groups:
        for species in (2, 3, 4):
            for steps in (64, 128):
                paths.update(
                    sorted_inputs(
                        root
                        / f"multiple-vapors/{split}/v{species}-s{steps}-k12"
                    )
                )


def cohort_files(root, config, paths, out, reference_seeds):
    """Bind all source trajectories and a common set of stochastic seeds."""
    splits, groups, reserved = (
        config[k] for k in ("splits", "groups", "reserved")
    )
    for split in splits:
        if "pulse" in groups:
            paths.update(sorted_inputs(root / "data" / split))
            paths.add(root / f"{split}-refinement.json")
        if "multicycle" in groups:
            paths.update(sorted_inputs(root / "data/multicycle" / split))
            paths.add(root / f"multicycle-{split}-refinement.npz")
        if "tails" in groups:
            prefix = "reserved-s" if reserved else "refinement-s"
            config["tail_prefix"] = prefix
            for n in (32, 64, 128):
                paths.update(
                    sorted_inputs(
                        root / f"composition-tails/{prefix}{n}" / split
                    )
                )
        vapor_inputs(root, groups, split, paths)
        colocated_inputs(root, groups, split, paths)
        species_inputs(root, groups, split, paths)
        if "stochastic" in groups and split != "challenge":
            src = root / "reserved" / split if reserved else root
            ref = out.parent / f"{out.stem}-{split}-ensemble.json"
            freeze_ensemble(src, ref, [64, 128, 256], reference_seeds, 32, 1024)
            config["ensemble_file"] = ref.name
            config["ensemble_sha256"] = digest(ref)
            config["ensemble_seeds"] = reference_seeds


def prepare(root, out, reserved, groups, reference_seeds):
    """Freeze independent data and models before reading forecast scores."""
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits = ("test", "challenge") if reserved else ("validation",)
    paths = set()
    config = {"splits": splits, "groups": groups, "reserved": reserved}
    model_files(root, config, paths, groups)
    cohort_files(root, config, paths, out, reference_seeds)
    config.update(
        files=paper_trail(root, paths),
        source=source_identity(),
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(  # noqa: S603
            [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
    )
    payload = json.dumps(config, indent=2, sort_keys=True) + "\n"
    with out.open("x") as f:
        f.write(payload)
    print(digest(out), len(paths), flush=True)


def baselines(root, mode, split, dst):
    """Compare persistence, sections and full particle states."""
    from multicycle_storage import forcing, inputs
    from node_fit import metrics_for_series
    from node_transfer import write_table
    from quadrature_validation import load_reference
    from reduced_exchange import rollout
    from tail_quadrature import decode, encode

    files, ids, t, pair, bound, ref, level = load_reference(root, mode, split)
    common, m, n, g = inputs(root, files, ids, mode, split)
    f = forcing(
        common["lengths"], common["gas_levels_kg_m3"], common["interval_s"]
    )
    truth = dict(
        time=torch.from_numpy(t.repeat(2, 0)),
        water=torch.from_numpy(ref.reshape(len(m), -1)),
        ref=(pair, bound),
    )
    rows = []
    for name, classes in (
        ("persistence", -1),
        ("size-only", 1),
        ("size-composition", 3),
        ("full-particle-nodes", 0),
    ):
        file = dst / f"{mode}-{name}.npz"
        if classes > 0:
            record = dict(shape=encode(m, n, 8, classes))
        elif classes == 0:
            record = dict(mass=m, counts=n)
        else:
            record = dict(
                initial_water=(m[..., 0] * n).sum(1) / common["volume_m3"]
            )
        np.savez_compressed(file, **record, **common)
        with np.load(file) as a:
            if classes < 0:
                water = np.broadcast_to(
                    a["initial_water"][:, None], (len(m), f.shape[1] + 1)
                ).copy()
                gas = np.zeros_like(water)
                gas[:, 0] = g
                for j, frame in enumerate(f.transpose(1, 0, 2)):
                    gas[:, j + 1] = np.where(
                        frame[:, 3] > 0, frame[:, 4], gas[:, j]
                    )
                w, gg = torch.from_numpy(water), torch.from_numpy(gas)
            else:
                mass, count = (
                    decode(a["shape"]) if classes else (a["mass"], a["counts"])
                )
                w, gg = rollout(
                    *map(
                        torch.from_numpy,
                        (mass, count / float(a["volume_m3"]), g, f),
                    ),
                    6,
                    True,
                )
        values = metrics_for_series(w, gg, truth, True)
        for schedule, out in values.items():
            rows.append(
                dict(
                    model=name,
                    schedule=schedule,
                    member_target=level,
                    bytes=file.stat().st_size,
                    archive_sha256=digest(file),
                    **out,
                )
            )
    write_table(rows, dst / "physical-controls.csv")


def qualify_joint(root, dst, manifest, split, group):
    """Score equal-marginal and equal-covariance constructions."""
    if group == "covariance":
        from dual_covariance_score import run

        run(
            root / "dual-covariance",
            dst / "joint-loading.csv",
            split,
            ["uniform", "size", "composition"],
            64 if split == "validation" else 48,
            [0, 5],
            ["marginal", "global", "size", "joint"],
            [4],
            [],
            ["3x3"],
            False,
        )
    else:
        from dual_ternary_score import run
        from learned_joint_score import evaluate

        labels = [
            "cubic-uniform",
            "quartic-uniform",
            "cubic-composition",
            "quartic-composition",
        ]
        families = 48 if split == "validation" else 36
        run(
            root / "dual-ternary",
            dst / "mixed-moments.csv",
            split,
            labels,
            families,
            [0, 5],
            ["marginal", "covariance", "third", "fourth"],
            ["global", "joint"],
            256,
            False,
            2,
        )
        evaluate(
            root / "dual-ternary",
            dst / "learned-features.csv",
            [root / x for x in manifest["ternary"]],
            split,
            labels,
            families,
            5,
            2,
            256,
            False,
        )


def qualify_single_vapor(root, dst, manifest, split, group):
    """Benchmark dynamics and storage on matched pulse histories."""
    from diagnostic_trajectory import score as diagnostics
    from multicycle_score import score as multi
    from projection_study import evaluate as projections
    from validate_horizons import compare as pulse

    cp = [root / x for x in manifest["forecasts"]]
    if group == "pulse":
        pulse(root, split, cp, dst / "horizons.csv", "cpu", bundles=dst)
    for file in cp:
        if group == "multicycle":
            multi(root, split, file, dst / (file.stem + ".csv"), "cpu")
        diagnostics(
            root,
            root,
            split,
            group,
            file,
            dst / (file.stem + "-diagnostics.csv"),
            False,
            "cpu",
        )
    baselines(root, group, split, dst)
    projections(
        root,
        root / "profile-quant/models",
        dst / "profile-records",
        dst / "compact-profiles.csv",
        5,
        manifest["widths"],
        [6],
        "int12",
        mode=group,
        split=split,
    )


def qualify_extended(root, dst, manifest, split, group):
    """Compare prolonged forcing, hydration history and new materials."""
    if group == "long-async":
        from dual_long_schedule import run as prolonged

        prolonged(
            root / "dual-schedules",
            dst / "long-pulses.csv",
            [root / x for x in manifest["long_async"]],
            split,
            64 if split == "validation" else 24,
            ["conditioned"],
            24,
            [0, 2, 4, 20],
        )
    elif group in ("history", "equal-volume"):
        from history_fit_score import score as hydration
        from history_recursive_eval import run as infer_allocation
        from history_response import prepare as observation_surface

        families = 192 if split == "validation" else 96
        home = root / (
            "history-restart" if group == "history" else "history-isovolume"
        )
        record = home / f"{split}-{families}.npz"
        surface = dst / "response-surface.npz"
        estimates = dst / "recursive-estimates"
        observation_surface(record, surface)
        infer_allocation(
            record,
            surface,
            estimates,
            [root / k for k in manifest["recursive_history"]],
            [0, 0.01],
            [3],
        )
        hydration(
            home,
            record,
            dst / "hydration-history.csv",
            [root / x for x in manifest["history"]],
            split,
            [0, 0.01],
            [2, 8],
            True,
            False,
            sorted(estimates.glob("*.npz")),
        )
    elif group == "multivapor":
        from multivapor_experiment import assess

        assess(
            root / "multiple-vapors",
            dst / "species-transfer.csv",
            split,
            [2, 3, 4],
            80 if split == "validation" else 48,
            [5],
            [1, 4],
            [root / x for x in manifest["multivapor"]],
            False,
        )


def evaluated_group(root, dst, manifest, split, group, ref):
    """Evaluate one frozen panel and retain the detailed tables."""
    dst.mkdir(parents=True, exist_ok=True)
    if group in ("pulse", "multicycle"):
        qualify_single_vapor(root, dst, manifest, split, group)
    elif group == "tails":
        from tail_compact import assess as compact
        from tail_fit_report import bulk
        from tail_hybrid import evaluate as hybrid
        from tail_study import compare

        prefix = manifest["tail_prefix"]
        compare(
            root,
            dst / "records",
            prefix,
            dst / "physical-and-direct.csv",
            [6, 8, 16],
            [1, 2, 3],
            12,
            ["first-full-cycle.pt", "temporal-recursive-pilot-w8-s1831.pt"],
            64,
            split,
        )
        compact(
            root,
            prefix,
            dst / "centered",
            dst / "compact.csv",
            [6, 8],
            [8, 12],
            12,
            split,
        )
        hybrid(
            root,
            [root / x for x in manifest["hybrid"]],
            dst / "hybrid",
            dst / "learned-shape.csv",
            prefix,
            6,
            [8, 12],
            split,
        )
        bulk(
            root,
            [root / x for x in manifest["bulk"]],
            dst / "bulk",
            dst / "learned-bulk.csv",
            prefix,
            split,
        )
    elif group == "dual":
        from dual_experiment import assess

        assess(
            root / "dual-vapor",
            dst / "two-vapor.csv",
            split,
            64 if split != "validation" else 96,
            [0, 3, 5, 8],
            [1, 2, 8],
            [root / x for x in manifest["dual"]],
            False,
            True,
        )
    elif group == "async":
        from dual_schedule_score import run as asynchronous

        asynchronous(
            root / "dual-schedules",
            dst / "twelve-plateau.csv",
            split,
            ["conditioned", "proportional"],
            3,
            64 if split == "validation" else 40,
            [5],
            [1, 2],
            [root / x for x in manifest["async"]],
            ["2x2", "3x2"],
            False,
            False,
            True,
            True,
        )
        asynchronous(
            root / "dual-schedules",
            dst / "implicit-refinement.csv",
            split,
            ["conditioned", "proportional"],
            3,
            64 if split == "validation" else 40,
            [0, 5],
            [4],
            [],
            ["3x3"],
            False,
            False,
            True,
            True,
        )
        from dual_hybrid import run as hybrid_forecast

        guarded_heads = [
            root / f"dual-schedules/models/{kind}-n1024-u2400-s3201.pt"
            for kind in ("ff", "gru", "spatiotemporal")
        ]
        hybrid_forecast(
            root / "dual-schedules",
            dst / "guided-mixture.csv",
            guarded_heads,
            split,
            ["conditioned", "proportional"],
            3,
            64 if split == "validation" else 40,
            5,
            [0, 2, 4, 8],
            4,
            ["hold"],
            False,
            True,
        )
        wider = [
            x
            for x in manifest["async"]
            if "capacity-4096/" in x and x.endswith("-c4.pt")
        ]
        hybrid_forecast(
            root / "dual-schedules",
            dst / "wide-guided.csv",
            [root / x for x in wider],
            split,
            ["conditioned", "proportional"],
            3,
            64 if split == "validation" else 40,
            5,
            [4],
            4,
            ["hold"],
            False,
            True,
        )
    elif group in ("long-async", "history", "equal-volume", "multivapor"):
        qualify_extended(root, dst, manifest, split, group)
    elif group in ("covariance", "ternary"):
        qualify_joint(root, dst, manifest, split, group)
    elif group == "stochastic":
        from ensemble_comparison import summarize
        from stochastic_accuracy import score

        target = root / "reserved" / split if manifest["reserved"] else root
        files = []
        for cp in manifest["stochastic_models"]:
            path = dst / (Path(cp).stem + ".csv")
            score(
                root,
                cp,
                path,
                [64, 128, 256],
                manifest["ensemble_seeds"],
                split,
                ref,
            )
            files.append(path)
        summarize(target, ref, files, dst / "architecture-comparison.csv")


def seal_scores(dst, manifest_file):
    """Identify completed score tables and the manifest used for them."""
    csvs = sorted(dst.rglob("*.csv"))
    if not csvs:
        raise ValueError(f"No scores in {dst}")
    counts = {}
    for path in csvs:
        with path.open() as f:
            key = str(path.relative_to(dst))
            counts[key] = sum(1 for _ in csv.DictReader(f))
        if counts[key] == 0:
            raise ValueError(f"Empty scores in {path}")
    (dst / "complete.json").write_text(
        json.dumps(
            dict(
                manifest_sha256=digest(manifest_file),
                rows=counts,
                forecast_sha256={
                    str(p.relative_to(dst)): digest(p)
                    for p in sorted(dst.rglob("*.npz"))
                },
                sha256={str(p.relative_to(dst)): digest(p) for p in csvs},
            ),
            indent=2,
        )
        + "\n"
    )


def replay(root, manifest_file, out, groups):
    """Verify every frozen digest before writing held-out scores."""
    m = json.loads(manifest_file.read_text())
    if m["source"] != source_identity():
        raise ValueError("Scoring source changed")
    for name, sha in m["files"].items():
        if digest(root / name) != sha:
            raise ValueError(f"Frozen input changed: {name}")
    ref = (
        manifest_file.parent / m["ensemble_file"]
        if "ensemble_file" in m
        else None
    )
    if ref and digest(ref) != m["ensemble_sha256"]:
        raise ValueError("Ensemble manifest changed")
    for split in m["splits"]:
        for group in groups or m["groups"]:
            if group not in m["groups"]:
                raise ValueError("Group outside frozen panel")
            if split == "challenge" and group == "stochastic":
                continue
            dst = out / split / group
            evaluated_group(root, dst, m, split, group, ref)
            seal_scores(dst, manifest_file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    for name in ("prepare", "replay"):
        a = sub.add_parser(name)
        a.add_argument("root", type=Path)
        if name == "replay":
            a.add_argument("manifest_file", type=Path)
        a.add_argument("out", type=Path)
        a.add_argument(
            "--groups",
            nargs="+",
            choices=GROUPS,
            default=list(GROUPS) if name == "prepare" else None,
        )
        if name == "prepare":
            a.add_argument("--reserved", action="store_true")
            a.add_argument("--reference-seeds", type=int, default=16)
    args = vars(p.parse_args())
    torch.set_num_threads(1)
    {"prepare": prepare, "replay": replay}[args.pop("action")](**args)
