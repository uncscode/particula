"""Score deterministic forecasts against native stochastic ensembles."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from forecast import read_encoder
from measures import observables
from stochastic_report import read


@torch.inference_mode()
def score(
    root, checkpoint, out, slots, min_seeds, split="validation", manifest=None
):
    """Match the initial population, mode, resolution and each endpoint."""
    path = root / checkpoint
    model, info = load_model(path, "cpu")
    encoder = read_encoder(root / info["encoder"], "cpu")
    dataset = root / "reserved" / split if split == "test" else root
    from ensemble_manifest import checksum, load_frozen

    native = (
        load_frozen(dataset, manifest, slots)
        if manifest
        else read(dataset / "stochastic")
    )
    reference_sha = checksum(manifest) if manifest else None
    model_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for k in slots:
        with np.load(dataset / "stochastic-inputs" / f"k{k}.npz") as d:
            x, initial, diag = observables(
                d["mass"],
                d["counts"],
                d["density"],
                float(d["volume"]),
                d["gas"],
                encoder.grid,
            )
            ids, density, mix = d["ids"], d["density_multiplier"], d["mix"]
            p = torch.as_tensor(initial)
            z = encoder.encode(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(d["counts"], dtype=torch.float32),
            )
            time = d["time"]
            for mode in ("coagulation", "coupled"):
                candidates = [
                    s
                    for (m, kk, s), group in native.items()
                    if m == mode
                    and kk == k
                    and len(group) == 2 * len(p)
                    and min(len(v) for v in group.values()) >= min_seeds
                ]
                if not candidates:
                    continue
                steps = max(candidates)
                group = native[mode, k, steps]
                f = p.new_zeros(len(p), len(time) - 1, 5)
                f[..., 0] = torch.as_tensor(np.diff(time))
                f[..., 1] = 1
                if mode == "coupled":
                    f[..., 2] = 1
                    f[:, 8, 3] = 1
                    f[:, 8, 4] = torch.as_tensor(d["reset"])
                zp, physical = model.rollout(
                    (z - model.z_mean) / model.z_std, p, f
                )
                estimate = encoder.diagnose(
                    zp * model.z_std + model.z_mean, physical
                )
                for leg, t in (("wet_end", 8), ("dry_end", -1)):
                    guess = np.stack(
                        [
                            (physical[:, t, 0] / p[:, 0]).numpy(),
                            estimate[:, t, -4].numpy(),
                            (
                                physical[:, t, 1] / physical[:, t, 2:4].sum(-1)
                            ).numpy(),
                        ],
                        -1,
                    )
                    persistence = np.stack(
                        [
                            np.ones(len(p)),
                            diag[:, -4],
                            initial[:, 1] / initial[:, 2:4].sum(-1),
                        ],
                        -1,
                    )
                    for b in range(len(p)):
                        realizations = np.stack(
                            list(
                                group[
                                    int(ids[b]),
                                    int(density[b]),
                                    int(mix[b]),
                                    leg,
                                ].values()
                            )
                        )
                        mean = realizations.mean(0)
                        se = realizations.std(0, ddof=1) / np.sqrt(
                            len(realizations)
                        )
                        for j, observable in enumerate(
                            (
                                "number_fraction",
                                "mean_wet_diameter_um",
                                "water_dry_ratio",
                            )
                        ):
                            rows.append(
                                dict(
                                    checkpoint=checkpoint,
                                    sha256=model_sha,
                                    reference_manifest_sha256=reference_sha,
                                    split=split,
                                    mode=mode,
                                    slots=k,
                                    native_steps=steps,
                                    leg=leg,
                                    seconds=time[t],
                                    family=ids[b],
                                    density_multiplier=density[b],
                                    mix=mix[b],
                                    seeds=len(realizations),
                                    observable=observable,
                                    forecast=guess[b, j],
                                    persistence=persistence[b, j],
                                    native_mean=mean[j],
                                    native_standard_error=se[j],
                                    forecast_minus_mean=guess[b, j] - mean[j],
                                    mae_over_realizations=np.abs(
                                        guess[b, j] - realizations[:, j]
                                    ).mean(),
                                )
                            )
    if not rows:
        raise ValueError("No complete native stochastic comparison cells")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(checkpoint, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument(
        "--split", choices=["validation", "test"], default="validation"
    )
    p.add_argument("checkpoint")
    p.add_argument("out", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--slots", nargs="+", type=int, default=[64, 128, 256, 512])
    p.add_argument("--min-seeds", type=int, default=8)
    score(**vars(p.parse_args()))
