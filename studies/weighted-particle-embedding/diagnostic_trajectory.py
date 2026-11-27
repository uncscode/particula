"""Compare diagnostic decoding of native states and freely forecast states."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from fit_forecast import predict
from forecast import read_encoder, trajectories


def time_groups(force):
    """Group first observations at each physical time within each box."""
    elapsed = torch.cat(
        [torch.zeros_like(force[:, :1, 0]), force[..., 0].cumsum(1)], 1
    )
    ticks = torch.round(elapsed * 1e9).long()
    first = torch.ones_like(ticks, dtype=torch.bool)
    first[:, 1:] = ticks[:, 1:] != ticks[:, :-1]
    for tick in torch.unique(ticks):
        yield (
            float(tick) * 1e-9,
            (first & (ticks == tick)).nonzero(as_tuple=True),
        )


def metrics(decoded, native, physical, true_physical, grid):
    """Measure errors in observable units without renormalizing decoded bins."""
    errors = (decoded - native).abs()
    nbin = 12 if grid == "pulse" else 20
    nmaterial = 1 if grid == "pulse" else 3
    fractions = decoded[:, :nbin]
    result = dict(
        diameter_mae_um=float(errors[:, -4].mean()),
        diameter_p95_ae_um=float(torch.quantile(errors[:, -4], 0.95)),
        diameter_bias_um=float((decoded[:, -4] - native[:, -4]).mean()),
        area_moment_mae_um2=float(errors[:, -3].mean()),
        dry_composition_second_moment_mae=float(errors[:, -2].mean()),
        water_composition_cross_moment_mae=float(errors[:, -1].mean()),
        number_bin_l1=float(errors[:, :nbin].sum(1).mean()),
        negative_number_fraction=float(
            fractions.clamp_max(0).abs().sum(1).mean()
        ),
        number_bin_sum_error=float((fractions.sum(1) - 1).abs().mean()),
        member_water_mae_mg_m3=float(
            (physical[:, 1] - true_physical[:, 1]).abs().mean() * 1e6
        ),
    )
    lanes = (3,) if grid == "pulse" else (1, 2, 3)
    for j, lane in enumerate(lanes, 1):
        present = true_physical[:, lane] > 0
        column = ("water", "dry_poor", "dry_rich")[lane - 1]
        result[f"{column}_bin_l1"] = (
            float(errors[present, j * nbin : (j + 1) * nbin].sum(1).mean())
            if present.any()
            else None
        )
    if grid == "tails":
        idx = (1 + nmaterial) * nbin
        result["rich_number_fraction_mae"] = float(errors[:, idx].mean())
    return result


@torch.inference_mode()
def score(root, dataset_root, split, mode, checkpoint, output, extra, device):
    """Compare both decoders on identical native boxes and forcing."""
    model, info = load_model(checkpoint, device)
    encoder_path = root / info["encoder"]
    encoder_digest = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    if info.get("encoder_sha256", encoder_digest) != encoder_digest:
        raise ValueError("Forecast and encoder identities differ")
    encoder = read_encoder(encoder_path, device)
    dataset_root = dataset_root or root
    folder = dataset_root / "data/pulse-extra" / split if extra else None
    if extra and mode != "pulse":
        raise ValueError("Additional two-leg families require pulse mode")
    data = trajectories(dataset_root, split, mode, encoder, device, folder)
    if info["kind"] == "relaxation":
        z = (data["z"][:, 0] - model.z_mean) / model.z_std
        z, p = model.rollout(z, data["p"][:, 0], data["force"])
    else:
        z, p = predict(model, data)
    estimates = [
        ("decoded_native", encoder.diagnose(data["z"], data["p"]), data["p"]),
        ("forecast", encoder.diagnose(z * model.z_std + model.z_mean, p), p),
    ]
    grid = getattr(encoder, "grid", "pulse")
    rows = []
    for time, index in time_groups(data["force"]):
        for label, diagnostics, physical in estimates:
            row = metrics(
                diagnostics[index],
                data["diag"][index],
                physical[index],
                data["p"][index],
                grid,
            )
            if not all(np.isfinite(v) for v in row.values() if v is not None):
                raise FloatingPointError("Nonfinite decoded observable")
            rows.append(
                dict(
                    mode=mode,
                    split=split,
                    grid=grid,
                    model=checkpoint.name,
                    decoder=label,
                    elapsed_s=time,
                    boxes=len(index[0]),
                    families=int(data["id"][index[0]].unique().numel()),
                    **row,
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(output, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--mode",
        choices=["pulse", "multicycle", "coagulation", "coupled"],
        default="pulse",
    )
    parser.add_argument("--extra", action="store_true")
    parser.add_argument("--device", default="cpu")
    score(**vars(parser.parse_args()))
