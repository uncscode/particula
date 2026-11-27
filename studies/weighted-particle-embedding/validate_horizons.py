"""Measure time-resolved contrasts against refined native data."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from fit_forecast import predict
from forecast import read_encoder, trajectories


def check_inventories(p, d, start):
    """Check exact dry state, closed water and every imposed gas reset."""
    if not torch.isfinite(p).all() or (p < 0).any():
        raise FloatingPointError("Invalid two-leg pulse inventory")
    np.testing.assert_allclose(
        p[..., (0, 2, 3)].cpu(),
        d["p"][:, start:, (0, 2, 3)].cpu(),
        rtol=1e-12,
        atol=1e-18,
    )
    total = p[..., 1] + p[..., 4]
    reset = d["force"][:, start:, 3] > 0
    np.testing.assert_allclose(
        torch.diff(total, dim=1)[~reset].cpu(), 0, rtol=0, atol=3e-15
    )
    np.testing.assert_allclose(
        p[:, 1:, 4][reset].cpu(),
        d["force"][:, start:, 4][reset].cpu(),
        rtol=1e-12,
        atol=1e-18,
    )


def member_water(ref, reference, order, target, d):
    """Resolve the timestep level of both water-inventory targets."""
    if "member_water" in ref[0]:
        members = np.stack([reference[k]["member_water"] for k in order])
        np.testing.assert_allclose(members[:, 1] - members[:, 0], target)
        member_reference = "finest"
    else:
        members = d["p"][..., 1].cpu().numpy().reshape(len(order), 2, -1)
        member_reference = "archived"
    return members, member_reference


def check_sources(folder, manifest):
    """Match the source files to the timestep-refinement run."""
    if "sources" in manifest:
        actual = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.glob("families-*.npz")
        }
        if actual != manifest["sources"]:
            raise ValueError("Refinement inputs differ from forecast inputs")


@torch.inference_mode()
def assess(
    root, split, path, device, extra=False, dataset_root=None, bundles=None
):
    """Score forecast history lengths at a shared starting observation."""
    dataset_root = dataset_root or root
    label = "extra-" + split if extra else split
    manifest = json.loads(
        (dataset_root / f"{label}-refinement.json").read_text()
    )
    ref = manifest["records"]
    ids = [r["id"] for r in ref]
    reference = {r["id"]: r for r in ref}
    info = torch.load(path, map_location=device, weights_only=True)
    encoder_path = root / info["encoder"]
    digest = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    if info.get("encoder_sha256", digest) != digest:
        raise ValueError("Forecast and encoder identities differ")
    enc = read_encoder(encoder_path, device)
    folder = (
        dataset_root / "data/pulse-extra" / split
        if extra
        else dataset_root / "data" / split
    )
    check_sources(folder, manifest)
    d = trajectories(root, split, "pulse", enc, device, folder)
    order = d["id"][::2].tolist()
    if len(ids) != len(set(ids)) or set(order) != set(ids):
        raise ValueError(
            "Reference and prediction family IDs must match exactly"
        )
    target = np.array([reference[k]["pair_contrast"] for k in order])
    bound = np.array([reference[k]["reference_error"] for k in order])
    members, member_reference = member_water(ref, reference, order, target, d)
    model, info = load_model(path, device)
    records = []
    starts = [(0, 1)]
    length = d["p"].shape[1]
    origin = min(15, length // 2 - 2)
    for observed in (1, 4, 16):
        if observed <= min(origin + 1, info.get("history", 1)):
            starts.append((origin, observed))
    for start, observed in starts:
        if info["kind"] == "relaxation":
            z = (d["z"][:, start] - model.z_mean) / model.z_std
            _, p = model.rollout(z, d["p"][:, start], d["force"][:, start:])
        else:
            _, p = predict(model, d, start=start, observed=observed)
        check_inventories(p, d, start)
        member_error = (
            np.abs(
                p[..., 1].cpu().numpy().reshape(len(order), 2, -1)
                - members[:, :, start:]
            )
            * 1e6
        )
        pred = (p[1::2, :, 1] - p[::2, :, 1]).cpu().numpy()
        elapsed = (
            torch.cat([d["force"][:1, :1, 0] * 0, d["force"][:1, start:, 0]], 1)
            .cumsum(1)[0]
            .cpu()
            .numpy()
        )
        if bundles is not None and start == 0:
            np.savez_compressed(
                bundles / (path.stem + ".npz"),
                ids=order,
                cluster=order,
                time=elapsed,
                forecast_water=p[..., 1]
                .cpu()
                .numpy()
                .reshape(len(order), 2, -1),
                native_water=members,
                pair_uncertainty=bound,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        chain = True
        for j in range(1, p.shape[1]):
            true = target[:, start + j]
            mask = np.abs(true) > 4 * bound[:, start + j]
            if not mask.any():
                continue
            error = pred[mask, j] - true[mask]
            ratio = np.abs(error).mean() / np.abs(true[mask]).mean()
            sign = np.mean(pred[mask, j] * true[mask] > 0)
            passed = bool(sign >= 0.90 and ratio <= 0.25)
            chain = chain and passed
            records.append(
                dict(
                    checkpoint=path.name,
                    split=split,
                    initial_frame=start,
                    observed_history=observed,
                    target_frame=start + j,
                    elapsed_s=float(elapsed[j]),
                    families=len(mask),
                    resolved_families=int(mask.sum()),
                    member_reference=member_reference,
                    member_mae_mg_m3=float(member_error[:, :, j].mean()),
                    member_p95_ae_mg_m3=float(
                        np.quantile(member_error[:, :, j], 0.95)
                    ),
                    sign_accuracy=sign,
                    contrast_error_over_signal=ratio,
                    contrast_mae_mg_m3=1e6 * np.abs(error).mean(),
                    contrast_p95_error_mg_m3=1e6
                    * np.quantile(np.abs(error), 0.95),
                    qualified=passed,
                    contiguous_qualified=chain,
                )
            )
    return records


def compare(
    root,
    split,
    paths,
    output,
    device,
    extra=False,
    dataset_root=None,
    bundles=None,
):
    """Tabulate qualified observations and contiguous forecast horizons."""
    rows = []
    for path in paths:
        rows.extend(
            assess(root, split, path, device, extra, dataset_root, bundles)
        )
    with output.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for path in paths:
        times = [
            r["elapsed_s"]
            for r in rows
            if r["checkpoint"] == path.name
            and r["initial_frame"] == 0
            and r["contiguous_qualified"]
        ]
        print(
            path.name, "contiguous qualified seconds", max(times, default=0.0)
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--split", default="validation")
    p.add_argument("--extra", action="store_true")
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--device", default="cpu")
    compare(**vars(p.parse_args()))
