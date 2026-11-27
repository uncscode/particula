"""Find the storage precision needed to retain signed pulse forecasts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from forecast import read_encoder, trajectories


def pack(shape, center, radius, bits):
    """Pack symmetric, coordinate-specific integer levels into a byte stream."""
    limit = (1 << (bits - 1)) - 1
    normalized = (shape - center) / radius
    clipped = float(np.mean(np.abs(normalized) > 1))
    codes = (
        np.rint(np.clip(normalized, -1, 1) * limit).astype(np.int32) + limit
    ).astype(np.uint16)
    binary = (codes.reshape(-1, 1) >> np.arange(bits, dtype=np.uint16)) & 1
    return np.packbits(binary.ravel(), bitorder="little"), clipped


def unpack(packed, center, radius, bits, dimensions):
    """Recover the exact stored levels, dropping trailing byte padding."""
    size = int(np.prod(dimensions))
    binary = np.unpackbits(packed, bitorder="little", count=size * bits)
    codes = binary.reshape(size, bits).astype(np.uint16) @ (
        1 << np.arange(bits, dtype=np.uint16)
    )
    limit = (1 << (bits - 1)) - 1
    return center + (codes.reshape(dimensions).astype(np.float32) - limit) * (
        radius / limit
    )


@torch.inference_mode()
def assess(root, checkpoint, split, extra):
    """Forecast from deserialized states using a training-only codebook."""
    model, info = load_model(root / checkpoint, "cpu")
    encoder = read_encoder(root / info["encoder"], "cpu")
    train = trajectories(root, "train", "pulse", encoder, "cpu")
    cal = train["z"].flatten(0, 1)
    center = cal.median(0).values
    radius = torch.quantile((cal - center).abs(), 0.999, dim=0).clamp_min(1e-5)
    center, radius = center.numpy(), radius.numpy()
    label = ("extra-" if extra else "") + split
    folder = (
        root / "data/pulse-extra" / split if extra else root / "data" / split
    )
    valid = trajectories(root, split, "pulse", encoder, "cpu", folder)
    ids = valid["id"][::2].tolist()
    reference = json.loads((root / f"{label}-refinement.json").read_text())[
        "records"
    ]
    by_id = {r["id"]: r for r in reference}
    if set(ids) != set(by_id) or len(ids) != len(by_id):
        raise ValueError("Native and encoded families differ")
    truth = np.array([by_id[key]["pair_contrast"] for key in ids])
    bound = np.array([by_id[key]["reference_error"] for key in ids])
    mask = np.abs(truth) > 4 * bound
    times = np.r_[0, np.cumsum(valid["force"][0, :, 0].numpy())]
    out = root / "precision-horizon" / Path(checkpoint).stem / label
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    frames = []
    for precision in ("float32", "float16", 12, 10, 8, 6, 4):
        shape = valid["z"].numpy()
        a = dict(
            physical=valid["p"].numpy(),
            force=valid["force"].numpy(),
            ids=np.array(ids),
            time=times,
            precision=str(precision),
            encoder_sha256=hashlib.sha256(
                (root / info["encoder"]).read_bytes()
            ).hexdigest(),
            dynamics_sha256=hashlib.sha256(
                (root / checkpoint).read_bytes()
            ).hexdigest(),
        )
        if isinstance(precision, int):
            a["codes"], clipped = pack(shape, center, radius, precision)
            a.update(center=center, radius=radius, dimensions=shape.shape)
        else:
            a["shape"] = shape.astype(precision)
            clipped = 0.0
        path = out / f"{precision}.npz"
        np.savez_compressed(path, **a)
        with np.load(path) as archive:
            z = (
                unpack(
                    archive["codes"],
                    archive["center"],
                    archive["radius"],
                    precision,
                    archive["dimensions"],
                )
                if isinstance(precision, int)
                else archive["shape"].astype(np.float32)
            )
            z = torch.as_tensor(z)
            physical = torch.as_tensor(archive["physical"])
            force = torch.as_tensor(archive["force"])
        _, p = model.rollout(
            (z[:, 0] - model.z_mean) / model.z_std, physical[:, 0], force
        )
        pred = p[..., 1].numpy()
        contrast = pred[1::2] - pred[::2]
        delta = np.abs(contrast - truth)
        ratio = (mask * delta).sum(0) / np.maximum(
            (mask * np.abs(truth)).sum(0), 1e-30
        )
        sign = ((contrast * truth > 0) & mask).sum(0) / np.maximum(
            mask.sum(0), 1
        )
        resolved = mask.any(0)
        failed = resolved & ((ratio > 0.25) | (sign < 0.9))
        chain = ~np.maximum.accumulate(failed)
        for j in np.flatnonzero(resolved):
            frames.append(
                dict(
                    precision=precision,
                    frame=j,
                    seconds=times[j],
                    resolved=int(mask[:, j].sum()),
                    relative=ratio[j],
                    sign_accuracy=sign[j],
                    contiguous=bool(chain[j]),
                )
            )
        if precision == "float32":
            baseline = pred
        summary.append(
            dict(
                precision=precision,
                families=len(ids),
                archive_bytes=path.stat().st_size,
                clipped_fraction=clipped,
                longest_ms=1000 * max(times[resolved & chain], default=0.0),
                individual_water_mae_mg_m3=1e6
                * np.abs(pred - physical[..., 1].numpy()).mean(),
                change_from_fp32_mg_m3=1e6 * np.abs(pred - baseline).mean(),
                end_contrast_mae_mg_m3=1e6 * delta[mask[:, -1], -1].mean(),
            )
        )
    for name, rows in [("summary.csv", summary), ("horizons.csv", frames)]:
        with (out / name).open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print([(r["precision"], r["longest_ms"]) for r in summary], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("checkpoint")
    p.add_argument("--split", default="validation")
    p.add_argument("--extra", action="store_true")
    assess(**vars(p.parse_args()))
