"""Encode device-resident populations with bounded temporary storage."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from forecast import read_encoder


def raw_features(mass):
    """Compute the same dry/wet diameter and fraction inputs on device."""
    dry = mass[..., 1:].sum(-1)
    total = dry + mass[..., 0]
    dry_volume = dry / 1400.0
    wet_volume = dry_volume + mass[..., 0] / 1000.0
    dry_um = 2e6 * (3 * dry_volume / (4 * torch.pi)).pow(1 / 3)
    wet_um = 2e6 * (3 * wet_volume / (4 * torch.pi)).pow(1 / 3)
    soluble = mass[..., 2] / torch.where(dry > 0, dry, 1.0)
    water = mass[..., 0] / torch.where(total > 0, total, 1.0)
    return torch.stack(
        [
            dry_um.clamp_min(0.001).log(),
            wet_um.clamp_min(0.001).log(),
            soluble,
            water,
        ],
        -1,
    ).float()


@torch.inference_mode()
def reduce_population(enc, mass, counts, gas, volume, chunk, device):
    """Accumulate exact physical records and weighted learned coordinates."""
    boxes, slots = counts.shape
    width = enc.width
    numerator = torch.zeros(boxes, width, dtype=torch.float64, device=device)
    number = torch.zeros(boxes, 1, dtype=torch.float64, device=device)
    inventory = torch.zeros(boxes, 3, dtype=torch.float64, device=device)
    for start in range(0, slots, chunk):
        m = mass[:, start : start + chunk].to(device)
        w = counts[:, start : start + chunk, None].to(device)
        x = raw_features(m)
        x = torch.where(w > 0, x, enc.x_mean)
        feature = enc.map_particles(x)
        numerator += (w * feature).sum(1)
        number += w.sum(1)
        inventory += (w * m).sum(1)
    z = (numerator / torch.where(number > 0, number, 1.0)).float()
    p = torch.cat(
        [number / volume[:, None], inventory / volume[:, None], gas[:, None]],
        -1,
    )
    return z, p


def population(boxes, slots):
    """Draw a reproducible polydisperse, heterogeneously weighted population."""
    rng = np.random.default_rng(7319)
    r = np.exp(rng.normal(np.log(0.63e-6), 0.3, (boxes, slots)))
    dry = 4 * np.pi / 3 * r**3 * 1400.0
    fraction = rng.uniform(0.05, 0.95, (boxes, slots))
    water = dry * rng.uniform(0.1, 0.8, (boxes, slots))
    m = np.stack([water, dry * (1 - fraction), dry * fraction], -1)
    w = rng.lognormal(0.0, 1.0, (boxes, slots))
    w[:, ::19] = 0.0
    gas = np.full(boxes, 0.014)
    volume = w.sum(1) / 1e11
    return tuple(torch.from_numpy(a) for a in (m, w, gas, volume))


def measure(root, checkpoint, boxes, slots, chunk, reps, output=None):
    """Time upload, resident reductions and compact readback separately."""
    from projection_study import digest

    enc = read_encoder(root / checkpoint, "cuda")
    mass, counts, gas, volume = population(boxes, slots)
    cuda_gas, cuda_volume = gas.cuda(), volume.cuda()
    start = time.perf_counter()
    resident_mass, resident_counts = mass.cuda(), counts.cuda()
    torch.cuda.synchronize()
    upload = time.perf_counter() - start
    records = []
    for mode in ("host_chunks", "resident"):
        m, w = (
            (mass, counts)
            if mode == "host_chunks"
            else (resident_mass, resident_counts)
        )
        for i in range(reps + 1):
            torch.cuda.synchronize()
            start = time.perf_counter()
            z, p = reduce_population(
                enc, m, w, cuda_gas, cuda_volume, chunk, "cuda"
            )
            torch.cuda.synchronize()
            reduce = time.perf_counter() - start
            result, physical = z.cpu().numpy(), p.cpu().numpy()
            elapsed = time.perf_counter() - start
            records.append(
                dict(
                    mode=mode,
                    repetition=i,
                    reduction_s=reduce,
                    through_readback_s=elapsed,
                )
            )
        if mode == "host_chunks":
            host = result.copy()
        else:
            np.testing.assert_allclose(result, host, rtol=1e-6, atol=1e-7)
    numbers = counts.numpy()
    expected = np.concatenate(
        [
            numbers.sum(1, keepdims=True) / volume.numpy()[:, None],
            np.einsum("bi,bis->bs", numbers, mass.numpy())
            / volume.numpy()[:, None],
            gas.numpy()[:, None],
        ],
        -1,
    )
    np.testing.assert_allclose(physical, expected, rtol=2e-12, atol=1e-25)
    out = output or root / "large-population"
    out.mkdir(parents=True, exist_ok=True)
    record = out / f"b{boxes}-s{slots}-c{chunk}-state.npz"
    sha = digest(root / checkpoint)
    np.savez_compressed(
        record,
        shape=result,
        physical=physical,
        volume_m3=volume.numpy(),
        species=("water", "dry_a", "dry_b"),
        encoder_sha256=sha,
        schema="weighted-pulse-v1",
    )
    with np.load(record) as a:
        np.testing.assert_array_equal(a["shape"], result)
        np.testing.assert_allclose(
            a["physical"], expected, rtol=2e-12, atol=1e-25
        )
        np.testing.assert_array_equal(a["volume_m3"], volume.numpy())
        if str(a["encoder_sha256"]) != sha:
            raise ValueError(
                "Saved particle encoder differs from the fitted model"
            )

    report = dict(
        slots_per_box=slots,
        boxes=boxes,
        chunk_slots_per_box=chunk,
        checkpoint=checkpoint,
        active_slots=int(np.count_nonzero(numbers)),
        encoder_sha256=sha,
        encoder_bytes=(root / checkpoint).stat().st_size,
        record_bytes=record.stat().st_size,
        record_sha256=digest(record),
        resident_upload_s=upload,
        measurements=records,
        native_particle_input_bytes=mass.numel() * mass.element_size()
        + counts.numel() * counts.element_size(),
        compact_output_bytes=physical.nbytes + result.nbytes,
        shape=result.tolist(),
        physical=physical.tolist(),
        device=torch.cuda.get_device_name(),
        peak_device_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    (out / f"b{boxes}-s{slots}-c{chunk}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(boxes, slots, report["compact_output_bytes"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("checkpoint")
    for k, v in (
        ("boxes", 1),
        ("slots", 1048576),
        ("chunk", 16384),
        ("reps", 7),
    ):
        p.add_argument("--" + k, type=int, default=v)
    p.add_argument("--output", type=Path)
    measure(**vars(p.parse_args()))
