"""Compare actual initial archives under a common sequence of vapor resets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from forecast import read_encoder
from large_population import raw_features
from multicycle_score import frames_by_time, summarize
from quadrature_validation import load_reference


def forcing(lengths, levels, interval):
    """Reconstruct only the externally prescribed schedule from the archive."""
    rows = []
    for duration, target in zip(lengths, levels, strict=True):
        steps = []
        for leg, n in enumerate(duration):
            if leg:
                steps.append([0.0, 0.0, 1.0, 1.0, target[leg]])
            steps.extend([[interval, 0.0, 1.0, 0.0, 0.0]] * int(n))
        rows.append(steps)
    return np.repeat(np.asarray(rows), 2, axis=0)


def inputs(root, files, ids, mode="multicycle", split="validation"):
    """Collect shared environmental metadata and the initial native arrays."""
    mass, number, gas, lengths, levels = [], [], [], [], []
    seen = []
    environment = None
    source = (
        root / ("data/multicycle" if mode == "multicycle" else "data") / split
    )
    for file in files:
        with np.load(source / file) as d:
            n = len(d["ids"])
            mass.append(d["mass"][0])
            number.append(d["counts"])
            gas.append(d["gas"][0])
            reset = np.r_[0, np.where(np.diff(d["time"]) == 0)[0] + 1]
            seen.extend(d["ids"])
            durations = np.diff(np.r_[reset, len(d["time"])]) - 1
            if "lengths" in d:
                np.testing.assert_array_equal(durations, d["lengths"])
            lengths.extend([durations] * n)
            imposed = d["gas"][reset].T
            np.testing.assert_array_equal(imposed[::2], imposed[1::2])
            levels.extend(imposed[::2])
            steps = int(d["steps"])
            interval = d["time"][1]
            density = d["density"]
            volume = float(d["volume"])
            reconstructed = forcing(lengths[-n:], levels[-n:], interval)
            current = (steps, interval, tuple(density), volume)
            if environment is not None and current != environment:
                raise ValueError(
                    "Archive has inconsistent cadence, densities or volume"
                )
            environment = current
            np.testing.assert_allclose(
                reconstructed[0, :, 0],
                np.diff(d["time"]),
                rtol=1e-12,
                atol=1e-14,
            )
            if "force" in d:
                np.testing.assert_allclose(
                    reconstructed,
                    np.swapaxes(d["force"], 0, 1),
                    rtol=1e-13,
                    atol=1e-16,
                )
    np.testing.assert_array_equal(seen, ids)
    common = dict(
        ids=np.asarray(ids),
        lengths=np.stack(lengths),
        gas_levels_kg_m3=np.stack(levels),
        interval_s=interval,
        volume_m3=volume,
        density=density,
        metadata=json.dumps(
            dict(
                temperature_K=293.0,
                pressure_Pa=101325.0,
                kappa=[0.0, 0.03, 0.75],
                materials=["water", "low-kappa", "high-kappa"],
                gas_to_material=[0],
                order="internal, external for each family",
                native_steps=steps,
            ),
            sort_keys=True,
        ),
    )
    return (
        common,
        np.concatenate(mass),
        np.concatenate(number),
        np.concatenate(gas),
    )


def metrics(pred, ref, bound, times, frames):
    """Score refined paired contrasts and every individual aerosol."""
    obs = []
    chain = True
    horizon = 0.0
    count = len(ref)
    for j, t in enumerate(times[1:], 1):
        ix = (np.arange(count), slice(None), frames[:, j])
        uncertainty = bound[np.arange(count), frames[:, j], None]
        m = summarize(pred[ix], ref[ix], uncertainty)
        if m["qualified"] is not None:
            chain = chain and m["qualified"]
            if chain:
                horizon = float(t)
        obs.append(m)
    return dict(
        contiguous_ms=horizon * 1000,
        mean_member_mae_mg_m3=np.mean([r["member_mae_mg_m3"] for r in obs]),
        terminal_member_mae_mg_m3=obs[-1]["member_mae_mg_m3"],
        terminal_pair_mae_mg_m3=obs[-1]["contrast_mae_mg_m3"],
    )


def physical_replay(root, files, ids, method, archive, common, physical):
    """Pack the same persisted coordinates used for the full native replays."""
    folder = root / "multicycle-quadrature-replay" / method / "validation"
    states = []
    predictions = []
    seen = []
    projected = []
    for file in files:
        with (
            np.load(folder / file) as initial,
            np.load(folder / ("forecast-" + file)) as future,
        ):
            np.testing.assert_array_equal(initial["ids"], future["ids"])
            seen.extend(initial["ids"])
            key = "moments" if "moments" in initial else "shape"
            states.append(initial[key])
            predictions.append(future["water"].T)
            if "moment_projections" in future:
                projected.extend(future["moment_projections"])
    np.testing.assert_array_equal(seen, ids)
    state = np.concatenate(states)
    np.savez_compressed(
        archive, **common, physical=physical, method=method, **{key: state}
    )
    return (
        np.concatenate(predictions).reshape(len(ids), 2, -1),
        0,
        int(np.prod(state.shape[1:])),
        (np.mean(np.asarray(projected) > 0) if key == "moments" else None),
    )


@torch.inference_mode()
def learned(
    root, model_name, archive, common, mass, counts, physical, precision
):
    """Reload the initial embedding and advance under the stored forcing."""
    cp = root / model_name
    model, info = load_model(cp, "cpu")
    ep = root / info["encoder"]
    enc = read_encoder(ep, "cpu")
    z = enc.encode(
        raw_features(torch.as_tensor(mass)),
        torch.as_tensor(counts, dtype=torch.float32),
    )
    np.savez_compressed(
        archive,
        **common,
        physical=physical,
        shape=z.numpy().astype(precision),
        model_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
        encoder_sha256=hashlib.sha256(ep.read_bytes()).hexdigest(),
    )
    with np.load(archive) as a:
        z = torch.tensor(a["shape"], dtype=torch.float32)
        p = torch.as_tensor(a["physical"])
        f = torch.as_tensor(
            forcing(a["lengths"], a["gas_levels_kg_m3"], float(a["interval_s"]))
        )
        _, future = model.rollout((z - model.z_mean) / model.z_std, p, f)
    return (
        future[..., 1].numpy().reshape(len(mass) // 2, 2, -1),
        cp.stat().st_size + ep.stat().st_size,
        z.shape[-1],
        None,
    )


def compare(root, methods, checkpoints, out=None):
    """Charge data and shared weights for one fixed validation cohort."""
    files, ids, time, _, bound, ref, _ = load_reference(
        root, "multicycle", "validation"
    )
    times, frames = frames_by_time(time)
    common, mass, counts, gas = inputs(root, files, ids)
    n = counts / common["volume_m3"]
    physical = np.column_stack([n.sum(1), (n[:, :, None] * mass).sum(1), gas])
    out = out or root / "multicycle-storage"
    out.mkdir(exist_ok=True)
    native = out / "native.npz"
    np.savez_compressed(native, mass=mass, counts=counts, gas=gas, **common)
    native_water = []
    for name in files:
        with np.load(root / "data/multicycle/validation" / name) as history:
            native_water.append(history["water"].T)
    rows = []

    patterns, cluster = np.unique(
        common["lengths"], axis=0, return_inverse=True
    )
    by_pattern = []

    def record(name, file, result):
        pred, weights, dims, projected = result
        subrows = []
        for i, durations in enumerate(patterns):
            selected = cluster == i
            m = metrics(
                pred[selected],
                ref[selected],
                bound[selected],
                times,
                frames[selected],
            )
            subrows.append(
                dict(
                    method=name,
                    durations=list(map(int, durations)),
                    families=int(selected.sum()),
                    **m,
                )
            )
        by_pattern.extend(subrows)
        rows.append(
            dict(
                method=name,
                duration_patterns=len(subrows),
                minimum_pattern_horizon_ms=min(
                    v["contiguous_ms"] for v in subrows
                ),
                families=len(ids),
                shape_width=dims,
                state_bytes=file.stat().st_size,
                weight_bytes=weights,
                bytes_per_batch_at_1000=file.stat().st_size + weights / 1000,
                fraction_members_projected=projected,
                **metrics(pred, ref, bound, times, frames),
            )
        )

    record(
        "native",
        native,
        (np.concatenate(native_water).reshape(len(ids), 2, -1), 0, 0, None),
    )
    for method in methods:
        path = out / (method + ".npz")
        record(
            method,
            path,
            physical_replay(root, files, ids, method, path, common, physical),
        )
    for checkpoint in checkpoints:
        for precision in ("float16", "float32"):
            name = Path(checkpoint).stem + "-" + precision
            path = out / (name + ".npz")
            record(
                name,
                path,
                learned(
                    root,
                    checkpoint,
                    path,
                    common,
                    mass,
                    counts,
                    physical,
                    precision,
                ),
            )
    for name, data in (("frontier", rows), ("by-duration-pattern", by_pattern)):
        with (out / (name + ".csv")).open("w", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=data[0], lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    print(out, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    compare(**vars(p.parse_args()))
