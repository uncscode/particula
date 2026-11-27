"""Score free multi-pulse forecasts at common physical times and leg ends."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from fit_forecast import predict
from forecast import read_encoder, trajectories
from measures import observables


def summarize(predicted, reference, uncertainty, contrast=None):
    """Measure paired contrast on resolved families and member error on all."""
    if contrast is None:
        contrast = reference[:, 1] - reference[:, 0]
    guess = predicted[:, 1] - predicted[:, 0]
    mask = np.abs(contrast) > 4 * uncertainty.sum(axis=1)
    result = dict(
        families=len(mask),
        resolved=int(mask.sum()),
        member_mae_mg_m3=np.abs(predicted - reference).mean() * 1e6,
    )
    if not mask.any():
        return dict(
            **result,
            sign_accuracy=None,
            contrast_mae_mg_m3=None,
            error_over_signal=None,
            qualified=None,
        )
    err = np.abs(guess[mask] - contrast[mask])
    sign = np.mean(guess[mask] * contrast[mask] > 0)
    ratio = err.mean() / np.abs(contrast[mask]).mean()
    return dict(
        **result,
        sign_accuracy=sign,
        contrast_mae_mg_m3=1e6 * err.mean(),
        error_over_signal=ratio,
        qualified=bool(sign >= 0.9 and ratio <= 0.25),
    )


def frames_by_time(time):
    """Choose the final positive-duration observation before a same-time
    reset.
    """
    # Particle mass is exactly unchanged at a reset. Its post-reset gas is
    # distinct.
    labels = np.rint(time * 1e9).astype(np.int64)
    ticks, first = np.unique(labels[0], return_index=True)
    selections = []
    for row in labels:
        local_ticks, local_first = np.unique(row, return_index=True)
        np.testing.assert_array_equal(local_ticks, ticks)
        selections.append(local_first)
    return time[0, first], np.asarray(selections)


def framewise(model, encoder, folder, device):
    """Use fixed query shapes and past boundaries for each forcing shard."""
    outputs = []
    for path in sorted(folder.glob("families-*.npz")):
        with np.load(path) as d:
            x, p, _ = observables(
                d["mass"][0],
                d["counts"],
                d["density"],
                float(d["volume"]),
                d["gas"][0],
                encoder.grid,
            )
            z = encoder.encode(
                torch.tensor(x, dtype=torch.float32, device=device),
                torch.tensor(d["counts"], dtype=torch.float32, device=device),
            )
            z = (z - model.z_mean) / model.z_std
            p = torch.as_tensor(p, device=device)
            force = torch.as_tensor(d["force"], device=device)
        if not torch.equal(
            force[..., 3], force[:, :1, 3].expand_as(force[..., 3])
        ):
            raise ValueError("Shard contains mixed reset times")
        memory = model.start_memory(z)
        age = p.new_zeros(len(z))
        sequence = [p]
        proc = force[0, :, 1:3]
        coeff = model.coefficients(z, p, proc, memory)
        for step in force:
            if not torch.equal(step[:, 1:3], proc):
                raise ValueError(
                    "Framewise vapor audit expects fixed process flags"
                )
            age = age + step[:, 0]
            zz, pp = model.propagate(
                z, p, proc, age[:, None], coeff=coeff, memory=memory
            )
            last = pp[:, 0]
            if step[0, 3] > 0:
                if memory is not None:
                    memory = model.advance_memory(
                        memory, z, p, zz[:, 0], last, proc, age
                    )
                last = torch.cat((last[:, :4], step[:, 4:5]), dim=1)
                z, p = zz[:, 0], last
                age = p.new_zeros(len(z))
                coeff = model.coefficients(z, p, proc, memory)
            sequence.append(last)
        outputs.append(torch.stack(sequence, 1))
    return torch.cat(outputs)


def forecast_state(model, info, encoder, data, folder, device, canonical):
    """Select the normal forecast or an audit with fixed query shapes."""
    if canonical:
        if info["kind"] != "relaxation":
            raise ValueError("Framewise audit requires a closed-leg model")
        p = framewise(model, encoder, folder, device)
    elif info["kind"] == "relaxation":
        z = (data["z"][:, 0] - model.z_mean) / model.z_std
        _, p = model.rollout(z, data["p"][:, 0], data["force"])
    else:
        _, p = predict(model, data)
    return p


@torch.inference_mode()
def score(
    root, split, checkpoint, out, device, dataset_root=None, canonical=False
):
    """Run from the initial particle population using only prescribed future
    forcing.
    """
    model, info = load_model(checkpoint, device)
    encoder_path = root / info["encoder"]
    digest = hashlib.sha256(encoder_path.read_bytes()).hexdigest()
    if "encoder_sha256" in info and digest != info["encoder_sha256"]:
        raise ValueError("Forecast and encoder identities differ")
    encoder = read_encoder(encoder_path, device)
    dataset_root = dataset_root or root
    data = trajectories(dataset_root, split, "multicycle", encoder, device)
    p = forecast_state(
        model,
        info,
        encoder,
        data,
        dataset_root / "data/multicycle" / split,
        device,
        canonical,
    )
    with np.load(
        dataset_root / f"multicycle-{split}-refinement.npz"
    ) as refined:
        ids = data["id"][::2].cpu().numpy()
        np.testing.assert_array_equal(ids, data["id"][1::2].cpu().numpy())
        np.testing.assert_array_equal(ids, refined["ids"])
        clusters = []
        input_hashes = []
        folder = dataset_root / "data/multicycle" / split
        for source in sorted(folder.glob("families-*.npz")):
            with np.load(source) as native:
                members = native["ids"]
                clusters.extend([members[0]] * len(members))
                input_hashes.extend(
                    [hashlib.sha256(source.read_bytes()).hexdigest()]
                    * len(members)
                )
        np.testing.assert_array_equal(refined["source_sha256"], input_hashes)
        time = refined["time"]
        source_time = data["force"][::2, :, 0].cumsum(1).cpu().numpy()
        np.testing.assert_allclose(source_time, time[:, 1:], rtol=0, atol=1e-12)
        water, uncertainty = refined["water"], refined["uncertainty"]
        end = refined["stage_ends"]
    if not torch.isfinite(p).all() or p.min() < 0:
        raise FloatingPointError("Forecast has invalid inventory")
    np.testing.assert_allclose(
        p[..., 2:4].cpu(), data["p"][..., 2:4].cpu(), rtol=1e-12, atol=1e-18
    )
    total = p[..., 1] + p[..., 4]
    closed = data["force"][..., 3] == 0
    np.testing.assert_allclose(
        torch.diff(total, dim=1)[closed].cpu(), 0.0, rtol=0, atol=3e-15
    )
    water_pred = p[..., 1].cpu().numpy().reshape(water.shape)
    rows = []
    times, frames = frames_by_time(time)
    family = np.arange(len(ids))

    def get_row(frame):
        return summarize(
            water_pred[family, :, frame],
            water[family, :, frame],
            uncertainty[family, :, frame],
        )

    chain = True
    for i in range(1, len(times)):
        row = get_row(frames[:, i])
        if row["qualified"] is not None:
            chain = chain and row["qualified"]
        rows.append(
            dict(
                group="elapsed",
                index=i,
                time_min_s=times[i],
                time_max_s=times[i],
                **row,
                contiguous=chain if row["qualified"] is not None else None,
            )
        )
    for i, frame in enumerate(end.T, 1):
        t = time[family, frame]
        rows.append(
            dict(
                group="leg_end",
                index=i,
                time_min_s=t.min(),
                time_max_s=t.max(),
                **get_row(frame),
                contiguous=None,
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out.with_suffix(".npz"),
        ids=ids,
        cluster=clusters,
        time=times,
        forecast_water=water_pred[family[:, None], :, frames].transpose(
            0, 2, 1
        ),
        native_water=water[family[:, None], :, frames].transpose(0, 2, 1),
        uncertainty=uncertainty[family[:, None], :, frames].transpose(0, 2, 1),
        stage_end_times=time[family[:, None], end],
        canonical_query_shape=canonical,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        checkpoint.name,
        "contiguous seconds",
        max((r["time_max_s"] for r in rows if r["contiguous"]), default=0),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--split", default="validation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--canonical", action="store_true")
    score(**vars(p.parse_args()))
