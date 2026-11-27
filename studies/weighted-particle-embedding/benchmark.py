"""Measure latent storage and pulse forecast time on CUDA."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from forecast import Dynamics, read_encoder
from measures import observables
from sections import sections


def write(path, **payload):
    """Write and flush a compressed history and return its file size."""
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path.stat().st_size


def load_model(path, device):
    """Restore the selected forecast weights on the measurement device."""
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved["kind"] == "continuous":
        from continuous import ContinuousODE

        m = ContinuousODE(
            saved["width"], saved["hidden"], saved["substeps"]
        ).to(device)
    elif saved["kind"] == "relaxation":
        from relaxation import Relaxation

        model_class = Relaxation
        mixture = saved.get("mixture") or {}
        if mixture:
            from mixture_exchange import MixtureExchange

            model_class = MixtureExchange
            if saved.get("temporal_experts"):
                from temporal_exchange import TemporalExchange

                model_class = TemporalExchange
        m = model_class(
            saved["width"],
            saved["hidden"],
            saved["terms"],
            saved.get("kelvin", False),
            saved.get("equilibrium_knots", 0),
            saved.get("rate_scaling", "pilot"),
            saved.get("match_flux", False),
            **mixture,
        ).to(device)
        saved["history"] = 1
    else:
        m = Dynamics(
            saved["width"],
            saved["kind"],
            saved["history"],
            saved["hidden"],
            saved.get("feedforward"),
            saved.get("thermal_transfer", False),
        ).to(device)
    m.load_state_dict(saved["weights"])
    return m.eval(), saved


@torch.inference_mode()
def benchmark(root, cell, checkpoint, repetitions):
    """Time forecast initialization, integration, decoding and output."""
    path = root / checkpoint
    device = "cuda"
    init_started = time.perf_counter()
    model, saved = load_model(path, device)
    encoder_path = root / saved["encoder"]
    enc = read_encoder(encoder_path, device)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - init_started
    out = cell / path.stem
    out.mkdir(exist_ok=True)
    with np.load(cell / "native-records.npz") as d:
        masses, gas, counts, time_s = (
            d[k] for k in ("mass", "gas", "counts", "time")
        )
        meta = str(d["metadata"])
    metadata = json.loads(meta)
    if metadata.get("grid", "pulse") != enc.grid:
        raise ValueError(
            "Native and learned timings require the same diagnostic grid"
        )
    density = np.asarray(metadata["density"])
    volume = metadata["volume_m3"]
    with np.load(cell / "native-observations.npz") as d:
        truth = d["observations"]
    samples = len(time_s) // 2 - 1
    boxes = counts.shape[0]

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    model_meta = json.dumps(
        dict(
            protocol=metadata,
            encoder_sha256=digest(encoder_path),
            forecast_sha256=digest(path),
            kind=saved["kind"],
            history=saved["history"],
            precision="fp64 inventory, fp32 shape",
        ),
        sort_keys=True,
    )
    # Use only the declared vapor-reset schedule during the forecast.
    wet = np.repeat([f["wet_rh"] for f in metadata["families"]], 2)[:boxes]
    dry = np.repeat([f["dry_rh"] for f in metadata["families"]], 2)[:boxes]
    replacement = gas[0] * dry / wet
    records = []
    for rep in range(repetitions + 1):
        start = time.perf_counter()
        x, physical, _ = observables(
            masses[0], counts, density, volume, gas[0], enc.grid
        )

        def to_device(a, dtype):
            return torch.as_tensor(a, dtype=dtype, device=device)

        z = enc.encode(
            to_device(x, torch.float32), to_device(counts, torch.float32)
        )
        p = to_device(physical, torch.float64)
        f = np.zeros((boxes, len(time_s) - 1, 5))
        f[..., 0] = np.diff(time_s)
        f[..., 2] = 1.0
        f[:, samples, 3] = 1.0
        f[:, samples, 4] = replacement
        force = to_device(f, torch.float64)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        states, ps = model.rollout((z - model.z_mean) / model.z_std, p, force)
        zs = states * model.z_std + model.z_mean
        torch.cuda.synchronize()
        integration = time.perf_counter() - begin
        obs = (
            torch.cat([ps, enc.diagnose(zs, ps).double()], -1)
            .transpose(0, 1)
            .cpu()
            .numpy()
        )
        # Common diagnostic file and metadata for native and learned timings.
        write(out / "observations.npz", observations=obs, metadata=meta)
        records.append(
            dict(
                repetition=rep,
                integration_s=integration,
                end_to_end_s=time.perf_counter() - start,
            )
        )
    latent = zs.transpose(0, 1).cpu().numpy()
    inventories = ps.transpose(0, 1).cpu().numpy()
    series_bytes = write(
        out / "forecast-records.npz",
        time=time_s,
        shape=latent,
        physical=inventories,
        metadata=model_meta,
    )
    # Observed encodings quantify storage independent of forecast smoothness.
    x, p, _ = observables(masses, counts, density, volume, gas, enc.grid)
    x, weights = (
        to_device(x, torch.float32),
        to_device(
            np.broadcast_to(counts.copy(), masses.shape[:-1]).copy(),
            torch.float32,
        ),
    )
    observed = enc.encode(x, weights)
    p_device = to_device(p, torch.float64)
    ideal = enc.diagnose(observed, p_device).cpu().numpy()
    encoded_bytes = write(
        out / "encoded-records.npz",
        time=time_s,
        shape=observed.cpu().numpy(),
        physical=p,
        metadata=model_meta,
    )
    section_bytes = {}
    for composition in (False, True):
        c = sections(x, weights, saved["width"], composition)
        label = "size-composition" if composition else "size"
        section_bytes[label] = write(
            out / (label + ".npz"),
            time=time_s,
            shape=c.cpu().numpy(),
            physical=p,
            metadata=json.dumps(
                dict(
                    protocol=metadata,
                    class_count=4 if composition else 1,
                    log_dry_um=[0.35, 3.5],
                )
            ),
        )
    init_frames = min(saved["history"], len(time_s))
    anchor = {}
    if saved["kind"] == "relaxation":
        anchor = dict(
            anchor_time=time_s[samples + 1],
            anchor_shape=latent[samples + 1],
            anchor_physical=inventories[samples + 1],
        )
    history_bytes = write(
        out / "restart-history.npz",
        **anchor,
        time=time_s[-init_frames:],
        shape=latent[-init_frames:],
        physical=inventories[-init_frames:],
        interval_forcing=f[:, -max(0, init_frames - 1) :]
        if init_frames > 1
        else f[:, :0],
    )
    native = (cell / "native-records.npz").stat().st_size
    exact = (cell / "native-checkpoint.npz").stat().st_size
    hours = (time_s[-1] - time_s[0]) / 3600
    # Per emitted batch history, one exact terminal checkpoint, and one model
    # bundle.
    errors = obs - truth
    data = dict(
        checkpoint=checkpoint,
        repetitions=records,
        model_load_and_device_init_s=model_load_s,
        boxes=boxes,
        simulated_hours_per_box=hours,
        bytes=dict(
            native_records=native,
            forecast_records=series_bytes,
            encoded_native_records=encoded_bytes,
            section_records=section_bytes,
            restart_history=history_bytes,
            exact_terminal_checkpoint=exact,
            encoder_weights=encoder_path.stat().st_size,
            forecast_weights=path.stat().st_size,
        ),
        water_mae_mg_m3=float(np.abs(errors[..., 1]).mean() * 1e6),
        max_water_error_mg_m3=float(np.abs(errors[..., 1]).max() * 1e6),
        wet_end_water_mae_mg_m3=float(
            np.abs(errors[samples, :, 1]).mean() * 1e6
        ),
        dry_end_water_mae_mg_m3=float(np.abs(errors[-1, :, 1]).mean() * 1e6),
        resolved_mean_wet_diameter_mae_um=float(np.abs(errors[..., -4]).mean()),
        decoded_true_mean_wet_diameter_mae_um=float(
            np.abs(ideal[..., -4] - truth[..., -4]).mean()
        ),
    )
    (out / "cost.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        checkpoint,
        boxes,
        "integration, end-to-end warm medians",
        [
            float(np.median([r[k] for r in records[1:]]))
            for k in ("integration_s", "end_to_end_s")
        ],
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "cell"):
        p.add_argument(name, type=Path)
    p.add_argument("checkpoint")
    p.add_argument("--repetitions", type=int, default=7)
    benchmark(**vars(p.parse_args()))
