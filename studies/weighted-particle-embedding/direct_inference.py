"""Forecast from a directly pooled neural embedding of weighted particles."""

import argparse
from pathlib import Path

import numpy as np
import torch
from benchmark import load_model
from compact_inference import particle_state, save, schedules
from forecast import read_encoder
from measures import observables
from projection_study import digest


@torch.inference_mode()
def encode(raw, encoder, out):
    """Encode all represented particles and preserve each physical inventory."""
    enc = read_encoder(encoder, "cpu")
    with np.load(raw) as d:
        m, n = particle_state(d)
        _, g = schedules(
            d["lengths"],
            d["gas_levels_kg_m3"],
            float(d["interval_s"]),
            float(d["volume_m3"]),
            len(n),
        )
        x, p, _ = observables(
            m, n, d["density"], float(d["volume_m3"]), g, enc.grid
        )
        z = enc.encode(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(n, dtype=torch.float32),
        )
        fields = {
            k: d[k]
            for k in ("lengths", "interval_s", "volume_m3", "gas_levels_kg_m3")
        }
        save(
            out,
            dict(
                **fields,
                shape=z.numpy(),
                physical=p,
                grid=enc.grid,
                encoder_sha256=digest(encoder),
                materials=("water", "low-kappa", "high-kappa"),
            ),
        )


@torch.inference_mode()
def forecast(record, encoder, checkpoint, out):
    """Reload the stored record and run the selected causal forecaster."""
    model, info = load_model(checkpoint, "cpu")
    enc = read_encoder(encoder, "cpu")
    sha = digest(encoder)
    with np.load(record) as d:
        if (
            str(d["encoder_sha256"]) != sha
            or info.get("encoder_sha256", sha) != sha
            or str(d["grid"]) != enc.grid
        ):
            raise ValueError("Record, forecast and learned encoder must agree")
        z = torch.from_numpy(d["shape"])
        p = torch.from_numpy(d["physical"])
        f, gas = schedules(
            d["lengths"],
            d["gas_levels_kg_m3"],
            float(d["interval_s"]),
            float(d["volume_m3"]),
            len(p),
        )
        np.testing.assert_array_equal(gas, p[:, 4])
    zp, pp = model.rollout(
        (z - model.z_mean) / model.z_std, p, torch.from_numpy(f)
    )
    physical = pp.numpy()
    shape = zp * model.z_std + model.z_mean
    diag = enc.diagnose(shape, pp).numpy()
    remain = physical[..., [0, 2, 3]]
    np.testing.assert_allclose(
        remain,
        np.broadcast_to(remain[:, :1], remain.shape),
        rtol=1e-12,
        atol=1e-18,
    )
    wet = physical[:, :, 1]
    vapor = physical[:, :, 4]
    reset = f[:, :, 3] > 0
    np.testing.assert_allclose(
        np.diff(wet + vapor, axis=1)[~reset], 0, atol=3e-15
    )
    np.testing.assert_allclose(
        vapor[:, 1:][reset], f[:, :, 4][reset], atol=1e-16, rtol=1e-12
    )
    if not np.isfinite(physical).all() or np.any(physical < 0):
        raise ArithmeticError("Invalid forecast inventory")
    save(
        out,
        dict(
            time_s=np.c_[np.zeros(len(p)), np.cumsum(f[:, :, 0], 1)],
            particle_water_kg_m3=wet,
            gas_water_kg_m3=vapor,
            number_m3=physical[:, :, 0],
            dry_material_kg_m3=physical[:, :, 2:4],
            shape=shape.numpy(),
            diagnostics=diag,
            grid=enc.grid,
            mean_wet_diameter_um=diag[..., -4],
            initial_state_sha256=digest(record),
            encoder_sha256=sha,
            checkpoint_sha256=digest(checkpoint),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    subs = p.add_subparsers(dest="action", required=True)
    for action, fields in (
        ("encode", ("raw", "encoder", "out")),
        ("forecast", ("record", "encoder", "checkpoint", "out")),
    ):
        sub = subs.add_parser(action)
        for f in fields:
            sub.add_argument(f, type=Path)
    args = vars(p.parse_args())
    if args["out"].resolve() in [
        v.resolve()
        for k, v in args.items()
        if k != "out" and isinstance(v, Path)
    ]:
        p.error("Choose a distinct output archive")
    torch.set_num_threads(1)
    {"encode": encode, "forecast": forecast}[args.pop("action")](**args)
