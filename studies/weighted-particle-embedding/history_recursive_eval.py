"""Predict hidden allocations with a frozen recursive estimator."""

import argparse
from pathlib import Path

import numpy as np
import torch
from history_recursive import Estimate, load
from projection_study import digest


@torch.inference_mode()
def run(record, surface, out, paths, noise, iterations):
    """Reopen the learned updates and free-run them at several depths."""
    static, x, resp, _, _ = load(record, surface)
    resp = resp.repeat_interleave(2, 0)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(record) as f:
        ids = f["ids"]
    for path in paths:
        d = torch.load(path, weights_only=True, map_location="cpu")
        net = Estimate(**d["config"])
        net.load_state_dict(d["model"])
        for sigma in noise:
            eps = np.random.default_rng(4912).normal(0, sigma, x.shape)
            eps[:, -1] = 0
            obs = x + torch.tensor(eps, dtype=x.dtype)
            for depth in iterations:
                a = net(
                    static,
                    obs,
                    resp,
                    static.new_full((len(static),), sigma),
                    depth,
                ).numpy()
                dst = out / f"{path.stem}-i{depth}-noise{sigma:g}.npz"
                np.savez_compressed(
                    dst,
                    ids=ids,
                    coef=a,
                    noise_mg_m3=sigma,
                    history_frames=x.shape[1],
                    input_sha256=digest(record),
                    checkpoint_sha256=digest(path),
                    sensitivity_sha256=digest(surface),
                    shared_bytes=path.stat().st_size,
                    seed=d["seed"],
                )
        print(path.stem, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for f in ("record", "surface", "out"):
        p.add_argument(f, type=Path)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument(
        "--noise", type=float, nargs="+", default=[0, 0.001, 0.01, 0.1]
    )
    p.add_argument("--iterations", type=int, nargs="+", default=[1, 3, 6])
    torch.set_num_threads(1)
    run(**vars(p.parse_args()))
