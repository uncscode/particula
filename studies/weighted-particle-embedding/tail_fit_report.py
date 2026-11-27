"""Replay fitted bulk-transfer corrections and learned particle statistics."""

import argparse
from pathlib import Path

import numpy as np
import torch
from multicycle_storage import forcing
from node_transfer import write_table
from projection_study import digest
from tail_learning import TailCorrection, features
from tail_quadrature import decode, encode
from tail_study import physical, read, score


@torch.no_grad()
def bulk(root, checkpoints, folder, out, prefix, split="validation"):
    """Rebuild the physical baseline and charge it to each neural predictor."""
    source = root / "composition-tails"
    fine = read(source / f"{prefix}128" / split)
    coarse = read(source / f"{prefix}64" / split)
    np.testing.assert_array_equal(fine["ids"], coarse["ids"])
    ids = fine["ids"]
    times = np.broadcast_to(fine["time"], (len(ids), len(fine["time"]))).copy()
    delta = abs(fine["w"] - coarse["w"])
    truth = dict(
        time=torch.from_numpy(times.repeat(2, 0)),
        water=torch.from_numpy(fine["w"]),
        ref=(fine["w"][1::2] - fine["w"][::2], delta[1::2] + delta[::2]),
    )
    x = torch.from_numpy(features(fine["m"], fine["n"])).float()
    count = torch.from_numpy(fine["n"])
    population = encode(fine["m"], fine["n"], 8, 2)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for cp in checkpoints:
        cfg = torch.load(cp, weights_only=True, map_location="cpu")
        if cfg["step"] != 1200:
            raise ValueError("Incomplete checkpoint")
        model = TailCorrection(cfg["width"], cfg["encoder"], cfg["head"])
        model.load_state_dict(cfg["model"])
        shape = model.encode(x, count).numpy()
        dest = folder / f"{cp.stem}.npz"
        np.savez_compressed(
            dest,
            population=population,
            shape=shape,
            ids=ids,
            time=fine["time"],
            levels=fine["levels"][::2],
            density=fine["density"],
            volume=fine["volume"],
            checkpoint_sha256=digest(cp),
        )
        with np.load(dest) as a:
            np.testing.assert_array_equal(a["ids"], ids)
            m, n = decode(a["population"])
            vol = float(a["volume"])
            dry = (m[:, :, 1:].sum(-1) * n).sum(1) / vol
            levels = a["levels"]
            steps = (len(a["time"]) - 2) // 2
            f = forcing(np.full((len(levels), 2), steps), levels, a["time"][1])
            base, gas = physical(m, n, levels[:, 0].repeat(2), f, vol, 12)
            w, g = model.advance(
                torch.as_tensor(a["shape"]),
                torch.as_tensor(dry),
                torch.as_tensor(f),
                base,
                gas,
            )
            stats = score(
                cp.stem,
                dest,
                cp.stat().st_size,
                w,
                g,
                truth,
                times,
                shape,
                digest(cp),
            )
            rows.append(
                dict(
                    domain=prefix,
                    encoder=cfg["encoder"],
                    head=cfg["head"],
                    seed=cfg["seed"],
                    train_ids_sha256=cfg["train_ids_sha256"],
                    parameters=sum(p.numel() for p in model.parameters()),
                    **stats,
                )
            )
    write_table(rows, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("root", "folder", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument(
        "--split",
        choices=("validation", "test", "challenge"),
        default="validation",
    )
    p.add_argument("--prefix", default="refinement-s")
    torch.set_num_threads(1)
    bulk(**vars(p.parse_args()))
