"""Fit learned particle records through finite-reservoir water exchange."""

import argparse
from pathlib import Path

import numpy as np
import torch
from conditional_geometry import baseline, geometry
from joint_fit import atomic_save
from learned_profile import ProfileCodec, particle_inputs, section_features
from moment_quadrature import encode, size_nodes
from projected_moments import context, expand_size
from projection_study import digest
from reduced_exchange import rollout
from torch.nn import functional


def quadrature_core(core, totals):
    """Precompute count fractions and dry mass at each exact radius node."""
    n = (core.shape[1] + 2) // 2
    fixed = np.zeros((len(core), n, 2))
    size = expand_size(core)
    for b in np.flatnonzero(totals[:, 0] > 0):
        u, w = size_nodes(np.r_[1.0, size[b]])
        fixed[b, : len(w), 0] = w
        fixed[b, : len(w), 1] = w * np.maximum(1 + u, 0) ** 3
    return fixed


def nodes_from_profile(profile, fixed, totals):
    """Differentiably allocate the known number and material inventories."""
    mean, spread, upper, wet, contrast = profile.unbind(-1)
    hi = upper.clamp(0, 1)
    mix = torch.stack((1 - hi, hi), -1)
    offset = torch.stack((-hi, 1 - hi), -1)
    f = (mean[..., None] + offset * spread[..., None]).clamp(0, 1)
    water = (wet[..., None] + offset * contrast[..., None]).clamp_min(0)
    count = fixed[..., 0, None] * mix
    dry = fixed[..., 1, None] * mix
    shares = torch.stack((count, dry * water, dry * (1 - f), dry * f), -1)
    shares = shares.flatten(1, 2)
    shares = torch.where(
        (shares.sum(1) == 0)[:, None] & (totals > 0)[:, None],
        dry.flatten(1)[:, :, None],
        shares,
    )
    norm = shares.sum(1, keepdim=True)
    contained = shares / torch.where(norm > 0, norm, 1.0) * totals[:, None]
    counts = contained[..., 0]
    mass = contained[..., 1:] / torch.where(counts > 0, counts, 1.0)[..., None]
    return mass, counts


def prepare(root, initial, out, degree, length):
    """Save consecutive native observations after wet and dry gas resets."""
    with np.load(initial) as trained:
        calibration_ids = trained["ids"]
        center, scale = trained["center"], trained["scale"]
    items = []
    seen = []
    for group in ("data/multicycle/train", "small-transfer/2ms/data/train"):
        for path in sorted((root / group).glob("families-*.npz")):
            with np.load(path) as d:
                full = "mass" in d
                history = d["mass" if full else "mass0"]
                gas, forces, water = d["gas"], d["force"], d["water"]
                counts, ids, volume = d["counts"], d["ids"], float(d["volume"])
                seen.extend(ids)
                starts = [0]
                if full:
                    starts.extend(np.flatnonzero(forces[:, 0, 3]) + 1)
                for leg, start in enumerate(starts):
                    selected = slice(start, start + length)
                    if (
                        len(forces[selected]) != length
                        or forces[selected, :, 3].any()
                    ):
                        raise ValueError(
                            "Response interval spans another gas reset"
                        )
                    mass = history[start] if full else history
                    codes, totals = encode(mass, counts, degree)
                    core, p = geometry(codes, totals, degree)
                    base = baseline(core, totals)
                    target = (p - base).reshape(len(p), -1)
                    x, w = particle_inputs(mass, counts)
                    items.append(
                        dict(
                            x=x,
                            weights=w,
                            known=context(core, totals)[:, 1:].astype(
                                np.float32
                            ),
                            fixed=quadrature_core(core, totals),
                            baseline=base,
                            profile=((target - center) / scale).astype(
                                np.float32
                            ),
                            totals=totals,
                            volume=np.full(len(x), volume),
                            force=forces[selected].swapaxes(0, 1),
                            gas=gas[start],
                            water=water[start : start + length + 1].T,
                            ids=ids.repeat(2),
                            leg=np.full(len(x), leg),
                        )
                    )
    if len(seen) != len(set(seen)) or set(seen) != set(calibration_ids):
        raise ValueError(
            "Response cases differ from the original training split"
        )
    slots = max(i["x"].shape[1] for i in items)
    for i in items:
        extra = slots - i["x"].shape[1]
        i["x"] = np.pad(i["x"], ((0, 0), (0, extra), (0, 0)))
        i["weights"] = np.pad(i["weights"], ((0, 0), (0, extra)))
    arrays = {k: np.concatenate([i[k] for i in items]) for k in items[0]}
    np.testing.assert_array_equal(arrays["ids"][::2], arrays["ids"][1::2])
    np.testing.assert_allclose(
        arrays["water"][:, 0],
        arrays["totals"][:, 1] / arrays["volume"],
        rtol=2e-12,
    )
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(out, initial_sha256=digest(initial), **arrays)
    print(out, len(seen), len(arrays["ids"]) // 2, flush=True)


def response_loss(pred, obs):
    """Scale member and paired errors by their reference excursions."""
    actual = (obs[:, 1:] - obs[:, :1]) * 1e6
    predicted = (pred[:, 1:] - pred[:, :1]) * 1e6

    def normalized(a, b):
        scale = b.abs().amax(dim=-1, keepdim=True).clamp_min(0.05)
        return functional.smooth_l1_loss(
            (a - b) / scale, torch.zeros_like(b), beta=0.2
        )

    return normalized(predicted, actual) + normalized(
        predicted[1::2] - predicted[::2], actual[1::2] - actual[::2]
    )


def initialize(data, cp, origins):
    """Choose a training origin distribution and fit its calibration."""
    state = cp["model"]
    if origins == "initial":
        selected = data["leg"] == 0
        data = {k: v[selected] for k, v in data.items()}
    axes = state["axes"]
    mean, std = state["context_mean"], state["context_std"]
    if origins == "calibrated":
        residual = data["profile"].double() * cp["scale"] + cp["center"]
        center = residual.mean(0)
        scale = residual.std(0, correction=0).clamp_min(1e-7)
        normalized = (residual - center) / scale
        _, _, full_axes = torch.linalg.svd(normalized, full_matrices=False)
        axes = full_axes[: len(axes)]
        data["profile"] = normalized.float()
        mean = data["known"].mean(0)
        std = data["known"].std(0).clamp_min(1e-7)
        cp.update(center=center, scale=scale, step=0)
    net = ProfileCodec(axes, mean, std, cp["kind"], cp["hidden"])
    if origins != "calibrated":
        net.load_state_dict(state)
    return data, net


def fit(cache, parent, out, updates, seed, strength, substeps, batch, origins):
    """Fine-tune on wet and dry responses with a geometry-only control."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cp = torch.load(parent, map_location="cpu", weights_only=True)
    with np.load(cache) as d:
        if cp["training_sha256"] != d["initial_sha256"].item():
            raise ValueError(
                "Training split differs between geometry and response fits"
            )
        data = {
            k: torch.from_numpy(d[k]) for k in d.files if k != "initial_sha256"
        }
    data, net = initialize(data, cp, origins)
    if cp["kind"] == "sections":
        data["sections"] = torch.cat(
            [
                section_features(x, w)
                for x, w in zip(
                    data["x"].split(128),
                    data["weights"].split(128),
                    strict=True,
                )
            ]
        )
    opt = torch.optim.AdamW(net.parameters(), lr=0.00025)
    for step in range(1, updates + 1):
        idx = (
            2 * rng.integers(len(data["ids"]) // 2, size=batch)[:, None]
            + np.arange(2)
        ).ravel()
        d = {k: v[idx] for k, v in data.items()}
        z = net.encode(
            d["known"], d["x"], d["weights"], d["profile"], d.get("sections")
        )
        # Use the deployed fp16 latent in the forward pass.
        z = z + (z.half().float() - z).detach()
        y = net.decode(z, d["known"])
        target = d["profile"]
        shape_loss = functional.smooth_l1_loss(
            y, target
        ) + 0.5 * functional.smooth_l1_loss(
            y[1::2] - y[::2], target[1::2] - target[::2]
        )
        loss = shape_loss
        if strength:
            p = (y.double() * cp["scale"] + cp["center"]).reshape(
                -1, cp["degree"], 5
            ) + d["baseline"]
            mass, count = nodes_from_profile(p, d["fixed"], d["totals"])
            water, _ = rollout(
                mass,
                count / d["volume"][:, None],
                d["gas"],
                d["force"],
                substeps,
                True,
            )
            loss = loss + strength * response_loss(water, d["water"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), 5, error_if_nonfinite=True
        )
        opt.step()
        if step == updates // 2:
            opt.param_groups[0]["lr"] *= 0.35
        if step % 100 == 0 or step == updates:
            print(step, loss.item(), shape_loss.item(), flush=True)
    cp.update(
        model=net.state_dict(),
        step=cp["step"] + updates,
        parent_sha256=digest(parent),
        response_sha256=digest(cache),
        response_strength=strength,
        response_substeps=substeps,
        response_batch_pairs=batch,
        response_seed=seed,
        response_origins=origins,
    )
    out.parent.mkdir(exist_ok=True, parents=True)
    atomic_save(cp, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    for arg in ("root", "initial", "out"):
        a.add_argument(arg, type=Path)
    a.add_argument("--degree", type=int, default=5)
    a.add_argument("--length", type=int, default=4)
    b = sub.add_parser("fit")
    for arg in ("cache", "parent", "out"):
        b.add_argument(arg, type=Path)
    b.add_argument("--updates", type=int, default=1200)
    b.add_argument("--seed", type=int, default=2301)
    b.add_argument("--strength", type=float, default=1.0)
    b.add_argument("--substeps", type=int, default=2)
    b.add_argument(
        "--origins",
        choices=["pretrained", "initial", "calibrated"],
        default="pretrained",
    )
    b.add_argument("--batch", type=int, default=16)
    args = vars(p.parse_args())
    torch.set_num_threads(1)
    action = args.pop("action")
    if action == "prepare":
        prepare(**args)
    else:
        fit(**args)
