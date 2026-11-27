"""Matched current inventories with different observable uptake histories."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from dual_materials import ACCOMMODATION, DENSITY, DIFFUSIVITY, MOLAR, SATURATED
from dual_reference import KELVIN, PATH

START = dict(
    train=30000000, validation=31000000, test=32000000, challenge=33000000
)
DT = 5e-6
HISTORY = 8


def design(ids, challenge=False):
    """Draw four dry classes and two within-box volatile loadings."""
    u = np.stack([np.random.default_rng([int(i), 573]).random(24) for i in ids])
    radius = np.array([0.6, 0.85, 1.15, 1.55]) * (0.90 + 0.2 * u[:, :4]) * 1e-6
    if challenge:
        radius *= 0.84
    dry = 1400 * 4 * np.pi / 3 * radius**3
    fraction = 0.05 + 0.90 * u[:, 4:8]
    fixed = dry[..., None] * np.stack([1 - fraction, fraction], -1)
    w = (0.4 + u[:, 8:12]) * (1e-4 + 7e-4 * u[:, 12:13])
    n = w / dry
    basis = np.log(radius / 1e-6)
    basis -= (basis * w).sum(1, keepdims=True) / w.sum(1, keepdims=True)
    basis /= abs(basis).max(1, keepdims=True)
    means = np.c_[0.26 + 0.38 * u[:, 13], 0.10 + 0.24 * u[:, 14]]
    magnitude = 0.15 + 0.65 * u[:, 15:17]
    magnitude[:, 1] *= np.where(u[:, 17] > 0.5, 1, -1)
    if challenge:
        magnitude *= 1.12
    coef = np.stack((-magnitude, magnitude), 1).reshape(-1, 2)
    previous = (0.15 + 0.40 * u[:, 18:20]) * SATURATED
    supply = (
        np.stack((0.48 + 0.43 * u[:, 20:22], 0.02 + 0.12 * u[:, 22:24]), 1)
        * SATURATED
    )
    return dict(
        ids=ids,
        fixed=fixed.repeat(2, 0),
        number=n.repeat(2, 0),
        basis=basis.repeat(2, 0),
        mean=means.repeat(2, 0),
        coef=coef,
        previous=previous.repeat(2, 0),
        levels=supply,
    )


def reconstruct(d, coef):
    """Recover positive node masses at fixed bulk inventories."""
    ratio = d["mean"][:, None] * (1 + d["basis"][..., None] * coef[:, None])
    return np.concatenate(
        (ratio * d["fixed"].sum(-1)[..., None], d["fixed"]), -1
    )


def prehistory(d, subdivisions=2):
    """Integrate a short reversible interval and verify it in forward time."""
    dry = d["fixed"].sum(-1)
    mass = reconstruct(d, d["coef"])
    w = d["number"] * dry
    q0 = mass[..., :2] / dry[..., None]
    bulk = d["mean"] * w.sum(1, keepdims=True)
    total = bulk + d["previous"]
    vol = (d["fixed"] / DENSITY[2:]).sum(-1)
    mol = (d["fixed"] / MOLAR[2:]).sum(-1) / dry

    def rate(q):
        gas = total - (w[..., None] * q).sum(1)
        r = np.cbrt(3 / (4 * np.pi) * (vol + dry * (q / DENSITY[:2]).sum(-1)))[
            ..., None
        ]
        kn = PATH / r
        gain = (
            4
            * np.pi
            * r
            * DIFFUSIVITY
            * 0.75
            * ACCOMMODATION
            * (1 + kn)
            / (kn**2 + (1 + 0.283 * ACCOMMODATION) * kn + 0.75 * ACCOMMODATION)
        )
        x = q / (
            MOLAR[:2]
            * (mol[..., None] + (q / MOLAR[:2]).sum(-1, keepdims=True))
        )
        return (
            gain
            / dry[..., None]
            * (gas[:, None] - SATURATED * np.exp(KELVIN / r) * x)
        )

    def advance(q, dt):
        for _ in range(subdivisions):
            h = dt / subdivisions
            a = rate(q)
            b = rate(q + h * a / 2)
            c = rate(q + h * b / 2)
            e = rate(q + h * c)
            q = q + h * (a + 2 * b + 2 * c + e) / 6
        if (
            not np.isfinite(q).all()
            or q.min() < 0
            or (total - (w[..., None] * q).sum(1)).min() < 0
        ):
            raise ArithmeticError(
                "History crossed a physical inventory boundary"
            )
        return q

    q = q0
    observations = [bulk]
    for _ in range(HISTORY - 1):
        q = advance(q, -DT)
        observations.append((w[..., None] * q).sum(1))
    past = q.copy()
    for _ in range(HISTORY - 1):
        q = advance(q, DT)
    np.testing.assert_allclose(q, q0, rtol=2e-11, atol=1e-13)
    return np.stack(observations[::-1], 1), past


def build(root, split, families):
    """Persist the cold-state record and observable past as separate arrays."""
    root.mkdir(parents=True, exist_ok=True)
    ids = START[split] + np.arange(families)
    data = design(ids, split == "challenge")
    paths = []
    for i in range(0, families, 256):
        local = design(ids[i : i + 256], split == "challenge")
        history, start = prehistory(local)
        paths.append((history, start))
    data.update(
        history=np.concatenate([a for a, _ in paths]),
        prehistory_ratio=np.concatenate([b for _, b in paths]),
        past_time_s=np.arange(1 - HISTORY, 1) * DT,
    )
    f = root / f"{split}-{families}.npz"
    tmp = f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(f)
    print(f, hashlib.sha256(f.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("split", choices=START)
    p.add_argument("families", type=int)
    build(**vars(p.parse_args()))
