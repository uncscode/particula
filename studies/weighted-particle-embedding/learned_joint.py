"""Learn weighted particle features for co-located water and organic mass."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dual_ternary_record import DUAL, POLY, SUPPORT, TERMS, encode_joint
from dual_transfer import DualExchange
from torch import nn


class JointFeatures(nn.Module):
    """A shared map on two particle loadings and a linear moment readout."""

    def __init__(self, width, hidden=32):
        """Keep the encoded dimension independent of the nine load cells."""
        super().__init__()
        self.width = width
        self.map = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
        )
        self.readout = nn.Linear(width, 4, bias=False)
        mesh = np.stack(np.meshgrid(SUPPORT, SUPPORT, indexing="ij"), -1)
        self.register_buffer("points", torch.from_numpy(mesh.copy()))
        self.double()

    def features(self):
        """Remove additive marginal terms for these known uniform marginals."""
        x = self.map(self.points)
        return (
            x
            - x.mean(0, keepdim=True)
            - x.mean(1, keepdim=True)
            + x.mean((0, 1))
        )

    def forward(self, p):
        """Pool the nine joint cells and reconstruct four interaction terms."""
        z = torch.einsum("bij,ijk->bk", p, self.features())
        return self.readout(z)


def tables(rng, count):
    """Sample smooth positive tables with identical row and column totals."""
    delta = rng.normal(size=(count, 3, 3))
    delta = (
        delta
        - delta.mean(1, keepdims=True)
        - delta.mean(2, keepdims=True)
        + delta.mean((1, 2), keepdims=True)
    )
    delta = delta / np.maximum(-delta.min((1, 2), keepdims=True), 1e-10)
    return (1 + rng.uniform(0.05, 0.98, (count, 1, 1)) * delta) / 9


@torch.inference_mode()
def flux_library(seed, count):
    """Generate dry states and exact particle fluxes."""
    rng = np.random.default_rng(seed)

    def log_sample(lo, hi):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), (count, 1)))

    dry = log_sample(2e-19, 1e-14)
    f = rng.uniform(0.04, 0.96, (count, 1))
    means = np.column_stack((log_sample(0.04, 1.2), log_sample(0.01, 0.35)))
    span = means * rng.uniform(0.4, 0.98, (count, 2))
    signs = np.stack(np.meshgrid(SUPPORT, SUPPORT, indexing="ij"), -1).reshape(
        9, 2
    )
    q = means[:, None] + span[:, None] * signs
    fixed = np.column_stack((dry * (1 - f), dry * f))
    m = np.concatenate(
        (dry[..., None] * q, np.broadcast_to(fixed[:, None], q.shape)), -1
    )
    native = DualExchange()
    number = torch.ones((count, 9), dtype=torch.float64)
    ctx = native.static(torch.from_numpy(m.copy()), number)
    gas = torch.from_numpy(rng.uniform(0.05, 0.95, (count, 2))) * native.sat
    rate = (
        native.linearization(torch.from_numpy(q), gas, ctx)[0]
        .reshape(count, 3, 3, 2)
        .numpy()
    )
    # Only the double-centered part can distinguish equal marginal loads.
    diff = (
        rate
        - rate.mean(1, keepdims=True)
        - rate.mean(2, keepdims=True)
        + rate.mean((1, 2), keepdims=True)
    )
    return torch.from_numpy(
        diff
        / np.maximum(np.linalg.norm(diff, axis=(1, 2), keepdims=True), 1e-12)
    )


def train(out, seed, widths, mixes, updates, batch, library, training_tables):
    """Fit matched-width features for joint shape and optional flux response."""
    out.mkdir(parents=True, exist_ok=True)
    d = torch.from_numpy(DUAL)
    p = torch.from_numpy(tables(np.random.default_rng(51203), training_tables))
    basis = torch.from_numpy(POLY)
    target = basis.T @ p @ basis
    flux = flux_library(41953, library)
    for width in widths:
        for mix in mixes:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = JointFeatures(width)
            opt = torch.optim.AdamW(model.parameters(), lr=0.003)
            for _ in range(updates):
                i = rng.integers(len(p), size=batch)
                f = flux[rng.integers(len(flux), size=batch)]
                raw = model(p[i]).reshape(-1, 2, 2)
                residual = d @ (raw - target[i]) @ d.T
                shape = (residual.square().sum((1, 2))).mean()
                transfer = (residual[..., None] * f).sum((1, 2)).square().mean()
                loss = shape * (1 - mix) + transfer * mix
                opt.zero_grad()
                loss.backward()
                opt.step()
            dst = out / f"joint-w{width}-flux{mix:g}-s{seed}.pt"
            tmp = dst.with_suffix(".tmp")
            torch.save(
                dict(
                    model=model.state_dict(),
                    width=width,
                    seed=seed,
                    flux_fraction=mix,
                    updates=updates,
                    batch=batch,
                    training_tables=training_tables,
                    flux_library=library,
                ),
                tmp,
            )
            tmp.replace(dst)
            print(dst.name, float(loss.detach()), flush=True)


@torch.inference_mode()
def compress(m, n, k, conditioning, model):
    """Pool the learned map by dry size and composition."""
    v = encode_joint(m, n, k, "marginal", conditioning)
    dry = m[..., 2:].sum(-1)
    w = n / n.sum(1, keepdims=True)
    means = np.column_stack((v["totals"][:, 1], v["organic"])) / v["totals"][
        :, 2:
    ].sum(1, keepdims=True)
    z = (m[..., :2] / dry[..., None] - means[:, None]) / v["spread"][:, None]
    labels = np.rint(z).astype(int)
    np.testing.assert_allclose(z, labels, atol=3e-6)
    if abs(labels).max() > 1:
        raise ValueError("Particle outside ternary training support")
    table = model.features().numpy()
    phi = table[labels[..., 0] + 1, labels[..., 1] + 1]
    u = np.cbrt(dry / (w * dry).sum(1, keepdims=True)) - 1
    powers = u[..., None] ** np.arange(1 if conditioning == "global" else k)
    f = m[..., 3] / dry
    cells = (
        np.stack((np.ones_like(f), f), -1)
        if conditioning == "joint"
        else np.ones_like(f[..., None])
    )
    v["features"] = np.einsum(
        "bi,biv,bij,bic->bvjc", w, phi, powers, cells, optimize=True
    ).astype(np.float32)
    del v["mixed"]
    return v


@torch.inference_mode()
def project(v, model):
    """Turn only the stored features into conditional mixed moments."""
    result = dict(v)
    # The readout is row-major, while the explicit hierarchy adds (2,1)
    # before (1,2). Align by the declared moment index.
    matrix = model.readout.weight.numpy().reshape(2, 2, -1)
    mat = np.stack([matrix[a, b] for a, b in TERMS["fourth"]])
    result["mixed"] = np.einsum("mv,bvjc->bmjc", mat, v["features"])
    result["detail"] = "fourth"
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=3701)
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--batch", type=int, default=192)
    p.add_argument("--library", type=int, default=2048)
    p.add_argument("--training-tables", type=int, default=65536)
    p.add_argument("--widths", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--mixes", nargs="+", type=float, default=[0, 0.9, 1])
    torch.set_num_threads(1)
    train(**vars(p.parse_args()))
