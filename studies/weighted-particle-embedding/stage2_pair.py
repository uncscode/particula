"""Predict symmetric, positive collision rates with optional charge."""

import numpy as np
import torch
from coupled_events import KT, MFP_AIR, MU_AIR
from torch import nn

# Electrostatic energy length e²/(4 pi epsilon_0 kT), in m at 293 K.
COULOMB_LENGTH = 1.602176634e-19**2 / (4 * np.pi * 8.8541878128e-12 * KT)


def pair_inputs(r, mass, charge, i, j):
    """Build permutation-equivariant particle and pair features."""
    lr = (r / 80e-9).log()
    rho = mass / (4 * torch.pi / 3 * r.pow(3))
    ld = (rho / 1200.0).log()
    kn = MFP_AIR / r
    slip = 1 + kn * (1.257 + 0.4 * torch.exp(-1.1 / kn))
    d = KT * slip / (6 * torch.pi * MU_AIR * r)
    speed = torch.sqrt(8 * KT / (torch.pi * mass))
    radius = r[:, i] + r[:, j]
    diff = d[:, i] + d[:, j]
    v = torch.hypot(speed[:, i], speed[:, j])
    kinetic = 4 * diff / (radius * v)
    base = 4 * torch.pi * diff * radius / (1 + kinetic)
    phi = -COULOMB_LENGTH * charge[:, i] * charge[:, j] / radius
    u = lr[:, i] - lr[:, j]
    w = ld[:, i] - ld[:, j]
    core = torch.stack(
        (
            (lr[:, i] + lr[:, j]) * 0.5,
            u.abs(),
            (ld[:, i] + ld[:, j]) * 0.5,
            w.abs(),
            u * w,
            kinetic.log(),
            torch.log1p(u.square()),
            torch.log1p(w.square()),
        ),
        -1,
    )
    node = torch.stack((lr, ld, torch.log1p(kn), torch.log1p(charge.abs())), -1)
    return base, phi, core, node


def physics_kernel(r, mass, i, j):
    """Exact Fuchs transition expression for the native Brownian control."""
    kn = MFP_AIR / r
    d = (
        KT
        * (1 + kn * (1.257 + 0.4 * torch.exp(-1.1 / kn)))
        / (6 * torch.pi * MU_AIR * r)
    )
    speed = (8 * KT / (torch.pi * mass)).sqrt()
    length = 8 * d / (torch.pi * speed)
    g = (
        (2 * r + length).pow(3) - (4 * r.square() + length.square()).pow(1.5)
    ) / (6 * r * length) - 2 * r
    rad = r[:, i] + r[:, j]
    diff = d[:, i] + d[:, j]
    return (
        4
        * torch.pi
        * diff
        * rad
        / (
            rad / (rad + torch.hypot(g[:, i], g[:, j]))
            + 4 * diff / (rad * torch.hypot(speed[:, i], speed[:, j]))
        )
    )


def charged_kernel(r, mass, z, i, j, hard=False):
    """Evaluate the native 2012 dimensional kernel on selected pairs."""
    from particula.util.constants import (
        ELECTRIC_PERMITTIVITY,
        ELEMENTARY_CHARGE_VALUE,
    )

    length = ELEMENTARY_CHARGE_VALUE**2 / (
        4 * np.pi * ELECTRIC_PERMITTIVITY * KT
    )
    kn = MFP_AIR / r
    friction = (
        6
        * torch.pi
        * MU_AIR
        * r
        / (1 + kn * (1.257 + 0.4 * torch.exp(-1.1 / kn)))
    )
    rm = mass[:, i] * mass[:, j] / (mass[:, i] + mass[:, j])
    rf = friction[:, i] * friction[:, j] / (friction[:, i] + friction[:, j])
    radii = r[:, i] + r[:, j]
    charges = z.to(r.dtype)
    phi = (-length * charges[:, i] * charges[:, j] / radii).clamp_min(-200.0)
    kinetic = torch.where(phi >= 0, 1 + phi, phi.exp())
    denom = -torch.expm1(-phi)
    continuum = torch.where(
        phi == 0, 1.0, phi / torch.where(phi == 0, 1.0, denom)
    )
    kd = torch.where(
        kinetic >= 1e-80,
        (KT * rm).sqrt() / rf * continuum / (radii * kinetic),
        0.0,
    )
    k2 = kd.square()
    k3 = kd * k2
    h = (
        4 * torch.pi * k2
        + 25.836 * k3
        + np.sqrt(8 * np.pi) * 11.211 * k2.square()
    ) / (1 + 3.502 * kd + 7.211 * k2 + 11.211 * k3)
    if not hard:
        small = torch.minimum(
            kd, 3 * kd / (2 * phi.clamp_min(1e-12))
        ).clamp_min(1e-16)
        h = torch.where(
            (phi > 0.5) & (small < 2.5),
            4 * torch.pi * k2 / (1 + 1.598 * small.pow(1.1709)),
            h,
        )
    return h * rf * radii.pow(3) * kinetic.square() / (rm * continuum)


class PairNet(nn.Module):
    """Predict nonnegative pair kernels with exchange symmetry."""

    def __init__(self, kind, width, charged=False):
        """Allocate the symmetric rate and particle context networks."""
        super().__init__()
        self.kind, self.width, self.charged = kind, width, charged
        if kind == "set-attention":
            self.q = nn.Linear(4, width, bias=False)
            self.k = nn.Linear(4, width, bias=False)
            self.v = nn.Linear(4, width)
        elif kind == "gru":
            self.gru = nn.GRUCell(4, width)
        elif kind == "causal":
            self.token = nn.Linear(4, width)
            self.q = nn.Linear(width, width, bias=False)
            self.k = nn.Linear(width, width, bias=False)
            self.v = nn.Linear(width, width)
            self.age = nn.Parameter(torch.zeros(4))
        elif kind != "ff":
            raise ValueError(f"Unknown collision model {kind}")
        features = 8 + (0 if kind == "ff" else 2 * width)
        self.correction = nn.Sequential(
            nn.Linear(features, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        if charged:
            self.charge_head = nn.Sequential(
                nn.Linear(features + 2, width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, 1),
            )
            nn.init.zeros_(self.charge_head[-1].weight)
            nn.init.zeros_(self.charge_head[-1].bias)
        self.double()

    def start(self, r):
        """Initialize causal state for the current particle slots."""
        w = (
            self.width * (3 if self.kind == "causal" else 1)
            if self.kind in ("gru", "causal")
            else 0
        )
        return r.new_zeros((*r.shape, w))

    def context(self, node, active, h):
        """Compute permutation-equivariant or causal particle context."""
        if self.kind == "set-attention":
            scores = (self.q(node) @ self.k(node).transpose(-1, -2)) / np.sqrt(
                self.width
            )
            scores = scores.masked_fill(~active[:, None], -1e4)
            return scores.softmax(-1) @ self.v(node), h
        if self.kind == "gru":
            h = self.gru(node.flatten(0, 1), h.flatten(0, 1)).reshape_as(h)
            return h, h
        token = self.token(node)
        seq = torch.cat(
            (h.reshape(*h.shape[:-1], 3, self.width), token.unsqueeze(-2)), -2
        )
        scores = (self.q(token).unsqueeze(-2) * self.k(seq)).sum(-1) / np.sqrt(
            self.width
        ) + self.age
        return token + (scores.softmax(-1).unsqueeze(-1) * self.v(seq)).sum(
            -2
        ), seq[..., 1:, :].flatten(-2)

    def forward(self, r, mass, z, active, i, j, h):
        # Absence has zero hazard. Give the padded node a finite temporary
        # state.
        """Evaluate valid pair rates with optional signed-charge correction."""
        r = torch.where(active, r, 80e-9)
        mass = torch.where(active, mass, 4 * np.pi / 3 * (80e-9) ** 3 * 1200.0)
        z = z if self.charged else torch.zeros_like(z)
        base, phi, x, node = pair_inputs(r, mass, z, i, j)
        dtype = self.correction[0].weight.dtype
        x = x.to(dtype)
        if self.kind != "ff":
            # The neutral backbone must be independent of charge. This also
            # preserves the exact neutral limit with an independently fitted
            # head.
            node = torch.cat(
                (node[..., :3], torch.zeros_like(node[..., 3:])), dim=-1
            )
            ctx, h = self.context(node.to(dtype), active, h.to(dtype))
            x = torch.cat(
                (
                    x,
                    (ctx[:, i] + ctx[:, j]) * 0.5,
                    (ctx[:, i] - ctx[:, j]).abs(),
                ),
                -1,
            )
        log_rate = self.correction(x).squeeze(-1).to(base.dtype)
        if self.charged:
            p = torch.asinh(phi)
            zfeatures = torch.cat(
                (x, p.unsqueeze(-1), p.square().unsqueeze(-1)), -1
            )
            # Repulsion/attraction and zero charge are anchored by the continuum
            # enhancement. Fit the remaining transition-regime correction.
            safe = torch.where(phi.abs() < 1e-6, torch.ones_like(phi), phi)
            enhance = torch.where(
                phi.abs() < 1e-6,
                1 + phi * 0.5,
                safe / (-torch.expm1(-safe.clamp(-80, 80))),
            )
            log_rate = (
                log_rate
                + enhance.log()
                + torch.tanh(p)
                * self.charge_head(zfeatures.to(dtype))
                .squeeze(-1)
                .to(base.dtype)
            )
        k = base * log_rate.clamp(-80, 50).exp()
        return k * (active[:, i] & active[:, j]), h.to(r.dtype)


def load_pair(file, device="cpu"):
    """Restore the fitted collision model on the requested device."""
    cp = torch.load(file, map_location="cpu", weights_only=False)
    if cp["kind"] == "jump-distance":
        from stage2_jump_distance import JumpDistance

        model = JumpDistance(cp["width"])
    else:
        model = PairNet(cp["kind"], cp["width"], cp["charged"])
    if cp.get("neural_dtype"):
        model.to(dtype=getattr(torch, cp["neural_dtype"]))
    model.load_state_dict(cp["state"])
    return model.to(device)
