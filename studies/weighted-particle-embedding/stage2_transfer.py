"""Define conservative vapor operators with water and temporal attention."""

from pathlib import Path

import numpy as np
import torch
from multivapor_transfer import (
    Rosenbrock4Exchange,
    SpeciesHead,
    from_checkpoint,
)
from torch import nn

HERE = Path(__file__).resolve().parent
PARENT = HERE / "examples/routing-fits/head-equal-s4901-u1200.pt"


class CausalHead(nn.Module):
    """Attend over three previous accepted tokens and the current token."""

    def __init__(self, width):
        """Allocate four-token attention and a scalar correction readout."""
        super().__init__()
        self.project = nn.Linear(14, width)
        self.key = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.age = nn.Parameter(torch.zeros(4))
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, 1),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)
        self.double()

    def forward(self, x, h, update):
        """Update the token history and predict a correction."""
        token = self.project(x)
        past = h.reshape(*token.shape[:-1], 3, token.shape[-1])
        seq = torch.cat((past, token.unsqueeze(-2)), -2)
        query = self.query(token).unsqueeze(-2)
        score = (query * self.key(seq)).sum(-1) / np.sqrt(
            token.shape[-1]
        ) + self.age
        value = token + (score.softmax(-1).unsqueeze(-1) * self.value(seq)).sum(
            -2
        )
        new = seq[..., 1:, :].flatten(-2)
        return self.ff(value).squeeze(-1).tanh(), torch.where(
            update[..., None, None], new, h
        )


class WaterHead(nn.Module):
    """Apply a water-specific residual to a common vapor network."""

    def __init__(self, width, water):
        """Allocate a shared vapor mixture and one water-specific readout."""
        super().__init__()
        self.shared = SpeciesHead("mixture", width)
        self.special = nn.Sequential(
            nn.Linear(14, width), nn.SiLU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.special[-1].weight)
        nn.init.zeros_(self.special[-1].bias)
        self.water = water
        self.double()

    def forward(self, x, h, update):
        """Combine the water correction with the shared organic prediction."""
        shared, h = self.shared(x, h, update)
        if self.water is None:
            return shared, h
        # Learn only the designated water lane. Every species has one flux.
        j = self.water
        water = (shared[..., j] + self.special(x[..., j, :]).squeeze(-1)).tanh()
        return torch.cat(
            (shared[..., :j], water[..., None], shared[..., j + 1 :]), -1
        ), h


class Transfer(Rosenbrock4Exchange):
    """Use a single finite-reservoir balance across the selected gas lanes."""

    def __init__(self, spec, kind="physical", width=9):
        """Select the transfer architecture and its history dimensions."""
        original = (
            kind
            if kind in ("physical", "ff", "gru", "attention", "mixture")
            else "physical"
        )
        super().__init__(spec, original, width)
        self.kind = kind
        if kind == "causal":
            self.head = CausalHead(width)
        if kind == "water":
            self.head = WaterHead(
                width,
                spec.names.index("water") if "water" in spec.names else None,
            )
        self.memory_width = (
            3 * width
            if kind == "causal"
            else width
            if kind in ("water", "gru", "mixture")
            else 0
        )
        self.active_lanes = None

    def start(self, number):
        """Initialize process memory for every particle and volatile lane."""
        return number.new_zeros(
            (*number.shape, len(self.molar), self.memory_width)
        )

    def linearization(self, q, gas, state, rate_only=False):
        """Restrict transfer to the active species."""
        values = super().linearization(q, gas, state, rate_only)
        if self.active_lanes is None:
            return values
        active = self.active_lanes
        if rate_only:
            return values * active
        rate, jac, gain, drive, radius = values
        return (
            rate * active,
            jac * active[:, None],
            gain * active,
            drive * active,
            radius,
        )


def restore(spec, model, device="cpu"):
    """Restore a physical solver, fitted head or two-specialist composition."""
    if str(model).startswith("nonlinear-") and not Path(model).is_file():
        from stage2_nonlinear import NonlinearExchange

        _, outer, reservoir, order = str(model).split("-")
        return NonlinearExchange(
            spec, int(outer), int(reservoir), int(order)
        ).to(device)
    if str(model).startswith("simplified-radau-") and not Path(model).is_file():
        from stage2_simplified_radau import SimplifiedRadau

        stages, iterations = map(int, str(model).split("-")[-2:])
        return SimplifiedRadau(spec, stages, iterations).to(device)
    if model in ("rank-physical", "rank-parent"):
        from stage2_lowrank import LowRankROS4

        if model == "rank-physical":
            return LowRankROS4(spec).to(device)
        cp = torch.load(PARENT, map_location="cpu", weights_only=False)
        net = LowRankROS4(spec, **cp["config"])
        net.head.load_state_dict(cp["head"])
        return net.to(device)
    if str(model).startswith("radau-") and not Path(model).is_file():
        from stage2_radau import Radau

        _, stages, iterations = str(model).split("-")
        return Radau(spec, int(stages), int(iterations)).to(device)
    if model == "parent":
        return from_checkpoint(
            spec, torch.load(PARENT, map_location="cpu", weights_only=False)
        ).to(device)
    if model == "physical":
        return Transfer(spec).to(device)
    cp = torch.load(model, map_location="cpu", weights_only=False)
    return restore_head(spec, cp, device)


def restore_head(spec, cp, device):
    """Recreate an archived neural head from its training configuration."""
    if cp.get("conductance"):
        from stage2_conductance import KineticRadau, KineticROS4

        cls = KineticROS4 if cp.get("integrator") == "ros4" else KineticRadau
        return cls(spec, cp).to(device)
    if cp.get("defect"):
        from stage2_defect import Defect

        net = Defect(spec, cp["base"], cp["kind"], cp["width"])
        if net.readout is not None:
            net.readout.load_state_dict(cp["readout"])
        if net.cell is not None:
            net.cell.load_state_dict(cp["cell"])
        return net.to(device)
    if cp.get("specialists"):
        from stage2_specialists import Specialists

        return Specialists(spec, cp["water"], cp["organic"], cp["order"]).to(
            device
        )
    if cp.get("radau"):
        from stage2_radau import Radau

        net = Radau(spec, *cp["radau"], cp["kind"], cp["width"])
    else:
        net = Transfer(spec, cp["kind"], cp["width"])
    net.head.load_state_dict(cp["head"])
    if cp.get("active", "all") != "all":
        net.active_lanes = torch.tensor(
            [(x == "water") == (cp["active"] == "water") for x in spec.names],
            device=device,
        )
    return net.to(device)


def substep(net, q, gas, static, dt, h, order="joint"):
    """Apply joint or sequential water and organic material substeps."""
    if order == "joint" or q.shape[-1] == 1:
        return net.step(q, gas, static, dt, h)
    if not isinstance(net, Transfer):
        raise ValueError("Masked species steps require the Stage 2 operator")
    water = torch.arange(q.shape[-1], device=q.device) == 0
    schedules = {
        "water-first": [(water, 1.0), (~water, 1.0)],
        "organic-first": [(~water, 1.0), (water, 1.0)],
        "symmetric": [(water, 0.5), (~water, 1.0), (water, 0.5)],
    }
    try:
        for active, fraction in schedules[order]:
            net.active_lanes = active
            q, gas, hnext = net.step(q, gas, static, dt * fraction, h)
            h = torch.where(active[None, None, :, None], hnext, h)
    finally:
        net.active_lanes = None
    return q, gas, h


def step_box(box, net, dt, substeps=1, memory=None, order="joint"):
    """Accepted mass and gas update on the current physical population."""
    m = torch.from_numpy(box.mass)[None]
    n = m.new_full(m.shape[:2], 1 / box.volume)
    g = torch.from_numpy(box.gas)[None]
    a = net.static(m, n)
    q = m[..., : len(box.gas)] / a[0][..., None]
    h = net.start(n) if memory is None else memory
    hdt = g.new_tensor([dt / substeps])
    for _ in range(substeps):
        q, g, h = substep(net, q, g, a, hdt, h, order)
    box.mass[:, : len(box.gas)] = (q * a[0][..., None])[0].detach().numpy()
    box.gas = g[0].detach().numpy().copy()
    return h
