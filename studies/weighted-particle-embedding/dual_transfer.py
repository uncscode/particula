"""A two-reservoir particle solver with shared instantaneous and recurrent
corrections.
"""

import numpy as np
import torch
from dual_materials import ACCOMMODATION, DENSITY, DIFFUSIVITY, MOLAR, SATURATED
from dual_reference import KELVIN, PATH
from torch import nn


def solve2(matrix, rhs):
    """Solve nonsingular two-by-two systems with multiple right-hand sides."""
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c, d = matrix[..., 1, 0], matrix[..., 1, 1]
    det = a * d - b * c
    if det.is_cuda:
        torch._assert_async((det.abs() >= 1e-12).all(), "Singular gas solve")
    elif torch.any(det.abs() < 1e-12):
        raise ArithmeticError("Singular two-vapor implicit system")
    x, y = rhs.unbind(-2)
    return torch.stack(
        (
            (d[..., None] * x - b[..., None] * y) / det[..., None],
            (-c[..., None] * x + a[..., None] * y) / det[..., None],
        ),
        -2,
    )


class ParticleContext(nn.Module):
    """Pool instantaneous state across the weighted, unordered population."""

    def __init__(self, width, attention):
        """Build value features and optional particle attention."""
        super().__init__()
        self.value = nn.Linear(13, width)
        self.query = nn.Linear(13, width, bias=False) if attention else None
        self.key = nn.Linear(13, width, bias=False) if attention else None
        self.response = nn.Sequential(
            nn.Linear(13 + width, width), nn.SiLU(), nn.Linear(width, 2)
        )
        nn.init.zeros_(self.response[-1].weight)
        nn.init.zeros_(self.response[-1].bias)

    def forward(self, features, weight):
        """Weight the represented population by dry mass."""
        mapped = self.value(features).tanh()
        if self.query is None:
            w = weight / weight.sum(1, keepdim=True).clamp_min(1e-30)
            context = (
                (mapped * w[..., None]).sum(1, keepdim=True).expand_as(mapped)
            )
        else:
            logits = self.query(features) @ self.key(features).transpose(1, 2)
            logits = logits / (mapped.shape[-1] ** 0.5)
            logits = torch.where(weight[:, None] > 0, logits, -1e30)
            scale = (logits - logits.amax(-1, keepdim=True)).exp() * weight[
                :, None
            ]
            context = (
                scale / scale.sum(-1, keepdim=True).clamp_min(1e-30)
            ) @ mapped
        return self.response(torch.cat((features, context), -1)).tanh()


class DualExchange(nn.Module):
    """Conserve both gas reservoirs and evolve one shared state per particle."""

    def __init__(self, kind="physical", width=12):
        """Construct the physical or learned transfer update."""
        super().__init__()
        if kind not in (
            "physical",
            "ff",
            "gru",
            "mixture",
            "set",
            "attention",
            "spatiotemporal",
        ):
            raise ValueError("Unknown two-vapor correction")
        self.kind = kind
        self.routing = "learned"
        self.spatial = (
            ParticleContext(width, kind != "set")
            if kind in ("set", "attention", "spatiotemporal")
            else None
        )
        self.width = width
        for name, v in (
            ("molar", MOLAR[:2]),
            ("density", DENSITY[:2]),
            ("sat", SATURATED),
            ("alpha", ACCOMMODATION),
            ("diffusion", DIFFUSIVITY),
            ("kelvin", KELVIN),
            ("path", PATH),
        ):
            self.register_buffer(name, torch.tensor(v, dtype=torch.float64))
        self.register_buffer("identity", torch.eye(2, dtype=torch.float64))
        self.cell = (
            nn.GRUCell(13, width)
            if kind in ("gru", "mixture", "spatiotemporal")
            else None
        )
        if kind in ("ff", "gru", "mixture", "spatiotemporal"):
            self.mapping = nn.Sequential(
                nn.Linear(13 + (width if self.cell else 0), width),
                nn.SiLU(),
                nn.Linear(width, 2),
            )
            nn.init.zeros_(self.mapping[-1].weight)
            nn.init.zeros_(self.mapping[-1].bias)
        if kind in ("mixture", "spatiotemporal"):
            self.instant = nn.Sequential(
                nn.Linear(13, width), nn.SiLU(), nn.Linear(width, 2)
            )
            self.gate = nn.Linear(13, 4 if kind == "spatiotemporal" else 3)
            nn.init.zeros_(self.instant[-1].weight)
            nn.init.zeros_(self.instant[-1].bias)
        self.double()

    def static(self, mass, number):
        """Compute the fixed dry volume, moles and represented dry mass."""
        fixed = torch.where((number > 0)[..., None], mass[..., 2:], 1e-18)
        dry = fixed.sum(-1)
        moles = (fixed / torch.as_tensor(MOLAR[2:], device=fixed.device)).sum(
            -1
        ) / dry
        volume = (
            fixed / torch.as_tensor(DENSITY[2:], device=fixed.device)
        ).sum(-1)
        return dry, moles, volume, number * dry, fixed[..., 1] / dry, number > 0

    def linearization(self, q, gas, state, rate_only=False):
        """Evaluate both fluxes and local and reservoir derivatives."""
        dry, base_moles, base_vol, weight, fraction, active = state
        volume = base_vol + dry * (q / self.density).sum(-1)
        radius = (3 * volume / (4 * torch.pi)).pow(1 / 3)
        kn = self.path / radius[..., None]
        denom = kn.square() + (1 + 0.283 * self.alpha) * kn + 0.75 * self.alpha
        gain = (
            4
            * torch.pi
            * radius[..., None]
            * self.diffusion
            * 0.75
            * self.alpha
            * (1 + kn)
            / (denom * dry[..., None])
        ) * active[..., None]
        mol = base_moles + (q / self.molar).sum(-1)
        x = q / (self.molar * mol[..., None])
        saturation = self.sat * torch.exp(self.kelvin / radius[..., None])
        drive = gas[:, None] - saturation * x
        rate = gain * drive
        if rate_only:
            return rate
        power = (
            1 - kn / (1 + kn) + (2 * kn + 1 + 0.283 * self.alpha) * kn / denom
        )
        dr = dry[..., None] / (
            4 * torch.pi * radius[..., None].square() * self.density
        )
        dx = (self.identity / self.molar - x[..., None] / self.molar) / mol[
            ..., None, None
        ]
        dx_sat = saturation[..., None] * (
            dx
            - (x * self.kelvin / radius[..., None].square())[..., None]
            * dr[..., None, :]
        )
        local = gain[..., None] * (
            (drive * power / radius[..., None])[..., None] * dr[..., None, :]
            - dx_sat
        )
        return rate, local, gain, drive, radius

    def increment(self, q, gas, state, dt):
        """Eliminate both shared vapor increments through a rank-two solve."""
        _, _, _, weight, _, _ = state
        rate, jac, gain, drive, radius = self.linearization(q, gas, state)
        h = dt[:, None, None]
        delta = self.solve_increment(weight, jac, gain, h, h * rate)
        stiffness = h * (
            jac.abs().sum(-1) + (weight[..., None] * gain).sum(1, keepdim=True)
        )
        return delta, stiffness, drive, radius

    def solve_increment(self, weight, jac, gain, h, rhs):
        """Apply the implicit two-vapor inverse to a supplied residual."""
        mat = self.identity - h[..., None] * jac
        sides = torch.cat((rhs[..., None], torch.diag_embed(h * gain)), -1)
        response = solve2(mat, sides)
        free, gas_response = response[..., 0], response[..., 1:]
        deposited = solve2(
            self.identity + (weight[..., None, None] * gas_response).sum(1),
            (weight[..., None] * free).sum(1)[..., None],
        )
        return free - (gas_response @ deposited[:, None]).squeeze(-1)

    def start(self, number):
        """Initialize particle memory for a new trajectory."""
        return number.new_zeros((*number.shape, self.width if self.cell else 0))

    def correction(self, features, weight, active, dt, memory):
        """Combine instantaneous, population and recurrent estimates."""
        if self.kind in ("set", "attention"):
            return self.spatial(features, weight), memory
        inputs = features
        if self.cell:
            cand = self.cell(
                features.flatten(0, 1), memory.flatten(0, 1)
            ).reshape_as(memory)
            memory = torch.where(
                ((dt[:, None] > 0) & active)[..., None], cand, memory
            )
            inputs = torch.cat((features, memory), -1)
        adjust = self.mapping(inputs).tanh()
        if self.kind in ("mixture", "spatiotemporal"):
            experts = [
                torch.zeros_like(adjust),
                self.instant(features).tanh(),
                adjust,
            ]
            if self.spatial is not None:
                experts.append(self.spatial(features, weight))
            if self.routing == "learned":
                gates = self.gate(features).softmax(-1)
            elif self.routing == "uniform":
                gates = torch.full_like(self.gate(features), 1 / len(experts))
            else:
                index = ("physical", "ff", "gru", "attention").index(
                    self.routing
                )
                return experts[index], memory
            adjust = (torch.stack(experts, -1) * gates[..., None, :]).sum(-1)
        return adjust, memory

    def step(self, q, gas, state, dt, memory):
        """Advance the node masses and both finite reservoirs."""
        delta, stiff, drive, radius = self.increment(q, gas, state, dt)
        delta, memory = self.adjustment(
            q, gas, state, dt, memory, delta, stiff, drive, radius
        )
        q, gas = self.balance(q, gas, state[3], delta)
        return q, gas, memory

    def adjustment(
        self, q, gas, state, dt, memory, delta, stiff, drive, radius
    ):
        """Apply the optional learned correction to a physical increment."""
        _, _, _, weight, fraction, active = state
        if self.kind != "physical":
            concentration = torch.log1p(
                weight.sum(1, keepdim=True) * 1e4
            ).expand_as(radius)
            features = torch.cat(
                (
                    torch.stack(
                        ((radius * 1e6).log(), fraction, concentration), -1
                    ),
                    torch.log1p(q),
                    (gas / self.sat)[:, None].expand_as(q),
                    drive / self.sat,
                    torch.log1p(stiff),
                    torch.asinh(delta),
                ),
                -1,
            )
            adjust, memory = self.correction(
                features, weight, active, dt, memory
            )
            delta = delta * (1 + 0.75 * adjust * stiff / (1 + stiff))
        return delta, memory

    @staticmethod
    def balance(q, gas, weight, delta):
        """Conserve both reservoirs with positive particle and gas masses."""
        loss = torch.minimum(q, (-delta).clamp_min(0))
        available = gas + (weight[..., None] * loss).sum(1)
        gain = delta.clamp_min(0)
        demand = (weight[..., None] * gain).sum(1)
        scale = (available / demand.clamp_min(1e-30)).clamp(0, 1)
        return (
            q - loss + gain * scale[:, None],
            (available - demand * scale).clamp_min(0),
        )

    def rollout(self, mass, number, levels, times, substeps=1, reset_mask=None):
        """Forecast from one initial state and prescribed gas resets."""
        if substeps < 1:
            raise ValueError("Need at least one advancing substep")
        state = self.static(mass, number)
        dry = state[0]
        q = mass[..., : len(self.molar)] / dry[..., None]
        reset_mask = (
            torch.ones_like(levels, dtype=torch.bool)
            if reset_mask is None
            else reset_mask
        )
        gas = levels[:, 0]
        memory = self.start(number)
        outputs = [q * dry[..., None]]
        gases = [gas]
        interval = 0
        for dt in np.diff(times):
            if dt == 0:
                interval += 1
                gas = torch.where(
                    reset_mask[:, interval], levels[:, interval], gas
                )
            else:
                step = gas.new_full((len(gas),), float(dt / substeps))
                for _ in range(substeps):
                    q, gas, memory = self.step(q, gas, state, step, memory)
            outputs.append(q * dry[..., None])
            gases.append(gas)
        return torch.stack(outputs, 1), torch.stack(gases, 1)
