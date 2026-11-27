"""Recursive nonlinear stages for a finite vapor reservoir and wet particles."""

import math

import torch
from reduced_exchange import balance
from rosenbrock_exchange import factors, solve, transport
from torch import nn
from water_constants import SATURATION


def tableau(order):
    """Return stiffly accurate, singly diagonally implicit RK coefficients."""
    if order == 2:
        g = 1 - 1 / math.sqrt(2)
        return g, ((), (1 - g,))
    if order == 3:
        g = 0.435866521508459
        first = (-6 * g * g + 16 * g - 1) / 4
        second = (6 * g * g - 20 * g + 5) / 4
        return g, ((), ((1 - g) / 2,), (first, second))
    raise ValueError("Expected second or third order")


class StageSolver(nn.Module):
    """Share one nonlinear iteration across stages, particles and time."""

    def __init__(self, order=2, iterations=2, kind="physical", width=12):
        """Build the stage tableau and optional learned Newton updates."""
        super().__init__()
        if iterations < 1 or kind not in ("physical", "ff", "gru", "mixture"):
            raise ValueError("Invalid nonlinear iteration configuration")
        self.gamma, self.rows = tableau(order)
        self.iterations = iterations
        self.kind = kind
        self.width = width
        self.cell = nn.GRUCell(9, width) if kind in ("gru", "mixture") else None
        self.mapping = None
        if kind != "physical":
            dim = 9 + (width if self.cell else 0)
            self.mapping = nn.Sequential(
                nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, 1)
            )
            nn.init.zeros_(self.mapping[-1].weight)
            nn.init.zeros_(self.mapping[-1].bias)
        if kind == "mixture":
            self.instant = nn.Sequential(
                nn.Linear(9, width), nn.SiLU(), nn.Linear(width, 1)
            )
            self.gate = nn.Linear(9, 3)
            nn.init.zeros_(self.instant[-1].weight)
            nn.init.zeros_(self.instant[-1].bias)
        self.double()

    def static(self, mass, number):
        """Dry masses in kg per represented particle, number in m⁻³."""
        dry = mass[..., 1:].sum(-1)
        volume = dry / 1400
        capacity = (1000 / 1400) * (0.03 * mass[..., 1] + 0.75 * mass[..., 2])
        radius = (volume * (3 / (4 * torch.pi))).clamp_min(1e-36).pow(1 / 3)
        return (
            dry,
            volume,
            capacity,
            torch.log(radius * 1e6),
            mass[..., 2] / dry.clamp_min(1e-36),
        )

    def start(self, number):
        """Create zero recurrent state for each particle."""
        return torch.zeros(
            (*number.shape, self.width if self.cell else 0),
            dtype=number.dtype,
            device=number.device,
        )

    def adjust(self, features, memory, active):
        """Mix instantaneous, recurrent and physical step corrections."""
        if self.cell:
            updated = self.cell(
                features.flatten(0, 1), memory.flatten(0, 1)
            ).reshape_as(memory)
            memory = torch.where(active[..., None], updated, memory)
            inputs = torch.cat((features, memory), -1)
        else:
            inputs = features
        correction = self.mapping(inputs).squeeze(-1).tanh()
        if self.kind == "mixture":
            weights = self.gate(features).softmax(-1)
            correction = (
                weights[..., 1] * self.instant(features).squeeze(-1).tanh()
                + weights[..., 2] * correction
            )
        return correction.double(), memory

    def step(self, water, gas, number, state, h, memory, *, diagnostics=False):
        """Solve implicit stages with Newton or learned relaxation."""
        dry, volume, capacity, size, fraction = state
        active = (h[:, None] > 0) & (number > 0)
        gd = self.gamma * h[:, None]
        rates = []
        diagnostics_sum = torch.zeros_like(gas)
        for stage, row in enumerate(self.rows):
            anchor = water
            for coeff, previous in zip(row, rates, strict=True):
                anchor = anchor + h[:, None] * coeff * previous
            # The previous stage provides a feasible nonlinear starting value.
            if stage == 0:
                wet, vapor = water, gas
            for j in range(self.iterations):
                rate, diagonal, conduct = transport(
                    wet, vapor, number, volume, capacity, True
                )
                mismatch = anchor - wet + gd * rate
                d, response, denominator, _ = factors(
                    diagonal, conduct, number, gd
                )
                dx = solve(mismatch, number, d, response, denominator)
                if self.mapping is not None:
                    stiffness = gd * (
                        diagonal.abs() + (conduct * number).sum(1, keepdim=True)
                    )
                    scale = dry.clamp_min(1e-36)
                    features = torch.stack(
                        (
                            size,
                            fraction,
                            torch.log1p(wet / scale),
                            (vapor / SATURATION)[:, None].expand_as(wet),
                            stiffness.log1p(),
                            torch.asinh(mismatch / scale),
                            torch.asinh(dx / scale),
                            torch.full_like(wet, stage / (len(self.rows) - 1)),
                            torch.full_like(
                                wet, j / max(1, self.iterations - 1)
                            ),
                        ),
                        -1,
                    )
                    adjustment, memory = self.adjust(features, memory, active)
                    dx = dx * (
                        1 + 0.75 * adjustment * stiffness / (1 + stiffness)
                    )
                wet, vapor = balance(wet, vapor, number, dx)
            if stage + 1 < len(self.rows) or diagnostics:
                final_rate = transport(
                    wet, vapor, number, volume, capacity, False
                )
                rates.append(final_rate)
                if diagnostics:
                    error = (wet - anchor - gd * final_rate).abs() * number
                    diagnostics_sum += error.sum(1)
        return (
            (wet, vapor, memory, diagnostics_sum)
            if diagnostics
            else (wet, vapor, memory)
        )

    def rollout(
        self,
        mass,
        number,
        gas,
        force,
        substeps=1,
        return_nodes=False,
        diagnostics=False,
    ):
        """Advance the initial particles and record the closed-leg histories."""
        if substeps < 1:
            raise ValueError("Need at least one integration step")
        water = mass[..., 0]
        memory = self.start(number)
        state = self.static(mass, number)
        output = [water if return_nodes else (water * number).sum(1)]
        gases, residual = [gas], torch.zeros_like(gas)
        for f in force.unbind(1):
            for _ in range(substeps):
                result = self.step(
                    water,
                    gas,
                    number,
                    state,
                    f[:, 0] / substeps,
                    memory,
                    diagnostics=diagnostics,
                )
                water, gas, memory = result[:3]
                if diagnostics:
                    residual += result[3]
            gas = torch.where(f[:, 3] > 0, f[:, 4], gas)
            output.append(water if return_nodes else (water * number).sum(1))
            gases.append(gas)
        series = torch.stack(output, 1), torch.stack(gases, 1)
        return (*series, residual) if diagnostics else series
