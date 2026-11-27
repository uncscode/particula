"""Learn bounded integration corrections on a compressed dry population."""

import torch
from reduced_exchange import balance, implicit_increment, implicit_step
from torch import nn
from water_constants import SATURATION


class NodeCorrection(nn.Module):
    """Share kinetic corrections across an unordered population of nodes."""

    def __init__(
        self,
        hidden=24,
        memory=0,
        scheme="implicit",
        mixture=False,
        physical_clock=False,
    ):
        """Initialize exactly at the untrained physical integration scheme."""
        super().__init__()
        if scheme not in ("implicit", "extrapolated"):
            raise ValueError("Unknown node integration scheme")
        self.scheme = scheme
        if mixture and not memory:
            raise ValueError("The mixed architecture requires node memory")
        self.memory = memory
        self.mixture = mixture
        self.routing = "learned"
        self.physical_clock = physical_clock
        if physical_clock and not memory:
            raise ValueError("A recurrent clock requires node memory")
        if memory:
            self.cell = nn.GRUCell(7, memory)
        self.mapping = nn.Sequential(
            nn.Linear(7 + memory, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.mapping[-1].weight)
        nn.init.zeros_(self.mapping[-1].bias)

        if mixture:
            self.instant = nn.Sequential(
                nn.Linear(7, hidden), nn.SiLU(), nn.Linear(hidden, 1)
            )
            nn.init.zeros_(self.instant[-1].weight)
            nn.init.zeros_(self.instant[-1].bias)
            self.gate = nn.Linear(7, 3)

    def correction(self, features, memory):
        """Mix physical, instantaneous and recurrent corrections per node."""
        inputs = torch.cat((features, memory), -1) if self.memory else features
        recurrent = self.mapping(inputs).squeeze(-1).double().tanh()
        if not self.mixture:
            return recurrent
        instant = self.instant(features).squeeze(-1).double().tanh()
        if self.routing == "instant":
            return instant
        if self.routing == "recurrent":
            return recurrent
        if self.routing == "uniform":
            return (instant + recurrent) / 3
        weights = self.gate(features).softmax(-1).double()
        # The physical expert supplies a zero correction.
        return weights[..., 1] * instant + weights[..., 2] * recurrent

    def static(self, mass, number):
        """Build the conserved dry-node state once per trajectory."""
        dry = mass[..., 1:].sum(-1)
        volume = dry / 1400
        capacity = (1000 / 1400) * (0.03 * mass[..., 1] + 0.75 * mass[..., 2])
        radius = (3 / (4 * torch.pi) * volume.clamp_min(1e-36)).pow(1 / 3)
        size = (radius * 1e6).clamp_min(1e-4).log()
        fraction = mass[..., 2] / dry.clamp_min(1e-36)
        concentration = torch.log1p(
            number.sum(1, keepdim=True) / 1e11
        ).expand_as(dry)
        return dry, volume, capacity, size, fraction, concentration

    def start(self, number):
        """Initialize node memory without forecast history."""
        return torch.zeros(
            (*number.shape, self.memory),
            dtype=torch.float32,
            device=number.device,
        )

    def step(self, water, gas, number, state, h, memory):
        """Predict one conservative step, preserving the infinitesimal flux."""
        dry, volume, capacity, size, fraction, concentration = state
        delta, local, hk, surface = implicit_increment(
            water, gas, number, volume, capacity, h
        )
        stiffness = local + (hk * number).sum(1, keepdim=True)
        drive = (gas[:, None] - surface) / SATURATION
        features = torch.stack(
            (
                size,
                fraction,
                concentration,
                water / dry.clamp_min(1e-36),
                gas[:, None].expand_as(water) / SATURATION,
                drive,
                stiffness.log1p(),
            ),
            -1,
        ).float()
        if self.memory:
            candidate = self.cell(
                features.flatten(0, 1), memory.flatten(0, 1)
            ).reshape_as(memory)
            if self.physical_clock:
                # Evolve memory on the local particle + shared gas timescale.
                fraction_elapsed = -torch.expm1(-stiffness).float()
                candidate = memory + fraction_elapsed[..., None] * (
                    candidate - memory
                )
            active = (h[:, None] > 0) & (number > 0)
            memory = torch.where(active[..., None], candidate, memory)
        adjustment = self.correction(features, memory)
        if self.scheme == "implicit":
            delta = delta * (1 + adjustment * stiffness / (1 + stiffness))
        else:
            coarse, _ = balance(water, gas, number, delta)
            half, half_gas = implicit_step(
                water, gas, number, volume, capacity, h / 2
            )
            fine, _ = implicit_step(
                half, half_gas, number, volume, capacity, h / 2
            )
            delta = (2 * fine - coarse - water) + adjustment * (fine - coarse)
        water, gas = balance(water, gas, number, delta)
        return water, gas, memory

    def rollout(self, mass, number, gas, force, substeps=1, return_nodes=False):
        """Apply only known forcing and return bulk particle and gas water."""
        if substeps < 1:
            raise ValueError("Expected positive substeps")
        state = self.static(mass, number)
        memory = self.start(number)
        water = mass[..., 0]
        series = [water if return_nodes else (number * water).sum(1)]
        gas_series = [gas]
        for f in force.unbind(1):
            for _ in range(substeps):
                water, gas, memory = self.step(
                    water, gas, number, state, f[:, 0] / substeps, memory
                )
            gas = torch.where(f[:, 3] > 0, f[:, 4], gas)
            series.append(water if return_nodes else (number * water).sum(1))
            gas_series.append(gas)
        return torch.stack(series, 1), torch.stack(gas_series, 1)
