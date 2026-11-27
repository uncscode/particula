"""Solve active vapor blocks while retaining all species in thermodynamics."""

import numpy as np
import torch
from stage2_transfer import restore
from vapor_properties import Materials


class BlockExchange:
    """Build active water/organic solvers from the current physical state."""

    def __init__(self, spec, device, model, organic_groups, water_partners=0):
        """Group organics by a fixed material-property timescale proxy."""
        water = np.array([0])
        org = np.arange(1, spec.species)
        order = np.argsort(
            -(
                spec.diffusivity[1:]
                * spec.accommodation[1:]
                * spec.saturation[1:]
            )
        )
        if organic_groups < 1:
            raise ValueError("Provide a positive number of organic groups")
        if not 0 <= water_partners <= len(org):
            raise ValueError("Water partners must fit the available organics")
        ranked = org[order]
        blocks = [np.r_[water, ranked[:water_partners]]]
        remaining = ranked[water_partners:]
        if len(remaining):
            blocks.extend(
                np.array_split(remaining, min(organic_groups, len(remaining)))
            )
        self.solvers = []
        self.s = spec.species
        self.molar = torch.as_tensor(spec.molar[: self.s], device=device)
        self.rho = torch.as_tensor(spec.density[: self.s], device=device)
        dry = np.arange(self.s, len(spec.molar))
        for active in blocks:
            keep = np.r_[active, dry]
            other = np.setdiff1d(np.arange(self.s), active)
            material = Materials(
                tuple(spec.names[i] for i in active),
                spec.molar[keep],
                spec.density[keep],
                spec.pressure[active],
                spec.diffusivity[active],
                spec.accommodation[active],
            )
            self.solvers.append(
                (
                    torch.as_tensor(active, device=device),
                    torch.as_tensor(other, device=device),
                    restore(material, model, device),
                )
            )

    def advance(
        self,
        state,
        full_model,
        dt,
        steps,
        mode,
        water_steps,
        organic_steps,
        work,
    ):
        """Compose active solves at their own rates with common mass and gas."""
        number = state.live / state.volume[:, None]
        dry, moles, volume, weight, composition, live = full_model.static(
            state.mass, number
        )
        indices = list(range(len(self.solvers)))
        if mode == "organic-outer":
            indices = indices[1:] + indices[:1]
        schedule = (
            [(i, 1.0) for i in indices]
            if mode == "water-first"
            else (
                [(i, 0.5) for i in indices[:-1]]
                + [(indices[-1], 1.0)]
                + [(i, 0.5) for i in reversed(indices[:-1])]
            )
        )
        for _ in range(steps):
            for idx, fraction in schedule:
                active, other, model = self.solvers[idx]
                frozen = state.mass[..., other]
                static = (
                    dry,
                    moles + (frozen / self.molar[other]).sum(-1) / dry,
                    volume + (frozen / self.rho[other]).sum(-1),
                    weight,
                    composition,
                    live,
                )
                q = state.mass[..., active] / dry[..., None]
                gas = state.gas[:, active]
                h = state.vapor_memory[..., active, :]
                sub = water_steps if idx == 0 else organic_steps
                elapsed = gas.new_full(
                    (len(gas),), dt * fraction / (steps * sub)
                )
                for _ in range(sub):
                    q, gas, h = model.step(q, gas, static, elapsed, h)
                state.mass[..., active] = q * dry[..., None]
                state.gas[:, active] = gas
                state.vapor_memory[..., active, :] = h
                key = "water_substeps" if idx == 0 else "organic_substeps"
                work[key] = work.get(key, 0) + sub
