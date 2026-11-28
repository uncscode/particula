"""Conservative grid inventories and native discrete boxes."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InventoryLayout:
    """Ordered gases, dry size classes and particle materials.

    Lanes contain extensive gas kg, then represented counts by bin, then
    particulate kg in bin-major/material-minor order. Native discrete boxes
    store gas kg/m³, represented counts and kg per individual particle.
    """

    gases: tuple
    bins: tuple = ()
    materials: tuple = ()

    def __post_init__(self):
        """Require unique named lanes and coherent bin/material axes."""
        for name in ("gases", "bins", "materials"):
            seq = tuple(getattr(self, name))
            if len(set(seq)) != len(seq) or any(
                not isinstance(s, str) or not s.strip() for s in seq
            ):
                raise ValueError("inventory names must be unique and nonempty")
            object.__setattr__(self, name, seq)
        if bool(self.bins) != bool(self.materials) or not (
            self.gases or self.bins
        ):
            raise ValueError("supply gases or paired bins and materials")

    @property
    def width(self):
        """Number of extensive lanes per cell."""
        return len(self.gases) + len(self.bins) * (1 + len(self.materials))

    def validate(self, inventory, n):
        """Reject nonphysical inventories including mass in an empty bin."""
        a = np.asarray(inventory, dtype=float)
        if (
            a.shape != (n, self.width)
            or not np.isfinite(a).all()
            or np.any(a < 0)
        ):
            raise ValueError("invalid grid inventories")
        if self.bins:
            g, b = len(self.gases), len(self.bins)
            mass = a[:, g + b :].reshape(n, b, len(self.materials))
            if np.any((a[:, g : g + b] == 0) != np.all(mass == 0, axis=2)):
                raise ValueError(
                    "particle number and material must occupy the same bins"
                )
        return a

    def _check_native(self, grid, gas, particles):
        n = len(grid.volume)
        if (
            tuple(gas.name) != self.gases
            or gas.concentration.shape != (n, len(self.gases))
            or particles.distribution_type != "discrete"
            or particles.masses.shape
            != (n, len(self.bins), len(self.materials))
            or particles.volume.shape != (n,)
            or not np.all(np.isfinite(particles.volume))
            or np.any(particles.volume <= 0)
            or np.any(particles.charge)
        ):
            raise ValueError(
                "need matching neutral discrete boxes and named gases"
            )

    def pack(self, grid, gas, particles):
        """Convert native neutral discrete boxes to extensive quantities."""
        self._check_native(grid, gas, particles)
        vol = grid.volume[:, None]
        counts = particles.population * (vol / particles.volume[:, None])
        mass = particles.masses * counts[:, :, None]
        return self.validate(
            np.column_stack(
                (gas.concentration * vol, counts, mass.reshape(len(vol), -1))
            ),
            len(vol),
        )

    def unpack(self, grid, inventory, gas, particles):
        """Return detached native gas and particle copies in the same order."""
        self._check_native(grid, gas, particles)
        n = len(grid.volume)
        a = self.validate(inventory, n)
        g, b = len(self.gases), len(self.bins)
        counts = a[:, g : g + b]
        gas, particles = gas.copy(), particles.copy()
        gas.concentration[:] = a[:, :g] / grid.volume[:, None]
        particles.population[:] = (
            counts * (particles.volume / grid.volume)[:, None]
        )
        if not np.all(np.isfinite(particles.population)):
            raise ValueError(
                "grid counts overflow the target sample population"
            )
        mass = a[:, g + b :].reshape(n, b, len(self.materials))
        np.divide(
            mass,
            counts[:, :, None],
            out=particles.masses,
            where=counts[:, :, None] > 0,
        )
        return gas, particles


@dataclass(frozen=True)
class Uptake:
    """First-order gas uptake (s⁻¹) into one occupied particle class."""

    gas: str
    size_bin: str
    material: str
    rate: float

    def bind(self, layout):
        """Resolve named lanes once before transport."""
        if not np.isfinite(self.rate) or self.rate < 0:
            raise ValueError("uptake rate must be finite and nonnegative")
        b = layout.bins.index(self.size_bin)
        m = layout.materials.index(self.material)
        g = len(layout.gases)
        return (
            layout.gases.index(self.gas),
            g + b,
            g + len(layout.bins) + b * len(layout.materials) + m,
        )
