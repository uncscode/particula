"""Weighted conductance moments of native isothermal water exchange at 293 K."""

import torch
from embedding import WeightedEncoder, mlp
from water_constants import KELVIN, MEAN_FREE_PATH, SATURATION

SCALE = 1e10


def conductance(radius):
    """Native transition-regime conductance for water in m³/s."""
    kn = MEAN_FREE_PATH / radius
    alpha = 0.2
    correction = (
        0.75
        * alpha
        * (1 + kn)
        / (kn.square() + kn + 0.283 * alpha * kn + 0.75 * alpha)
    )
    return 4 * torch.pi * radius * 2.4e-5 * correction


def flux_moments(x):
    """Return scaled transport K and surface-saturation-weighted K in m³/s."""
    radius = (0.5e-6 * x[..., 1].double().exp()).clamp_min(5e-10)
    transport = conductance(radius)
    water = x[..., 3].double()
    kappa = 0.03 + 0.72 * x[..., 2].double()
    activity = water / (water + 1000 / 1400 * kappa * (1 - water)).clamp_min(
        1e-30
    )
    surface = activity * torch.exp(KELVIN / radius)
    return SCALE * torch.stack([transport, transport * surface], -1)


def bulk_rate(shape, p):
    """Compute native particle-water tendency in kg m⁻³ s⁻¹."""
    return (
        p[..., 0]
        / SCALE
        * (p[..., 4] * shape[..., 0] - SATURATION * shape[..., 1])
    )


class FluxEncoder(WeightedEncoder):
    """Reserve two coordinates for instantaneous water transfer."""

    def __init__(self, width, dim):
        """Share the remaining shape budget with the learned particle map."""
        super().__init__(width, dim)
        if width <= 2:
            raise ValueError(
                "Flux encoding needs at least one learned coordinate"
            )
        self.phi = mlp([4, 64, 64, width - 2])

    def map_particles(self, x):
        """Preserve the two physical moments alongside a learned shape map."""
        return torch.cat(
            [flux_moments(x).float(), super().map_particles(x)], -1
        )
