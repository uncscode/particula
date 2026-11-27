"""Store weighted dry-particle corrections to finite-reservoir equilibrium."""

import math

import torch
from activation import KELVIN_LENGTH
from embedding import WeightedEncoder, mlp

SATURATION = 0.01713206014665131


class EquilibriumEncoder(WeightedEncoder):
    """Reserve a few shape coordinates for known dry-particle thermodynamics."""

    def __init__(self, width, diag_dim, coordinates):
        """Keep the overall storage width fixed between representations."""
        if coordinates < 4 or width <= coordinates:
            raise ValueError(
                "Need four equilibrium coordinates and a learned coordinate"
            )
        super().__init__(width, diag_dim)
        self.equilibrium_knots = coordinates
        grid = -torch.expm1(
            -torch.linspace(
                -math.log(0.9),
                -math.log(0.007),
                coordinates,
                dtype=torch.float64,
            )
        )
        self.register_buffer("rh_knots", grid)
        self.phi = mlp([4, 64, 64, width - coordinates])

    def map_particles(self, x):
        """Evaluate the per-particle dry curvature correction at RH knots."""
        radius = (x[..., 0:1].double().exp() * 0.5).clamp_min(0.0005)
        kappa = 0.03 + 0.72 * x[..., 2:3].double()
        a = KELVIN_LENGTH * 1e6 / radius
        rh = self.rh_knots
        water = kappa * rh / (1 - rh + torch.expm1(a))
        for _ in range(9):
            delta = torch.expm1(a / (1 + water).pow(1 / 3))
            water = kappa * rh / (1 - rh + delta)
        volume = 4 * torch.pi / 3 * radius.pow(3)
        curve = (
            volume
            * kappa
            * (delta / (KELVIN_LENGTH * 1e6))
            * (1 - rh)
            / (1 - rh + delta)
        )
        return torch.cat([curve.float(), super().map_particles(x)], -1)


def correction_at_rh(shape, rh):
    """Cubic interpolation on uniformly spaced log saturation deficits."""
    width = shape.shape[-1]
    low, high = -math.log(0.9), -math.log(0.007)
    pos = ((-torch.log1p(-rh) - low) / (high - low) * (width - 1)).clamp(
        0, width - 1
    )
    start = (pos.floor().long() - 1).clamp(0, width - 4)
    t = pos - start
    y = shape.gather(1, start[:, None] + torch.arange(4, device=shape.device))
    coeff = torch.stack(
        [
            -(t - 1) * (t - 2) * (t - 3) / 6,
            t * (t - 2) * (t - 3) / 2,
            -t * (t - 1) * (t - 3) / 2,
            t * (t - 1) * (t - 2) / 6,
        ],
        1,
    )
    return (y * coeff).sum(1)


def spectral_equilibrium(p, shape):
    """Close the finite reservoir with differentiable curvature corrections."""
    cap = 1000 / 1400 * (0.03 * p[:, 2] + 0.75 * p[:, 3])
    total = p[:, 1] + p[:, 4]
    linear = SATURATION - total + cap
    planar = (
        2
        * cap
        * total
        / (linear + torch.sqrt(linear.square() + 4 * cap * total)).clamp_min(
            1e-30
        )
    )
    rh = ((total - planar) / SATURATION).clamp(1e-8, 0.996)
    scale = p[:, 0] * 1e-15 * (KELVIN_LENGTH * 1e6)
    for _ in range(6):
        deficit = 1 - rh
        uptake = (
            cap * rh / deficit
            - scale * correction_at_rh(shape, rh) * rh / deficit.square()
        )
        residual = uptake + SATURATION * rh - total
        rh = (rh - residual / (SATURATION + cap / deficit.square())).clamp(
            0, 0.996
        )
    return torch.minimum(total, (total - SATURATION * rh).clamp_min(0))
