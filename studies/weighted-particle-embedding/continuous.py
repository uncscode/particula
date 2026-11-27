"""Continuous latent vector field with a conservative exponential midpoint
step.
"""

import torch
from embedding import mlp
from forecast import physical_features
from relaxation import equilibrium_water
from torch import nn


class ContinuousODE(nn.Module):
    """Separate binary-collision and gas-exchange rates in physical seconds."""

    def __init__(self, width, hidden=96, substeps=2):
        """Build instantaneous rates independent of the chosen integration
        step.
        """
        super().__init__()
        self.width = width
        self.history = 1
        self.substeps = substeps
        self.register_buffer("z_mean", torch.zeros(width))
        self.register_buffer("z_std", torch.ones(width))
        self.coagulation = mlp([width, hidden, hidden, width + 1])
        self.exchange = mlp([width + 5, hidden, hidden, 2 * width + 3])

    def features(self, z, p, f):
        """Expose state for runners that also evaluate history-based models."""
        return torch.cat([z, physical_features(p), f[..., 1:3].float()], -1)

    def coefficients(self, z, p, process):
        """Evaluate the continuous collision and phase-transfer coefficients."""
        coag = self.coagulation(z)
        cond = self.exchange(torch.cat([z, physical_features(p)], -1))
        coag_enabled, cond_enabled = process[:, 0], process[:, 1]
        k = nn.functional.softplus(coag[:, -1]).double() * 1e-15 * coag_enabled
        change = coag[:, : self.width] * coag_enabled[:, None].float()
        destination = cond[:, : self.width]
        shape_rate = (
            nn.functional.softplus(cond[:, self.width : 2 * self.width]) * 200
            + 0.1
        ) * cond_enabled[:, None].float()
        water_rate = (
            nn.functional.softplus(cond[:, -3]).double() * 200 + 0.1
        ) * cond_enabled
        curvature = (1 + 0.85 * cond[:, -2].tanh()).double()
        equilibrium = equilibrium_water(p, curvature)
        dry = p[:, 2:4].sum(1)
        # A bounded transient flux can depend on the evolving size/composition
        # shape.
        target = torch.minimum(
            p[:, 1] + p[:, 4],
            (equilibrium + 0.25 * dry * cond[:, -1].double().tanh()).clamp_min(
                0
            ),
        )
        return k, change, destination, shape_rate, water_rate, target

    def advance(self, z, p, dt, coeff, mid_number):
        """Integrate frozen rates with positive phase inventories."""
        k, coag, eq, rate, water_rate, target = coeff
        number = p[:, 0] / (1 + dt * k * p[:, 0])
        water = p[:, 1] + (-torch.expm1(-dt * water_rate)) * (target - p[:, 1])
        transfer = water - p[:, 1]
        # Exact paired update guards both available inventories at roundoff.
        transfer = torch.minimum(p[:, 4], torch.maximum(-p[:, 1], transfer))
        next_p = torch.stack(
            [number, p[:, 1] + transfer, p[:, 2], p[:, 3], p[:, 4] - transfer],
            -1,
        )
        h = dt[:, None].float()
        arg = h * rate
        decay = -torch.expm1(-arg)
        response = torch.where(
            arg > 1e-7, decay / arg.clamp_min(1e-7), 1 - arg / 2
        )
        new_z = (
            z
            + decay * (eq - z)
            + h * response * coag * (mid_number[:, None] * 1e-14).float()
        )
        return new_z, next_p

    def rollout(self, z, p, force, history=None):
        """Integrate the vector field and apply resets at observation
        boundaries.
        """
        zs, ps = [z], [p]
        for i in range(force.shape[1]):
            f = force[:, i]
            h = f[:, 0] / self.substeps
            for _ in range(self.substeps):
                plain = self.coefficients(z, p, f[:, 1:3])
                mid_z, mid_p = self.advance(z, p, h / 2, plain, p[:, 0])
                midpoint = self.coefficients(mid_z, mid_p, f[:, 1:3])
                z, p = self.advance(z, p, h, midpoint, mid_p[:, 0])
            p = torch.cat(
                [p[:, :4], torch.where(f[:, 3] > 0, f[:, 4], p[:, 4])[:, None]],
                1,
            )
            zs.append(z)
            ps.append(p)
        return torch.stack(zs, 1), torch.stack(ps, 1)
