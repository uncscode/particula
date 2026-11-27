"""Solve coupled vapor and gas trajectories with Radau collocation."""

import numpy as np
import torch
from stage2_transfer import Transfer


def coefficients(stages):
    """Return the two- or three-stage Radau IIA collocation matrix."""
    if stages == 2:
        return np.array([[5 / 12, -1 / 12], [3 / 4, 1 / 4]], dtype=float)
    if stages == 3:
        r = np.sqrt(6.0)
        return np.array(
            [
                [
                    (88 - 7 * r) / 360,
                    (296 - 169 * r) / 1800,
                    (-2 + 3 * r) / 225,
                ],
                [
                    (296 + 169 * r) / 1800,
                    (88 + 7 * r) / 360,
                    (-2 - 3 * r) / 225,
                ],
                [(16 - r) / 36, (16 + r) / 36, 1 / 9],
            ]
        )
    raise ValueError("Use Radau IIA with two or three stages")


class Radau(Transfer):
    """Newton solve on local stage blocks with a shared gas Schur complement."""

    def __init__(self, spec, stages=2, iterations=3, kind="physical", width=9):
        """Bind materials and collocation coefficients."""
        super().__init__(spec, kind, width)
        self.stages, self.newton = stages, iterations
        self.register_buffer("A", torch.as_tensor(coefficients(stages)))
        self.register_buffer(
            "stage_identity",
            torch.eye(stages * spec.species, dtype=torch.float64),
        )

    @staticmethod
    def stage_static(state, stages):
        """Broadcast each static particle property over collocation stages."""
        return tuple(
            x[:, None].expand(-1, stages, *x.shape[1:]).flatten(0, 1)
            for x in state
        )

    def block_solve(self, jac, gain, w, dt, residual):
        """Eliminate particle blocks and solve the shared gas system."""
        b, a, n, s = residual.shape

        def blocks(x):
            return (
                torch.einsum("ac,bcnjk->bnajck", self.A, x)
                * dt[:, None, None, None, None, None]
            ).reshape(b, n, a * s, a * s)

        local = self.stage_identity - blocks(jac)
        shared = blocks(torch.diag_embed(gain))
        rhs = torch.cat(
            (residual.transpose(1, 2).reshape(b, n, a * s, 1), shared), -1
        )
        result, info = torch.linalg.solve_ex(local, rhs)
        torch._assert_async(
            (info == 0).all(), "Singular local collocation block"
        )
        free, shared = result[..., :1], result[..., 1:]
        correction, info = torch.linalg.solve_ex(
            self.stage_identity + (w[..., None, None] * shared).sum(1),
            (w[..., None, None] * free).sum(1),
        )
        torch._assert_async(
            (info == 0).all(), "Singular shared collocation block"
        )
        return (
            (free - shared @ correction[:, None])
            .reshape(b, n, a, s)
            .transpose(1, 2)
        )

    def step(self, q, gas, state, dt, memory, source_at=None):
        """Solve the coupled stage equations with positive damped iterates."""
        w = state[3]
        b, n, s = q.shape
        a = self.stages
        fixed = self.stage_static(state, a)
        y = q[:, None].expand(-1, a, -1, -1)
        # An external integrated dose enters each stage before partitioning.
        # The final Radau stage lies at the physical step endpoint.
        supply = gas[:, None] if source_at is None else gas[:, None] + source_at
        stage_g = supply.expand(-1, a, -1)
        for iteration in range(self.newton):
            rate, jac, gain, drive, r = self.linearization(
                y.reshape(b * a, n, s), stage_g.reshape(b * a, s), fixed
            )
            rate = rate.reshape(b, a, n, s)
            jac = jac.reshape(b, a, n, s, s)
            gain = gain.reshape(b, a, n, s)
            residual = (
                q[:, None]
                - y
                + dt[:, None, None, None]
                * torch.einsum("ac,bcns->bans", self.A, rate)
            )
            delta = self.block_solve(jac, gain, w, dt, residual)
            if iteration == 0 and self.head is not None:
                # Reconcile a single causal memory at the accepted endpoint.
                # Apply its bounded update factor to the first Newton
                # directions.
                stiff = dt[:, None, None] * (
                    jac[:, -1].abs().sum(-1)
                    + (w[..., None] * gain[:, -1]).sum(1, keepdim=True)
                )
                first = delta[:, -1]
                learned, memory = self.adjustment(
                    q,
                    gas,
                    state,
                    dt,
                    memory,
                    first,
                    stiff,
                    drive.reshape(b, a, n, s)[:, -1],
                    r.reshape(b, a, n)[:, -1],
                )
                safe = torch.where(first.abs() > 1e-14, first, 1.0)
                factor = torch.where(first.abs() > 1e-14, learned / safe, 1.0)
                delta = delta * factor[:, None]
            dg = -(w[:, None, :, None] * delta).sum(2)
            # One damping coefficient per box protects all collocation stages
            # and reservoirs together. Full Newton steps resume inside the cone.
            lim_y = torch.where(
                delta < 0, -y / delta.clamp_max(-1e-300), 2.0
            ).amin((1, 2, 3))
            lim_g = torch.where(
                dg < 0, -stage_g / dg.clamp_max(-1e-300), 2.0
            ).amin((1, 2))
            alpha = torch.minimum(lim_y, lim_g).clamp_min(0)
            alpha = torch.where(alpha < 1.0, 0.98 * alpha, 1.0)
            y = y + alpha[:, None, None, None] * delta
            stage_g = supply - (w[:, None, :, None] * (y - q[:, None])).sum(2)
        yn, gn = self.balance(q, supply[:, -1], w, y[:, -1] - q)
        return yn, gn, memory
