"""Diagonalize Radau stages and reuse the ideal-mixture low-rank Jacobian."""

import numpy as np
import torch
from stage2_lowrank import LowRankROS4
from stage2_radau import Radau, coefficients


class SimplifiedRadau(LowRankROS4):
    """Factor frozen collocation stages by eigenmode."""

    def __init__(self, spec, stages, iterations):
        """Prepare the independent real and complex eigenmodes."""
        super().__init__(spec)
        a = coefficients(stages)
        eigen, vectors = np.linalg.eig(a)
        inverse = np.linalg.inv(vectors)
        keep = np.flatnonzero(eigen.imag >= 0)
        self.register_buffer("a", torch.from_numpy(a))
        self.register_buffer("values", torch.from_numpy(eigen[keep]))
        self.register_buffer("left", torch.from_numpy(inverse[keep].copy()))
        self.register_buffer(
            "right", torch.from_numpy(vectors[:, keep].T.copy())
        )
        self.real = tuple(bool(eigen[i].imag == 0) for i in keep)
        self.iterations, self.stages = iterations, stages

    def solvers(self, q, gas, state, dt):
        """Factor the frozen reservoir system in stage coordinates."""
        _, p, u, v, gain, _, _ = self.factors(q, gas, state, dt)
        args = (p, u, v, gain)
        modes = []
        for i, real in enumerate(self.real):
            value = self.values[i]
            dtype = q.dtype if real else torch.complex128
            h = dt[:, None, None] * (value.real if real else value) / self.gamma
            solve = self.factorized(*(a.to(dtype) for a in args), state[3], h)
            left, right = self.left[i], self.right[i]
            if real:
                left, right = left.real, right.real
            modes.append((solve, left, right, real))
        return modes

    @staticmethod
    def update(modes, residual):
        """Reconstruct a real stage increment from the independent solves."""
        delta = torch.zeros_like(residual)
        for solve, left, right, real in modes:
            rhs = (left[None, :, None, None] * residual).sum(1)
            piece = right[None, :, None, None] * solve(rhs)[:, None]
            delta = delta + (piece if real else 2 * piece.real)
        return delta

    def step(self, q, gas, state, dt, memory):
        """Iterate the positive nonlinear collocation stages."""
        modes = self.solvers(q, gas, state, dt)
        a, w = self.stages, state[3][:, None, :, None]
        b, n, s = q.shape
        fixed = Radau.stage_static(state, a)
        y = q[:, None].expand(-1, a, -1, -1)
        g = gas[:, None].expand(-1, a, -1)
        for _ in range(self.iterations):
            rate = self.linearization(
                y.reshape(b * a, n, s),
                g.reshape(b * a, s),
                fixed,
                rate_only=True,
            ).reshape(b, a, n, s)
            residual = (
                q[:, None]
                - y
                + dt[:, None, None, None]
                * torch.einsum("ac,bcns->bans", self.a, rate)
            )
            delta = self.update(modes, residual)
            dg = -(w * delta).sum(2)
            lim = torch.minimum(
                torch.where(delta < 0, -y / delta.clamp_max(-1e-300), 2.0).amin(
                    (1, 2, 3)
                ),
                torch.where(dg < 0, -g / dg.clamp_max(-1e-300), 2.0).amin(
                    (1, 2)
                ),
            )
            weight = torch.where(lim < 1.0, 0.98 * lim.clamp_min(0), 1.0)
            y = y + weight[:, None, None, None] * delta
            g = gas[:, None] - (w * (y - q[:, None])).sum(2)
        yn, gn = self.balance(q, gas, state[3], y[:, -1] - q)
        return yn, gn, memory
