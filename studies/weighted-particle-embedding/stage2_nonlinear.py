"""Test positive implicit relaxation and extrapolated vapor exchange."""

import torch
from stage2_transfer import Transfer


class NonlinearExchange(Transfer):
    """Iterate wet composition while closing each shared gas reservoir."""

    def __init__(
        self, spec, outer=3, reservoir=3, order=2, kind="physical", width=9
    ):
        """Set the local composition and shared-reservoir iteration budgets."""
        super().__init__(spec, kind, width)
        self.outer, self.reservoir, self.extrapolation = outer, reservoir, order

    @staticmethod
    def root(capacity, uptake, scale, sat, guess):
        # q - q_initial = h G [gas - E q/(A+q)] is a quadratic.
        """Solve the positive backward-Euler uptake quadratic."""
        load = uptake + scale * guess[:, None]
        b = capacity + scale * sat - load
        disc = (b.square() + 4 * capacity * load).clamp_min(1e-100).sqrt()
        positive = 2 * capacity * load / (disc + b).clamp_min(1e-60)
        return torch.where(b >= 0, positive, (disc - b) * 0.5)

    def implicit(self, q, gas, state, dt):
        """Resolve the gas reservoir and local material partition."""
        dry, base_moles, base_vol, w, _, live = state
        wet = q
        gg = gas
        inventory = gas + (w[..., None] * q).sum(1)
        for _ in range(self.outer):
            vol = base_vol + dry * (wet / self.density).sum(-1)
            r = (3 / (4 * torch.pi) * vol).pow(1 / 3)
            kn = self.path / r[..., None]
            den = (
                kn.square() + (1 + 0.283 * self.alpha) * kn + 0.75 * self.alpha
            )
            gain = (
                4
                * torch.pi
                * r[..., None]
                * self.diffusion
                * 0.75
                * self.alpha
                * (1 + kn)
                / (den * dry[..., None])
            )
            scale = dt[:, None, None] * gain * live[..., None]
            total = base_moles + (wet / self.molar).sum(-1)
            capacity = (total[..., None] * self.molar - wet).clamp_min(1e-15)
            sat = self.sat * torch.exp(self.kelvin / r[..., None])
            if self.active_lanes is not None:
                scale = scale * self.active_lanes
            for _ in range(self.reservoir):
                wet = self.root(capacity, q, scale, sat, gg)
                residual = gg + (w[..., None] * wet).sum(1) - inventory
                dq = scale / (
                    1 + scale * sat * capacity / (capacity + wet).square()
                )
                gg = (
                    gg - residual / (1 + (w[..., None] * dq).sum(1))
                ).clamp_min(0)
                gg = torch.minimum(gg, inventory)
            wet = self.root(capacity, q, scale, sat, gg)
        return self.balance(q, gas, w, wet - q)

    def repeated(self, q, g, state, dt, n):
        """Compose closed implicit steps over one observation interval."""
        for _ in range(n):
            q, g = self.implicit(q, g, state, dt / n)
        return q

    def step(self, q, gas, state, dt, memory):
        """Extrapolate successively refined nonlinear transfer increments."""
        coarse, g = self.implicit(q, gas, state, dt)
        if self.extrapolation == 1:
            return coarse, g, memory
        fine = self.repeated(q, gas, state, dt, 2)
        err = fine - coarse
        if self.extrapolation == 3:
            best = self.repeated(q, gas, state, dt, 4)
            low = 2 * fine - coarse
            fine = 2 * best - fine
            err = (fine - low) / 3
        if self.head is not None:
            # Scale the defect between distinct causal resolutions. The shared
            # projection is applied once to the final sum of the species fluxes.
            _, jac, gain, drive, r = self.linearization(q, gas, state)
            stiff = dt[:, None, None] * (
                jac.abs().sum(-1)
                + (state[3][..., None] * gain).sum(1, keepdim=True)
            )
            err, memory = self.adjustment(
                q, gas, state, dt, memory, err, stiff, drive, r
            )
        y, g = self.balance(q, gas, state[3], fine - q + err)
        return y, g, memory
