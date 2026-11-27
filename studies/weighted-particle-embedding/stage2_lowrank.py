"""Use the two local cross-species couplings of ideal-molar vapor exchange."""

import torch
from stage2_transfer import Transfer


class LowRankROS4(Transfer):
    """Factor the two local ideal-mixture couplings separately."""

    def transport_gain(self, rr, dry, live):
        """Evaluate the transition conductance and its wet-radius slope."""
        kn = self.path / rr
        denom = kn.square() + (1 + 0.283 * self.alpha) * kn + 0.75 * self.alpha
        gain = (
            4
            * torch.pi
            * rr
            * self.diffusion
            * 0.75
            * self.alpha
            * (1 + kn)
            / (denom * dry[..., None])
            * live[..., None]
        )
        power = (
            1 - kn / (1 + kn) + (2 * kn + 1 + 0.283 * self.alpha) * kn / denom
        )
        return gain, power

    def factors(self, q, gas, state, dt):
        """Separate the local loss, wet-radius and mixture-mole derivatives."""
        dry, moles0, volume0, w, _, live = state
        volume = volume0 + dry * (q / self.density).sum(-1)
        radius = (3 * volume / (4 * torch.pi)).pow(1 / 3)
        rr = radius[..., None]
        gain, power = self.transport_gain(rr, dry, live)
        moles = moles0 + (q / self.molar).sum(-1)
        x = q / (self.molar * moles[..., None])
        ke = self.sat * torch.exp(self.kelvin / rr)
        drive = gas[:, None] - ke * x
        rate = gain * drive
        p = gain * ke / (moles[..., None] * self.molar)
        u = torch.stack(
            (gain * (drive * power + ke * x * self.kelvin / rr), gain * ke * x),
            -1,
        )
        v = torch.stack(
            (
                dry[..., None] / (4 * torch.pi * rr.pow(3) * self.density),
                1 / (moles[..., None] * self.molar),
            ),
            -1,
        )
        return rate, p, u, v, gain, drive, radius

    def factorized(self, p, u, v, gain, w, h):
        # Eliminating local two-coordinate responses leaves one shared gas
        # system. Its diagonal plus rank 2N form is cheaper when 2N < S.
        """Factor the gas system using the smaller species or particle rank."""
        b, n, s = p.shape
        scale = self.gamma * h
        inv = (1 + scale * p).reciprocal()
        ell = scale * gain * inv
        a = (scale * inv)[..., None] * u
        vt = v.transpose(-1, -2)
        local = torch.eye(2, device=p.device, dtype=p.dtype) - vt @ a
        x00, x01 = local[..., 0, 0], local[..., 0, 1]
        x10, x11 = local[..., 1, 0], local[..., 1, 1]
        det = x00 * x11 - x01 * x10
        inverse = (
            torch.stack((x11, -x01, -x10, x00), -1).reshape(b, n, 2, 2)
            / det[..., None, None]
        )
        au = a @ inverse
        weighted = (
            (w[..., None, None] * au).permute(0, 2, 1, 3).reshape(b, s, 2 * n)
        )
        vgas = (vt * ell[..., None, :]).reshape(b, 2 * n, s)
        gasdiag = 1 + (w[..., None] * ell).sum(1)
        small = 2 * n < s
        if small:
            left = weighted / gasdiag[..., None]
            system = (
                torch.eye(2 * n, device=p.device, dtype=p.dtype) + vgas @ left
            )
        else:
            system = torch.diag_embed(gasdiag) + weighted @ vgas
        lu, piv, status = torch.linalg.lu_factor_ex(system)
        torch._assert_async((status == 0).all(), "Singular gas factor")

        def solve(rhs):
            e = (rhs * inv)[..., None]
            free = e + au @ (vt @ e)
            d = (w[..., None, None] * free).sum(1)
            if small:
                scaled = d / gasdiag[..., None]
                d = scaled - left @ torch.linalg.lu_solve(
                    lu, piv, vgas @ scaled
                )
            else:
                d = torch.linalg.lu_solve(lu, piv, d)
            removed = ell[..., None] * d[:, None]
            return (free - removed - au @ (vt @ removed)).squeeze(-1)

        return solve

    def step(self, q, gas, state, dt, memory):
        """Advance four Rosenbrock stages using shared low-rank factors."""
        h = dt[:, None, None]
        rate, p, u, v, gain, drive, r = self.factors(q, gas, state, dt)
        solve = self.factorized(p, u, v, gain, state[3], h)
        first = solve(h * rate)
        lower, error = self.increments(q, gas, state, h, first, solve)
        if self.head:
            # Identical embeddings and stiffness features allow checkpoint
            # reuse.
            jac = u @ v.transpose(-1, -2) - torch.diag_embed(p)
            stiff = h * (
                jac.abs().sum(-1)
                + (state[3][..., None] * gain).sum(1, keepdim=True)
            )
            error, memory = self.adjustment(
                q, gas, state, dt, memory, error, stiff, drive, r
            )
        y, g = self.balance(q, gas, state[3], lower + error)
        return y, g, memory
