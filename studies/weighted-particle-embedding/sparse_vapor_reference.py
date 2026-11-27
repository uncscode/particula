"""Integrate explicit gas and particle states with a sparse Jacobian."""

import numpy as np
from multivapor_reference import Rates
from scipy import sparse
from scipy.integrate import solve_ivp


class ExplicitReservoir:
    """Retain one additional gas concentration per volatile in the state."""

    def __init__(self, m, n, volume, spec):
        """Fix each particle's dry mass and each gas concentration scale."""
        self.local = Rates(m, n, volume, spec)
        self.s = spec.species
        self.scale = spec.saturation
        self.lanes = np.tile(np.arange(self.s), len(n))

    def rates(self, _, y):
        """Give the two phases equal and opposite mass rates for every gas."""
        gas = y[-self.s :] * self.scale
        dq = self.local.local(y[: -self.s], gas)
        dg = -(self.local.w @ dq.reshape(-1, self.s)) / self.scale
        return np.r_[dq, dg]

    def jacobian(self, _, y):
        """Assemble local derivative blocks and sparse shared gas couplings."""
        s = self.s
        a, b = self.local.local(y[:-s], y[-s:] * self.scale, True)
        n = len(a)
        particle = sparse.bsr_matrix(
            (a, np.arange(n), np.arange(n + 1)), shape=(n * s, n * s)
        )
        gas_to_q = sparse.coo_matrix(
            ((b * self.scale).ravel(), (np.arange(n * s), self.lanes)),
            shape=(n * s, s),
        )
        q_to_g = (
            (-a * self.local.w[:, None, None] / self.scale[None, :, None])
            .transpose(1, 0, 2)
            .reshape(s, -1)
        )
        gas = sparse.diags(-(self.local.w @ b))
        return sparse.bmat(
            [[particle, gas_to_q], [sparse.csr_matrix(q_to_g), gas]],
            format="csc",
        )


def integrate(m, n, volume, levels, mask, t, spec, tol):
    """Integrate and independently check per-species conservation."""
    valid = n > 0
    m, n = m[valid], n[valid]
    model = ExplicitReservoir(m, n, volume, spec)
    s = spec.species
    current = (m[:, :s] / model.local.dry[:, None]).ravel()
    gas = levels[0].copy()
    starts = np.r_[0, np.flatnonzero(np.diff(t) == 0) + 1]
    wet, gases = [], []
    for leg, (i, j) in enumerate(
        zip(starts, np.r_[starts[1:], len(t)], strict=True)
    ):
        gas = np.where(mask[leg], levels[leg], gas)
        q0 = np.r_[current, gas / model.scale]
        times = t[i:j] - t[i]
        sol = solve_ivp(
            model.rates,
            (0, times[-1]),
            q0,
            method="BDF",
            jac=model.jacobian,
            t_eval=times,
            rtol=tol,
            atol=tol * 1e-3,
        )
        if (
            not sol.success
            or not np.isfinite(sol.y).all()
            or sol.y.min() < -tol * 1e-2
        ):
            raise ArithmeticError(sol.message)
        q = sol.y[:-s].T.reshape(len(times), len(m), s)
        g = sol.y[-s:].T * model.scale
        bulk = np.einsum("i,tis->ts", model.local.w, q)
        np.testing.assert_allclose(
            bulk + g,
            np.broadcast_to(bulk[0] + g[0], g.shape),
            rtol=2e-11,
            atol=1e-16,
        )
        current = q[-1].ravel()
        gas = g[-1]
        wet.append(bulk)
        gases.append(g)
    return np.concatenate(wet), np.concatenate(gases)
