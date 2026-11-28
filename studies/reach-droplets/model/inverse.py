"""Differentiable size-resolved chamber model and instrument mapping for Figure 7."""

from condensation import budget_error, remap
import numpy as np
from workspace import WORK, background

HERE = WORK / "inversion"
from chamber import (
    Chamber,
    torch,
    Grid,
    MESHES,
    air,
    hologram,
    gas_path,
    temperature,
    inputs,
    HEIGHT0,
)


def project(centers, width, edges):
    """Map bin inventories conservatively, assuming constant density within a model bin."""
    left = centers - width / 2
    right = centers + width / 2
    return (
        np.maximum(0, np.minimum(right[:, None], edges[1:]) - np.maximum(left[:, None], edges[:-1]))
        / width
    )


def interpolator(times, knots):
    return np.stack([np.interp(times, knots, e) for e in np.eye(len(knots))], axis=1)


def pair_basis(d, left, right):
    # Bilinear interpolation in log parent diameter. Off-diagonal terms include both orderings.
    weights = interpolator(np.log(d), np.log([15.0, 35.0, 60.0]))
    rows = []
    for a, b in zip(*np.triu_indices(3)):
        row = weights[left, a] * weights[right, b]
        if a != b:
            row = row + weights[left, b] * weights[right, a]
        rows.append(row)
    out = np.stack(rows)
    assert np.allclose(out.sum(0), 1.0) and out.min() >= 0
    return out


def train_mask(times, split):
    if split == "early":
        return (times >= 3) & (times < 5)
    return (times >= 3) & ((split == "all") | ((times < 6.25) & ((times < 5) | (times >= 5.5))))


class Forward:
    def __init__(self, run, mesh, width, dt, device):
        self.run = run
        self.width = width
        self.dt = dt
        self.model = Chamber(run, width, dt, device)
        m = self.model
        put = m.tensor
        self.put = put
        grid = Grid(*MESHES[str(mesh)])
        d = m.d
        self.grid = grid
        self.pair_weights = put(pair_basis(d, m.ij.cpu().numpy(), m.jj.cpu().numpy()))
        t, self.p, base, chi, _, speed, oldloss, init, _, _ = inputs(run, "local", d, dt)
        self.t = t
        self.f = put(grid.v[:, None])
        w = hologram(grid, d, z=HEIGHT0)[0]
        self.opt = put(w)
        self.chord = put(gas_path(grid, 0.230))
        x = background(run)
        T = base[:, None] + x[0] + x[1] * temperature(grid, run, t)
        self.T = put(T)
        self.chi = put(chi)
        # Extra temperature gradient is centered on the vapor path and scaled to unit maximum.
        radial = np.broadcast_to(grid.rc / 0.254, grid.shape).ravel().copy()
        radial -= gas_path(grid, 0.230) @ radial
        self.mode = put(radial / max(abs(radial)))
        pr = np.exp(
            x[4] * (np.broadcast_to(grid.rc / 0.254, grid.shape).ravel() - 1 / 12) / (5 / 6)
        )[:, None]
        self.init = put(grid.v[:, None] * pr * init / (w * pr).sum(0))
        side = oldloss[11] * (11 / 36)
        end = oldloss[0] / 3
        loss = np.zeros((*grid.shape, len(d)))
        loss[:, :, -1] += side / (1 - (1 - 1 / grid.nr) ** 2)
        loss[0] += grid.nz * end
        loss[-1] += grid.nz * end
        self.survive = put(np.exp(-dt * loss.reshape(grid.n, -1)))
        fall = speed * dt / grid.dz
        source = np.arange(grid.nz)[:, None] + np.floor(fall).astype(int)

        def gather(index):
            return put(np.minimum(index, grid.nz - 1))[:, None, None, :].expand(*grid.shape, len(d))

        self.i0 = gather(source)
        self.i1 = gather(source + 1)
        self.w0 = put((1 - fall % 1) * (source < grid.nz))[:, None, None, :]
        self.w1 = put((fall % 1) * (source + 1 < grid.nz))[:, None, None, :]
        self.ident = put(np.eye(grid.n))
        dif = air(grid, run, "jets", scales=(0, 1)).toarray()
        self.diff = put(dif)
        self.circ = put(air(grid, run, "jets").toarray() - dif)

    def predict(
        self, logs, transport, thermal, diagnostics=False, analysis_update=None, birth=None
    ):
        m = self.model
        dt = self.dt
        put = self.put
        grid = self.grid
        change = 10**logs
        if transport.ndim == 1:
            transfer = torch.linalg.solve(
                self.ident
                - dt * (torch.exp(transport[0]) * self.circ + torch.exp(transport[1]) * self.diff),
                self.ident,
            )
        else:
            a = torch.exp(transport)[:, :, None, None]
            transfer = torch.linalg.solve(
                self.ident[None] - dt * (a[:, 0] * self.circ + a[:, 1] * self.diff),
                self.ident.expand(len(self.t), -1, -1),
            )
        T = self.T + thermal[:, None] * self.mode
        factor = self.p * 0.01801528 / (8.314462618 * T)
        c = T - 273.15
        es = 611.21 * torch.exp((18.678 - c / 234.5) * c / (257.14 + c))
        diff = 2.11e-5 * (T / 273.15) ** 1.94 * 101325 / self.p
        conduct = 0.024 * (T / 273.15) ** 0.9
        resistance = 1000 * (
            461.5 * T / (diff * es) + (2.5e6 / (461.5 * T) - 1) * 2.5e6 / (conduct * T)
        )
        pop = self.init
        gas = factor[0] * self.chi[0]
        rows = []
        budgets = []
        if birth is not None:
            from scipy.special import ndtr

            edges = np.clip(np.arange(len(m.d) + 1) * self.width, 4, 10)
            weights = np.diff(ndtr(np.log(edges / 7.0) / np.log(1.2)))
            small = put(weights / weights.sum())
            pop = pop + birth[0] * self.f * small
        for i in range(len(self.t)):
            gas = gas + factor[i] * (self.chi[i] - (self.chord @ gas) / (self.chord @ factor[i]))
            if birth is not None:
                new_count = dt * birth[1][i] * small
                pop = pop + self.f * new_count
                gas = gas - 1000 * np.pi / 6 * 1e-12 * (new_count @ m.vol)
            gain = 8e12 * ((gas / factor[i]) * self.p / es[i] - 1) / resistance[i]
            new2 = m.diam**2 + dt * gain[:, None]
            wet, uptake = remap(pop, new2, m.vol)
            vgas = gas - uptake / self.f[:, 0]
            if diagnostics:
                residual = budget_error(pop, wet, m.vol, gas, vgas, self.f[:, 0])
            conc = wet / self.f
            if logs.shape[1] == 6:
                kernel = m.base * 10 ** (logs[i] @ self.pair_weights)
            else:
                kernel = (
                    change[i, 0] * (m.base - m.kernel["midpairs"])
                    + change[i, 1] * m.kernel["midpairs"]
                )
            merged, events = m.collision(conc, kernel, dt)
            local = (merged * self.f * self.survive).reshape(*grid.shape, len(m.d))
            settled = (
                local.gather(0, self.i0) * self.w0 + local.gather(0, self.i1) * self.w1
            ).reshape(grid.n, -1)
            move = transfer if transport.ndim == 1 else transfer[i]
            moved = move @ torch.cat([settled, (vgas * self.f[:, 0])[:, None]], 1)
            after = moved[:, :-1]
            gas = moved[:, -1] / self.f[:, 0]
            rows.append(((pop + after) / 2 / self.f * self.opt).sum(0))
            if diagnostics:
                budgets.append(
                    torch.stack(
                        [
                            (after - settled).sum(0).abs().max(),
                            (gas * self.f[:, 0] - vgas * self.f[:, 0]).sum().abs(),
                            ((merged - conc) @ m.vol * self.f[:, 0]).sum().abs(),
                            (merged - conc).sum(1).add(events).abs().max(),
                            torch.minimum(vgas.min(), gas.min()),
                            after.min(),
                            residual,
                        ]
                    )
                )
            pop = analysis_update(i, after, rows) if analysis_update is not None else after
        spec = torch.stack(rows)
        return (spec, torch.stack(budgets)) if diagnostics else spec
