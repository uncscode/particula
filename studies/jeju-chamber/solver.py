"""Six air histories with per-cell particle and vapor transport. Numerical arrays use chamber-condition concentrations."""

import flow as flow_module
import numpy as np
import precise
import warp as wp
from microphysics import MODES, solve
from native import PRESSURES, TEMPS, make_tables
from observations import extract, forcing
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtri
from settling import floor_flux, sediment

FAMILIES = {
    "previous": (0, 0, 0, 0),
    "native_only": (1, 0, 0, 0),
    "native_no_collection": (1, 0, 0, 0),
    "native_brownian": (1, 1, 0, 0),
    "native_vdw": (1, 2, 0, 0),
    "native_settling_collisions": (1, 0, 1, 0),
    "native_shear": (1, 0, 0, 1),
    "native_mechanical": (1, 0, 1, 1),
    "native_brownian_full": (1, 1, 1, 1),
    "native_full": (1, 2, 1, 1),
    "native_zero_hamaker_full": (1, 3, 1, 1),
    "legacy_wall_full": (0, 2, 1, 1),
}
LEGACY = {"previous", "native_only"}


class Experiment:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.runs = extract()
        self.force = forcing(self.runs)
        self.d = self.runs[0]["diameter"]
        self.edges = self.runs[0]["edges"]
        self.nb = len(self.d)
        self.nmin = self.force.shape[1] // 60
        self.obs = np.full((6, self.nmin, self.nb), np.nan)
        self.cpi = np.zeros((6, self.nmin, 8))
        self.vapor = np.full((6, self.nmin), np.nan)
        for i, r in enumerate(self.runs):
            n = len(r["observed"])
            self.obs[i, :n] = r["observed"]
            self.cpi[i, :n] = r["cpi"]
            a = r["vapor"].reshape(-1, 60)
            v = np.sum(np.isfinite(a), axis=1)
            self.vapor[i, :n] = np.where(
                v >= 55, np.nansum(a, axis=1) / np.maximum(v, 1), np.nan
            )
        self.valid = np.all(np.isfinite(self.obs), axis=2)
        self.initial = self.arr(np.array([r["initial"] for r in self.runs]))
        self.d_device = self.arr(self.d)
        self.ff = self.arr(self.force)

    def arr(self, a, dtype=wp.float32):
        return wp.array(np.asarray(a), dtype=dtype, device=self.device)


def brackets(value, nodes):
    lookup = np.interp(value, nodes, np.arange(len(nodes)))
    ix = min(int(lookup), len(nodes) - 2)
    return ix, lookup - ix


class Fit(Experiment):
    def __init__(self, cache, device="cuda:0", bins=36):
        super().__init__(device)
        cache.mkdir(parents=True, exist_ok=True)
        self.bins = bins
        wall, thermal, nodes = make_tables(self, bins, cache)
        self.wall = self.arr(wall)
        self.lower = np.log(nodes[0])
        self.spacing = np.log(nodes[1] / nodes[0])
        nn = thermal.shape[-1]
        lookup = np.empty(
            (6 * self.nmin, thermal.shape[1], nn, nn), dtype=np.float32
        )
        for e, r in enumerate(self.runs):
            for m in range(self.nmin):
                i = min(60 * m + 30, r["valid_stop"])
                ti, tf = brackets(r["temp"][i, 0], TEMPS)
                pi, pf = brackets(r["pressure"][i], PRESSURES)
                a = ti * len(PRESSURES) + pi
                b = a + len(PRESSURES)
                lookup[e * self.nmin + m] = (1.0 - tf) * (
                    (1.0 - pf) * thermal[a] + pf * thermal[a + 1]
                ) + tf * ((1.0 - pf) * thermal[b] + pf * thermal[b + 1])
        self.thermal = self.arr(lookup)


class Circulation(Fit):
    """Run local microphysics and the prescribed conservative exchange."""

    def __init__(self, cache, device="cuda:0", bins=36):
        super().__init__(cache, device=device, bins=bins)
        self.step = solve

    def parcels(
        self,
        params,
        flow,
        grid,
        physics="previous",
        dt=5,
        nc=20,
        ni=54,
        ns=12,
        interval=20,
        floor=False,
        step_inputs=(),
        mixing_matrices=None,
        state_edit=None,
    ):
        """Return local observables, retained ledgers, and grid transport."""
        p = np.atleast_2d(params)
        flow = np.atleast_2d(flow)
        assert len(p) == len(flow) and np.isfinite(flow).all()
        assert interval % dt == 0 and 60 % dt == 0
        mix_particles, mix_air = (
            (precise.particle_mix, precise.air_mix)
            if dt == 1
            else (flow_module.mix_particles, flow_module.mix_air)
        )
        old_flags = list(MODES["full"])
        if physics not in LEGACY:
            old_flags[6:8] = 0, 0
        flags = self.arr(
            np.tile(old_flags + list(FAMILIES[physics]), (len(p), 1)), wp.int32
        )
        nprod = self.bins if any(FAMILIES[physics][1:]) else 0
        nl = nc + ns + self.nb + nprod
        if state_edit is not None:
            nl += state_edit.extra_slots
        width = max(nl, ni)
        cells = grid.size
        n = len(p) * 6 * cells
        parameters = self.arr(p)
        reference = parameters
        if floor:
            if physics not in ("previous", "legacy_wall_full"):
                raise ValueError(
                    "Explicit floor transport requires the legacy wall law."
                )
            p = p.copy()
            p[:, [10, 22]] = 0.0
            parameters = self.arr(p)
        config = self.arr(flow)
        layout = self.arr(np.stack((grid.thermal, grid.source), axis=-1))
        matrices = (
            np.stack(
                [
                    grid.matrix(u, diffusivity, dt)
                    for u, diffusivity, _, _ in flow
                ]
            )
            if mixing_matrices is None
            else mixing_matrices
        )
        assert matrices.min() >= -1e-14
        np.testing.assert_allclose(matrices.sum(axis=-2), 1.0, atol=1e-12)
        np.testing.assert_allclose(matrices.sum(axis=-1), 1.0, atol=1e-12)
        if matrices.ndim == 5:
            minute_weights = [
                self.arr(m.reshape(-1, cells, cells)) for m in matrices
            ]
        else:
            minute_weights = None
            weights = self.arr(matrices)
        buffer = dict(device=self.device, dtype=float)
        a, b = [wp.zeros((n, 8, width), **buffer) for _ in range(2)]
        current, next_history = [wp.zeros((n, 8), **buffer) for _ in range(2)]
        out = wp.zeros((n, self.nmin, 4 * self.nb), **buffer)
        diag = wp.zeros((n, self.nmin, 24), **buffer)
        book = wp.zeros((n, 19), **buffer)
        scratch = wp.zeros((n, 10, nl + ni if nprod else 1), **buffer)
        ground = wp.zeros((n, 2), **buffer) if floor else None
        prefix = [
            parameters,
            flags,
            self.ff,
            self.initial,
            self.d_device,
            self.arr(ndtri((np.arange(nc) + 0.5) / nc)),
            self.arr(ndtri((np.arange(ns) + 0.5) / ns)),
            self.wall,
            self.thermal,
            scratch,
            self.bins,
            nprod,
            interval,
            self.lower,
            self.spacing,
        ]
        total = int(max(r["valid_stop"] for r in self.runs)) + 1
        for k in range(0, total, dt):
            if state_edit is not None and k in state_edit.seconds:
                state_edit.apply(k, a, current, book, p, grid, nc, ni, nl)
            args = prefix + [
                a,
                out,
                diag,
                book,
                dt,
                k,
                cells,
                config,
                layout,
                current,
                ni,
                self.nb,
                self.edges[0],
                float(np.mean(np.diff(np.log(self.edges)))),
            ]
            wp.launch(
                self.step, n, args + list(step_inputs), device=self.device
            )
            if cells > 1:
                if minute_weights is not None:
                    weights = minute_weights[k // 60]
                wp.launch(
                    mix_particles,
                    (n, width),
                    [a, b, weights, nl, ni, cells],
                    device=self.device,
                )
                # The local cumulative water source is a ledger, not a tracer.
                wp.launch(
                    mix_air,
                    (n, 8),
                    [current, next_history, weights, cells],
                    device=self.device,
                )
                a, b = b, a
                current, next_history = next_history, current
            if floor:
                settling = [
                    reference,
                    self.ff,
                    k,
                    dt,
                    grid.nx,
                    grid.nz,
                    nl,
                    ni,
                    grid.dz,
                ]
                wp.launch(
                    floor_flux,
                    n,
                    [a, book, ground] + settling,
                    device=self.device,
                )
                wp.launch(
                    sediment, (n, width), [a, b] + settling, device=self.device
                )
                a, b = b, a
        shape = len(p), 6, cells
        y = out.numpy().reshape(*shape, self.nmin, 4, self.nb)
        x = diag.numpy().reshape(*shape, self.nmin, 24)
        budget = book.numpy().reshape(*shape, 19)
        if floor:
            budget = np.concatenate(
                (budget, ground.numpy().reshape(*shape, 2)),
                axis=-1,
            )
        assert np.isfinite(y).all() and y.min() >= 0
        assert np.isfinite(x).all() and np.isfinite(budget).all()
        return gaussian_filter1d(y, 0.6, axis=-1, mode="reflect"), x, budget


def instrument(grid, y, x, location="port"):
    """Sample Welas/CPI at the underside and use volume-mean vapor."""
    if location == "volume":
        weights = np.full(grid.size, 1 / grid.size)
    elif location == "bottom":
        weights = grid.bottom
    else:
        weights = grid.port
    volume = np.einsum("b,crbmt->crmt", np.full(grid.size, 1 / grid.size), x)
    sampled = np.einsum("b,crbmt->crmt", weights, x)
    sampled[..., 16:] = volume[..., 16:]
    return np.einsum("b,crbmfd->crmfd", weights, y), sampled
