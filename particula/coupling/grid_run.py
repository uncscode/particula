"""Conservative fixed-grid execution and completed-interval checkpoints."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import warp as wp

from particula.coupling import grid_warp as kernels
from particula.coupling.flow_replay import (
    LiveGrid,
    save_archive,
    validated_interval,
)
from particula.dynamics.parcel_motion import Particle


@dataclass(frozen=True)
class Release:
    """One stable particle record, slip (m/s) and launch interval.

    A launch interval of -1 means initially airborne. Other records reserve
    the specified particle inventory in the bed until that interval begins.
    """

    particle: Particle
    slip: float
    interval: int = -1


@dataclass(frozen=True)
class HostTransport:
    """Already transported extensive inventories at a completed host interval.

    imports and exports are nonnegative per-cell boundary totals over the
    interval. Internal redistribution belongs to the host.
    """

    start: float
    end: float
    inventories: np.ndarray
    imports: np.ndarray
    exports: np.ndarray


class GridRun:
    """Carry named inventory and stable particles across accepted intervals.

    CPU uses NumPy for conservative transport. Particle face traversal uses
    the same compiled Warp operation on CPU and CUDA. Observation, checkpoint
    and host-owned input are explicit synchronization boundaries.
    """

    def __init__(
        self,
        forcing,
        layout,
        inventory,
        *,
        releases=(),
        bed=None,
        surface=None,
        inlet=None,
        uptake=None,
        dispersion=None,
        owner="particula",
        device="cpu",
    ):
        """Bind the forcing, native inventory layout and execution device."""
        self.forcing = forcing
        self.grid = forcing.grid
        self.layout = layout
        self.device = wp.get_device(device)
        self.gpu = self.device.is_cuda
        if owner not in ("particula", "host"):
            raise ValueError("invalid field transport owner")
        self.owner = owner
        self.index = 0
        self._faulted = False
        self.dispersion = dispersion
        self.uptake = uptake
        self.lanes = uptake.bind(layout) if uptake else ()
        self.releases = tuple(releases)
        self.surface = surface
        self.manual_records = len(self.releases)
        if surface is not None:
            if owner == "host" and surface.representation == "binned":
                raise ValueError(
                    "binned surface emission requires Particula transport"
                )
            self.releases += surface.bind(forcing, layout)
        n = len(self.grid.volume)
        a = layout.validate(inventory, n)
        inlet = np.broadcast_to(
            0.0 if inlet is None else inlet, (n, 6, layout.width)
        ).copy()
        layout.validate(inlet.reshape(n * 6, -1), n * 6)
        self.inlet = inlet
        self.neighbor = self.grid.neighbors
        self.volume = self.grid.volume
        self.state = a.copy()
        self.imports = np.zeros_like(a)
        self.exports = np.zeros_like(a)
        cfg = dict(
            layout=asdict(layout),
            owner=owner,
            uptake=asdict(uptake) if uptake else None,
            releases=[asdict(r) for r in self.releases],
            forcing=forcing.identity,
        )
        if dispersion is not None:
            cfg["dispersion"] = dispersion.configuration()
        if surface is not None:
            cfg["surface"] = surface.identity
        self._init_particles(bed)
        digest = hashlib.sha256(
            json.dumps(cfg, sort_keys=True, allow_nan=False).encode()
        )
        for arr in (inlet, self.bed_residual):
            digest.update(arr.tobytes())
        self.signature = digest.hexdigest()
        if self.gpu:
            for name in ("state", "imports", "exports", "inlet", "volume"):
                setattr(self, name, self._device_array(getattr(self, name)))
            self.neighbor = self._device_array(self.neighbor, wp.int32)
            self.scratch = wp.empty_like(self.state)
        if self.releases and self.dispersion:
            self.jump_neighbors = (
                self.neighbor
                if self.gpu
                else self._device_array(self.neighbor, wp.int32)
            )

    def _device_array(self, a, dtype=wp.float64):
        return wp.array(
            np.ascontiguousarray(a), dtype=dtype, device=self.device
        )

    def _init_particles(self, bed):
        grid = self.grid
        releases = self.releases
        ids = [r.particle.id for r in releases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate particle ID")
        materials = (
            len(releases[0].particle.masses)
            if releases
            else len(bed)
            if bed is not None
            else len(self.surface.materials)
            if self.surface is not None
            else 0
        )
        if self.surface is not None and materials != len(
            self.surface.materials
        ):
            raise ValueError(
                "resolved particle and surface material order must match"
            )
        self.payload = np.zeros((len(ids), materials + 2))
        cells = []
        positions = []
        for i, r in enumerate(releases):
            p = r.particle
            if (
                type(r.interval) is not int
                or r.interval < -1
                or (
                    not isinstance(self.forcing, LiveGrid)
                    and r.interval >= len(self.forcing.frames)
                )
                or not np.isfinite(r.slip)
                or r.slip < 0
                or len(p.masses) != materials
            ):
                raise ValueError("invalid particle release")
            ref = np.array(p.position, dtype=float)
            ref[2] = grid.bounds[2][0] + (
                ref[2] - grid.ground(ref[0])
            ) / grid.jacobian(ref[0])
            cell = int(p.cell)
            if not 0 <= cell < len(grid.volume):
                raise ValueError("invalid release cell ID")
            ix = np.array(np.unravel_index(cell, grid.shape))
            lo = np.array(grid.bounds)[:, 0] + grid.spacing * ix
            if np.any(ref < lo - 1e-10) or np.any(
                ref > lo + grid.spacing + 1e-10
            ):
                raise ValueError("particle position disagrees with its cell")
            positions.append(np.clip(ref, lo, lo + grid.spacing))
            cells.append(cell)
            self.payload[i] = [
                p.count,
                p.count * p.charge,
                *np.multiply(p.count, p.masses),
            ]
        status = np.array(
            [int(r.interval < 0) for r in releases], dtype=np.int32
        )
        # The patch inventory independently funds every generated source slot.
        reserved = self.payload[: self.manual_records, 2:][
            status[: self.manual_records] == 0
        ].sum(axis=0)
        bed = np.array(reserved if bed is None else bed, dtype=float)
        if (
            bed.shape != reserved.shape
            or not np.isfinite(bed).all()
            or np.any(bed < reserved)
        ):
            raise ValueError("bed cannot supply scheduled particles")
        self.bed_residual = bed - reserved
        self.max_slip = max((r.slip for r in releases), default=0.0)
        if releases:
            self.positions = self._device_array(np.array(positions))
            self.cells = self._device_array(cells, wp.int32)
            self.status = self._device_array(status, wp.int32)
            self.event_time = self._device_array(np.full(len(ids), -1.0))
            self.slip = self._device_array([r.slip for r in releases])
            self.release = self._device_array(
                [r.interval for r in releases], wp.int32
            )
            self.p_inventory = self._device_array(self.payload)
            codes = {"closed": 0, "periodic": 1, "deposit": 2, "open": 3}
            self.bc = self._device_array(
                [codes[b] for b in grid.boundaries], wp.int32
            )
            self.jac = self._device_array(grid.jacobian(grid.coordinates[:, 0]))
            keys = self.dispersion.keys(ids) if self.dispersion else []
            self.dispersion_keys = self._device_array(keys, wp.uint64)
            if self.dispersion:
                self.edge_jac = self._device_array(grid.jacobian(grid.edges[0]))

    def advance(self, host=None):
        """Advance one accepted interval, then apply the local uptake once."""
        if self.index >= len(self.forcing.frames):
            raise ValueError("forcing exhausted")
        f = self.forcing.frames[self.index]
        return self._advance_frame(f, host)

    def advance_interval(self, frame, host=None):
        """Accept one live host interval and retain forcing for exact replay."""
        if not isinstance(self.forcing, LiveGrid):
            raise TypeError("advance_interval requires a live grid binding")
        if (
            self.index != len(self.forcing.frames)
            or frame.start != self.forcing.times[-1]
        ):
            raise ValueError(
                "live host and grid must start at the same boundary"
            )
        f = validated_interval(self.grid, frame)
        end = self._advance_frame(f, host)
        self.forcing.commit(f)
        return end

    def _advance_frame(self, f, host):
        self._require_consistent()
        if (host is not None) != (self.owner == "host"):
            raise ValueError(
                "supply transported fields exactly for a host-owned interval"
            )
        dt = f.end - f.start
        outgoing = np.maximum(f.transfer, 0) + f.mixing
        incoming = np.maximum(-f.transfer, 0) + f.mixing
        # Limit total outgoing fraction, including all directional diffusion.
        cfl = (
            np.max(outgoing.sum(axis=1) / self.grid.volume)
            if host is None
            else 0
        )
        sub, k, jumps = self._step_schedule(f, dt, cfl)
        host_data = self._validate_host(f, host) if host is not None else None
        self._faulted = True
        self._emit_binned()
        if host is not None:
            self._install_host(*host_data)
        else:
            outgoing = outgoing / sub
            incoming = incoming / sub
            if self.gpu:
                outgoing = self._device_array(outgoing)
                incoming = self._device_array(incoming)
        speed = self._device_array(f.speed) if self.releases else None
        for i in range(sub):
            if host is None:
                self._transport(outgoing, incoming)
            if self.releases:
                self._advance_particles(
                    i, f.start + i * dt / sub, dt / sub, speed, k, jumps
                )
        if self.uptake:
            self._apply_uptake(dt)
        if self.gpu:
            wp.synchronize_device(self.device)
        self._faulted = False
        self.index += 1
        return f.end

    def _step_schedule(self, f, dt, cfl):
        """Bound the outgoing fraction and stochastic first-contact step."""
        if self.releases:
            rate = np.max(np.abs(f.speed), axis=0)
            rate[2] += self.max_slip / min(
                self.grid.jacobian(self.grid.coordinates[:, 0])
            )
            cfl = max(cfl, dt * np.max(rate / self.grid.spacing))
        k = np.zeros(3)
        jumps = None
        dispersion_steps = 1
        if self.dispersion and self.releases:
            k, jumps = self.dispersion.coefficients(self.grid, f)
            if jumps is not None:
                cfl = max(cfl, np.max(jumps.sum(axis=1)))
            if np.any(k) or (jumps is not None and np.any(jumps)):
                dispersion_steps = int(np.ceil(dt / self.dispersion.max_step))
        sub = max(1, dispersion_steps, int(np.ceil(cfl / 0.8)))
        if jumps is not None:
            jumps = self._device_array(jumps / sub)
        return sub, k, jumps

    def _advance_particles(self, i, start, step_dt, speed, k, jumps):
        """Traverse physical boundaries, then sample mapped mixing faces."""
        g = self.grid
        wp.launch(
            kernels.move_particles,
            len(self.releases),
            [
                self.index,
                i,
                self.dispersion_keys,
                wp.vec3d(*k),
                start,
                step_dt,
                wp.vec3i(*g.shape),
                wp.vec3d(*np.asarray(g.bounds)[:, 0]),
                wp.vec3d(*g.spacing),
                self.bc,
                self.jac,
                speed,
                self.slip,
                self.release,
                self.positions,
                self.cells,
                self.status,
                self.event_time,
            ],
            device=self.device,
        )
        if jumps is not None:
            wp.launch(
                kernels.diffuse_cells,
                len(self.releases),
                [
                    self.index,
                    i,
                    start + step_dt,
                    self.dispersion_keys,
                    jumps,
                    self.jump_neighbors,
                    wp.vec3i(*g.shape),
                    wp.vec3d(*np.asarray(g.bounds)[:, 0]),
                    wp.vec3d(*g.spacing),
                    self.edge_jac,
                    self.bc,
                    self.positions,
                    self.cells,
                    self.status,
                    self.event_time,
                ],
                device=self.device,
            )

    def _emit_binned(self):
        if self.surface is not None and self.surface.representation == "binned":
            increment = self.surface.field_increment(
                self.index, self.layout, len(self.grid.volume)
            )
            if self.gpu:
                wp.launch(
                    kernels.accumulate,
                    self.state.shape,
                    [self.state, self._device_array(increment)],
                    device=self.device,
                )
            else:
                self.state += increment

    def _apply_uptake(self, dt):
        fraction = -np.expm1(-self.uptake.rate * dt)
        if self.gpu:
            wp.launch(
                kernels.uptake,
                len(self.grid.volume),
                [self.state, *self.lanes, fraction],
                device=self.device,
            )
        else:
            gas, num, mat = self.lanes
            amount = fraction * self.state[:, gas] * (self.state[:, num] > 0)
            self.state[:, gas] -= amount
            self.state[:, mat] += amount

    def _validate_host(self, f, h):
        if (h.start, h.end) != (f.start, f.end):
            raise ValueError("host and aerosol interval endpoints must match")
        n = len(self.grid.volume)
        a = self.layout.validate(h.inventories, n)
        imp = np.asarray(h.imports, dtype=float)
        exp = np.asarray(h.exports, dtype=float)
        if any(
            v.shape != a.shape or not np.isfinite(v).all() or np.any(v < 0)
            for v in (imp, exp)
        ):
            raise ValueError("invalid host boundary ledgers")
        previous = self._field_total(self.state)
        received = previous + imp.sum(axis=0)
        accounted = a.sum(axis=0) + exp.sum(axis=0)
        if not np.allclose(received, accounted, rtol=1e-11, atol=0):
            raise ValueError(
                "host inventories disagree with boundary transfers"
            )
        return a, imp, exp

    def _install_host(self, a, imp, exp):
        if self.gpu:
            self.state.assign(a)
            for ledger, increment in ((self.imports, imp), (self.exports, exp)):
                wp.launch(
                    kernels.accumulate,
                    ledger.shape,
                    [ledger, self._device_array(increment)],
                    device=self.device,
                )
        else:
            self.state[:] = a
            self.imports += imp
            self.exports += exp

    def _transport(self, outgoing, incoming):
        if self.gpu:
            wp.launch(
                kernels.transport,
                self.state.shape,
                [
                    self.neighbor,
                    incoming,
                    outgoing,
                    self.volume,
                    self.inlet,
                    self.state,
                    self.scratch,
                    self.imports,
                    self.exports,
                ],
                device=self.device,
            )
            self.state, self.scratch = self.scratch, self.state
            return
        c = self.state / self.volume[:, None]
        new = self.state * (1 - outgoing.sum(axis=1) / self.volume)[:, None]
        for side in range(6):
            nb = self.neighbor[:, side]
            valid = nb >= 0
            new[valid] += incoming[valid, side, None] * c[nb[valid]]
            external = ~valid
            added = incoming[external, side, None] * self.inlet[external, side]
            new[external] += added
            self.imports[external] += added
            self.exports[external] += (
                outgoing[external, side, None] * c[external]
            )
        self.state = new

    def _require_consistent(self):
        if self._faulted:
            raise RuntimeError(
                "incomplete grid step; restore a completed checkpoint"
            )

    def _history_id(self, index):
        if isinstance(self.forcing, LiveGrid) and index:
            from particula.coupling.flow_replay import GridReplay

            return GridReplay(
                self.grid, self.forcing.frames[:index], self.forcing.provenance
            ).identity
        return self.forcing.identity

    def _read(self, a):
        return a.numpy() if self.gpu else a.copy()

    def _field_total(self, a):
        if not self.gpu:
            return a.sum(axis=0)
        total = wp.zeros(
            self.layout.width, dtype=wp.float64, device=self.device
        )
        wp.launch(kernels.field_totals, a.shape, [a, total], device=self.device)
        return total.numpy()

    def budget(self):
        """Reduce fields and resolved count, charge and kg by fate."""
        self._require_consistent()
        result = {}
        for name in ("state", "imports", "exports"):
            a = getattr(self, name)
            result[name] = self._field_total(a)
        totals = np.zeros((4, self.payload.shape[1]))
        if self.releases:
            device_total = self._device_array(totals)
            wp.launch(
                kernels.particle_totals,
                self.payload.shape,
                [self.status, self.p_inventory, device_total],
                device=self.device,
            )
            totals = device_total.numpy()
        for key, row in zip(
            ("pending", "airborne", "deposited", "escaped"), totals, strict=True
        ):
            result[key] = row
        result["bed_mass"] = self.bed_residual + totals[0, 2:]
        if self.surface is not None:
            remaining = self.surface.remaining[self.index]
            result["surface_bed_kg"] = remaining.copy()
            result["surface_emitted_kg"] = self.surface.emitted[
                : self.index
            ].sum(axis=0)
            # Future resolved slots are already represented in "pending".
            free = (
                self.surface.remaining[-1]
                if self.surface.representation == "resolved"
                else remaining
            )
            result["bed_mass"] += free.sum(axis=0) @ np.array(
                [d.fractions for d in self.surface.classes]
            )
        return result

    def snapshot(self):
        """Download fields, stable IDs, xyz and event times."""
        self._require_consistent()
        data = {
            k: self._read(getattr(self, k))
            for k in ("state", "imports", "exports")
        }
        if self.releases:
            data.update(
                {
                    k: getattr(self, k).numpy().copy()
                    for k in ("positions", "cells", "status", "event_time")
                }
            )
            data["world"] = self.grid.world(data["positions"])
            data["ids"] = np.array([r.particle.id for r in self.releases])
        return data

    def checkpoint(self, path):
        """Write the state at a completed coupling boundary."""
        data = self.snapshot()
        data["bed_mass"] = self.bed_residual.copy()
        if self.releases:
            data["bed_mass"] += self.payload[data["status"] == 0, 2:].sum(
                axis=0
            )
        if self.surface is not None:
            data["surface_bed_kg"] = self.surface.remaining[self.index]
        data.pop("world", None)
        save_archive(
            path,
            dict(
                schema=1,
                dispersion=self.dispersion.configuration()
                if self.dispersion
                else None,
                index=self.index,
                signature=self.signature,
                history=self._history_id(self.index),
            ),
            **data,
        )

    def restore(self, path):
        """Restore matching state at an accepted interval boundary."""
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            idx = meta["index"]
            if (
                meta["schema"] != 1
                or meta.get("dispersion")
                != (
                    self.dispersion.configuration() if self.dispersion else None
                )
                or meta["signature"] != self.signature
                or type(idx) is not int
                or not 0 <= idx <= len(self.forcing.frames)
                or meta["history"] != self._history_id(idx)
            ):
                raise ValueError("checkpoint configuration or time mismatch")
            if self.surface is not None and not np.array_equal(
                archive["surface_bed_kg"], self.surface.remaining[idx]
            ):
                raise ValueError(
                    "checkpoint surface bed disagrees with source phase"
                )
            names = ["state", "imports", "exports"]
            if self.releases:
                names += ["positions", "cells", "status", "event_time"]
            values = {k: archive[k] for k in names}
            bed_mass = archive["bed_mass"]
            if self.releases and archive["ids"].tolist() != [
                r.particle.id for r in self.releases
            ]:
                raise ValueError("checkpoint particle IDs do not match")
        for k, v in values.items():
            if v.shape != getattr(self, k).shape or not np.isfinite(v).all():
                raise ValueError("invalid checkpoint array")
        self.layout.validate(values["state"], len(self.grid.volume))
        if np.any(values["imports"] < 0) or np.any(values["exports"] < 0):
            raise ValueError("invalid checkpoint boundary inventories")
        self._check_checkpoint_particles(values, bed_mass, idx)
        for k, v in values.items():
            a = getattr(self, k)
            if isinstance(a, np.ndarray):
                a[:] = v
            else:
                a.assign(v)
        self.index = idx
        self._faulted = False

    def _check_checkpoint_particles(self, values, bed_mass, index):
        expected = self.bed_residual.copy()
        if self.releases:
            cells, fates = values["cells"], values["status"]
            if (
                cells.dtype.kind not in "iu"
                or fates.dtype.kind not in "iu"
                or np.any((cells < 0) | (cells >= len(self.grid.volume)))
                or np.any((fates < 0) | (fates > 3))
            ):
                raise ValueError("invalid checkpoint particle state")
            pending = np.array([r.interval >= index for r in self.releases])
            if not np.array_equal(pending, fates == 0):
                raise ValueError("checkpoint disagrees with launch schedule")
            expected += self.payload[pending, 2:].sum(axis=0)
            events = values["event_time"]
            if (
                np.any(events[fates < 2] != -1)
                or np.any(events[fates >= 2] < self.forcing.times[0])
                or np.any(
                    events[fates >= 2] > self.forcing.times[index] + 1e-12
                )
            ):
                raise ValueError("invalid particle event times")
            keys = np.stack(np.unravel_index(cells, self.grid.shape), axis=1)
            low = np.asarray(self.grid.bounds)[:, 0] + keys * self.grid.spacing
            local = (values["positions"] - low) / self.grid.spacing
            if np.any(local < -1e-9) or np.any(local > 1 + 1e-9):
                raise ValueError(
                    "checkpoint positions disagree with grid cells"
                )
        if not np.array_equal(expected, bed_mass):
            raise ValueError(
                "checkpoint bed inventory disagrees with particles"
            )
