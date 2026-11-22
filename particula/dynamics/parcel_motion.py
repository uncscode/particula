"""Resolved Stokes settling with event-time migration in a translating mesh."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

from particula.dynamics.parcel import RD
from particula.gas.properties.dynamic_viscosity import get_dynamic_viscosity
from particula.gpu.kernels.parcel_motion import arrivals, displace
from particula.particles.properties.settling_velocity import (
    get_particle_settling_velocity_via_system_state,
)


@dataclass(frozen=True)
class Particle:
    """Stable ID, world position (m), radius (m), density (kg/m³) and inventory.

    Count is represented number, masses are kg per particle, and charge is
    elementary charges per particle. Cell IDs are independent of storage slots.
    """

    id: str
    cell: str
    position: tuple
    radius: float
    density: float
    count: float
    masses: tuple
    charge: float

    def __post_init__(self):
        """Validate the physical inputs and retain owned values."""
        object.__setattr__(self, "position", tuple(self.position))
        object.__setattr__(self, "masses", tuple(self.masses))
        if (
            not self.id
            or len(self.position) != 3
            or not self.masses
            or not np.isfinite(
                [
                    *self.position,
                    self.radius,
                    self.density,
                    self.count,
                    *self.masses,
                    self.charge,
                ]
            ).all()
            or min(self.radius, self.density, self.count) <= 0
            or min(self.masses) < 0
            or sum(self.masses) <= 0
        ):
            raise ValueError(
                "invalid particle identity, position or extensive inventory"
            )


@dataclass(frozen=True)
class Crossing:
    """Face event at absolute seconds with its represented inventory."""

    time: float
    particle: str
    source: str
    destination: str | None
    axis: int
    kind: str
    record: Particle


@dataclass(frozen=True)
class MotionState:
    """Restart boundary with world coordinates and cumulative events."""

    time: float
    particles: tuple[Particle, ...]
    events: tuple[Crossing, ...] = ()
    geometry: str = ""
    retained: tuple = ()  # (particle ID, axis, outward sign) in wall contact

    def save(self, path):
        """Write the completed checkpoint to JSON."""
        Path(path).write_text(json.dumps(asdict(self), allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        """Read a checkpoint for validation at continuation."""
        fields = json.loads(Path(path).read_text())
        particles = tuple(Particle(**p) for p in fields["particles"])
        events = tuple(
            Crossing(**{**e, "record": Particle(**e["record"])})
            for e in fields["events"]
        )
        return cls(
            fields["time"],
            particles,
            events,
            fields["geometry"],
            tuple(map(tuple, fields.get("retained", ()))),
        )

    def rebind_geometry(
        self, cluster, *, mesh_velocity=(0.0, 0.0, 0.0), boundaries=None
    ):
        """Adopt explicitly updated bounds at a completed motion boundary.

        Each live particle must still lie in its declared cell at this time.
        Subsequent crossings use the new shared faces and boundary policies.
        """
        mesh = _vector(mesh_velocity)
        boxes = {k: np.asarray(v) for k, v in cluster.bounds.items()}
        _, digest = _boundary_rules(boxes, mesh, boundaries)
        for particle in self.particles:
            if particle.cell not in boxes:
                raise ValueError(
                    "rebound layout must contain every occupied cell"
                )
            lo, hi = boxes[particle.cell] + mesh * self.time
            x = np.asarray(particle.position)
            if np.any(x < lo) or np.any(x > hi):
                raise ValueError("updated geometry excludes a live particle")
        return replace(self, geometry=digest, retained=())


def _vector(v):
    v = np.asarray(v, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("require finite xyz velocity in m/s")
    return v


def _destination(cluster, cell, axis, sign, x, boxes, periodic=False):
    """Find an oriented face patch containing the trajectory intersection."""
    face = boxes[cell][int(sign > 0), axis]
    if periodic:
        face = (
            min(b[0, axis] for b in boxes.values())
            if sign > 0
            else max(b[1, axis] for b in boxes.values())
        )
    candidates = []
    for k, b in boxes.items():
        shared = (cell, k, axis) if sign > 0 else (k, cell, axis)
        if (not periodic and shared not in cluster.faces) or b[
            int(sign < 0), axis
        ] != face:
            continue
        others = np.arange(3) != axis
        if np.all(x[others] >= b[0, others]) and np.all(
            x[others] <= b[1, others]
        ):
            candidates.append(k)
    if periodic and not candidates:
        raise ValueError("unpaired periodic face")
    return min(candidates) if candidates else None, face


def _initial_motion(
    state, boxes, air_velocity, mesh, capacity, temperature, pressure
):
    velocity = {
        k: _vector(
            air_velocity[k] if isinstance(air_velocity, dict) else air_velocity
        )
        for k in boxes
    }
    limits = {
        k: capacity[k] if isinstance(capacity, dict) else capacity
        for k in boxes
    }
    if any(type(v) is not int or v < 0 for v in limits.values()):
        raise ValueError("capacity must be a nonnegative integer per cell")
    live = {p.id: p for p in state.particles}
    if len(live) != len(state.particles):
        raise ValueError("duplicate live particle ID")
    occupancy = Counter(p.cell for p in live.values())
    if any(k not in boxes or n > limits[k] for k, n in occupancy.items()):
        raise ValueError("initial cell or occupancy invalid")
    positions = {
        k: np.asarray(p.position) - mesh * state.time for k, p in live.items()
    }
    for k, x in positions.items():
        b = boxes[live[k].cell]
        tolerance = (
            32 * np.finfo(float).eps * np.maximum(1, np.max(np.abs(b), axis=0))
        )
        if np.any(x < b[0] - tolerance) or np.any(x > b[1] + tolerance):
            raise ValueError("particle lies outside its claimed cell")
        positions[k] = np.clip(x, *b)
    speed = {}
    viscosity = get_dynamic_viscosity(temperature)
    for k, p in live.items():
        v = float(
            get_particle_settling_velocity_via_system_state(
                p.radius, p.density, temperature, pressure
            )
        )
        # The selected Stokes-Cunningham law requires creeping particle flow.
        if 2 * p.radius * v * (pressure / (RD * temperature)) / viscosity > 0.1:
            raise ValueError(
                "terminal settling exceeds the creeping-flow range"
            )
        speed[k] = v
    return velocity, limits, live, occupancy, positions, speed


def _advance_to_face(
    keys, live, positions, boxes, v, now, finish, opts, device
):
    pos = wp.array(np.stack([positions[k] for k in keys]), **opts)
    vel = wp.array(v, **opts)
    bounds = np.stack([boxes[live[k].cell] for k in keys])
    times = wp.empty(v.shape, **opts)
    wp.launch(
        arrivals,
        v.shape,
        [
            pos,
            vel,
            wp.array(bounds[:, 0], **opts),
            wp.array(bounds[:, 1], **opts),
            float("inf"),
        ],
        [times],
        device=device,
    )
    arrivals_host = times.numpy()
    first = float(arrivals_host.min())
    remaining = finish - now
    tol = 16 * np.spacing(max(abs(now), abs(finish), 1.0))
    if first > remaining + tol or not np.isfinite(first):
        h = remaining
        hits = []
    else:
        h = min(first, remaining)
        hits = np.argwhere(arrivals_host <= first + tol).tolist()
    out = wp.empty(v.shape, **opts)
    wp.launch(displace, v.shape, [pos, vel, h], [out], device=device)
    positions.update(zip(keys, out.numpy(), strict=True))
    return finish if h == remaining else now + h, hits


def _face_rule(cluster, cell, a, sign, x, boxes, policies):
    dst, face = _destination(cluster, cell, a, sign, x, boxes)
    kind = "cross"
    if dst is None:
        kind = policies.get((cell, a, sign), "retain")
        if kind == "periodic":
            dst, face = _destination(cluster, cell, a, sign, x, boxes, True)
        elif kind == "retain":
            dst = cell
    return dst, face, kind


def _update_wall_contacts(
    blocked, record, dst, relative, position, cluster, boxes, policies
):
    """Keep a sliding contact where the destination continues its wall."""
    for axis in range(3):
        key = (record.id, axis)
        sign = blocked.get(key)
        if sign is None:
            continue
        side = int(sign > 0)
        same_wall = boxes[record.cell][side, axis] == boxes[dst][side, axis]
        if same_wall and sign * relative[axis] > 0:
            _, _, rule = _face_rule(
                cluster, dst, axis, sign, position, boxes, policies
            )
            if rule == "retain":
                continue
        del blocked[key]


def _cross_faces(
    cluster,
    hits,
    keys,
    live,
    v,
    positions,
    boxes,
    policies,
    blocked,
    occupancy,
    mesh,
    now,
    limits,
    events,
    velocity,
    speed,
):
    # Process one axis at a time. Re-locate corner hits in the new cell.
    axis = min(a for _, a in hits)
    pending = []
    next_occupancy = occupancy.copy()
    for i, a in ((i, a) for i, a in hits if a == axis):
        k = keys[i]
        p = live[k]
        sign = 1 if v[i, a] > 0 else -1
        x = positions[k]
        x[a] = boxes[p.cell][int(sign > 0), a]
        dst, face, kind = _face_rule(
            cluster, p.cell, a, sign, x, boxes, policies
        )
        if kind == "retain":
            blocked[k, a] = sign
        if dst != p.cell:
            next_occupancy[p.cell] -= 1
            if dst is not None:
                next_occupancy[dst] += 1
        record = replace(p, position=tuple(x + mesh * now))
        pending.append((k, a, dst, face, kind, record, sign))
    if any(n > limits[k] for k, n in next_occupancy.items()):
        raise ValueError("destination capacity exhausted before crossing")
    occupancy = next_occupancy
    for k, a, dst, face, kind, record, sign in pending:
        events.append(Crossing(now, k, record.cell, dst, a, kind, record))
        if dst is None:
            del live[k]
        else:
            live[k] = replace(record, cell=dst)
            if kind == "periodic":
                positions[k][a] = face
            relative = _relative_velocity(velocity[dst], mesh, speed[k])
            if sign * relative[a] < 0:
                raise ValueError(
                    "discontinuous air velocities reverse at shared face"
                )
            _update_wall_contacts(
                blocked,
                record,
                dst,
                relative,
                positions[k],
                cluster,
                boxes,
                policies,
            )
    return occupancy


def _boundary_rules(boxes, mesh, boundaries):
    policies = dict(boundaries or {})
    for (k, a, sign), mode in policies.items():
        if (
            k not in boxes
            or a not in range(3)
            or sign not in (-1, 1)
            or mode not in ("retain", "periodic", "open", "deposit")
        ):
            raise ValueError("invalid face boundary")
    config = json.dumps(
        [
            sorted((k, b.tolist()) for k, b in boxes.items()),
            mesh.tolist(),
            sorted((list(k), v) for k, v in policies.items()),
        ]
    )
    digest = hashlib.sha256(config.encode()).hexdigest()
    return policies, digest


def _relative_velocity(air, mesh, settling):
    return air - mesh - [0, 0, settling]


def _retained_contacts(retained, live, velocity, mesh, speed):
    blocked = {}
    for k, a, sign in retained:
        if k not in live or a not in range(3) or sign not in (-1, 1):
            raise ValueError("invalid retained contact in checkpoint")
        rel = _relative_velocity(velocity[live[k].cell], mesh, speed[k])[a]
        if sign * rel > 0:
            blocked[k, a] = sign
    return blocked


def advance_motion(
    cluster,
    state,
    *,
    duration,
    air_velocity,
    capacity,
    device="cpu",
    mesh_velocity=(0.0, 0.0, 0.0),
    boundaries=None,
    temperature=288.15,
    pressure=101325.0,
    gradient=False,
):
    """Advance constant air velocity per cell in a uniformly translating mesh.

    Bounds are at time zero. Boundary keys are (cell, axis, sign), with
    axis 0/1/2 and sign −1/+1. Modes are retain, periodic, open and deposit.
    Coincident arrivals are resolved by axis then stable ID. Capacity is per
    destination cell. A rejected call leaves its input checkpoint intact.
    CPU or native CUDA computes arrival times and displacements; the host
    reserves slots and records events at each crossing boundary.
    """
    if gradient:
        raise ValueError("gradients require smooth physics without migration")
    if (
        not cluster.bounds
        or not np.isfinite([state.time, duration, temperature, pressure]).all()
        or duration <= 0
        or min(temperature, pressure) <= 0
    ):
        raise ValueError(
            "require geometry, positive duration, pressure and temperature"
        )
    mesh = _vector(mesh_velocity)
    boxes = {k: np.asarray(v) for k, v in cluster.bounds.items()}
    policies, digest = _boundary_rules(boxes, mesh, boundaries)
    if state.geometry and state.geometry != digest:
        raise ValueError("restart geometry or face rules have changed")
    velocity, limits, live, occupancy, positions, speed = _initial_motion(
        state, boxes, air_velocity, mesh, capacity, temperature, pressure
    )
    events = list(state.events)
    opts = dict(dtype=wp.float64, device=device)
    now = state.time
    finish = now + duration
    blocked = _retained_contacts(state.retained, live, velocity, mesh, speed)
    # Accepted right-endpoint crossings update cell membership before restart.
    while live:
        keys = sorted(live)
        v = np.stack(
            [
                _relative_velocity(velocity[live[k].cell], mesh, speed[k])
                for k in keys
            ]
        )
        for i, k in enumerate(keys):
            for a in range(3):
                if (k, a) in blocked:
                    v[i, a] = 0
        now, hits = _advance_to_face(
            keys, live, positions, boxes, v, now, finish, opts, device
        )
        if not hits:
            break
        occupancy = _cross_faces(
            cluster,
            hits,
            keys,
            live,
            v,
            positions,
            boxes,
            policies,
            blocked,
            occupancy,
            mesh,
            now,
            limits,
            events,
            velocity,
            speed,
        )
    return MotionState(
        finish,
        tuple(
            replace(p, position=tuple(positions[k] + mesh * finish))
            for k, p in sorted(live.items())
        ),
        tuple(events),
        digest,
        tuple(
            (k, a, sign)
            for (k, a), sign in sorted(blocked.items())
            if k in live
        ),
    )
