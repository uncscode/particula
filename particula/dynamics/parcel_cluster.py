"""Parcel geometry and directed, conservative exchange under prescribed flow."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from particula.dynamics.parcel_trace import ParcelTrace


@dataclass(frozen=True)
class Connection:
    """Directed flow (m³/s) of organic gas, particle bins or water."""

    donor: str
    receiver: str
    flow: float
    gas: bool = True
    particles: bool = False
    water: bool = False
    active_intervals: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        """Validate half-open contact intervals in absolute seconds."""
        intervals = tuple(tuple(interval) for interval in self.active_intervals)
        last_end = -np.inf
        for a, b in intervals:
            if not np.isfinite((a, b)).all() or a >= b or a < last_end:
                raise ValueError(
                    "contact intervals must be ordered and disjoint"
                )
            last_end = b
        object.__setattr__(self, "active_intervals", intervals)


@dataclass(frozen=True)
class ParcelCluster:
    """Traces keyed by stable ID, with centers and cell bounds in metres."""

    traces: dict[str, ParcelTrace]
    centers: dict
    connections: tuple
    bounds: dict
    faces: tuple

    @property
    def geometric_volumes(self):
        """Return bounding-cell volumes in m³ for particle location.

        Prescribed gas exchange uses the thermodynamic volumes in the traces.
        """
        return {
            key: float(np.prod(np.subtract(hi, lo)))
            for key, (lo, hi) in self.bounds.items()
        }

    @classmethod
    def custom(cls, traces, *, centers, connections=(), bounds=None):
        """Declare exchange and optional boxes for geometric crossings."""
        ids = tuple(traces)
        if (
            not ids
            or set(ids) != set(centers)
            or any(traces[k].parcel_id != k for k in ids)
        ):
            raise ValueError("centers and trace IDs must match")
        xyz = {k: tuple(map(float, centers[k])) for k in ids}
        if any(len(v) != 3 or not np.isfinite(v).all() for v in xyz.values()):
            raise ValueError("require finite three-dimensional centers")
        edges = _connections(traces, connections)
        boxes, faces = _geometry(ids, xyz, bounds)
        return cls(
            MappingProxyType(dict(traces)),
            MappingProxyType(xyz),
            edges,
            MappingProxyType(boxes),
            tuple(faces),
        )

    @classmethod
    def regular(cls, traces, *, shape, widths, connections=()):
        """Build a regular grid from counts, widths and ordered traces."""
        if (
            len(shape) != 3
            or any(type(n) is not int or n <= 0 for n in shape)
            or np.prod(shape) != len(traces)
        ):
            raise ValueError("shape must contain one cell for each trace")
        widths = np.asarray(widths, dtype=float)
        if (
            widths.shape != (3,)
            or not np.isfinite(widths).all()
            or np.any(widths <= 0)
        ):
            raise ValueError("three positive physical cell widths required")
        boxes = {}
        centers = {}
        for k, index in zip(traces, np.ndindex(*shape), strict=True):
            lo = np.asarray(index) * widths
            # Compute shared coordinates identically on both sides of each face.
            hi = (np.asarray(index) + 1) * widths
            boxes[k] = (lo, hi)
            centers[k] = (lo + hi) / 2
        return cls.custom(
            traces, centers=centers, connections=connections, bounds=boxes
        )


def _connections(traces, connections):
    edges = tuple(connections)
    pairs = set()
    for edge in edges:
        pair = (edge.donor, edge.receiver)
        if (
            pair in pairs
            or any(k not in traces for k in pair)
            or pair[0] == pair[1]
            or not np.isfinite(edge.flow)
            or edge.flow < 0
        ):
            raise ValueError("invalid or duplicate directed connection")
        pairs.add(pair)
        if edge.gas or edge.particles:
            if traces[pair[0]].organic.name != traces[pair[1]].organic.name:
                raise ValueError("connected organic species must match by name")
        if edge.particles:
            a, b = (traces[k].initial for k in pair)
            if (
                not np.array_equal(a.dry_radius, b.dry_radius)
                or a.kappa != b.kappa
                or traces[pair[0]].dry_density is None
                or traces[pair[0]].dry_density != traces[pair[1]].dry_density
            ):
                raise ValueError(
                    "transport requires matching bins and dry density"
                )
    return edges


def _geometry(ids, xyz, bounds):
    boxes = {}
    faces = []
    if bounds is not None:
        if set(bounds) != set(ids):
            raise ValueError("bounds must cover all cells")
        for k in ids:
            box = np.asarray(bounds[k], dtype=float)
            if (
                box.shape != (2, 3)
                or not np.isfinite(box).all()
                or np.any(box[0] >= box[1])
                or np.any(np.asarray(xyz[k]) < box[0])
                or np.any(np.asarray(xyz[k]) > box[1])
            ):
                raise ValueError(
                    "require finite ordered bounds around cell center"
                )
            boxes[k] = tuple(map(tuple, box))
        for i, k in enumerate(ids):
            for other in ids[i + 1 :]:
                a, b = np.asarray(boxes[k]), np.asarray(boxes[other])
                overlap = np.minimum(a[1], b[1]) - np.maximum(a[0], b[0])
                if np.all(overlap > 0):
                    raise ValueError("cell interiors overlap")
                for axis in range(3):
                    if overlap[axis] == 0 and np.all(
                        np.delete(overlap, axis) > 0
                    ):
                        lo, hi = (
                            (k, other)
                            if a[1, axis] == b[0, axis]
                            else (other, k)
                        )
                        faces.append((lo, hi, axis))
    return boxes, faces


def run_cluster(
    cluster, *, max_step, device="cpu", gradient=None, humidity_amplitude=0.0
):
    """Advance all parcels across the union of their forcing records.

    A gradient tuple (donor, receiver, observation_cell[, 'gas']) selects
    a flow rate and terminal organic particle mass (or gas mass).
    ("humidity", parcel_id) observes terminal liquid in that parcel as all
    histories undergo the same fractional vapor perturbation.
    """
    from particula.dynamics._parcel_experiment import integrate

    if gradient and any(edge.active_intervals for edge in cluster.connections):
        raise ValueError("gradients require fixed connectivity")
    humidity_cell = None
    if (
        isinstance(gradient, tuple)
        and len(gradient) == 2
        and gradient[0] == "humidity"
    ):
        humidity_cell = gradient[1]
        if humidity_cell not in cluster.traces:
            raise ValueError("unknown humidity observation parcel")
        gradient = None
    if not cluster.connections and gradient is None:
        return _independent_groups(
            cluster, max_step, device, humidity_amplitude, humidity_cell
        )
    return integrate(
        cluster.traces,
        connections=cluster.connections,
        max_step=max_step,
        device=device,
        humidity_target=humidity_cell,
        flow_target=gradient,
        amplitude=humidity_amplitude,
    )


def _independent_groups(cluster, max_step, device, amplitude, humidity_cell):
    """Batch common forcing schedules and preserve independent step meshes."""
    from particula.dynamics._parcel_experiment import ClusterResult, integrate

    groups = {}
    for key, trace in cluster.traces.items():
        groups.setdefault(trace.time.tobytes(), {})[key] = trace
    results = []
    for traces in groups.values():
        results.append(
            integrate(
                traces,
                max_step=max_step,
                device=device,
                amplitude=amplitude,
                humidity_target=humidity_cell
                if humidity_cell in traces
                else None,
            )
        )
    if len(results) == 1:
        return results[0]
    if any(
        r.time[0] != results[0].time[0] or r.time[-1] != results[0].time[-1]
        for r in results
    ):
        raise ValueError(
            "all parcel histories must cover the common experiment duration"
        )
    common = results[0].time
    for r in results[1:]:
        common = np.intersect1d(common, r.time)
    totals = {
        k: sum(r.totals[k][np.searchsorted(r.time, common)] for r in results)
        for k in results[0].totals
    }
    return ClusterResult(
        common,
        {k: p for r in results for k, p in r.parcels.items()},
        {},
        {},
        totals,
        np.unique(np.concatenate([r.schedule for r in results])),
        None,
    )
