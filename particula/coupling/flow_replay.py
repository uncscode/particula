"""Accepted grid intervals and native trajectory replay, with SI metadata."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from particula.coupling.flows import (
    FlowStrategy,
    GridInterval,
)
from particula.coupling.grid import FlowGrid
from particula.dynamics.parcel_trace import ParcelTrace

_UNITS = {
    "air": ["m/s", "m/s", "m/s", "K", "Pa", "kg/kg dry air"],
    "time": "s",
    "coordinates": "m",
    "transfer": "m3 outward",
    "mixing": "m3",
    "diffusivity": "m2/s along grid lines",
    "speed": "m/s reference",
}


def save_archive(path, metadata, **arrays):
    """Atomically replace an archive after all arrays have been written."""
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            np.savez_compressed(
                f, metadata=json.dumps(metadata, allow_nan=False), **arrays
            )
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def frozen_array(value, shape):
    """Validate, copy and freeze a finite float64 array."""
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"expected finite data with shape {shape}")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class TrajectoryReplay(FlowStrategy):
    """Offer one named trajectory using the existing CSV/JSON record."""

    trace: ParcelTrace

    def trajectory(self, parcel_id):
        """Retrieve the recorded forcing without spatial extrapolation."""
        if parcel_id != self.trace.parcel_id:
            raise ValueError("unknown trajectory identity")
        return self.trace


def validated_interval(grid, f):
    """Own one accepted interval and check volume and mixing conservation."""
    n = len(grid.volume)
    if not np.isfinite([f.start, f.end]).all() or f.end <= f.start:
        raise ValueError("invalid accepted times")
    air = frozen_array(f.air, (n, 6))
    speed = frozen_array(f.speed, (n, 3))
    transfer = frozen_array(f.transfer, (n, 6))
    mix = frozen_array(f.mixing, (n, 6))
    diffusivity = frozen_array(f.diffusivity, (n, 3))
    if np.any(diffusivity < 0):
        raise ValueError("diffusivity must be nonnegative")
    if (air[:, 3:5] <= 0).any() or (air[:, 5] < 0).any() or (mix < 0).any():
        raise ValueError("invalid air or mixing fields")
    for side in range(6):
        nb = grid.neighbors[:, side]
        inside = nb >= 0
        if not np.array_equal(
            transfer[inside, side], -transfer[nb[inside], side ^ 1]
        ) or not np.array_equal(mix[inside, side], mix[nb[inside], side ^ 1]):
            raise ValueError("shared face transfers must agree")
        if np.any(mix[~inside, side]) or (
            grid.boundaries[side] != "open" and np.any(transfer[~inside, side])
        ):
            raise ValueError("invalid exterior transfers")
    return GridInterval(f.start, f.end, air, transfer, mix, speed, diffusivity)


class GridReplay(FlowStrategy):
    """Immutable forcing with fixed geometry and exact interval transfers."""

    def __init__(self, grid, frames, provenance):
        """Validate field alignment, shared faces and accepted times."""
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError("supply grid forcing provenance")
        self.grid = grid
        self.provenance = provenance
        self.frames = tuple(validated_interval(grid, f) for f in frames)
        if not self.frames or any(
            a.end != b.start
            for a, b in zip(self.frames[:-1], self.frames[1:], strict=True)
        ):
            raise ValueError("grid forcing times must be contiguous")
        self.times = np.array(
            [self.frames[0].start, *[f.end for f in self.frames]],
            dtype=np.float64,
        )
        self.times.setflags(write=False)
        digest = hashlib.sha256(
            json.dumps(asdict(grid), sort_keys=True).encode()
        )
        digest.update(self.times.tobytes())
        for frame in self.frames:
            for key in ("air", "speed", "transfer", "mixing", "diffusivity"):
                digest.update(getattr(frame, key).tobytes())
        self.identity = digest.hexdigest()

    @classmethod
    def record(
        cls, grid, provider, times, mixing=None, *, provenance, device="cpu"
    ):
        """Record forcing through each endpoint, honoring short advances."""
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or len(times) < 2
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("supply increasing finite coupling times")

        def accepted():
            start = times[0]
            for stop in times[1:]:
                while start < stop:
                    f = provider.interval(
                        grid, start, stop, mixing, device=device
                    )
                    if f.start != start or not start < f.end <= stop:
                        raise ValueError(
                            "provider returned an invalid accepted interval"
                        )
                    yield f
                    start = f.end

        return cls(grid, accepted(), provenance)

    def interval(self, grid, start, end, mixing=None, *, device="cpu"):
        """Return the exact next recorded interval on matching geometry."""
        k = np.searchsorted(self.times, start)
        if (
            not np.isfinite([start, end]).all()
            or mixing is not None
            or grid != self.grid
            or k >= len(self.frames)
            or start != self.times[k]
            or end < self.times[k + 1]
        ):
            raise ValueError(
                "replay needs matching geometry, mixing and interval endpoints"
            )
        return self.frames[k]

    def save(self, path):
        """Write the forcing, geometry, conventions and provenance."""
        meta = dict(
            schema=1,
            grid=asdict(self.grid),
            provenance=self.provenance,
            units=_UNITS,
            cell_id="C-order xyz",
            ownership="provider owns air and face forcing",
        )
        save_archive(
            path,
            meta,
            time=self.times,
            **{
                k: np.stack([getattr(f, k) for f in self.frames])
                for k in ("air", "speed", "transfer", "mixing", "diffusivity")
            },
        )

    @classmethod
    def load(cls, path):
        """Read versioned numerical data without pickle."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if (
                meta["schema"] != 1
                or meta["units"] != _UNITS
                or meta["ownership"] != "provider owns air and face forcing"
                or meta["cell_id"] != "C-order xyz"
            ):
                raise ValueError("unsupported grid forcing conventions")
            grid = FlowGrid(**meta["grid"])
            t = data["time"]
            arrays = [
                data[k]
                for k in ("air", "transfer", "mixing", "speed", "diffusivity")
            ]
            if any(len(a) != len(t) - 1 for a in arrays):
                raise ValueError("misaligned forcing archive")
            frames = [
                GridInterval(t[i], t[i + 1], *(a[i] for a in arrays))
                for i in range(len(t) - 1)
            ]
            return cls(grid, frames, meta["provenance"])


class LiveGrid:
    """A host-driven grid binding, retaining accepted forcing for replay.

    Supply a matched recorded prefix when reopening a host restart. The source
    ID identifies the host experiment and its authoritative field schema.
    """

    def __init__(self, grid, start, source_id, history=None):
        """Bind geometry and optionally restore its accepted history."""
        if (
            not np.isfinite(start)
            or not isinstance(source_id, str)
            or not source_id.strip()
        ):
            raise ValueError("supply finite initial time and a stable host ID")
        self.grid = grid
        self.provenance = source_id
        self.start = float(start)
        digest = hashlib.sha256(
            json.dumps(asdict(grid), sort_keys=True).encode()
        )
        digest.update(json.dumps((self.start, source_id)).encode())
        self.identity = digest.hexdigest()
        if history is not None and (
            history.grid != grid
            or history.times[0] != start
            or history.provenance != source_id
        ):
            raise ValueError("host restart history disagrees with binding")
        self.frames = list(history.frames) if history else []
        self.times = [self.start, *[f.end for f in self.frames]]

    def recording(self):
        """Export the accepted history through the standard replay path."""
        return GridReplay(self.grid, self.frames, self.provenance)

    def commit(self, frame):
        """Retain the accepted immutable forcing after a successful step."""
        self.frames.append(frame)
        self.times.append(frame.end)
