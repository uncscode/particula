"""Time-dependent profile forcing for moving air and fixed finite volumes."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

from particula.coupling.flows import DuneFlowStrategy, SoundingFlow, SpatialFlow
from particula.coupling.prescribed import SOUNDING_COLUMNS, DuneFlow, Sounding


@dataclass(frozen=True)
class SoundingSequence:
    """Profiles on a common height datum, timed in seconds since time_origin.

    Profiles may have different vertical sampling. All queries use the
    intersection of their height ranges. The caller supplies the common
    height datum and measurement or synthetic origin in the provenance.
    """

    times: tuple
    profiles: tuple
    time_origin: str
    provenance: str

    def __post_init__(self):
        """Check that every profile shares a nonempty height/time domain."""
        times = np.asarray(self.times, dtype=float)
        profiles = tuple(self.profiles)
        if (
            times.ndim != 1
            or len(times) < 2
            or len(times) != len(profiles)
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError(
                "supply at least two ordered finite sounding times"
            )
        if not all(isinstance(p, Sounding) for p in profiles):
            raise TypeError("each profile must be a normalized SI Sounding")
        if not all(
            isinstance(s, str) and s.strip()
            for s in (self.time_origin, self.provenance)
        ):
            raise ValueError("supply the time origin and profile provenance")
        lo = max(p.height[0] for p in profiles)
        hi = min(p.height[-1] for p in profiles)
        if lo >= hi:
            raise ValueError("sounding profiles need overlapping height ranges")
        # The union of vertical knots preserves every piecewise linear segment.
        heights = np.unique(np.concatenate([p.height for p in profiles]))
        heights = heights[(heights >= lo) & (heights <= hi)]
        aligned = tuple(
            Sounding(
                heights,
                *(
                    np.interp(heights, p.height, getattr(p, k))
                    for k in (
                        "temperature",
                        "pressure",
                        "vapor",
                        "east",
                        "north",
                    )
                ),
            )
            for p in profiles
        )
        object.__setattr__(self, "profiles", aligned)
        object.__setattr__(self, "times", tuple(times))

    @classmethod
    def from_csv(cls, path, *, time_origin, provenance):
        """Read profiles grouped by increasing time_s using named SI columns."""
        table = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
        if not {"time_s", *SOUNDING_COLUMNS}.issubset(table.dtype.names or ()):
            raise ValueError("missing time_s or required SI sounding columns")
        times = table["time_s"]
        if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
            raise ValueError("order sounding rows by finite time then height")
        knots = np.unique(times)
        profiles = tuple(
            Sounding(*(block[k] for k in SOUNDING_COLUMNS))
            for block in np.split(table, np.searchsorted(times, knots[1:]))
        )
        return cls(tuple(knots), profiles, time_origin, provenance)

    def bracket(self, time):
        """Return the lower interval index, including at the last record."""
        if not np.isfinite(time) or not self.times[0] <= time <= self.times[-1]:
            raise ValueError("time outside sounding sequence")
        return min(
            int(np.searchsorted(self.times, time, side="right")) - 1,
            len(self.times) - 2,
        )

    def at(self, time):
        """Return the interpolated SI column at one time in seconds."""
        i = self.bracket(time)
        a, b = self.profiles[i : i + 2]
        if time == self.times[i]:
            return a
        if time == self.times[i + 1]:
            return b
        w = (time - self.times[i]) / (self.times[i + 1] - self.times[i])
        return Sounding(
            a.height,
            *(
                getattr(a, k) * (1 - w) + getattr(b, k) * w
                for k in ("temperature", "pressure", "vapor", "east", "north")
            ),
        )


@dataclass(frozen=True)
class ChangingSoundingFlow(SpatialFlow):
    """Interpolate atmospheric profiles over level ground or a fixed dune."""

    sequence: SoundingSequence
    terrain: tuple | None = None  # crest, Gaussian width, level lid, all m

    def __post_init__(self):
        """Check that the common profile spans the entire prescribed dune."""
        if not isinstance(self.sequence, SoundingSequence):
            raise TypeError("supply a SoundingSequence")
        if self.terrain is not None:
            self._provider(self.sequence.profiles[0])

    def _provider(self, profile):
        if self.terrain is None:
            return SoundingFlow(profile)
        return DuneFlowStrategy(DuneFlow(profile, *self.terrain))

    def sample(self, xyz, time, *, device="cpu"):
        """Sample world positions within the common profile coverage."""
        return self._provider(self.sequence.at(time)).sample(
            xyz, time, device=device
        )

    def reference_air(self, grid, coord, time, device):
        """Sample pointwise atmosphere on the fixed grid geometry."""
        return self._provider(self.sequence.at(time)).reference_air(
            grid, coord, time, device
        )

    def parcel_trace(self, parcel_id, xyz, time, aerosol, *, provenance):
        """Integrate each linear time segment and retain the profile knots."""
        from particula.coupling.prescribed import Track, make_parcel_trace

        time = np.asarray(time, dtype=float)
        if (
            time.ndim != 1
            or len(time) < 2
            or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0)
        ):
            raise ValueError("supply increasing finite sample times")
        self.sequence.bracket(time[0])
        self.sequence.bracket(time[-1])
        knots = np.array(
            [
                time[0],
                *[s for s in self.sequence.times if time[0] < s < time[-1]],
                time[-1],
            ]
        )
        samples = np.union1d(time, knots)
        initial = np.asarray(xyz, dtype=float)
        if initial.shape != (3,) or not np.isfinite(initial).all():
            raise ValueError("give a finite world position")
        # The mapped coordinate eta is constant for an air center. Evaluating
        # above the analytical bed in eta also resolves near-surface tracks.
        if self.terrain is not None:
            dune = self._provider(self.sequence.profiles[0]).dune
            bed = dune.ground(initial[0])
            eta = (initial[2] - bed) / (1 - bed / dune.lid)
            if not 0 <= eta <= dune.lid:
                raise ValueError("initial position outside the dune layer")
        else:
            eta = initial[2]

        # Along a streamline J(x) dx/dt = U(eta,t). Integrating J once
        # separates terrain geometry from the time integral of the wind.
        wind = np.array(
            [
                [np.interp(eta, p.height, field) for field in (p.east, p.north)]
                for p in self.sequence.profiles
            ]
        )
        self.sequence.profiles[0].thermodynamics(eta)
        velocity = np.column_stack(
            [np.interp(samples, self.sequence.times, field) for field in wind.T]
        )
        increments = (
            0.5 * (velocity[:-1] + velocity[1:]) * np.diff(samples)[:, None]
        )
        xy = initial[:2] + np.vstack(
            (np.zeros(2), np.cumsum(increments, axis=0))
        )
        z = np.full(len(xy), eta)
        if self.terrain is not None:
            width = dune.width * np.sqrt(2)
            amplitude = dune.height / dune.lid * dune.width * np.sqrt(np.pi / 2)

            def primitive(x):
                return x - amplitude * erf(x / width)

            targets = primitive(initial[0]) + xy[1:, 0] - initial[0]
            if amplitude > 0:
                xy[1:, 0] = [
                    brentq(
                        lambda x, goal=target: primitive(x) - goal,
                        np.nextafter(target - amplitude, -np.inf),
                        np.nextafter(target + amplitude, np.inf),
                    )
                    for target in targets
                ]
            bed = dune.ground(xy[:, 0])
            z = bed + (1 - bed / dune.lid) * eta
        track = Track(samples, np.column_stack((xy, z)), None)
        fields = np.array(
            [
                self.sequence.at(t).thermodynamics(h)
                for t, h in zip(samples, z, strict=True)
            ]
        )
        return track, make_parcel_trace(
            parcel_id,
            track,
            aerosol,
            "; ".join(
                (
                    provenance,
                    self.sequence.time_origin,
                    self.sequence.provenance,
                )
            ),
            *fields.T,
        )

    def interval(self, grid, start, end, mixing=None, *, device="cpu"):
        """Split at knots and face reversals before integrating linear flux."""
        i = self.sequence.bracket(start)
        self.sequence.bracket(end)
        if end <= start:
            raise ValueError("require increasing finite seconds")
        a, b = self.sequence.times[i : i + 2]
        # For flat and terrain-following grids the horizontal volume factors
        # are positive. The zero of a face rate is therefore the zero of its
        # upstream component at the layer center. Layer-normal flux is zero.
        levels = (grid.edges[2][:-1] + grid.edges[2][1:]) / 2
        stop = min(b, end)
        for axis, field in enumerate(("east", "north")):
            sides = grid.boundaries[2 * axis : 2 * axis + 2]
            if grid.shape[axis] == 1 and all(
                mode in ("closed", "deposit") for mode in sides
            ):
                continue
            left, right = (
                np.interp(levels, p.height, getattr(p, field))
                for p in self.sequence.profiles[i : i + 2]
            )
            flip = ((left < 0) & (right > 0)) | ((left > 0) & (right < 0))
            roots = a + (b - a) * (left[flip] / (left[flip] - right[flip]))
            stop = min(
                stop, float(np.min(roots[roots > start], initial=np.inf))
            )
        provider = self._provider(self.sequence.at(start + (stop - start) / 2))
        return provider.interval(grid, start, stop, mixing, device=device)
