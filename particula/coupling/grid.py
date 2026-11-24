"""Stationary structured finite volumes. SI lengths; C-order (x,y,layer) IDs."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class FlowGrid:
    """Uniform reference edges and optional piecewise-linear bed elevations.

    A singleton axis has the physical thickness in the supplied bounds. For a
    terrain grid the third coordinate spans zero to the level lid in metres.
    Boundaries are ordered west/east, south/north, bottom/top. Open faces
    accept prescribed inlet concentrations; closed faces have zero transfer.
    """

    shape: tuple
    bounds: tuple
    boundaries: tuple = ("closed",) * 6
    bed: tuple = ()

    def __post_init__(self):
        """Validate geometry and opposite periodic faces before allocation."""
        if len(self.shape) != 3 or any(
            type(n) is not int or n < 1 for n in self.shape
        ):
            raise ValueError("supply three positive integer cell counts")
        bounds = np.asarray(self.bounds, dtype=float)
        if (
            bounds.shape != (3, 2)
            or not np.isfinite(bounds).all()
            or (bounds[:, 1] <= bounds[:, 0]).any()
        ):
            raise ValueError("supply increasing finite xyz bounds")
        if len(self.boundaries) != 6 or any(
            b not in ("open", "closed", "periodic", "deposit")
            for b in self.boundaries
        ):
            raise ValueError("invalid face boundaries")
        for axis in range(3):
            pair = self.boundaries[2 * axis : 2 * axis + 2]
            if pair.count("periodic") == 1 or ("deposit" in pair and axis != 2):
                raise ValueError(
                    "periodicity needs a pair; deposition needs a vertical face"
                )
        bed = np.asarray(self.bed, dtype=float)
        if bed.size:
            if (
                bed.shape != (self.shape[0] + 1,)
                or not np.isfinite(bed).all()
                or bounds[2, 0] != 0
                or np.any(bed >= bounds[2, 1])
            ):
                raise ValueError(
                    "bed needs one elevation per x edge below a level lid"
                )
            if self.boundaries[0] == "periodic" and bed[0] != bed[-1]:
                raise ValueError("periodic terrain must join")
        object.__setattr__(self, "bounds", tuple(map(tuple, bounds)))
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        object.__setattr__(self, "bed", tuple(bed))

    @cached_property
    def edges(self):
        """Reference edge coordinates in metres."""
        return tuple(
            np.linspace(*b, n + 1)
            for b, n in zip(self.bounds, self.shape, strict=True)
        )

    @cached_property
    def spacing(self):
        """Reference cell widths in metres."""
        return np.array(
            [
                (b - a) / n
                for (a, b), n in zip(self.bounds, self.shape, strict=True)
            ]
        )

    @cached_property
    def coordinates(self):
        """Reference cell centers (ncell,3) in metres."""
        return np.stack(
            np.meshgrid(
                *[(e[1:] + e[:-1]) / 2 for e in self.edges], indexing="ij"
            ),
            axis=-1,
        ).reshape(-1, 3)

    def ground(self, x):
        """Piecewise-linear physical bed ordinate in metres."""
        if self.bed:
            return np.interp(x, self.edges[0], self.bed)
        return np.zeros_like(x, dtype=float) + self.bounds[2][0]

    def jacobian(self, x):
        """Ratio of physical to reference column thickness."""
        return (self.bounds[2][1] - self.ground(x)) / (
            self.bounds[2][1] - self.bounds[2][0]
        )

    def world(self, coord):
        """Convert (...,3) reference metres into world xyz."""
        out = np.array(coord, dtype=float, copy=True)
        x = out[..., 0]
        out[..., 2] = self.ground(x) + (
            out[..., 2] - self.bounds[2][0]
        ) * self.jacobian(x)
        return out

    @cached_property
    def centers(self):
        """Physical center coordinates (m), one row per cell."""
        return self.world(self.coordinates)

    @cached_property
    def volume(self):
        """Exact physical volume (m³) of each mapped cell."""
        return np.prod(self.spacing) * self.jacobian(self.coordinates[:, 0])

    @cached_property
    def neighbors(self):
        """Six neighboring row IDs; exterior faces use -1."""
        ids = np.arange(np.prod(self.shape), dtype=np.int32).reshape(self.shape)
        result = []
        for side, bc in enumerate(self.boundaries):
            axis = side // 2
            sign = 1 if side % 2 else -1
            neighbor = np.roll(ids, -sign, axis=axis)
            if bc != "periodic":
                index = [slice(None)] * 3
                index[axis] = -1 if sign == 1 else 0
                neighbor[tuple(index)] = -1
            result.append(neighbor.ravel())
        return np.stack(result, axis=1)

    def face_geometry(self, side):
        """Return reference centers and oriented physical area vectors (m²)."""
        axis = side // 2
        sign = 1 if side % 2 else -1
        ref = self.coordinates.copy()
        ref[:, axis] += sign * self.spacing[axis] / 2
        area = np.zeros_like(ref)
        dx, dy, dz = self.spacing
        jac = self.jacobian(ref[:, 0])
        if axis == 0:
            area[:, 0] = dy * dz * jac
        elif axis == 1:
            area[:, 1] = dx * dz * jac
        else:
            area[:, 2] = dx * dy
            if self.bed:
                slope = np.repeat(
                    np.diff(self.bed) / dx, self.shape[1] * self.shape[2]
                )
                area[:, 0] = (
                    -dx * dy * slope * (1 - ref[:, 2] / self.bounds[2][1])
                )
        return ref, sign * area
