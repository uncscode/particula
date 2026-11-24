"""Finite-bed dust emission under a prescribed piecewise-linear wind."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from particula.coupling.flow_replay import GridReplay
from particula.dynamics.parcel_motion import Particle


@dataclass(frozen=True)
class DustClass:
    """Properties of an emitted dry spherical class in SI units.

    threshold is friction velocity (m/s), coefficient is kg/m²/s above
    normalized excess shear, diameter is m, density is kg/m³, slip is
    prescribed downward terminal speed (m/s), fractions are dry mass fractions.
    """

    name: str
    threshold: float
    coefficient: float
    diameter: float
    density: float
    fractions: tuple
    slip: float = 0.0

    def __post_init__(self):
        """Validate the dry mass and material partition."""
        object.__setattr__(self, "fractions", tuple(self.fractions))
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not np.isfinite(
                [
                    self.threshold,
                    self.coefficient,
                    self.diameter,
                    self.density,
                    self.slip,
                    *self.fractions,
                ]
            ).all()
            or min(self.threshold, self.diameter, self.density) <= 0
            or min(self.coefficient, self.slip, *self.fractions) < 0
            or not np.isclose(sum(self.fractions), 1.0, rtol=0.0, atol=1e-14)
        ):
            raise ValueError("invalid emitted dust class")

    @property
    def mass(self):
        """Return kg per physical grain of the emitted diameter."""
        return self.density * np.pi / 6 * self.diameter**3


@dataclass(frozen=True)
class BedPatch:
    """A bottom cell's erodible slope area (m²) and initial kg per class."""

    name: str
    cell: int
    area: float
    available: tuple

    def __post_init__(self):
        """Check patch identity, physical surface area and inventory."""
        object.__setattr__(self, "available", tuple(self.available))
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or type(self.cell) is not int
            or self.cell < 0
            or not self.available
            or not np.isfinite([self.area, *self.available]).all()
            or self.area <= 0
            or min(self.available) < 0
        ):
            raise ValueError("invalid bed patch")


def _excess_integral(u0, u1, threshold, cd):
    """Integrate positive normalized excess shear over unit linear-wind time."""
    d = u1 - u0
    a, b, c = cd * (d @ d), 2 * cd * (u0 @ d), cd * (u0 @ u0) - threshold**2
    cuts = [0.0, 1.0]
    disc = b * b - 4 * a * c
    if a and disc <= 0:
        # Convex shear stays positive except possibly at one tangent point.
        return max(0.0, c + b / 2 + a / 3) / threshold**2
    if a:
        q = -0.5 * (b + np.copysign(np.sqrt(disc), b))
        cuts += [r for r in (q / a, c / q) if 0 < r < 1]
    cuts.sort()
    result = 0.0
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
        mid = (lo + hi) / 2
        if (a * mid + b) * mid + c > 0:
            # A quadratic's exact mean equals midpoint value + curvature term.
            result += (hi - lo) * (
                (a * mid + b) * mid + c + a * (hi - lo) ** 2 / 12
            )
    return result / threshold**2


class SurfaceEmission:
    """Prescribed bed exchange integrated over accepted replay intervals.

    Each interval's integrated release enters at its left endpoint. Refining
    accepted intervals refines the resulting transport timing. Forcing and
    emission have independent outputs and share the same fixed grid and clock.
    """

    def __init__(
        self,
        forcing,
        flow,
        patches,
        classes,
        materials,
        *,
        drag_coefficient,
        reference_height,
        ejection_height,
        representation,
        capacity=None,
        device="cpu",
    ):
        """Sample wind above the bed, integrate gusts and reserve record slots.

        reference_height and ejection_height are physical metres above bed.
        capacity limits all resolved source records across the recording.
        Per-class rates use piecewise linear horizontal winds between knots.
        """
        if not isinstance(forcing, GridReplay):
            raise TypeError(
                "surface emission needs a recorded forcing schedule"
            )
        grid = forcing.grid
        self.patches, self.classes = tuple(patches), tuple(classes)
        self.materials = tuple(materials)
        for labels in (
            [p.name for p in self.patches],
            [d.name for d in self.classes],
            self.materials,
        ):
            if (
                not labels
                or len(set(labels)) != len(labels)
                or any(not isinstance(s, str) or not s.strip() for s in labels)
            ):
                raise ValueError(
                    "supply unique named patches, classes and materials"
                )
        if (
            representation not in ("binned", "resolved")
            or not np.isfinite(
                [drag_coefficient, reference_height, ejection_height]
            ).all()
            or drag_coefficient < 0
            or min(reference_height, ejection_height) <= 0
            or any(len(p.available) != len(self.classes) for p in self.patches)
            or any(
                len(d.fractions) != len(self.materials) for d in self.classes
            )
            or (
                capacity is not None
                and (type(capacity) is not int or capacity < 0)
            )
        ):
            raise ValueError("invalid surface emission configuration")
        self.representation = representation
        self.forcing_id = forcing.identity
        ref = self._reference_points(grid, ejection_height, reference_height)
        knots = forcing.times
        sequence = getattr(flow, "sequence", None)
        if sequence is not None:
            knots = np.union1d(
                knots, [t for t in sequence.times if knots[0] < t < knots[-1]]
            )
        wind = np.array(
            [flow.reference_air(grid, ref, t, device)[:, :2] for t in knots]
        )
        if not np.isfinite(wind).all():
            raise ValueError("patch winds must be finite")
        self.wind_times = knots.copy()
        self.friction_velocity = np.sqrt(drag_coefficient) * np.linalg.norm(
            wind, axis=2
        )
        self._integrate(forcing, knots, wind, drag_coefficient)
        self.records = (
            np.argwhere(self.emitted > 0)
            if representation == "resolved"
            else np.empty((0, 3), dtype=int)
        )
        if capacity is not None and len(self.records) > capacity:
            raise ValueError("insufficient resolved surface record capacity")
        config = [
            self.forcing_id,
            representation,
            list(self.materials),
            [asdict(p) for p in self.patches],
            [asdict(d) for d in self.classes],
            drag_coefficient,
            reference_height,
            ejection_height,
        ]
        digest = hashlib.sha256(
            json.dumps(config, sort_keys=True, allow_nan=False).encode()
        )
        for a in (
            self.emitted,
            self.remaining,
            self.positions,
            self.cells,
            self.wind_times,
            self.friction_velocity,
            self.records,
        ):
            digest.update(a.tobytes())
            a.setflags(write=False)
        self.identity = digest.hexdigest()

    def _reference_points(self, grid, ejection_height, reference_height):
        self.cells = np.array([p.cell for p in self.patches], dtype=int)
        if np.any(self.cells >= len(grid.centers)) or np.any(
            self.cells % grid.shape[2]
        ):
            raise ValueError("bed patches must occupy bottom cells")
        area = np.bincount(
            self.cells,
            weights=[p.area for p in self.patches],
            minlength=len(grid.volume),
        )
        face_area = np.linalg.norm(grid.face_geometry(4)[1], axis=1)
        if np.any(area > face_area * (1 + 1e-14)):
            raise ValueError("erodible area exceeds physical bed face area")
        ref = grid.coordinates[self.cells].copy()
        jac = grid.jacobian(ref[:, 0])
        if np.any(ejection_height >= jac * grid.spacing[2]) or np.any(
            reference_height > jac * (grid.bounds[2][1] - grid.bounds[2][0])
        ):
            raise ValueError(
                "ejection must be inside the bottom cell; sample below the lid"
            )
        ref[:, 2] = grid.bounds[2][0] + ejection_height / jac
        self.positions = grid.world(ref)
        ref[:, 2] = grid.bounds[2][0] + reference_height / jac
        return ref

    def _integrate(self, forcing, knots, wind, drag_coefficient):
        mass = np.array([p.available for p in self.patches], dtype=float)
        self.remaining = np.empty((len(forcing.times), *mass.shape))
        self.remaining[0] = mass
        self.emitted = np.zeros((len(forcing.frames), *mass.shape))
        for k, (start, end) in enumerate(
            zip(knots[:-1], knots[1:], strict=True)
        ):
            i = np.searchsorted(forcing.times, start, side="right") - 1
            for p, patch in enumerate(self.patches):
                for j, dust in enumerate(self.classes):
                    integral = _excess_integral(
                        wind[k, p],
                        wind[k + 1, p],
                        dust.threshold,
                        drag_coefficient,
                    )
                    dm = min(
                        mass[p, j],
                        patch.area
                        * dust.coefficient
                        * (end - start)
                        * integral,
                    )
                    mass[p, j] -= dm
                    self.emitted[i, p, j] += dm
            if end == forcing.times[i + 1]:
                self.remaining[i + 1] = mass

    def bind(self, forcing, layout):
        """Check compatibility and construct stable weighted resolved slots."""
        from particula.coupling.grid_run import Release

        if forcing.identity != self.forcing_id or (
            layout.materials and layout.materials != self.materials
        ):
            raise ValueError("surface forcing or material order mismatch")
        if self.representation == "binned" and not {
            d.name for d in self.classes
        }.issubset(layout.bins):
            raise ValueError("missing surface size class in grid bins")
        releases = []
        for i, p, j in self.records:
            d = self.classes[j]
            grain = Particle(
                id=f"surface/{i}/{p}/{j}",
                cell=str(self.cells[p]),
                position=tuple(self.positions[p]),
                radius=d.diameter / 2,
                density=d.density,
                count=self.emitted[i, p, j] / d.mass,
                masses=tuple(d.mass * np.asarray(d.fractions)),
                charge=0.0,
            )
            releases.append(Release(grain, d.slip, int(i)))
        return tuple(releases)

    def field_increment(self, index, layout, n):
        """Convert emitted kg to count and dry-material lanes."""
        a = np.zeros((n, layout.width))
        ng, nb, nm = len(layout.gases), len(layout.bins), len(layout.materials)
        for j, d in enumerate(self.classes):
            b = layout.bins.index(d.name)
            kg = self.emitted[index, :, j]
            np.add.at(a[:, ng + b], self.cells, kg / d.mass)
            start = ng + nb + b * nm
            np.add.at(
                a[:, start : start + nm], self.cells, kg[:, None] * d.fractions
            )
        return a
