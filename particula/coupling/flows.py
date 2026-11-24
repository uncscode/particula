"""Flow strategies for point observations and accepted grid intervals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from particula.coupling.dispersion import _BRIGGS
from particula.coupling.prescribed import DuneFlow, Sounding


@dataclass(frozen=True)
class GridInterval:
    """Accepted forcing. Face values are outward signed m³ and mixing m³.

    air columns are u,v,w (m/s), T (K), p (Pa), vapor (kg/kg dry air).
    speed contains cell-centered reference-coordinate velocities (m/s).
    """

    start: float
    end: float
    air: np.ndarray
    transfer: np.ndarray
    mixing: np.ndarray
    speed: np.ndarray
    diffusivity: np.ndarray


class FlowStrategy:
    """A forcing provider exposing the operations its data support."""

    def interval(self, grid, start, end, mixing, *, device="cpu"):
        """Offer a grid interval or reject an incompatible provider."""
        raise TypeError("this flow has no gridded forcing")

    def trajectory(self, parcel_id):
        """Return a recorded native parcel history by stable identity."""
        raise TypeError("this flow has no recorded trajectory")


class SpatialFlow(FlowStrategy, ABC):
    """Spatial air sampling and fixed-grid face quadrature."""

    @abstractmethod
    def sample(self, xyz, time, *, device="cpu"):
        """Sample the six air fields at (n,3) points in metres and seconds."""

    def parcel_trace(self, parcel_id, xyz, time, aerosol, *, provenance):
        """Follow air with the point sampler and build a native replay trace."""
        from scipy.integrate import solve_ivp

        from particula.coupling.prescribed import Track, make_parcel_trace

        t = np.asarray(time, dtype=float)
        if (
            t.ndim != 1
            or len(t) < 2
            or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)
        ):
            raise ValueError("require increasing finite sample times")
        initial = _points(np.asarray(xyz)[None, :], t[0])[0]

        def wind(now, position):
            return self.sample(position[None, :], now)[0, :3]

        result = solve_ivp(
            wind, (t[0], t[-1]), initial, t_eval=t, rtol=1e-8, atol=1e-10
        )
        if not result.success:
            raise RuntimeError(result.message)
        track = Track(t, result.y.T, None)
        air = np.stack(
            [
                self.sample(x[None, :], now)[0, 3:]
                for now, x in zip(t, track.xyz, strict=True)
            ]
        )
        return track, make_parcel_trace(
            parcel_id, track, aerosol, provenance, *air.T
        )

    def reference_air(self, grid, coord, time, device):
        """Sample physical fields on a flat grid."""
        if grid.bed:
            raise ValueError("mapped terrain requires a matching dune provider")
        return self.sample(grid.world(coord), time, device=device)

    def interval(self, grid, start, end, mixing, *, device="cpu"):
        """Integrate steady face volume flux and mixing across one interval."""
        if not np.isfinite([start, end]).all() or not end > start:
            raise ValueError("require increasing finite seconds")
        dt = end - start
        if mixing is None:
            mixing = DirectionalMixing()
        mid = start + dt / 2
        air = self.reference_air(grid, grid.coordinates, mid, device)
        speed = air[:, :3].copy()
        if grid.bed:
            # Layers follow the prescribed dune streamlines.
            speed[:, 2] = 0
        q = np.empty((len(air), 6))
        mix = np.zeros_like(q)
        for side in range(6):
            axis = side // 2
            ref, area = grid.face_geometry(side)
            face = self.reference_air(grid, ref, mid, device)
            q[:, side] = np.einsum("ij,ij->i", face[:, :3], area) * dt
            if grid.bed and axis == 2:
                q[:, side] = 0
            neighbor = grid.neighbors[:, side]
            if grid.boundaries[side] != "open":
                q[neighbor < 0, side] = 0
            # Directional diffusion along grid lines. The transmissibility
            # uses face area projected perpendicular to the center spacing.
            k = mixing.coefficient(self, grid, ref, axis)
            width = grid.spacing[axis]
            if axis == 2:
                width = width * grid.jacobian(ref[:, 0])
            mix[:, side] = dt * k * np.abs(area[:, axis]) / width
            mix[neighbor < 0, side] = 0
        # Own each shared face once, including the periodic seam.
        for side in (1, 3, 5):
            nb = grid.neighbors[:, side]
            active = nb >= 0
            q[nb[active], side - 1] = -q[active, side]
            mix[nb[active], side - 1] = mix[active, side]
        coefficients = np.empty((len(air), 3))
        for axis in range(3):
            coefficients[:, axis] = mixing.coefficient(
                self, grid, grid.coordinates, axis
            )
        return GridInterval(start, end, air, q, mix, speed, coefficients)


def _points(xyz, time):
    points = np.asarray(xyz, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or not np.isfinite(time)
    ):
        raise ValueError("sample finite (n,3) xyz and time")
    return points


@dataclass(frozen=True)
class UniformFlow(SpatialFlow):
    """Uniform velocity (m/s), T (K), p (Pa) and dry-air vapor ratio."""

    velocity: tuple
    temperature: float
    pressure: float
    vapor: float

    def __post_init__(self):
        """Check velocity and positive thermodynamic inputs."""
        values = np.asarray(
            (*self.velocity, self.temperature, self.pressure, self.vapor),
            dtype=float,
        )
        if (
            values.shape != (6,)
            or not np.isfinite(values).all()
            or min(values[3:5]) <= 0
            or values[5] < 0
        ):
            raise ValueError("invalid uniform flow")
        object.__setattr__(self, "velocity", tuple(values[:3]))

    def sample(self, xyz, time, *, device="cpu"):
        """Broadcast uniform air to each world point."""
        xyz = _points(xyz, time)
        row = [*self.velocity, self.temperature, self.pressure, self.vapor]
        if device != "cpu":
            from particula.coupling.grid_warp import uniform_sample

            return uniform_sample(len(xyz), row, device)
        return np.tile(row, (len(xyz), 1))


@dataclass(frozen=True)
class SoundingFlow(SpatialFlow):
    """Steady horizontally uniform measured or synthetic vertical profile."""

    sounding: Sounding

    def sample(self, xyz, time, *, device="cpu"):
        """Interpolate wind and thermodynamics against geometric height."""
        xyz = _points(xyz, time)
        z = xyz[:, 2]
        return _profile(
            self.sounding, z, z, np.ones_like(z), np.zeros_like(z), device
        )


def _profile(sounding, z, eta, jac, gradient, device):
    for level in (z, eta):
        if np.any(level < sounding.height[0] - 1e-10) or np.any(
            level > sounding.height[-1] + 1e-10
        ):
            raise ValueError("grid outside sounding coverage")
    if device != "cpu":
        from particula.coupling.grid_warp import profile_sample

        return profile_sample(sounding, z, eta, jac, gradient, device)
    u = np.interp(eta, sounding.height, sounding.east) / jac
    return np.column_stack(
        (
            u,
            np.interp(eta, sounding.height, sounding.north),
            u * gradient,
            *(
                np.interp(z, sounding.height, v)
                for v in (
                    sounding.temperature,
                    sounding.pressure,
                    sounding.vapor,
                )
            ),
        )
    )


@dataclass(frozen=True)
class DuneFlowStrategy(SpatialFlow):
    """Adapt the moving-parcel dune to point and terrain-grid sampling."""

    dune: DuneFlow

    def parcel_trace(self, parcel_id, xyz, time, aerosol, *, provenance):
        """Use the terrain-coordinate integrator on continuous streamlines."""
        return self.dune.parcel_trace(
            parcel_id, xyz, time, aerosol, provenance=provenance
        )

    def sample(self, xyz, time, *, device="cpu"):
        """Sample the continuous Gaussian dune at physical xyz."""
        xyz = _points(xyz, time)
        x, z = xyz[:, 0], xyz[:, 2]
        h = self.dune.ground(x)
        jac = 1 - h / self.dune.lid
        eta = (z - h) / jac
        if np.any(eta < 0) or np.any(z > self.dune.lid):
            raise ValueError("point outside the dune layer")
        grad = -x * h / self.dune.width**2 * (1 - eta / self.dune.lid)
        return _profile(self.dune.sounding, z, eta, jac, grad, device)

    def reference_air(self, grid, coord, time, device):
        """Evaluate the same streamlines on the piecewise-linear grid bed."""
        if (
            not grid.bed
            or grid.bounds[2] != (0, self.dune.lid)
            or not np.allclose(
                grid.bed,
                self.dune.ground(grid.edges[0]),
                rtol=1e-12,
                atol=1e-14,
            )
        ):
            raise ValueError("dune grid must match bed vertices and lid")
        x, eta = coord[:, 0], coord[:, 2]
        ix = np.clip(
            np.searchsorted(grid.edges[0], x, side="right") - 1,
            0,
            grid.shape[0] - 1,
        )
        grad = (
            np.diff(grid.bed)[ix] / grid.spacing[0] * (1 - eta / self.dune.lid)
        )
        return _profile(
            self.dune.sounding,
            grid.world(coord)[:, 2],
            eta,
            grid.jacobian(x),
            grad,
            device,
        )


@dataclass(frozen=True)
class DirectionalMixing:
    """Prescribed directional K in m²/s along x, y and layer coordinates."""

    diffusivity: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        """Require three finite nonnegative coefficients."""
        k = np.array(self.diffusivity, dtype=float)
        if k.shape != (3,) or not np.isfinite(k).all() or (k < 0).any():
            raise ValueError("give three nonnegative diffusivities")
        object.__setattr__(self, "diffusivity", tuple(k))

    def coefficient(self, flow, grid, ref, axis):
        """Return K for one face direction."""
        return self.diffusivity[axis]


@dataclass(frozen=True)
class StabilityMixing:
    """Effective transverse K = U/2 d(sigma²)/dx for a flat rural plume."""

    stability: str
    source_x: float

    def coefficient(self, flow, grid, ref, axis):
        """Differentiate the Briggs open-country variance at positive fetch."""
        if (
            not isinstance(flow, UniformFlow)
            or flow.velocity[0] <= 0
            or any(flow.velocity[1:])
            or grid.bed
        ):
            raise ValueError(
                "Briggs mixing needs uniform positive x wind over flat terrain"
            )
        if (
            self.stability not in _BRIGGS
            or not np.isfinite(self.source_x)
            or self.source_x >= grid.bounds[0][0]
            or grid.boundaries[0] == "periodic"
        ):
            raise ValueError(
                "supply class A-F and a source upstream of a nonperiodic inlet"
            )
        if axis == 0:
            return 0.0
        ay, az, bz, pz = _BRIGGS[self.stability]
        a, b, p = (ay, 0.0001, -0.5) if axis == 1 else (az, bz, pz)
        x = ref[:, 0] - self.source_x
        return (
            flow.velocity[0]
            * a
            * a
            * x
            * (1 + b * x) ** (2 * p - 1)
            * (1 + (1 + p) * b * x)
        )
