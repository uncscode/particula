"""One-way, terrain-following air and settling-particle trajectories.

Map upstream streamline height eta to z = h(x) + J(x) eta with
J = 1 - h/lid. Conserving horizontal volume flux gives u = U(eta)/J.
The fixed lid and Gaussian ridge bound a kinematic, attached flow.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from particula.dynamics.parcel import RD, RV, ParcelInitialState, _initial_radii
from particula.dynamics.parcel_trace import ParcelTrace

SOUNDING_COLUMNS = (
    "height_m",
    "temperature_K",
    "pressure_Pa",
    "vapor_kg_per_kg_dry_air",
    "east_m_s",
    "north_m_s",
)


@dataclass(frozen=True)
class Sounding:
    """Height above a common datum (m), T (K), p (Pa), vapor (kg/kg dry air)
    and east/north wind components (m/s) at each level.
    """

    height: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    vapor: np.ndarray
    east: np.ndarray
    north: np.ndarray

    def __post_init__(self):
        """Own a finite sorted column and reject invalid thermodynamics."""
        for name in self.__dataclass_fields__:
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                a.ndim != 1
                or a.shape != np.shape(self.height)
                or not np.isfinite(a).all()
            ):
                raise ValueError("sounding needs aligned finite columns")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if len(self.height) < 2 or np.any(np.diff(self.height) <= 0):
            raise ValueError("sounding heights must increase")
        if (
            np.any(self.temperature <= 0)
            or np.any(self.pressure <= 0)
            or np.any(self.vapor < 0)
        ):
            raise ValueError("invalid sounding temperature, pressure or vapor")

    @classmethod
    def from_csv(cls, path):
        """Read a normalized radiosonde table with named SI columns."""
        table = np.genfromtxt(path, delimiter=",", names=True, ndmin=1)
        if not set(SOUNDING_COLUMNS).issubset(table.dtype.names or ()):
            raise ValueError("missing required SI sounding columns")
        return cls(*(table[name] for name in SOUNDING_COLUMNS))

    def thermodynamics(self, z):
        """Interpolate T, p and vapor at geometric heights in metres."""
        if (
            not np.isfinite(z).all()
            or np.any(np.asarray(z) < self.height[0])
            or np.any(np.asarray(z) > self.height[-1])
        ):
            raise ValueError("height outside sounding")
        return tuple(
            np.interp(z, self.height, field)
            for field in (self.temperature, self.pressure, self.vapor)
        )


@dataclass(frozen=True)
class Track:
    """World positions (m), sample times (s), and optional impact time (s)."""

    time: np.ndarray
    xyz: np.ndarray
    deposition_time: float | None


@dataclass(frozen=True)
class DuneFlow:
    """Gaussian ridge centered at x=0 and uniform along y, in metres."""

    sounding: Sounding
    height: float
    width: float  # Gaussian standard deviation
    lid: float

    def __post_init__(self):
        """Require an open passage and sounding coverage of the layer."""
        if not np.isfinite([self.height, self.width, self.lid]).all() or not (
            0 <= self.height < self.lid and self.width > 0
        ):
            raise ValueError(
                "require positive width and 0 <= dune height < lid"
            )
        self.sounding.thermodynamics([0.0, self.lid])

    def ground(self, x):
        """Return bed elevation above the sounding datum in metres."""
        return self.height * np.exp(-0.5 * (np.asarray(x) / self.width) ** 2)

    def _wind(self, x, eta):
        jac = 1 - self.ground(x) / self.lid
        u = np.interp(eta, self.sounding.height, self.sounding.east) / jac
        v = np.interp(eta, self.sounding.height, self.sounding.north)
        return u, v, jac

    def velocity(self, x, z):
        """Sample east, north and vertical wind (m/s) at one world position."""
        bed = self.ground(x)
        if not np.isfinite([x, z]).all() or not bed <= z <= self.lid:
            raise ValueError("wind sample outside the flow layer")
        eta = (z - bed) / (1 - bed / self.lid)
        u, v, _ = self._wind(x, eta)
        slope = -x * bed / self.width**2
        return np.array([u, v, (1 - eta / self.lid) * slope * u])

    def track(self, xyz, time, *, settling=0.0):
        """Advect one center or particle with constant downward slip (m/s).

        xyz is a world position in metres and time contains increasing sample
        seconds. After first impact a settling particle stays at its impact
        point. A zero-slip air marker follows its initial streamline.
        """
        xyz = np.asarray(xyz, dtype=float)
        t = np.array(time, dtype=float, copy=True)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("give one finite xyz position")
        if (
            t.ndim != 1
            or len(t) < 2
            or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)
        ):
            raise ValueError("give increasing finite sample times")
        if not np.isfinite(settling) or settling < 0:
            raise ValueError("settling speed must be finite and nonnegative")
        x, y, z = xyz
        self.velocity(x, z)
        bed = self.ground(x)
        eta = (z - bed) / (1 - bed / self.lid)

        def tendency(_, state):
            u, v, jac = self._wind(state[0], state[2])
            return u, v, -settling / jac

        def hit_bed(_, state):
            return state[2]

        hit_bed.terminal = True
        hit_bed.direction = -1
        # Resolve even a narrow ridge far from the initial point.
        max_speed = np.max(np.abs(self.sounding.east)) / (
            1 - self.height / self.lid
        )
        max_step = self.width / (4 * max_speed) if max_speed else np.inf
        sol = solve_ivp(
            tendency,
            (t[0], t[-1]),
            (x, y, eta),
            events=hit_bed if settling else None,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
            max_step=max_step,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        impact = float(sol.t[-1]) if sol.status == 1 else None
        x, y, eta = sol.sol(np.minimum(t, sol.t[-1]))
        if impact is not None:
            eta[t >= impact] = 0
        bed = self.ground(x)
        return Track(
            t, np.column_stack((x, y, bed + (1 - bed / self.lid) * eta)), impact
        )

    def parcel_trace(
        self, parcel_id, xyz, time, aerosol: ParcelInitialState, *, provenance
    ):
        """Return a moving center and replay trace with fixed dry-air mass.

        The sounding supplies initial and subsequent thermodynamics. The
        aerosol template supplies dry mass, represented counts, radii and κ.
        Volume follows the moist gas law. Replay owns aerosol water exchange.
        """
        track = self.track(xyz, time)
        z = track.xyz[:, 2]
        temperature, pressure, vapor = self.sounding.thermodynamics(z)
        return track, make_parcel_trace(
            parcel_id, track, aerosol, provenance, temperature, pressure, vapor
        )

    def pickup(
        self,
        x,
        *,
        reference_height,
        drag_coefficient,
        threshold,
        available_mass,
        pulse_mass,
    ):
        """Return lifted mass (kg) from a finite dry bed patch in one pulse.

        Diagnose friction velocity as sqrt(Cd) times the horizontal wind at
        reference_height (m above bed). Above threshold (m/s), release the
        requested pulse capped by available mass (both kg). The caller debits
        the bed once. Cd, threshold and pulse size are experiment inputs.
        """
        params = (
            reference_height,
            drag_coefficient,
            threshold,
            available_mass,
            pulse_mass,
        )
        if (
            not np.isfinite(params).all()
            or min(params) < 0
            or reference_height == 0
        ):
            raise ValueError(
                "require positive height and nonnegative pickup inputs"
            )
        u, v, _ = self.velocity(x, self.ground(x) + reference_height)
        friction = np.sqrt(drag_coefficient) * np.hypot(u, v)
        return min(available_mass, pulse_mass) if friction > threshold else 0.0


def make_parcel_trace(
    parcel_id, track, aerosol, provenance, temperature, pressure, vapor
):
    """Bind sampled SI thermodynamics and fixed dry air to native replay."""
    z = track.xyz[:, 2]
    initial = replace(
        aerosol,
        height=z[0],
        temperature=temperature[0],
        pressure=pressure[0],
        vapor=vapor[0],
    )
    volume = initial.dry_air_mass * (RD + RV * vapor) * temperature / pressure
    return ParcelTrace(
        parcel_id,
        provenance,
        track.time,
        z,
        temperature,
        pressure,
        volume,
        vapor,
        initial,
        _initial_radii(initial),
    )
