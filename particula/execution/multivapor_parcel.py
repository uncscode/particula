"""CUDA conditional-graph solver for the finite multivapor parcel.

The resident coordinates are pressure, temperature and analytical material
masses. Each RHS reconstructs its finite gas inventories by conservation.
This formulation and embedded solvers are independent of the Radau oracle,
which integrates the gas reservoirs explicitly.
"""

import json
from dataclasses import asdict, dataclass, fields
from time import perf_counter

import numpy as np
import warp as wp

from particula import __version__
from particula.dynamics.multivapor_parcel import (
    CP,
    RD,
    ParcelTendencies,
    multivapor_trajectory,
)
from particula.dynamics.parcel import ParcelRK23, ParcelROS3
from particula.execution import _multivapor_parcel_kernels as k
from particula.execution.archive import read_archive, write_archive
from particula.particles.prepared_activity import provider_identity
from particula.util.constants import STANDARD_GRAVITY


@dataclass(frozen=True)
class ParcelReplayCheckpoint:
    """Accepted SI restart boundary with finite reservoirs and counters."""

    identity: dict
    time: float
    height: float
    pressure: float
    temperature: float
    masses: np.ndarray
    vapor: np.ndarray
    coordinates: np.ndarray
    next_step: float
    counters: tuple[int, ...]
    completed_segments: int

    def save(self, path):
        """Save an accepted parcel to an atomic numeric restart archive."""
        payload = {
            key: getattr(self, key)
            for key in ("coordinates", "vapor", "masses")
        }
        metadata = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in payload
        }
        write_archive(
            path,
            kind="restart",
            arrays=payload,
            metadata={"schema": "multivapor-parcel-v1", "boundary": metadata},
        )

    @classmethod
    def load(cls, path):
        """Read detached numeric arrays without initializing a device."""
        metadata, arrays, blobs = read_archive(
            path, kind="restart", max_bytes=64 * 1024**2
        )
        if (
            metadata.get("schema") != "multivapor-parcel-v1"
            or blobs
            or set(arrays) != {"vapor", "masses", "coordinates"}
        ):
            raise ValueError("unsupported parcel restart archive")
        try:
            boundary = dict(metadata["boundary"])
            boundary["counters"] = tuple(boundary["counters"])
            return cls(**boundary, **arrays)
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed parcel restart boundary") from exc


class ResidentMultiVaporParcel:
    """Prepare one isolated device workspace for a constant-updraft parcel."""

    def __init__(
        self,
        initial,
        *,
        device,
        adaptive,
        timestep,
        updraft,
        phase_exchange=True,
    ):
        """Allocate bounded scratch and capture the non-advancing trial."""
        if not isinstance(adaptive, (ParcelRK23, ParcelROS3)):
            raise TypeError(
                "use ParcelRK23 or ParcelROS3 for a resident parcel"
            )
        if (
            not np.isfinite(timestep)
            or timestep <= 0
            or not np.isfinite(updraft)
        ):
            raise ValueError("invalid output timestep or constant updraft")
        self.device = wp.get_device(device)
        if not self.device.is_cuda or not wp.is_conditional_graph_supported():
            raise ValueError(
                "resident multi-vapor parcel requires CUDA conditional graphs"
            )
        if provider_identity(initial.activity_strategy) is None:
            raise ValueError(
                "multi-vapor CUDA needs a device activity provider"
            )
        self.time = 0.0
        self.initial = initial
        self.controls = adaptive
        self.timestep = float(timestep)
        self.updraft = float(updraft)
        self.phase_exchange = bool(phase_exchange)
        self.segments = 0
        self.boundary_readbacks = 0
        self.replay_seconds = 0.0
        self.rhs = ParcelTendencies(initial, updraft, phase_exchange)
        self.lanes = self.rhs.targets
        nclass, nmat = initial.masses.shape
        nv = len(self.lanes)
        self.nclass = nclass
        self.n = 2 + nclass * nv
        self.ros = isinstance(adaptive, ParcelROS3)
        if self.ros and self.n**2 * 8 > 32 * 1024**2:
            raise ValueError(
                "dense parcel ROS3 Jacobian exceeds its 32 MiB workspace budget"
            )
        self.scale = np.maximum(
            initial.masses[:, self.lanes],
            np.maximum(initial.masses.sum(axis=1, keepdims=True) * 1e-6, 1e-30),
        )
        self.initial_y = np.r_[
            1.0, 1.0, (initial.masses[:, self.lanes] / self.scale).ravel()
        ]
        self.binding = initial.activity_strategy.bind_device(
            (1, nclass, nmat), self.lanes, self.device
        )
        if (
            self.ros
            and getattr(self.binding, "hybrid", False)
            and not callable(getattr(self.binding, "evaluate_direct", None))
        ):
            raise ValueError("hybrid ROS3 requires direct device linearization")
        self._arrays = []

        def upload(x, dtype=wp.float64):
            a = wp.array(
                np.ascontiguousarray(x), dtype=dtype, device=self.device
            )
            self._arrays.append(a)
            return a

        def zeros(shape, dtype=wp.float64):
            a = wp.zeros(shape, dtype=dtype, device=self.device)
            self._arrays.append(a)
            return a

        self._m0 = upload(initial.masses)
        self._q0 = upload(initial.vapor)
        self._counts = upload(initial.number.reshape(1, -1))
        self._scale = upload(self.scale)
        self._lanes = upload(self.lanes, wp.int32)
        self._density = upload(initial.density)
        self._props = upload(
            np.column_stack(
                (
                    self.rhs.gas_r,
                    self.rhs.latent,
                    self.rhs.p0,
                    self.rhs.t0,
                    self.rhs.diffusivity * self.rhs.enabled,
                )
            )
        )
        self.y = upload(self.initial_y)
        self.clock = upload([0.0, self.timestep, self.timestep, self.timestep])
        self.counters = zeros(5, wp.int32)
        self.cond = zeros(1, wp.int32)
        self.invalid = zeros(1, wp.int32)
        self.pressure, self.temp, self.vol, self.dp = [
            zeros(1) for _ in range(4)
        ]
        self.vapor = zeros(nv)
        self.masses = zeros((1, nclass, nmat))
        self.free, self.slopes, self.flux = [
            zeros((nclass, nv)) for _ in range(3)
        ]
        self.trial, self.error, *self.stages = [zeros(self.n) for _ in range(6)]
        if self.ros:
            self.jac = zeros((self.n, self.n))
            self.pivots = zeros(self.n, wp.int32)
            self.delta = zeros(1)
            self.f0, self.fr = zeros(self.n), zeros(self.n)
            self.jac_base = (
                zeros(self.n)
                if getattr(self.binding, "hybrid", False)
                else self.f0
            )
        # Compile the trial on scratch before capturing the acceptance loop.
        self._trial()
        self.counters.zero_()
        self.binding.detail.zero_()
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.capture_while(self.cond, self._iteration)
        self.graph = capture.graph
        self.scratch_bytes = sum(
            a.size * wp.types.type_size_in_bytes(a.dtype) for a in self._arrays
        )

    def _launch(self, kernel, n, *inputs):
        wp.launch(kernel, dim=n, inputs=list(inputs), device=self.device)

    def _unpack(self, state, stage):
        p = self.initial
        self._launch(
            k.unpack,
            1,
            state,
            self._m0,
            self._q0,
            self._scale,
            self._lanes,
            self._counts,
            p.pressure,
            p.temperature,
            p.dry_air_mass,
            self.pressure,
            self.temp,
            self.vapor,
            self.masses,
            self.invalid,
            self.binding.detail,
            stage,
        )

    def _fluxes(self):
        self._launch(
            k.fluxes,
            self.nclass,
            self.pressure,
            self.temp,
            self.vapor,
            self.vol,
            self.masses,
            self._counts,
            self._density,
            self._lanes,
            self._props,
            self.initial.surface_tension,
            self.binding.values,
            self.free,
            self.slopes,
            self.flux,
        )

    def _rate(self, state, out, stage, *, linearize=False):
        self._unpack(state, stage)
        hybrid = getattr(self.binding, "hybrid", False)
        evaluate = (
            self.binding.evaluate_direct
            if linearize and hybrid
            else self.binding.evaluate
        )
        evaluate(
            self.masses, self._counts, self.temp, self.invalid, stage=stage
        )
        self._launch(
            k.hydro,
            1,
            self.pressure,
            self.temp,
            self.vapor,
            self._props,
            RD,
            self.rhs.loading * STANDARD_GRAVITY * self.updraft,
            self.vol,
            self.dp,
        )
        self._fluxes()
        if hybrid and not linearize:
            self._launch(
                k.flux_guard,
                self.nclass,
                self.pressure,
                self.temp,
                self.vapor,
                self.vol,
                self.masses,
                self._density,
                self._lanes,
                self._props,
                self.free,
                self.slopes,
                self.flux,
                self.binding.errors,
                self.binding.uncertain,
            )
            self.binding.evaluate(
                self.masses,
                self._counts,
                self.temp,
                self.invalid,
                stage=stage,
                refine=True,
            )
            self._fluxes()
        self._launch(
            k.accumulate,
            1,
            self.vol,
            self.dp,
            self.flux,
            self._counts,
            self._props,
            self._scale,
            self.initial.dry_air_mass,
            self.initial.pressure,
            self.initial.temperature,
            CP,
            out,
            self.counters,
            self.invalid,
        )

    def _trial(self):
        self._launch(
            k.trial,
            1,
            self.clock,
            self.counters,
            self.invalid,
            self.binding.detail,
        )
        a, b, c, d = self.stages
        if self.ros:
            self._rate(self.y, self.f0, 1)
            if self.jac_base is not self.f0:
                self._rate(self.y, self.jac_base, 1, linearize=True)
            self._launch(k.mark_base, 1, self.invalid, self.counters)
            for j in range(self.n):
                self._launch(
                    k.perturb,
                    self.n,
                    self.y,
                    self.trial,
                    j,
                    self.delta,
                    self._q0,
                    self._m0,
                    self._lanes,
                    self._scale,
                    self._counts,
                    self.initial.dry_air_mass,
                )
                self._rate(self.trial, self.fr, 2 + j, linearize=True)
                self._launch(
                    k.linearize,
                    self.n,
                    self.jac_base,
                    self.fr,
                    self.delta,
                    self.clock,
                    self.jac,
                    j,
                )
            self._launch(k.factor, 1, self.jac, self.pivots, self.invalid)
            self._launch(
                k.solve,
                1,
                self.jac,
                self.pivots,
                self.clock,
                self.f0,
                a,
                b,
                1,
                a,
            )
            self._launch(k.ros_stage, self.n, self.y, a, self.trial)
            self._rate(self.trial, self.fr, 2 + self.n)
            self._launch(
                k.solve,
                1,
                self.jac,
                self.pivots,
                self.clock,
                self.fr,
                a,
                b,
                2,
                b,
            )
            self._launch(
                k.solve,
                1,
                self.jac,
                self.pivots,
                self.clock,
                self.fr,
                a,
                b,
                3,
                c,
            )
            self._launch(
                k.ros_finish, self.n, self.y, a, b, c, self.trial, self.error
            )
            self._rate(self.trial, self.fr, 3 + self.n)
        else:
            self._rate(self.y, a, 1)
            self._launch(k.mark_base, 1, self.invalid, self.counters)
            for stage, rate in ((1, b), (2, c), (3, d)):
                self._launch(
                    k.rk_stage,
                    self.n,
                    self.y,
                    self.clock,
                    a,
                    b,
                    c,
                    stage,
                    self.trial,
                )
                self._rate(self.trial, rate, stage + 1)
            self._launch(k.rk_error, self.n, self.clock, a, b, c, d, self.error)

    def _iteration(self):
        self._trial()
        self._launch(
            k.decide,
            1,
            self.y,
            self.trial,
            self.error,
            self.clock,
            self.invalid,
            self.counters,
            self.cond,
            self.controls.rtol,
            self.controls.min_step,
            self.controls.max_trials,
        )

    def identity(self):
        """Record the material, thermophysical, solver and device identities."""
        state = self.initial
        return dict(
            scheme="parcel-continuum-linearized-surface-v1",
            precision="float64",
            particula_version=__version__,
            device=(
                self.device.alias,
                self.device.name,
                self.device.arch,
                wp.__version__,
                str(self.device.uuid),
            ),
            method=type(self.controls).__name__,
            controls=asdict(self.controls),
            timestep=self.timestep,
            updraft=self.updraft,
            phase_exchange=self.phase_exchange,
            height=state.height,
            dry_air_mass=state.dry_air_mass,
            temperature=state.temperature,
            pressure=state.pressure,
            surface_tension=state.surface_tension,
            materials=tuple(state.material_names),
            density=state.density.tolist(),
            initial_masses=state.masses.tolist(),
            initial_vapor=state.vapor.tolist(),
            number=state.number.tolist(),
            vapors=tuple(asdict(v) for v in state.vapors),
            provider=state.activity_strategy.prepared_identity(),
        )

    def _decode(self, y):
        state = self.initial
        masses = state.masses.copy()
        masses[:, self.lanes] = y[2:].reshape(self.scale.shape) * self.scale
        q = (
            state.vapor
            + (
                state.number
                @ (state.masses[:, self.lanes] - masses[:, self.lanes])
            )
            / state.dry_air_mass
        )
        return y[0] * state.pressure, y[1] * state.temperature, q, masses

    def advance(self, stop):
        """Replay to an absolute time, then read observation and status."""
        if not np.isfinite(stop) or stop <= self.time:
            raise ValueError("parcel segment end must advance time")
        before = perf_counter()
        self._launch(
            k.start, 1, self.clock, self.counters, float(stop), self.cond
        )
        wp.capture_launch(self.graph)
        self.last_counts = tuple(int(x) for x in self.counters.numpy())
        self.boundary_readbacks += 1
        self.replay_seconds += perf_counter() - before
        if self.last_counts[3]:
            detail = self.binding.detail.numpy()
            raise RuntimeError(
                f"parcel failed at box 0, slot {detail[2]}, "
                f"solver stage {detail[3]}, backend status {detail[4]}, "
                f"controller status {self.last_counts[3]}"
            )
        self.time = float(stop)
        self.segments += 1
        self.boundary_readbacks += 1
        return self._decode(self.y.numpy())

    def checkpoint(self):
        """Copy an accepted boundary and next step for recapture."""
        counters = tuple(int(x) for x in self.counters.numpy())
        if counters[3]:
            raise ValueError(
                "checkpoint requires a successful accepted boundary"
            )
        time, next_step = self.clock.numpy()[:2]
        y = self.y.numpy()
        pressure, temp, vapor, mass = self._decode(y)
        return ParcelReplayCheckpoint(
            self.identity(),
            float(time),
            self.initial.height + self.updraft * time,
            pressure,
            temp,
            mass,
            vapor,
            y,
            float(next_step),
            counters,
            self.segments,
        )

    def restore(self, checkpoint):
        """Validate restart coordinates and identity before device writes."""
        if json.dumps(checkpoint.identity, sort_keys=True) != json.dumps(
            self.identity(), sort_keys=True
        ):
            raise ValueError("incompatible multi-vapor parcel restart identity")
        y = np.asarray(checkpoint.coordinates, dtype=float)
        if y.shape != (self.n,) or not np.isfinite(y).all():
            raise ValueError("incompatible parcel coordinates")
        p, t, q, m = self._decode(y)
        if (
            not np.array_equal(m, checkpoint.masses)
            or not np.array_equal(q, checkpoint.vapor)
            or p != checkpoint.pressure
            or t != checkpoint.temperature
            or p <= 0
            or t <= 0
            or np.any(m < 0)
            or np.any(q < 0)
            or checkpoint.height
            != self.initial.height + self.updraft * checkpoint.time
            or not np.isfinite(checkpoint.next_step)
            or checkpoint.next_step <= 0
            or not np.isfinite(checkpoint.time)
            or checkpoint.time < 0
            or checkpoint.completed_segments < 0
            or len(checkpoint.counters) != 5
            or min(checkpoint.counters) < 0
            or checkpoint.counters[3]
        ):
            raise ValueError(
                "invalid parcel physical or controller restart state"
            )
        self.y.assign(y)
        self.clock.assign(
            np.array(
                [
                    checkpoint.time,
                    checkpoint.next_step,
                    checkpoint.time,
                    checkpoint.next_step,
                ]
            )
        )
        self.counters.assign(np.asarray(checkpoint.counters, dtype=np.int32))
        self.time, self.segments = (
            checkpoint.time,
            checkpoint.completed_segments,
        )
        self.last_counts = checkpoint.counters


def run_multivapor_cuda(
    initial, time, updraft, phase_exchange, device, adaptive
):
    """Run resident RK23 or ROS3 with observations at the requested times."""
    owner = ResidentMultiVaporParcel(
        initial,
        device=device,
        adaptive=adaptive,
        timestep=time[1] - time[0],
        updraft=updraft,
        phase_exchange=phase_exchange,
    )
    states = [
        (initial.pressure, initial.temperature, initial.vapor, initial.masses)
    ]
    for stop in time[1:]:
        states.append(owner.advance(stop))
    p, t, q, m = zip(*states, strict=True)
    return multivapor_trajectory(initial, time, p, t, q, m, updraft)
