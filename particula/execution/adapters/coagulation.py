"""Provide concrete-only Brownian coagulation resource carriers.

These P2 carriers retain caller-owned CPU and resident-Warp resources by
identity.  Frozen dataclasses prohibit field rebinding only; retained resources
remain mutable.  They neither execute a kernel or ``Coagulation`` runnable nor
transfer, synchronize, allocate, select a backend, or seed, reset, or advance
RNG state.  In particular, the persistent RNG sidecar records future dispatch
intent only: a future direct call may seed it only when ``initialize_rng`` is
true, otherwise it must reuse it even for the same seed.  Callers own
synchronization, recovery, and the lack of rollback after a future launch.
"""

from dataclasses import dataclass
from math import prod
from typing import Any, cast

from particula.aerosol import Aerosol


@dataclass(frozen=True, eq=False)
class BrownianCoagulationConfig:
    """Denote the direct-kernel default Brownian coagulation mechanism."""


@dataclass(frozen=True, eq=False)
class CPUCoagulationState:
    """Retain one caller-owned CPU Brownian coagulation request."""

    config: BrownianCoagulationConfig
    aerosol: Aerosol

    def __post_init__(self) -> None:
        """Validate carrier types without inspecting aerosol state."""
        if type(self.config) is not BrownianCoagulationConfig:
            raise TypeError("config must be a BrownianCoagulationConfig.")
        if not isinstance(self.aerosol, Aerosol):
            raise TypeError("aerosol must be an Aerosol.")

    @property
    def backend_payload(self) -> Aerosol:
        """Return the retained caller-owned aerosol by identity."""
        return self.aerosol


@dataclass(frozen=True, eq=False)
class CPUCoagulationResult:
    """Retain a CPU result which preserves its state's aerosol identity."""

    state: CPUCoagulationState
    aerosol: Aerosol

    def __post_init__(self) -> None:
        """Validate result ownership without inspecting the aerosol."""
        if type(self.state) is not CPUCoagulationState:
            raise TypeError("state must be a CPUCoagulationState.")
        if not isinstance(self.aerosol, Aerosol):
            raise TypeError("aerosol must be an Aerosol.")
        if self.aerosol is not self.state.aerosol:
            raise ValueError("aerosol must be state.aerosol.")

    @property
    def backend_payload(self) -> Aerosol:
        """Return the retained caller-owned aerosol by identity."""
        return self.aerosol


def _dtype_itemsize(dtype: object, wp: Any) -> int | None:
    """Return a supported Warp scalar size, deferring unknown metadata."""
    if dtype is wp.float64:
        return 8
    if dtype is wp.int32 or dtype is wp.uint32:
        return 4
    return None


def _memory_range(array: object, wp: Any) -> tuple[int, int] | None:
    """Return a usable contiguous storage range, otherwise defer validation.

    This metadata-only helper deliberately treats unknown or malformed arrays as
    native-kernel validation concerns.  It never reads device data.
    """
    dtype = getattr(array, "dtype", None)
    itemsize = _dtype_itemsize(dtype, wp)
    shape = getattr(array, "shape", None)
    strides = getattr(array, "strides", None)
    ptr = getattr(array, "ptr", None)
    capacity = getattr(array, "capacity", None)
    if (
        itemsize is None
        or not isinstance(shape, tuple)
        or not isinstance(strides, tuple)
        or len(shape) != len(strides)
        or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        )
        or any(type(stride) is not int for stride in strides)
        or type(ptr) is not int
        or type(capacity) is not int
    ):
        return None
    if prod(shape) == 0:
        return None
    expected: list[int] = []
    stride = itemsize
    for dimension in reversed(shape):
        expected.insert(0, stride)
        stride *= dimension
    required = prod(shape) * itemsize
    if (
        ptr <= 0
        or capacity < required
        or tuple(strides) != tuple(expected)
        or ptr % itemsize != 0
    ):
        return None
    return ptr, ptr + required


def _overlaps(
    first: tuple[int, int] | None, second: tuple[int, int] | None
) -> bool:
    """Return whether two usable, nonempty storage ranges overlap."""
    return (
        first is not None
        and second is not None
        and first[0] < second[1]
        and second[0] < first[1]
    )


def _available_fields(
    resource: object, names: tuple[str, ...]
) -> tuple[object, ...]:
    """Return protected fields without validating their native schema."""
    return tuple(
        getattr(resource, name) for name in names if hasattr(resource, name)
    )


def _validate_ownership(
    wp: Any,
    protected: tuple[object, ...],
    collision_pairs: object | None,
    n_collisions: object | None,
    rng_states: object,
) -> None:
    """Reject aliases between writable sidecars and protected fields."""
    sidecars = (collision_pairs, n_collisions, rng_states)
    for sidecar in sidecars:
        if sidecar is None:
            continue
        for primary in protected:
            if sidecar is primary:
                raise ValueError("caller-owned Warp resources must not alias.")
        sidecar_range = _memory_range(sidecar, wp)
        if any(
            _overlaps(sidecar_range, _memory_range(primary, wp))
            for primary in protected
        ):
            raise ValueError("caller-owned Warp resources must not alias.")
    supplied = tuple(sidecar for sidecar in sidecars if sidecar is not None)
    for index, sidecar in enumerate(supplied):
        for other in supplied[index + 1 :]:
            if sidecar is other or _overlaps(
                _memory_range(sidecar, wp), _memory_range(other, wp)
            ):
                raise ValueError("caller-owned Warp resources must not alias.")


@dataclass(frozen=True, eq=False)
class WarpBrownianCoagulationState:
    """Retain a resident direct-Brownian request without native validation.

    Only primary kind, environment/direct-input form, required persistent RNG,
    and metadata-detectable sidecar ownership are checked.  Physical values,
    array schemas, and direct-kernel ordering remain native concerns.
    """

    config: BrownianCoagulationConfig
    particles: object
    temperature: object | None
    pressure: object | None
    time_step: object
    volume: object | None = None
    collision_pairs: object | None = None
    n_collisions: object | None = None
    rng_states: object | None = None
    rng_seed: object = 0
    initialize_rng: object = False
    environment: object | None = None

    def __post_init__(self) -> None:
        """Perform ordered kind, form, required-RNG, and ownership checks."""
        if type(self.config) is not BrownianCoagulationConfig:
            raise TypeError("config must be a BrownianCoagulationConfig.")
        try:
            import warp as wp
        except ModuleNotFoundError as error:
            if error.name != "warp":
                raise
            raise RuntimeError(
                "Warp is required to construct WarpBrownianCoagulationState."
            ) from error
        from particula.gpu.warp_types import (
            WarpEnvironmentData,
            WarpParticleData,
        )

        particle_type = cast(Any, WarpParticleData).cls
        environment_type = cast(Any, WarpEnvironmentData).cls
        if not isinstance(self.particles, particle_type):
            raise TypeError("particles must be a WarpParticleData.")
        environment_form = self.environment is not None
        if environment_form and not isinstance(
            self.environment, environment_type
        ):
            raise TypeError("environment must be a WarpEnvironmentData.")
        if environment_form:
            valid_form = self.temperature is None and self.pressure is None
        else:
            valid_form = (
                self.temperature is not None and self.pressure is not None
            )
        if not valid_form:
            raise ValueError(
                "provide either environment or both temperature and pressure."
            )
        if self.rng_states is None:
            raise ValueError("rng_states must be supplied.")
        protected = _available_fields(
            self.particles,
            ("masses", "concentration", "charge", "density", "volume"),
        )
        if environment_form:
            protected += _available_fields(
                self.environment,
                ("temperature", "pressure", "saturation_ratio"),
            )
        else:
            protected += tuple(
                value
                for value in (self.temperature, self.pressure, self.volume)
                if _memory_range(value, wp) is not None
            )
        _validate_ownership(
            wp,
            protected,
            self.collision_pairs,
            self.n_collisions,
            self.rng_states,
        )

    @property
    def backend_payload(self) -> object:
        """Return the caller-owned particle container by identity."""
        return self.particles


@dataclass(frozen=True, eq=False)
class WarpBrownianCoagulationResult:
    """Retain a future direct-Warp result and supplied identities."""

    state: WarpBrownianCoagulationState
    particles: object
    collision_pairs: object | None
    n_collisions: object | None

    def __post_init__(self) -> None:
        """Validate state and any state-supplied result sidecar identities."""
        if type(self.state) is not WarpBrownianCoagulationState:
            raise TypeError("state must be a WarpBrownianCoagulationState.")
        if self.particles is not self.state.particles:
            raise ValueError("particles must be state.particles.")
        if (
            self.state.collision_pairs is not None
            and self.collision_pairs is not self.state.collision_pairs
        ):
            raise ValueError("collision_pairs must be state.collision_pairs.")
        if (
            self.state.n_collisions is not None
            and self.n_collisions is not self.state.n_collisions
        ):
            raise ValueError("n_collisions must be state.n_collisions.")

    @property
    def backend_payload(self) -> object:
        """Return the caller-owned particle container by identity."""
        return self.particles
