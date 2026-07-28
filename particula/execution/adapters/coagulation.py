"""Provide concrete-only CPU and resident-Warp Brownian adapters.

P2 carriers and P3 execution states retain caller-owned resources by identity.
Frozen dataclasses prohibit field rebinding only; retained resources remain
mutable. The adapters dispatch only their already-selected CPU runnable or
resident-Warp kernel. They do not select another backend, transfer,
synchronize, convert, retry, recover, or roll back caller-owned state.

The persistent RNG sidecar records direct-Warp dispatch intent. A dispatch may
seed it only when ``initialize_rng`` is true; otherwise it reuses the supplied
sidecar, including when the seed is unchanged. Callers own synchronization and
recovery after a kernel launch.

CPU and resident-Warp calls have independent stochastic trajectories. This
concrete boundary provides no seed-by-seed cross-backend trajectory comparison.
"""

from dataclasses import dataclass
from math import prod
from numbers import Integral, Real
from typing import Any, cast

from particula.aerosol import Aerosol
from particula.dynamics import Coagulation
from particula.dynamics.coagulation.coagulation_strategy import (
    brownian_coagulation_strategy,
)
from particula.execution import (
    BackendResult,
    ExecutionResult,
    ExecutionState,
    MutationDeclaration,
    MutationScope,
    _isfinite_real,
)

BrownianCoagulationStrategy = (
    brownian_coagulation_strategy.BrownianCoagulationStrategy
)


@dataclass(frozen=True, eq=False)
class BrownianCoagulationConfig:
    """Denote the direct-kernel default Brownian coagulation mechanism.

    This fieldless, concrete-only marker selects no backend and performs no
    mechanism or physical-value validation.
    """


@dataclass(frozen=True, eq=False)
class CPUCoagulationState:
    """Retain one caller-owned CPU Brownian coagulation request.

    Construction validates the exact marker type and the aerosol type without
    inspecting aerosol state or strategy. It does not execute a runnable, copy
    resources, or validate coagulation physics. Frozen status prevents field
    rebinding only; the retained aerosol remains mutable and caller-owned.

    Args:
        config: Exact Brownian mechanism marker.
        aerosol: Caller-owned aerosol retained by identity.

    Raises:
        TypeError: If ``config`` is not an exact ``BrownianCoagulationConfig``
            or ``aerosol`` is not an ``Aerosol``.
    """

    config: BrownianCoagulationConfig
    aerosol: Aerosol

    def __post_init__(self) -> None:
        """Validate carrier types without inspecting aerosol state.

        Raises:
            TypeError: If the configuration marker or aerosol has an invalid
                type.
        """
        if type(self.config) is not BrownianCoagulationConfig:
            raise TypeError("config must be a BrownianCoagulationConfig.")
        if not isinstance(self.aerosol, Aerosol):
            raise TypeError("aerosol must be an Aerosol.")

    @property
    def backend_payload(self) -> Aerosol:
        """Return the retained caller-owned aerosol by identity.

        Returns:
            The exact aerosol retained by this state.
        """
        return self.aerosol


@dataclass(frozen=True, eq=False)
class CPUCoagulationResult:
    """Retain a CPU result that preserves its state's aerosol identity.

    This carrier is write-free and validates that its result aerosol is the
    exact resource retained by the supplied CPU state.

    Args:
        state: Exact CPU request state retained by identity.
        aerosol: The state's caller-owned aerosol.

    Raises:
        TypeError: If ``state`` is not an exact ``CPUCoagulationState`` or
            ``aerosol`` is not an ``Aerosol``.
        ValueError: If ``aerosol`` is not ``state.aerosol``.
    """

    state: CPUCoagulationState
    aerosol: Aerosol

    def __post_init__(self) -> None:
        """Validate result ownership without inspecting the aerosol.

        Raises:
            TypeError: If the state or aerosol has an invalid type.
            ValueError: If the result aerosol differs from the state's aerosol.
        """
        if type(self.state) is not CPUCoagulationState:
            raise TypeError("state must be a CPUCoagulationState.")
        if not isinstance(self.aerosol, Aerosol):
            raise TypeError("aerosol must be an Aerosol.")
        if self.aerosol is not self.state.aerosol:
            raise ValueError("aerosol must be state.aerosol.")

    @property
    def backend_payload(self) -> Aerosol:
        """Return the retained caller-owned aerosol by identity.

        Returns:
            The exact aerosol retained by this result.
        """
        return self.aerosol


@dataclass(frozen=True, eq=False)
class CPUCoagulationExecutionState:
    """Retain selected CPU coagulation controls and runnable by identity.

    This concrete-only P3 carrier validates only exact P2-state and runnable
    types. Execution controls remain opaque until dispatch, and retained
    resources remain caller-owned and mutable.

    Args:
        state: Exact P2 CPU state retained by identity.
        time_step: Original execution time step in s, validated at dispatch.
        sub_steps: Original number of runnable substeps, validated at dispatch.
        runnable: Exact caller-owned ``Coagulation`` runnable.

    Raises:
        TypeError: If ``state`` or ``runnable`` is not the required exact type.
    """

    state: CPUCoagulationState
    time_step: object
    sub_steps: object
    runnable: Coagulation

    def __post_init__(self) -> None:
        """Validate exact P2-state and runnable carrier types."""
        if type(self.state) is not CPUCoagulationState:
            raise TypeError("state must be an exact CPUCoagulationState.")
        if type(self.runnable) is not Coagulation:
            raise TypeError("runnable must be an exact Coagulation.")

    @property
    def backend_payload(self) -> Aerosol:
        """Return the exact caller-owned CPU aerosol payload.

        Returns:
            The aerosol retained by the P2 state.
        """
        return self.state.backend_payload


def _validate_time_step(time_step: object) -> None:
    """Validate a non-boolean, finite, nonnegative real time step.

    Args:
        time_step: Candidate execution time step in s.

    Raises:
        TypeError: If ``time_step`` is not a non-boolean real scalar.
        ValueError: If ``time_step`` is non-finite or negative.
    """
    if isinstance(time_step, bool) or not isinstance(time_step, Real):
        raise TypeError("time_step must be a real scalar.")
    if not _isfinite_real(time_step) or time_step < 0:
        raise ValueError("time_step must be finite and nonnegative.")


def _validate_cpu_brownian_runnable(runnable: Coagulation) -> None:
    """Validate the selected CPU Brownian particle-resolved capability.

    Args:
        runnable: Exact caller-owned coagulation runnable selected for dispatch.

    Raises:
        ValueError: If the runnable is not the supported Brownian,
            particle-resolved configuration.
    """
    strategy = runnable.coagulation_strategy
    if (
        type(strategy) is not BrownianCoagulationStrategy
        or strategy.distribution_type != "particle_resolved"
    ):
        raise ValueError(
            "runnable must use Brownian particle_resolved coagulation."
        )


class CPUCoagulationExecutionAdapter:
    """Dispatch one selected CPU coagulation request exactly once.

    This concrete-only adapter validates local controls and invokes the
    caller-owned runnable once without splitting, converting, retrying, or
    recovering. Successful results retain the original aerosol and declare
    state mutation. Delegate exceptions propagate unchanged.
    """

    def execute(self, state: ExecutionState) -> ExecutionResult:
        """Execute one exact CPU P3 state without fallback or recovery.

        Args:
            state: Exact selected CPU P3 execution state.

        Returns:
            A result retaining ``state`` and the original aerosol by identity.

        Raises:
            TypeError: If ``state`` or its time step has an invalid type.
            ValueError: If controls are invalid or the runnable returns another
                aerosol.
        """
        if type(state) is not CPUCoagulationExecutionState:
            raise TypeError("state must be a CPUCoagulationExecutionState.")
        _validate_time_step(state.time_step)
        if (
            isinstance(state.sub_steps, bool)
            or not isinstance(state.sub_steps, Integral)
            or state.sub_steps <= 0
        ):
            raise ValueError("sub_steps must be a positive integer.")
        _validate_cpu_brownian_runnable(state.runnable)
        aerosol = state.runnable.execute(
            state.state.aerosol,
            cast(float, state.time_step),
            cast(int, state.sub_steps),
        )
        cpu_result = CPUCoagulationResult(state.state, aerosol)
        return ExecutionResult(
            state,
            (),
            MutationDeclaration(frozenset({MutationScope.STATE})),
            BackendResult(cpu_result),
        )


def _dtype_itemsize(dtype: object, wp: Any) -> int | None:
    """Return a supported Warp scalar size, deferring unknown metadata.

    Args:
        dtype: Candidate Warp scalar dtype metadata.
        wp: Imported Warp module containing supported scalar dtype singletons.

    Returns:
        The scalar size in bytes, or ``None`` when metadata is unsupported.
    """
    if dtype is wp.float64:
        return 8
    if dtype is wp.float32 or dtype is wp.int32 or dtype is wp.uint32:
        return 4
    return None


def _memory_range(array: object, wp: Any) -> tuple[int, int] | None:
    """Return a usable contiguous storage range, otherwise defer validation.

    This metadata-only helper deliberately treats unknown or malformed arrays as
    native-kernel validation concerns.  It never reads device data.

    Args:
        array: Candidate resource with optional Warp-array metadata.
        wp: Imported Warp module used to recognize scalar dtypes.

    Returns:
        A half-open byte range for usable contiguous storage, or ``None`` when
        metadata cannot establish a range.
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
    """Return whether two usable, nonempty storage ranges overlap.

    Args:
        first: Optional first half-open byte range.
        second: Optional second half-open byte range.

    Returns:
        True when both ranges are present and share at least one byte.
    """
    return (
        first is not None
        and second is not None
        and first[0] < second[1]
        and second[0] < first[1]
    )


def _available_fields(
    resource: object, names: tuple[str, ...]
) -> tuple[object, ...]:
    """Return protected fields without validating their native schema.

    Args:
        resource: Primary resource whose available attributes are protected.
        names: Attribute names to retrieve when present.

    Returns:
        Available attribute values in ``names`` order.
    """
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
    """Reject aliases between writable sidecars and protected fields.

    Unknown metadata defers to native validation, but identical objects and
    metadata-detectable overlapping byte ranges are rejected. This function
    reads metadata only and does not mutate or synchronize device resources.

    Args:
        wp: Imported Warp module used for metadata interpretation.
        protected: Primary resources that writable sidecars cannot alias.
        collision_pairs: Optional caller-owned collision-pair output.
        n_collisions: Optional caller-owned collision-count output.
        rng_states: Required caller-owned persistent RNG sidecar.

    Raises:
        ValueError: If writable sidecars alias a protected resource or each
            other.
    """
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

    Only marker, container kind, environment/direct-input form, selected time,
    required persistent RNG, and metadata-detectable sidecar ownership are
    checked. Particle, environment/volume, output-buffer, device, dtype,
    capacity, and detailed RNG schemas remain native-kernel concerns.

    Construction is write-free: it does not import a kernel, transfer,
    synchronize, allocate, seed, reset, or advance ``rng_states``. A future
    dispatch may seed the caller-owned ``(n_boxes,)`` ``wp.uint32`` sidecar only
    when ``initialize_rng`` is true, and otherwise must reuse it even for the
    same seed. Callers own synchronization and recovery after a future launch.

    Args:
        config: Exact Brownian mechanism marker.
        particles: Caller-owned resident ``WarpParticleData`` container.
        temperature: Direct caller-owned temperature, or ``None`` with
            ``environment``.
        pressure: Direct caller-owned pressure, or ``None`` with
            ``environment``.
        time_step: Finite, nonnegative real direct-kernel time step in s.
        volume: Optional opaque direct-kernel volume input.
        collision_pairs: Optional caller-owned collision-pair output.
        n_collisions: Optional caller-owned collision-count output.
        rng_states: Required caller-owned persistent RNG sidecar.
        rng_seed: Opaque future seed intent retained without interpretation.
        initialize_rng: Opaque future reset intent retained without mutation.
        environment: Caller-owned ``WarpEnvironmentData``, or ``None`` for
            direct temperature and pressure inputs.

    Raises:
        TypeError: If the configuration, particles, or non-``None`` environment
            has an invalid type.
        RuntimeError: If the optional Warp runtime is unavailable after valid
            configuration preflight.
        ValueError: If input form or selected time is invalid, no RNG sidecar
            is supplied, or metadata establishes forbidden caller-resource
            aliasing.
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
        """Perform ordered kind, form, time, required-RNG, and ownership checks.

        The configuration check occurs before the lazy Warp import, preserving
        CPU-only construction and deterministic error ordering.

        Raises:
            TypeError: If a selection-owned kind check fails.
            RuntimeError: If Warp is unavailable after configuration preflight.
            ValueError: If resource form, selected time, RNG presence, or
                ownership is invalid.
        """
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
        _validate_time_step(self.time_step)
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
        """Return the caller-owned particle container by identity.

        Returns:
            The exact particle container retained by this state.
        """
        return self.particles


@dataclass(frozen=True, eq=False)
class WarpBrownianCoagulationResult:
    """Retain a future direct-Warp result and supplied resource identities.

    State-supplied diagnostic outputs must be returned by identity. Diagnostics
    omitted from the state remain optional for a future direct call to allocate
    or return according to its native contract.

    Args:
        state: Exact resident-Warp request state retained by identity.
        particles: The state's caller-owned particle container.
        collision_pairs: Returned collision-pair output, when applicable.
        n_collisions: Returned collision-count output, when applicable.

    Raises:
        TypeError: If ``state`` is not an exact
            ``WarpBrownianCoagulationState``.
        ValueError: If particles or a state-supplied diagnostic lacks the
            required identity.
    """

    state: WarpBrownianCoagulationState
    particles: object
    collision_pairs: object | None
    n_collisions: object | None

    def __post_init__(self) -> None:
        """Validate state and state-supplied result-sidecar identities.

        Raises:
            TypeError: If the state is not the exact required carrier type.
            ValueError: If a required particle or diagnostic identity differs.
        """
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
        """Return the caller-owned particle container by identity.

        Returns:
            The exact particle container retained by this result.
        """
        return self.particles


@dataclass(frozen=True, eq=False)
class WarpBrownianCoagulationExecutionState:
    """Retain one selected resident-Warp coagulation request by identity.

    This concrete-only P3 carrier validates only the exact P2 state. It neither
    imports a kernel nor inspects or synchronizes caller-owned resident
    resources.

    Args:
        state: Exact P2 resident-Warp state retained by identity.

    Raises:
        TypeError: If ``state`` is not an exact
            ``WarpBrownianCoagulationState``.
    """

    state: WarpBrownianCoagulationState

    def __post_init__(self) -> None:
        """Validate the exact P2 Warp state carrier type."""
        if type(self.state) is not WarpBrownianCoagulationState:
            raise TypeError(
                "state must be an exact WarpBrownianCoagulationState."
            )

    @property
    def backend_payload(self) -> object:
        """Return the exact caller-owned resident particle payload.

        Returns:
            The particle container retained by the P2 state.
        """
        return self.state.backend_payload


def _get_coagulation_step_gpu() -> Any:
    """Lazily resolve the optional direct Warp coagulation kernel.

    Returns:
        The direct ``coagulation_step_gpu`` kernel entry point.

    Raises:
        ImportError: If optional Warp kernel dependencies are unavailable.
    """
    from particula.gpu.kernels import coagulation_step_gpu

    return coagulation_step_gpu


class WarpBrownianCoagulationExecutionAdapter:
    """Dispatch one selected resident-Warp Brownian request exactly once.

    This concrete-only adapter resolves the selected kernel lazily after local
    state validation and calls it once. It forwards P2 resources by identity
    without conversion, synchronization, fallback, retry, or recovery. Kernel
    exceptions and any post-launch mutation remain governed by the kernel.
    """

    def execute(self, state: ExecutionState) -> ExecutionResult:
        """Execute one exact Warp P3 state without fallback or recovery.

        Args:
            state: Exact selected resident-Warp P3 execution state.

        Returns:
            A result retaining ``state`` and native result resources by
            identity.

        Raises:
            TypeError: If ``state`` is not the required exact P3 state.
            ValueError: If the direct kernel returns resources that violate P2
                result identity requirements. Direct kernel errors propagate
                unchanged.
            ImportError: If optional Warp kernel dependencies are unavailable
                after local preflight.
        """
        if type(state) is not WarpBrownianCoagulationExecutionState:
            raise TypeError(
                "state must be a WarpBrownianCoagulationExecutionState."
            )
        p2_state = state.state
        coagulation_step_gpu = _get_coagulation_step_gpu()
        particles, collision_pairs, n_collisions = coagulation_step_gpu(
            p2_state.particles,
            p2_state.temperature,
            p2_state.pressure,
            p2_state.time_step,
            p2_state.volume,
            rng_seed=p2_state.rng_seed,
            collision_pairs=p2_state.collision_pairs,
            n_collisions=p2_state.n_collisions,
            rng_states=p2_state.rng_states,
            initialize_rng=p2_state.initialize_rng,
            environment=p2_state.environment,
        )
        warp_result = WarpBrownianCoagulationResult(
            p2_state,
            particles,
            collision_pairs,
            n_collisions,
        )
        return ExecutionResult(
            state,
            (),
            MutationDeclaration(frozenset({MutationScope.STATE})),
            BackendResult(warp_result),
        )
