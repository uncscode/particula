"""Provide concrete-only condensation state carriers and selected adapters.

Import these carriers from ``particula.execution.adapters.condensation``, not
from ``particula.execution`` or top-level ``particula``. They retain
caller-owned resources by identity. P2 carriers perform read-only
construction-time metadata checks; P3 adapters make one selected native call.
Neither selection nor these adapters transfer, allocate, restore, synchronize,
retry, or fall back. Frozen fields prevent rebinding only; retained resources
remain mutable and caller-owned. Native calls may mutate particle masses, gas
concentration or vapor pressure, and writable sidecars. Callers own resource
lifetime, synchronization, concurrency, and any post-launch recovery limits.
"""

from dataclasses import dataclass
from math import prod
from numbers import Integral, Real
from typing import Any, cast

from particula.aerosol import Aerosol
from particula.dynamics import MassCondensation
from particula.execution import (
    Backend,
    BackendResult,
    CondensationConfiguration,
    ExecutionResult,
    ExecutionState,
    MutationDeclaration,
    MutationScope,
    _isfinite_real,
    require_condensation_profile,
)


@dataclass(frozen=True, eq=False)
class CondensationExecutionConfig:
    """Retain an exact semantic condensation configuration by identity.

    This concrete-only P2 carrier does not select a profile or inspect its
    configuration. A future selection caller remains responsible for calling
    ``require_condensation_profile()`` when that semantic check is needed.
    Frozen status prevents field rebinding but does not copy the configuration.

    Args:
        configuration: Exact concrete condensation configuration.

    Raises:
        TypeError: If ``configuration`` is not an exact
            ``CondensationConfiguration`` instance.
    """

    configuration: CondensationConfiguration

    def __post_init__(self) -> None:
        """Validate the exact configuration carrier type.

        Raises:
            TypeError: If ``configuration`` is not an exact
                ``CondensationConfiguration`` instance.
        """
        if type(self.configuration) is not CondensationConfiguration:
            raise TypeError(
                "configuration must be an exact CondensationConfiguration."
            )


@dataclass(frozen=True, eq=False)
class CPUCondensationState:
    """Retain caller-owned CPU condensation state without executing it.

    This concrete-only P2 carrier validates only the configuration and aerosol
    types, in that order. It does not inspect aerosol backing data, validate
    condensation physics, add execution controls, copy resources, or invoke a
    runnable. Frozen status prevents field rebinding only; ``aerosol`` remains
    caller-owned and mutable.

    Args:
        config: Exact P2 condensation execution configuration.
        aerosol: Caller-owned aerosol retained by identity.

    Raises:
        TypeError: If ``config`` is not an exact
            ``CondensationExecutionConfig`` or ``aerosol`` is not an
            ``Aerosol``.
    """

    config: CondensationExecutionConfig
    aerosol: Aerosol

    def __post_init__(self) -> None:
        """Validate state inputs in configuration then aerosol order.

        Raises:
            TypeError: If ``config`` is not an exact
                ``CondensationExecutionConfig`` or ``aerosol`` is not an
                ``Aerosol``.
        """
        if type(self.config) is not CondensationExecutionConfig:
            raise TypeError(
                "config must be an exact CondensationExecutionConfig."
            )
        if not isinstance(self.aerosol, Aerosol):
            raise TypeError("aerosol must be an Aerosol.")

    @property
    def backend_payload(self) -> Aerosol:
        """Return the caller-owned aerosol without inspecting it.

        Returns:
            The exact aerosol retained by this state.
        """
        return self.aerosol


def _validate_array(
    name: str,
    array: Any,
    dtype: Any,
    shape: tuple[int, ...] | None,
    device: Any | None = None,
) -> None:
    """Validate Warp-array metadata without device operations.

    Args:
        name: Qualified field name used in validation errors.
        array: Candidate Warp array whose metadata is read.
        dtype: Required Warp dtype.
        shape: Required array shape, or ``None`` to validate only metadata.
        device: Required device, or ``None`` when no device comparison applies.

    Raises:
        ValueError: If the candidate lacks required Warp metadata or its dtype,
            shape, or device is invalid.
    """
    attributes = ("dtype", "shape", "device")
    if not all(hasattr(array, attribute) for attribute in attributes):
        raise ValueError(f"{name} must be a Warp array.")
    if array.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype}.")
    if not isinstance(array.shape, tuple):
        raise ValueError(f"{name} must be a Warp array.")
    if shape is None:
        return
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if device is not None and str(array.device) != str(device):
        raise ValueError(f"{name} device must match particles.masses device.")


def _memory_range(
    name: str,
    array: Any,
    itemsize: int,
    *,
    require_contiguous: bool = True,
) -> tuple[int, int] | None:
    """Return a metadata-only storage range, or ``None`` for an empty array.

    Args:
        name: Qualified field name used in ownership errors.
        array: Metadata-valid Warp array whose storage is inspected.
        itemsize: Size in bytes of one array element.
        require_contiguous: Whether to reject non-contiguous storage.

    Returns:
        The half-open byte range occupied by nonempty storage, or ``None`` for
        an empty array.

    Raises:
        ValueError: If required pointer or stride metadata is absent, or if
            contiguous storage is required but unavailable.
    """
    strides = getattr(array, "strides", None)
    if not hasattr(array, "ptr"):
        raise ValueError(
            f"{name} must be a contiguous Warp array for ownership."
        )
    expected: list[int] = []
    stride = itemsize
    for dimension in reversed(array.shape):
        expected.insert(0, stride)
        stride *= dimension
    if require_contiguous and (
        strides is None or tuple(strides) != tuple(expected)
    ):
        raise ValueError(f"{name} must be contiguous for ownership checks.")
    if prod(array.shape) == 0:
        return None
    start = int(array.ptr)
    if strides is None:
        raise ValueError(f"{name} must provide strides for ownership checks.")
    offsets = tuple(
        (dimension - 1) * stride
        for dimension, stride in zip(array.shape, strides, strict=True)
    )
    return (
        start + sum(min(0, offset) for offset in offsets),
        start + sum(max(0, offset) for offset in offsets) + itemsize,
    )


def _validate_output_ownership(
    mass_transfer: Any | None,
    energy_transfer: Any | None,
    primary_fields: tuple[Any, ...],
) -> None:
    """Reject writable outputs overlapping primary fields or each other.

    This check reads pointer, shape, stride, and dtype metadata only. Omitted
    and empty outputs have no storage range and therefore cannot overlap.

    Args:
        mass_transfer: Optional writable mass-transfer output.
        energy_transfer: Optional writable energy-transfer output.
        primary_fields: Primary state arrays that outputs must not overlap.

    Raises:
        ValueError: If a supplied output is non-contiguous or aliases primary
            state or the other supplied output.
    """
    outputs = (
        ("mass_transfer", mass_transfer),
        ("energy_transfer", energy_transfer),
    )
    output_ranges = [
        (name, array, _memory_range(name, array, 8))
        for name, array in outputs
        if array is not None
    ]
    primary_ranges = [
        _memory_range(
            "primary state",
            array,
            _dtype_itemsize(array.dtype),
            require_contiguous=False,
        )
        for array in primary_fields
    ]
    for name, array, output_range in output_ranges:
        if output_range is None:
            continue
        for primary, primary_range in zip(
            primary_fields, primary_ranges, strict=True
        ):
            if array is primary or _overlaps(output_range, primary_range):
                raise ValueError(f"{name} must not overlap primary state.")
        for other_name, other_array, other_range in output_ranges:
            if name != other_name and (
                array is other_array or _overlaps(output_range, other_range)
            ):
                raise ValueError(f"{name} must not overlap {other_name}.")


def _dtype_itemsize(dtype: Any) -> int:
    """Return the fixed byte size of a P2 primary Warp dtype.

    Args:
        dtype: P2-validated ``wp.float64`` or ``wp.int32`` dtype metadata.

    Returns:
        The dtype item size in bytes.
    """
    import warp as wp

    return {
        wp.float64: 8,
        wp.int32: 4,
    }.get(dtype, 8)


def _overlaps(first: tuple[int, int], second: tuple[int, int] | None) -> bool:
    """Return whether two half-open byte ranges overlap.

    Args:
        first: Nonempty first byte range.
        second: Optional second byte range, with ``None`` representing empty
            storage.

    Returns:
        True if both ranges contain at least one common byte.
    """
    return second is not None and first[0] < second[1] and second[0] < first[1]


@dataclass(frozen=True, eq=False)
class WarpCondensationState:
    """Retain validated caller-owned resident Warp condensation resources.

    This concrete-only P2 carrier lazily imports Warp during construction. It
    validates primary-container metadata and writable-output ownership only.
    Thermodynamics need only be non-None. ``latent_heat`` and deferred
    ``thermal_work`` remain opaque, while writable ``mass_transfer`` and
    ``energy_transfer`` receive metadata and ownership checks. The selected
    Warp adapter forwards all three thermal sidecars by identity; the direct
    kernel owns their dependency, schema, value, and execution validation.
    Construction does not select a profile, execute a kernel or runnable,
    transfer, allocate, synchronize, or validate direct-kernel physics.

    Rejection occurs before a writer launch and mutates nothing. That
    construction guarantee does not promise post-launch rollback for a future
    adapter. Frozen status prevents field rebinding only. All retained
    containers and sidecars remain caller-owned and mutable.

    Args:
        config: Exact P2 condensation execution configuration.
        particles: Resident ``WarpParticleData`` primary container.
        gas: Resident ``WarpGasData`` primary container.
        environment: Resident ``WarpEnvironmentData`` primary container.
        thermodynamics: Required opaque thermodynamics reference.
        activity_surface: Optional opaque activity and surface reference.
        scratch_buffers: Optional opaque scratch-buffer reference.
        mass_transfer: Optional writable mass-transfer output sidecar.
        latent_heat: Optional opaque latent-heat reference.
        energy_transfer: Optional writable energy-transfer output sidecar.
        thermal_work: Optional opaque thermal-work reference.

    Raises:
        TypeError: If the configuration or a primary container has an invalid
            type.
        RuntimeError: If the optional Warp runtime is unavailable when state
            validation requires it.
        ValueError: If primary metadata, required thermodynamics, writable
            output metadata, or writable-output ownership is invalid.
    """

    config: CondensationExecutionConfig
    particles: object
    gas: object
    environment: object
    thermodynamics: object
    activity_surface: object | None = None
    scratch_buffers: object | None = None
    mass_transfer: object | None = None
    latent_heat: object | None = None
    energy_transfer: object | None = None
    thermal_work: object | None = None

    def __post_init__(self) -> None:
        """Perform ordered metadata and writable-output ownership validation.

        The lazy imports make importing this concrete module and constructing
        CPU-only carriers independent of the optional Warp runtime.

        Raises:
            TypeError: If the configuration or primary-container types are
                invalid.
            RuntimeError: If Warp is unavailable.
            ValueError: If required metadata, thermodynamics, or writable
                output ownership is invalid.
        """
        if type(self.config) is not CondensationExecutionConfig:
            raise TypeError(
                "config must be an exact CondensationExecutionConfig."
            )

        try:
            import warp as wp
        except ModuleNotFoundError as error:
            if error.name != "warp":
                raise
            raise RuntimeError(
                "WarpCondensationState requires the optional Warp runtime."
            ) from error
        from particula.gpu.warp_types import (
            WarpEnvironmentData,
            WarpGasData,
            WarpParticleData,
        )

        # ``@wp.struct`` exposes a Warp ``Struct`` descriptor.  Its callable
        # creates a generated instance subclass, so validate against the
        # descriptor's underlying Python class instead of exact type identity.
        particle_type = cast(Any, WarpParticleData).cls
        gas_type = cast(Any, WarpGasData).cls
        environment_type = cast(Any, WarpEnvironmentData).cls
        if not isinstance(self.particles, particle_type):
            raise TypeError("particles must be a WarpParticleData.")
        if not isinstance(self.gas, gas_type):
            raise TypeError("gas must be a WarpGasData.")
        if not isinstance(self.environment, environment_type):
            raise TypeError("environment must be a WarpEnvironmentData.")

        masses = self._validate_primary_arrays(wp)
        boxes, particles, species = masses.shape
        device = masses.device
        if self.thermodynamics is None:
            raise ValueError("thermodynamics must not be None.")
        self._validate_optional_outputs(wp, boxes, particles, species, device)

    def _validate_primary_arrays(self, wp: Any) -> Any:
        """Validate all primary Warp arrays and return the mass storage.

        Args:
            wp: Imported Warp module used for dtype metadata.

        Returns:
            The validated ``particles.masses`` array.

        Raises:
            ValueError: If a required primary array is missing or malformed.
        """
        particles = cast(Any, self.particles)
        gas = cast(Any, self.gas)
        environment = cast(Any, self.environment)
        if not hasattr(particles, "masses"):
            raise ValueError("particles.masses must be a Warp array.")
        masses = particles.masses
        _validate_array("particles.masses", masses, wp.float64, None)
        _validate_array(
            "particles.masses",
            masses,
            wp.float64,
            tuple(masses.shape),
        )
        if len(masses.shape) != 3:
            raise ValueError("particles.masses must have shape (B, N, S).")
        boxes, particle_count, species = masses.shape
        device = masses.device
        _validate_array(
            "particles.concentration",
            particles.concentration,
            wp.float64,
            (boxes, particle_count),
            device,
        )
        _validate_array(
            "particles.charge",
            particles.charge,
            wp.float64,
            (boxes, particle_count),
            device,
        )
        _validate_array(
            "particles.density",
            particles.density,
            wp.float64,
            (species,),
            device,
        )
        _validate_array(
            "particles.volume",
            particles.volume,
            wp.float64,
            (boxes,),
            device,
        )
        _validate_array(
            "gas.molar_mass",
            gas.molar_mass,
            wp.float64,
            (species,),
            device,
        )
        _validate_array(
            "gas.concentration",
            gas.concentration,
            wp.float64,
            (boxes, species),
            device,
        )
        _validate_array(
            "gas.vapor_pressure",
            gas.vapor_pressure,
            wp.float64,
            (boxes, species),
            device,
        )
        _validate_array(
            "gas.partitioning",
            gas.partitioning,
            wp.int32,
            (boxes, species),
            device,
        )
        _validate_array(
            "environment.temperature",
            environment.temperature,
            wp.float64,
            (boxes,),
            device,
        )
        _validate_array(
            "environment.pressure",
            environment.pressure,
            wp.float64,
            (boxes,),
            device,
        )
        _validate_array(
            "environment.saturation_ratio",
            environment.saturation_ratio,
            wp.float64,
            (boxes, species),
            device,
        )
        return masses

    def _validate_optional_outputs(
        self,
        wp: Any,
        boxes: int,
        particle_count: int,
        species: int,
        device: Any,
    ) -> None:
        """Validate optional outputs and ownership against primary state.

        Args:
            wp: Imported Warp module used for dtype metadata.
            boxes: Number of validated boxes in the primary state.
            particles: Number of validated particles in the primary state.
            species: Number of validated species in the primary state.
            device: Device metadata from the validated mass storage.
        """
        particles = cast(Any, self.particles)
        gas = cast(Any, self.gas)
        environment = cast(Any, self.environment)
        if self.mass_transfer is not None:
            _validate_array(
                "mass_transfer",
                self.mass_transfer,
                wp.float64,
                (boxes, particle_count, species),
                device,
            )
        if self.energy_transfer is not None:
            _validate_array(
                "energy_transfer",
                self.energy_transfer,
                wp.float64,
                (boxes, species),
                device,
            )

        _validate_output_ownership(
            self.mass_transfer,
            self.energy_transfer,
            (
                particles.masses,
                particles.concentration,
                particles.charge,
                particles.density,
                particles.volume,
                gas.molar_mass,
                gas.concentration,
                gas.vapor_pressure,
                gas.partitioning,
                environment.temperature,
                environment.pressure,
                environment.saturation_ratio,
            ),
        )

    @property
    def backend_payload(self) -> tuple[object, object, object]:
        """Return the three caller-owned primary containers by identity.

        Returns:
            The ``(particles, gas, environment)`` primary-container tuple.
        """
        return self.particles, self.gas, self.environment


@dataclass(frozen=True, eq=False)
class CPUCondensationExecutionState:
    """Retain selected CPU condensation controls and runnable by identity.

    Import this concrete-only P3 carrier from
    ``particula.execution.adapters.condensation``. Construction validates only
    exact P2 state and runnable types; execution controls remain opaque until
    dispatch. The retained runnable and aerosol are caller-owned and mutable;
    frozen fields prevent rebinding only.

    Args:
        state: Exact P2 CPU state retained by identity.
        time_step: Original execution time step in s, validated at dispatch.
        sub_steps: Original number of CPU substeps, validated at dispatch.
        runnable: Exact caller-owned ``MassCondensation`` runnable.

    Raises:
        TypeError: If ``state`` or ``runnable`` is not the required exact type.
    """

    state: CPUCondensationState
    time_step: object
    sub_steps: object
    runnable: MassCondensation

    def __post_init__(self) -> None:
        """Validate exact P2-state and runnable carrier types.

        Raises:
            TypeError: If ``state`` or ``runnable`` is not the required exact
                type.
        """
        if type(self.state) is not CPUCondensationState:
            raise TypeError("state must be an exact CPUCondensationState.")
        if type(self.runnable) is not MassCondensation:
            raise TypeError("runnable must be an exact MassCondensation.")

    @property
    def backend_payload(self) -> Aerosol:
        """Return the exact caller-owned CPU aerosol payload.

        Returns:
            The aerosol retained by the P2 state.
        """
        return self.state.backend_payload


@dataclass(frozen=True, eq=False)
class WarpCondensationExecutionState:
    """Retain selected resident-Warp condensation controls by identity.

    This concrete-only P3 carrier preserves the exact P2 state and time step.
    Construction neither imports a kernel nor inspects retained resources.

    Args:
        state: Exact P2 Warp state retained by identity.
        time_step: Original execution time step in s, validated at dispatch.

    Raises:
        TypeError: If ``state`` is not an exact ``WarpCondensationState``.
    """

    state: WarpCondensationState
    time_step: object

    def __post_init__(self) -> None:
        """Validate the exact P2 Warp state carrier type.

        Raises:
            TypeError: If ``state`` is not an exact ``WarpCondensationState``.
        """
        if type(self.state) is not WarpCondensationState:
            raise TypeError("state must be an exact WarpCondensationState.")

    @property
    def backend_payload(self) -> tuple[object, object, object]:
        """Return the exact caller-owned resident primary payload.

        Returns:
            The ``(particles, gas, environment)`` tuple from the P2 state.
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


def _require_isothermal(configuration: CondensationConfiguration) -> None:
    """Reject semantic latent heat at the selected isothermal boundary.

    Args:
        configuration: Selected condensation configuration to inspect.

    Raises:
        ValueError: If the configuration enables latent heat.
    """
    if configuration.latent_heat:
        raise ValueError(
            "isothermal condensation execution requires latent_heat=False."
        )


def _get_condensation_step_gpu() -> Any:
    """Lazily resolve the optional direct Warp kernel after preflight.

    Returns:
        The direct ``condensation_step_gpu`` kernel entry point.

    Raises:
        ImportError: If the optional Warp kernel dependencies are unavailable.
    """
    from particula.gpu.kernels import condensation_step_gpu

    return condensation_step_gpu


class CPUCondensationExecutionAdapter:
    """Dispatch one selected isothermal CPU condensation request exactly once.

    This concrete-only adapter performs local preflight, then invokes the
    caller-owned ``MassCondensation`` runnable once without splitting controls.
    It neither converts state nor catches delegate exceptions. The runnable must
    return the original aerosol, and successful results declare state mutation.

    Backend exceptions propagate unchanged. This adapter is concrete-only and
    must be imported from ``particula.execution.adapters.condensation``.
    """

    def execute(self, state: ExecutionState) -> ExecutionResult:
        """Execute one exact CPU P3 state without fallback or recovery.

        Args:
            state: Exact selected CPU P3 execution state.

        Returns:
            A result retaining ``state`` and the original aerosol by identity.

        Raises:
            TypeError: If ``state`` or its time step has an invalid type.
            ValueError: If controls or the selected profile are invalid, latent
                heat is enabled, or the runnable returns another aerosol.
        """
        if type(state) is not CPUCondensationExecutionState:
            raise TypeError("state must be a CPUCondensationExecutionState.")
        _validate_time_step(state.time_step)
        if (
            isinstance(state.sub_steps, bool)
            or not isinstance(state.sub_steps, Integral)
            or state.sub_steps <= 0
        ):
            raise ValueError("sub_steps must be a positive integer.")
        configuration = state.state.config.configuration
        require_condensation_profile(Backend.CPU, configuration)
        _require_isothermal(configuration)
        aerosol = state.runnable.execute(
            state.state.aerosol,
            cast(float, state.time_step),
            cast(int, state.sub_steps),
        )
        if aerosol is not state.state.aerosol:
            raise ValueError("CPU runnable must return the original aerosol.")
        return ExecutionResult(
            state,
            (),
            MutationDeclaration(frozenset({MutationScope.STATE})),
            BackendResult(aerosol),
        )


class WarpCondensationExecutionAdapter:
    """Dispatch one selected Warp request to its direct kernel.

    Preflight completes before the optional kernel import. The adapter makes one
    native call without conversion, allocation, restoration, synchronization,
    fallback, or exception recovery. Profile preflight completes before lazy
    resolution. It forwards caller-owned ``latent_heat``, ``energy_transfer``,
    and deferred ``thermal_work`` sidecars by identity. The direct kernel owns
    thermal-sidecar validation, execution, and post-launch mutation limits.

    This concrete-only adapter is imported from
    ``particula.execution.adapters.condensation``. It forwards the exact
    native kernel tuple without reconstructing it.
    """

    def execute(self, state: ExecutionState) -> ExecutionResult:
        """Execute one exact Warp P3 state with no post-launch recovery.

        Profile preflight occurs before lazy kernel resolution. After that
        preflight, this method forwards the caller-owned thermal sidecars by
        identity and leaves their validation and any post-launch behavior to the
        direct kernel.

        Args:
            state: Exact selected Warp P3 execution state.

        Returns:
            A result retaining ``state`` and the native kernel result by
            identity.

        Raises:
            TypeError: If ``state`` or its time step has an invalid type.
            ValueError: If controls or the selected profile are invalid, or if
                the direct kernel rejects thermal sidecars. Direct errors
                propagate unchanged.
            ImportError: If the optional Warp kernel cannot be imported after
                successful preflight.
        """
        if type(state) is not WarpCondensationExecutionState:
            raise TypeError("state must be a WarpCondensationExecutionState.")
        _validate_time_step(state.time_step)
        p2_state = state.state
        configuration = p2_state.config.configuration
        require_condensation_profile(Backend.WARP, configuration)
        condensation_step_gpu = _get_condensation_step_gpu()
        value = condensation_step_gpu(
            p2_state.particles,
            p2_state.gas,
            None,
            None,
            state.time_step,
            mass_transfer=p2_state.mass_transfer,
            environment=p2_state.environment,
            thermodynamics=p2_state.thermodynamics,
            activity_surface=p2_state.activity_surface,
            scratch_buffers=p2_state.scratch_buffers,
            latent_heat=p2_state.latent_heat,
            energy_transfer=p2_state.energy_transfer,
            thermal_work=p2_state.thermal_work,
        )
        return ExecutionResult(
            state,
            (),
            MutationDeclaration(frozenset({MutationScope.STATE})),
            BackendResult(value),
        )


# Short direct-module aliases retain compatibility with the P2 carrier naming.
CPUCondensationAdapter = CPUCondensationExecutionAdapter
WarpCondensationAdapter = WarpCondensationExecutionAdapter
