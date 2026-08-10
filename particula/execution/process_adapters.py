"""Delegate resident processes through established concrete resource views.

This direct-import-only module retains resident sessions, registries, and
published resource views by identity. Its adapters perform metadata-only
preflight and call one supported direct GPU kernel without transfers,
synchronization, acquisition, fallback, retry, rollback, or physics. These
concrete names are deliberately not exported from :mod:`particula.execution`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

from particula.execution.gpu_session import ResidentSession

if TYPE_CHECKING:
    from particula.execution.gpu_resources import (
        GPUResourceRegistry,
        NucleationResources,
        WallLossResources,
    )


def _resource_types() -> tuple[type[object], type[object], type[object]]:
    """Import concrete resource types only when a carrier needs them."""
    from particula.execution.gpu_resources import (
        GPUResourceRegistry,
        NucleationResources,
        WallLossResources,
    )

    return GPUResourceRegistry, WallLossResources, NucleationResources


@dataclass(frozen=True, eq=False)
class ResidentDilutionRequest:
    """Retain one dilution call's resident binding and opaque kernel inputs.

    This concrete-only carrier retains references without inspecting physical
    values. The direct dilution kernel owns all numerical validation and writer
    failure semantics. It neither copies nor recovers the retained state.

    Attributes:
        session: Exact concrete resident-session reference. Execution validates
            that it is active and pinned by ``registry``.
        registry: Exact concrete registry reference. Execution validates its
            binding to ``session``.
        coefficient: Opaque dilution coefficient for the direct kernel.
        time_step: Opaque duration for the direct kernel.
    """

    session: ResidentSession
    registry: GPUResourceRegistry
    coefficient: object
    time_step: object

    def __post_init__(self) -> None:
        """Validate exact resident dependency types without kernel imports.

        Raises:
            TypeError: If ``session`` or ``registry`` is not its exact required
                concrete type.
        """
        if type(self.session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        registry_type, _, _ = _resource_types()
        if type(self.registry) is not registry_type:
            raise TypeError("registry must be an exact GPUResourceRegistry.")


@dataclass(frozen=True, eq=False)
class ResidentWallLossRequest:
    """Retain one wall-loss call and its established RNG resource view.

    This concrete-only carrier preserves every reference by identity. It
    validates scheduler-owned logical-box selection and requires direct-kernel
    RNG reset to remain disabled; physical validation and writer failure
    semantics remain with the direct step. It neither copies retained state nor
    provides recovery.

    Attributes:
        session: Exact concrete resident-session reference. Execution validates
            that it is active and pinned by ``registry``.
        registry: Exact concrete registry reference. Execution validates its
            binding to ``session``.
        resources: Exact concrete wall-loss resource view. Execution validates
            that it is the established publication.
        config: Opaque direct-kernel configuration.
        time_step: Opaque duration for the direct kernel.
        rng_seed: Opaque seed forwarded unchanged to the direct kernel.
        initialize_rng: Legacy reset flag. Resident dispatch accepts only the
            literal ``False`` and always disables direct-kernel resets.
        enabled_box_indices: Scheduler-resolved, ascending logical box indices
            to dispatch. ``None`` retains the legacy all-box selection.
    """

    session: ResidentSession
    registry: GPUResourceRegistry
    resources: WallLossResources
    config: object
    time_step: object
    rng_seed: object = 0
    initialize_rng: object = False
    enabled_box_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate exact dependencies without inspecting kernel arguments.

        Raises:
            TypeError: If a resident dependency or resource view is not its
                exact required concrete type.
        """
        if type(self.session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        registry_type, wall_loss_type, _ = _resource_types()
        if type(self.registry) is not registry_type:
            raise TypeError("registry must be an exact GPUResourceRegistry.")
        if type(self.resources) is not wall_loss_type:
            raise TypeError("resources must be an exact WallLossResources.")
        if self.initialize_rng is not False:
            raise ValueError("resident wall_loss initialize_rng must be False.")
        self.validate_enabled_box_indices()

    def validate_enabled_box_indices(self) -> tuple[int, ...]:
        """Validate and return the scheduler-owned wall-loss launch set.

        ``None`` selects every logical box in ascending order. An explicit tuple
        must already be strictly ascending, unique, and within the resident
        box range; this preserves the one-to-one logical-lane mapping used by
        the persistent wall-loss RNG sidecar.

        Returns:
            Ascending selected logical-box indices. ``None`` returns all
            resident boxes in ascending order.

        Raises:
            TypeError: If an explicit selection is not a tuple or contains a
                non-integral or boolean index.
            ValueError: If an index is negative, repeated, unordered, or
                outside the resident box range.
        """
        if self.enabled_box_indices is None:
            return tuple(range(self.session.dimensions.n_boxes))
        indices = self.enabled_box_indices
        if type(indices) is not tuple:
            raise TypeError("enabled_box_indices must be a tuple or None.")
        previous = -1
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, Integral):
                raise TypeError(
                    "enabled_box_indices must contain integral indices."
                )
            if index <= previous or index < 0:
                raise ValueError(
                    "enabled_box_indices must be sorted unique nonnegative "
                    "indices."
                )
            if index >= self.session.dimensions.n_boxes:
                raise ValueError(
                    "enabled_box_indices must be within session boxes."
                )
            previous = index
        return tuple(indices)


@dataclass(frozen=True, eq=False)
class ResidentNucleationRequest:
    """Retain one nucleation call and its established sidecar resource view.

    This concrete-only carrier preserves references by identity and delegates
    controls, physical validation, and writer failure semantics unchanged. It
    neither copies retained state nor provides recovery.

    Attributes:
        session: Exact concrete resident-session reference. Execution validates
            that it is active and pinned by ``registry``.
        registry: Exact concrete registry reference. Execution validates its
            binding to ``session``.
        resources: Exact concrete nucleation resource view. Execution validates
            that it is the established publication.
        config: Opaque direct-kernel configuration.
        time_step: Opaque duration for the direct kernel.
        exhaustion_controls: Opaque direct-kernel exhaustion controls.
    """

    session: ResidentSession
    registry: GPUResourceRegistry
    resources: NucleationResources
    config: object
    time_step: object
    exhaustion_controls: object

    def __post_init__(self) -> None:
        """Validate exact dependencies without inspecting kernel arguments.

        Raises:
            TypeError: If a resident dependency or resource view is not its
                exact required concrete type.
        """
        if type(self.session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        registry_type, _, nucleation_type = _resource_types()
        if type(self.registry) is not registry_type:
            raise TypeError("registry must be an exact GPUResourceRegistry.")
        if type(self.resources) is not nucleation_type:
            raise TypeError("resources must be an exact NucleationResources.")


def _get_dilution_step_gpu() -> Callable[..., object]:
    """Lazily import the sole supported direct dilution boundary."""
    from particula.gpu.kernels import dilution_step_gpu

    return dilution_step_gpu


def _get_wall_loss_step_gpu() -> Callable[..., object]:
    """Lazily import the sole supported direct wall-loss boundary."""
    from particula.gpu.kernels import wall_loss_step_gpu

    return wall_loss_step_gpu


def _get_wall_loss_selected_boxes_step_gpu() -> Callable[..., object]:
    """Lazily import the private batched resident wall-loss dispatch seam."""
    from particula.gpu.kernels.wall_loss import (
        wall_loss_selected_boxes_step_gpu,
    )

    return wall_loss_selected_boxes_step_gpu


def _get_nucleation_step_gpu() -> Callable[..., object]:
    """Lazily import the sole supported direct nucleation boundary."""
    from particula.gpu.kernels import nucleation_step_gpu

    return nucleation_step_gpu


class ResidentDilutionAdapter:
    """Delegate one dilution request through its exact pinned session.

    This concrete-only adapter resolves one supported direct kernel after
    metadata-only preflight. It does not acquire resources, transfer data,
    synchronize, retry, roll back, or recover writer failures.
    """

    def execute(self, request: object) -> object:
        """Validate and delegate one dilution request.

        Args:
            request: Exact dilution request retaining the pinned session and
                registry.

        Returns:
            Native result returned by the supported direct dilution kernel.

        Raises:
            TypeError: If ``request`` is not an exact dilution request.
            ValueError: If the request session is not the registry's active,
                pinned session.

        Direct-kernel exceptions and mutations propagate without adapter retry,
        rollback, recovery, transfer, or synchronization.
        """
        if type(request) is not ResidentDilutionRequest:
            raise TypeError("request must be an exact ResidentDilutionRequest.")
        request.registry.validate_pinned_session(request.session)
        step = _get_dilution_step_gpu()
        return step(
            request.session.particles,
            request.session.gas,
            request.coefficient,
            request.time_step,
        )


class ResidentWallLossAdapter:
    """Delegate one wall-loss request through its exact published view.

    This concrete-only adapter resolves one supported direct kernel after
    metadata-only preflight. It preserves container and sidecar identity and
    dispatches selected logical boxes through a private selected-box kernel when
    selection is not all boxes, preventing disabled lanes from reaching the
    public direct kernel.
    It provides no acquisition, transfer, synchronization, rollback, or
    recovery.
    """

    def execute(self, request: object) -> object:
        """Validate and delegate one wall-loss request.

        Args:
            request: Exact wall-loss request retaining the pinned published
                view.

        Returns:
            Native result returned by the supported direct wall-loss kernel.

        Raises:
            TypeError: If ``request`` is not an exact wall-loss request or its
                resource view has the wrong concrete type.
            ValueError: If the session or resource view is no longer the active
                registry-pinned publication.

        An empty selection returns without resolving the kernel. For a partial
        selection, direct dispatch receives a private selected-box launch set.
        Direct-kernel exceptions and mutations propagate without adapter retry,
        rollback, recovery, transfer, or synchronization.
        """
        if type(request) is not ResidentWallLossRequest:
            raise TypeError("request must be an exact ResidentWallLossRequest.")
        request.registry.validate_pinned_session(request.session)
        request.registry.validate_wall_loss_resources(
            request.session, request.resources
        )
        enabled_logical_boxes = request.validate_enabled_box_indices()
        if not enabled_logical_boxes:
            return request.session.particles
        if len(enabled_logical_boxes) != request.session.dimensions.n_boxes:
            import warp as wp

            stream = request.session.metadata.stream
            enabled_physical_lanes = (
                enabled_logical_boxes
                if stream.n_boxes == 0
                else tuple(
                    stream.lanes[index] for index in enabled_logical_boxes
                )
            )
            selected_boxes = wp.array(
                enabled_physical_lanes,
                dtype=wp.int32,
                device=request.resources.rng_states.device,
            )
            return _get_wall_loss_selected_boxes_step_gpu()(
                request.session.particles,
                None,
                None,
                request.time_step,
                config=request.config,
                rng_seed=request.rng_seed,
                rng_states=request.resources.rng_states,
                selected_boxes=selected_boxes,
                environment=request.session.environment,
            )
        step = _get_wall_loss_step_gpu()
        return step(
            request.session.particles,
            None,
            None,
            request.time_step,
            config=request.config,
            rng_seed=request.rng_seed,
            rng_states=request.resources.rng_states,
            initialize_rng=False,
            environment=request.session.environment,
        )


class ResidentNucleationAdapter:
    """Delegate one nucleation request through its exact published view.

    This concrete-only adapter resolves one supported direct kernel after
    metadata-only preflight. It preserves container and sidecar identity and
    provides no acquisition, transfer, synchronization, or recovery.
    """

    def execute(self, request: object) -> object:
        """Validate and delegate one nucleation request without altering result.

        Args:
            request: Exact nucleation request retaining the pinned published
                view.

        Returns:
            Native result returned by the supported direct nucleation kernel.

        Raises:
            TypeError: If ``request`` is not an exact nucleation request or its
                resource view has the wrong concrete type.
            ValueError: If the session or resource view is no longer the active
                registry-pinned publication.

        Direct-kernel exceptions and mutations propagate without adapter retry,
        rollback, recovery, transfer, or synchronization.
        """
        if type(request) is not ResidentNucleationRequest:
            raise TypeError(
                "request must be an exact ResidentNucleationRequest."
            )
        request.registry.validate_pinned_session(request.session)
        request.registry.validate_nucleation_resources(
            request.session, request.resources
        )
        step = _get_nucleation_step_gpu()
        return step(
            request.session.particles,
            request.session.gas,
            request.config,
            request.time_step,
            scratch=request.resources.scratch,
            finalized_demand=request.resources.finalized_demand,
            diagnostics=request.resources.diagnostics,
            exhaustion_controls=request.exhaustion_controls,
            exhaustion_buffers=request.resources.exhaustion,
            temperature=None,
            saturation=None,
            environment=request.session.environment,
        )
