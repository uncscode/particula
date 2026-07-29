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

    This concrete-only carrier preserves every reference by identity and leaves
    physical validation, RNG behavior, and writer failures to the direct step.
    It neither copies retained state nor provides recovery.

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
        initialize_rng: Opaque reset flag forwarded unchanged to the kernel.
    """

    session: ResidentSession
    registry: GPUResourceRegistry
    resources: WallLossResources
    config: object
    time_step: object
    rng_seed: object = 0
    initialize_rng: object = False

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
    metadata-only preflight. It preserves container, sidecar, and RNG identity
    and provides no acquisition, transfer, synchronization, or recovery.
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

        Direct-kernel exceptions and mutations propagate without adapter retry,
        rollback, recovery, transfer, or synchronization.
        """
        if type(request) is not ResidentWallLossRequest:
            raise TypeError("request must be an exact ResidentWallLossRequest.")
        request.registry.validate_pinned_session(request.session)
        request.registry.validate_wall_loss_resources(
            request.session, request.resources
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
            initialize_rng=request.initialize_rng,
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
