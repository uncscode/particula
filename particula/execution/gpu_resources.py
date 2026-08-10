"""Allocate concrete reusable Warp sidecars for one active resident session.

This direct-import-only, Warp-dependent boundary pins complete fixed-shape
native sidecar families to one exact ``ACTIVE`` :class:`ResidentSession`.
It allocates and validates resources only: it neither executes a process,
transfers, synchronizes, nor resizes. Coagulation and wall-loss acquisition
initialize distinct P1-derived persistent RNG streams exactly once before
publishing their resident resources. The checkpoint-private restoration seam
instead publishes prevalidated current stream words without reseeding. The
manifests and views here are concrete-only and are deliberately not exported
from :mod:`particula.execution`.

The registry retains array identities and performs metadata-only schema and
nonaliasing checks. It does not establish allocator provenance, execute a
kernel, or change session lifecycle. Explicit lifecycle methods may inspect
frozen stream metadata or reset selected published lanes without hidden
transfer or synchronization.
``validate_pinned_session`` is the narrow direct-module-only integration seam
for resident timestep guards. It requires the exact retained session, then
revalidates its active lifecycle, pinned container and primary-array identities,
and schema metadata without inspecting payloads, acquiring sidecars, allocating,
 transferring, or synchronizing.

For concrete checkpointing, the private deterministic enumeration seam exposes
established live sidecars only after the same active pinned-session validation.
Checkpoint code owns any immutable host copy it creates; this registry retains
caller- or registry-owned device arrays. Enumeration neither copies nor
transfers payloads, and this module offers no restart, migration, or rollback
after launched device work.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import warp as wp

from particula.execution.communication import (
    CommunicationConfiguration,
    CommunicationTransportMode,
    validate_communication_configuration,
)
from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentSession,
)
from particula.execution.rng import (
    StreamManifest,
    StreamRegistry,
    _resolve_stream_selection,
    _StreamWriterError,
)
from particula.gpu.kernels.communication import (
    GasCommunicationBuffers,
    ParticleCommunicationBuffers,
)
from particula.gpu.kernels.condensation import CondensationScratchBuffers
from particula.gpu.kernels.exhaustion import ResamplingBuffers
from particula.gpu.kernels.nucleation import (
    NucleationDiagnosticBuffers,
    NucleationExhaustionBuffers,
    NucleationFinalizedDemandBuffers,
    NucleationScratchBuffers,
)

__all__ = [
    "ManifestEntry",
    "ResourceManifest",
    "PublishedStreamManifest",
    "GPUResourceRegistry",
    "CondensationResources",
    "CoagulationResources",
    "WallLossResources",
    "NucleationResources",
    "CommunicationResources",
]

_INT32_MAX = 2**31 - 1
_MAX_SIZE = (1 << 63) - 1
_ShapeKind = Literal["b", "bn", "bs", "bns", "bc2", "e", "en", "status"]


@wp.kernel
def _scan_diagnostic_accounting(
    values: wp.array2d(dtype=wp.float64),
    require_nonnegative: bool,
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Record invalid diagnostic accounting values in one device status lane."""
    box, species = wp.tid()
    value = values[box, species]
    if not wp.isfinite(value) or (require_nonnegative and value < 0.0):
        wp.atomic_max(invalid, 0, 1)


@dataclass(frozen=True)
class ManifestEntry:
    """Describe one fixed-shape concrete sidecar role."""

    role: str
    family: str
    dtype: Any
    shape_kind: _ShapeKind


@dataclass(frozen=True)
class ResourceManifest:
    """Declare immutable sidecar schemas for one resource family."""

    family: str
    entries: tuple[ManifestEntry, ...]


@dataclass(frozen=True)
class PublishedStreamManifest:
    """Describe immutable identity metadata for currently published streams.

    No live device arrays, pointers, device values, or current stream words are
    exposed by this inspection carrier.
    """

    stream: StreamManifest
    published_process_ids: tuple[str, ...]
    sidecar_roles: tuple[tuple[str, str], ...]


@dataclass(frozen=True, eq=False)
class CondensationResources:
    """Expose a complete native condensation scratch record."""

    scratch_buffers: CondensationScratchBuffers


@dataclass(frozen=True, eq=False)
class CoagulationResources:
    """Expose native coagulation outputs and one P1-initialized RNG sidecar.

    The registry publishes this identity-bound view only after validating all
    supplied sidecars and initializing the coagulation-only ``rng_states`` once
    from immutable resident stream metadata. Repeated compatible acquisition
    returns this view and its arrays by identity without allocation, reseeding,
    readback, transfer, or synchronization. The sidecar has no wall-loss,
    public checkpoint, reset, or inspection API. Schema-v3 checkpoint restart
    can privately restore fresh bindings from captured current words.
    """

    collision_capacity: int
    collision_pairs: Any
    n_collisions: Any
    rng_states: Any


@dataclass(frozen=True, eq=False)
class WallLossResources:
    """Expose one independently initialized wall-loss RNG sidecar.

    The registry publishes this identity-bound view only after initializing its
    ``rng_states`` once from the wall-loss process namespace. Compatible
    reacquisition returns the same view and sidecar by identity without
    allocation or reseeding. The sidecar is distinct from coagulation state and
    has no public reset or inspection API. Schema-v3 checkpoint restart can
    privately restore fresh bindings from captured current words.
    """

    rng_states: Any


@dataclass(frozen=True, eq=False)
class NucleationResources:
    """Expose complete native nucleation sidecar records."""

    scratch: NucleationScratchBuffers
    finalized_demand: NucleationFinalizedDemandBuffers
    diagnostics: NucleationDiagnosticBuffers
    exhaustion: NucleationExhaustionBuffers


@dataclass(frozen=True, eq=False)
class CommunicationResources:
    """Expose one exact resident communication configuration and work record.

    The registry publishes this identity-bound view only after P1 configuration
    validation and complete sidecar schema/nonaliasing checks. The
    configuration, native work record, and optional final volumes remain
    caller- or registry-owned device state; this view performs no transfer,
    synchronization, payload inspection, or mutation. It represents either the
    GAS or PARTICLES family, never a combined or open-map configuration.

    Attributes:
        configuration: Exact closed-map configuration retained by identity.
        buffers: Exact mode-matched native work record retained by identity.
        final_volumes: Optional pinned ``float64`` per-box target volumes.
    """

    configuration: CommunicationConfiguration
    buffers: GasCommunicationBuffers | ParticleCommunicationBuffers
    final_volumes: Any | None
    execution_state: "ResidentCommunicationState"


@dataclass(frozen=True, eq=False)
class ResidentCommunicationState:
    """Expose registry-pinned status and snapshot storage for barriers."""

    invalid: Any
    active_or_demand: Any
    volume_invalid: Any
    volume_changed: Any
    initial_masses: Any | None = None
    initial_concentration: Any | None = None
    initial_charge: Any | None = None


class _RestoredStreamRegistry:
    """Represent one restored published stream without a sibling allocation."""

    def __init__(
        self,
        root_seed: int,
        logical_box_ids: tuple[str, ...],
        lanes: tuple[int, ...],
        process_id: str,
        state: Any,
    ) -> None:
        """Retain validated continuation metadata and authoritative state."""
        self._root_seed = root_seed
        self._logical_box_ids = logical_box_ids
        self._lanes = lanes
        self._process_id = process_id
        self._state = state

    def inspect(self) -> StreamManifest:
        """Return the frozen descriptor metadata for the restored process."""
        from particula.execution.rng import StreamDescriptor, StreamKey

        return StreamManifest(
            self._root_seed,
            self._logical_box_ids,
            self._lanes,
            tuple(
                StreamDescriptor(StreamKey(1, self._process_id, name), lane)
                for name, lane in zip(
                    self._logical_box_ids, self._lanes, strict=True
                )
            ),
        )

    def preflight_selected(
        self, *, process_ids: tuple[str, ...], logical_box_ids: tuple[str, ...]
    ) -> None:
        """Validate explicit reset selectors and the retained state binding."""
        selected, ids = _resolve_stream_selection(
            process_ids,
            logical_box_ids,
            registered_logical_box_ids=self._logical_box_ids,
        )
        if selected != (self._process_id,) or not ids:
            raise ValueError("Requested RNG stream has not been acquired.")
        from particula.execution.rng import _validate_warp_state_array

        _validate_warp_state_array(
            self._state, self._process_id, len(self._lanes), wp
        )

    def initialize_selected(
        self, *, process_ids: tuple[str, ...], logical_box_ids: tuple[str, ...]
    ) -> None:
        """Explicitly derive and reset only selected restored stream lanes."""
        self.preflight_selected(
            process_ids=process_ids, logical_box_ids=logical_box_ids
        )
        from particula.execution.rng import StreamKey, _derive_initial_word

        lane_by_id = dict(zip(self._logical_box_ids, self._lanes, strict=True))
        words = np.asarray(
            [
                _derive_initial_word(
                    self._root_seed, StreamKey(1, self._process_id, name)
                )
                for name in logical_box_ids
            ],
            dtype=np.uint32,
        )
        lanes = np.asarray(
            [lane_by_id[name] for name in logical_box_ids], dtype=np.int32
        )
        from particula.execution.rng import _selected_write_kernel

        wp.launch(
            _selected_write_kernel(wp),
            dim=len(logical_box_ids),
            inputs=[
                self._state,
                wp.array(lanes, dtype=wp.int32, device="cpu"),
                wp.array(words, dtype=wp.uint32, device="cpu"),
            ],
            device=self._state.device,
        )

    def state_array_for(self, process_id: str) -> Any:
        """Return the sole restored state binding by identity."""
        if process_id != self._process_id:
            raise ValueError("process_id is unsupported.")
        return self._state

    def word_for(self, process_id: str, logical_box_id: str) -> int:
        """Derive one initial word for explicit reset inspection only."""
        if process_id != self._process_id:
            raise ValueError("process_id is unsupported.")
        if logical_box_id not in self._logical_box_ids:
            raise LookupError(
                "No stream is registered for process and logical ID."
            )
        from particula.execution.rng import StreamKey, _derive_initial_word

        return _derive_initial_word(
            self._root_seed, StreamKey(1, process_id, logical_box_id)
        )

    def words_by_lane(self, process_id: str) -> tuple[int, ...]:
        """Return derived initial words indexed by physical lane."""
        words = [0] * len(self._lanes)
        for logical_box_id, lane in zip(
            self._logical_box_ids, self._lanes, strict=True
        ):
            words[lane] = self.word_for(process_id, logical_box_id)
        return tuple(words)


_PublishedStreamRegistry = StreamRegistry | _RestoredStreamRegistry


def _entry(
    role: str, family: str, dtype: Any, shape_kind: _ShapeKind
) -> ManifestEntry:
    """Create one terse canonical manifest entry."""
    return ManifestEntry(role, family, dtype, shape_kind)


_CONDENSATION = ResourceManifest(
    "condensation",
    (
        _entry("work_mass_transfer", "condensation", wp.float64, "bns"),
        _entry("total_mass_transfer", "condensation", wp.float64, "bns"),
        _entry("dynamic_viscosity", "condensation", wp.float64, "b"),
        _entry("mean_free_path", "condensation", wp.float64, "b"),
        _entry(
            "positive_mass_transfer_demand", "condensation", wp.float64, "bs"
        ),
        _entry(
            "negative_mass_transfer_release", "condensation", wp.float64, "bs"
        ),
        _entry(
            "positive_mass_transfer_scale", "condensation", wp.float64, "bs"
        ),
    ),
)
_COAGULATION = ResourceManifest(
    "coagulation",
    (
        _entry("collision_pairs", "coagulation", wp.int32, "bc2"),
        _entry("n_collisions", "coagulation", wp.int32, "b"),
        _entry("rng_states", "coagulation", wp.uint32, "b"),
    ),
)
_WALL_LOSS = ResourceManifest(
    "wall_loss", (_entry("rng_states", "wall_loss", wp.uint32, "b"),)
)
_RESAMPLING_ENTRIES = (
    _entry("retained_counts", "nucleation", wp.int32, "b"),
    _entry("released_counts", "nucleation", wp.int32, "b"),
    _entry("retained_indices", "nucleation", wp.int32, "bn"),
    _entry("released_indices", "nucleation", wp.int32, "bn"),
    _entry("sorted_indices", "nucleation", wp.int32, "bn"),
    _entry("replacement_masses", "nucleation", wp.float64, "bns"),
    _entry("replacement_concentration", "nucleation", wp.float64, "bn"),
    _entry("replacement_charge", "nucleation", wp.float64, "bn"),
    _entry("source_radii", "nucleation", wp.float64, "bn"),
    _entry("radius_cubed_relative_error", "nucleation", wp.float64, "b"),
    _entry("mean_radius_relative_error", "nucleation", wp.float64, "b"),
    _entry("surface_relative_error", "nucleation", wp.float64, "b"),
    _entry("diversity_absolute_error", "nucleation", wp.float64, "b"),
    _entry("planning_status", "nucleation", wp.int32, "b"),
)
_NUCLEATION = ResourceManifest(
    "nucleation",
    (
        _entry("precursor_number_concentration", "nucleation", wp.float64, "b"),
        _entry("potential_rate", "nucleation", wp.float64, "b"),
        _entry("potential_demand", "nucleation", wp.float64, "b"),
        _entry("accepted_counts", "nucleation", wp.int32, "b"),
        _entry("accepted_demand", "nucleation", wp.float64, "b"),
        _entry("precursor_mass_change", "nucleation", wp.float64, "bs"),
        _entry("gate_codes", "nucleation", wp.int32, "b"),
        _entry("selected_slot_indices", "nucleation", wp.int32, "bn"),
        _entry("free_slot_indices", "nucleation", wp.int32, "bn"),
        _entry("active_slot_counts", "nucleation", wp.int32, "b"),
        _entry("free_slot_counts", "nucleation", wp.int32, "b"),
        *_RESAMPLING_ENTRIES,
        _entry("demand_workspace", "nucleation", wp.float64, "b"),
        _entry("final_demand", "nucleation", wp.float64, "b"),
        _entry("requested_scale", "nucleation", wp.float64, "b"),
        _entry("minimum_scale", "nucleation", wp.float64, "b"),
        _entry("minimum_volume", "nucleation", wp.float64, "b"),
        _entry("resolved_scale", "nucleation", wp.float64, "b"),
        _entry("resampling_releasable_counts", "nucleation", wp.int32, "b"),
        _entry("required_release_counts", "nucleation", wp.int32, "b"),
        _entry("scaling_required", "nucleation", wp.int32, "b"),
        _entry("final_counts", "nucleation", wp.int32, "b"),
        _entry("final_selected_slot_indices", "nucleation", wp.int32, "bn"),
    ),
)

_GAS_COMMUNICATION = ResourceManifest(
    "communication_gas",
    (
        _entry("source_boxes", "communication_gas", wp.int32, "e"),
        _entry("destination_boxes", "communication_gas", wp.int32, "e"),
        _entry("enabled", "communication_gas", wp.int32, "e"),
        _entry("rates", "communication_gas", wp.float64, "e"),
        _entry("amounts", "communication_gas", wp.float64, "bs"),
        _entry("amount_deltas", "communication_gas", wp.float64, "bs"),
        _entry("outbound_amounts", "communication_gas", wp.float64, "bs"),
        _entry("invalid", "communication_gas", wp.int32, "status"),
        _entry("active_or_demand", "communication_gas", wp.int32, "status"),
        _entry("volume_invalid", "communication_gas", wp.int32, "status"),
        _entry("volume_changed", "communication_gas", wp.int32, "status"),
    ),
)
_PARTICLE_COMMUNICATION = ResourceManifest(
    "communication_particles",
    (
        _entry("source_boxes", "communication_particles", wp.int32, "e"),
        _entry("destination_boxes", "communication_particles", wp.int32, "e"),
        _entry("enabled", "communication_particles", wp.int32, "e"),
        _entry("rates", "communication_particles", wp.float64, "e"),
        _entry("source_debits", "communication_particles", wp.float64, "bn"),
        _entry(
            "destination_credits", "communication_particles", wp.float64, "bn"
        ),
        _entry("assignments", "communication_particles", wp.int32, "en"),
        _entry(
            "request_concentrations",
            "communication_particles",
            wp.float64,
            "en",
        ),
        _entry("invalid", "communication_particles", wp.int32, "status"),
        _entry(
            "active_or_demand", "communication_particles", wp.int32, "status"
        ),
        _entry("volume_invalid", "communication_particles", wp.int32, "status"),
        _entry("volume_changed", "communication_particles", wp.int32, "status"),
        _entry("initial_masses", "communication_particles", wp.float64, "bns"),
        _entry(
            "initial_concentration", "communication_particles", wp.float64, "bn"
        ),
        _entry("initial_charge", "communication_particles", wp.float64, "bn"),
    ),
)


def _primary_arrays(session: ResidentSession) -> tuple[Any, ...]:
    """Return the protected resident primary arrays in canonical order."""
    particles = cast(Any, session.particles)
    gas = cast(Any, session.gas)
    environment = cast(Any, session.environment)
    return (
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
    )


def _item_size(dtype: Any) -> int:
    """Return the supported manifest item size without dtype coercion."""
    if dtype == wp.float64:
        return 8
    if dtype == wp.int32 or dtype == wp.uint32:
        return 4
    raise ValueError("Unsupported manifest dtype.")


class GPUResourceRegistry:
    """Pin reusable complete native sidecars to one exact active session.

    Publication pins caller- or registry-allocated Warp objects by role. This
    validates identity and nonaliasing, not unverifiable allocator provenance.
    No payload is read, copied, synchronized, or mutated by acquisition, except
    that first coagulation or wall-loss acquisition initializes its distinct
    P1-derived RNG sidecar before publication. Its
    concrete-only :meth:`validate_pinned_session` seam lets lifecycle guards
    verify the exact active binding without resource acquisition or execution.
    Its private checkpoint enumeration reports ordinary acquired sidecars in
    manifest order and published RNG bindings in canonical process order,
    without changing ownership or creating host copies. Checkpoint restart may
    privately publish prevalidated fresh RNG bindings without reseeding.
    """

    def __init__(self, session: ResidentSession) -> None:
        """Create a sidecar registry pinned to one active resident session.

        Args:
            session: Exact active resident session that supplies the fixed
                dimensions, device, and protected primary-array identities.

        Raises:
            TypeError: If ``session`` or its lifecycle carriers are not exact
                resident-session types.
            ValueError: If the session is not active or fails its metadata
                validation.
        """
        if type(session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        self._session = session
        self._validate_session_state()
        self._signature = self._session_signature()
        self._bindings: dict[str, dict[str, Any]] = {}
        self._views: dict[str, Any] = {}
        self._nucleation_records: tuple[Any, ...] | None = None
        self._capacities: dict[str, int] = {}
        self._open_step_token: Any | None = None
        self._coagulation_stream_registry: _PublishedStreamRegistry | None = (
            None
        )
        self._wall_loss_stream_registry: _PublishedStreamRegistry | None = None

    @property
    def manifests(self) -> tuple[ResourceManifest, ...]:
        """Return the canonical immutable direct-module manifest set.

        Returns:
            All established sidecar manifests, including the mutually exclusive
            gas and particle communication families.
        """
        return (
            _CONDENSATION,
            _COAGULATION,
            _WALL_LOSS,
            _NUCLEATION,
            _GAS_COMMUNICATION,
            _PARTICLE_COMMUNICATION,
        )

    def _session_signature(self) -> tuple[Any, ...]:
        """Build the pinned lifecycle, dimension, device, and identity
        signature.

        Returns:
            Immutable metadata used to detect session drift before acquisition.
        """
        particles = cast(Any, self._session.particles)
        return (
            self._session.lifecycle,
            self._session.dimensions,
            particles.masses.device,
            id(self._session.particles),
            id(self._session.gas),
            id(self._session.environment),
            *(id(value) for value in _primary_arrays(self._session)),
        )

    def _validate_session_signature(self) -> None:
        """Reject a fabricated, inactive, or replaced resident session state."""
        self._validate_session_carriers()
        if self._session_signature() != self._signature:
            raise ValueError("ResidentSession signature changed.")

        self._session.__post_init__()

    def validate_pinned_session(self, session: ResidentSession) -> None:
        """Validate the exact active pinned session without acquiring resources.

        This direct-module-only guard seam first requires ``session is`` the
        registry's retained session. It then performs the existing metadata-only
        active-lifecycle, container identity, primary-array identity, and schema
        validation. It does not inspect payloads, acquire sidecars, allocate,
        transfer, synchronize, execute, or mutate registry bindings or views.

        Args:
            session: The exact resident session retained at registry creation.

        Raises:
            ValueError: If ``session`` is not the retained object or its active
                lifecycle, schema, protected container identity, or primary
                array identity signature changed.
        """
        if session is not self._session:
            raise ValueError("session must be the pinned ResidentSession.")
        self._validate_session_signature()

    def _stream_metadata(self) -> tuple[int, tuple[str, ...], tuple[int, ...]]:
        """Return normalized host stream identity for sessions."""
        stream = self._session.metadata.stream
        if stream.n_boxes == 0 and self._session.dimensions.n_boxes:
            boxes = self._session.dimensions.n_boxes
            return (
                0,
                tuple(str(index) for index in range(boxes)),
                tuple(range(boxes)),
            )
        return stream.root_seed, stream.logical_box_ids, stream.lanes

    def _published_stream_registry(
        self, process_id: str
    ) -> _PublishedStreamRegistry | None:
        """Return the published process registry without exposing sidecars."""
        if process_id == "coagulation":
            return self._coagulation_stream_registry
        return self._wall_loss_stream_registry

    def inspect_published_streams(
        self, session: ResidentSession
    ) -> PublishedStreamManifest:
        """Return frozen metadata for currently published resident streams."""
        self.validate_pinned_session(session)
        root_seed, logical_box_ids, lanes = self._stream_metadata()
        published = tuple(
            process_id
            for process_id in ("coagulation", "wall_loss")
            if self._published_stream_registry(process_id) is not None
        )
        descriptors: tuple[Any, ...] = ()
        for process_id in published:
            registry = self._published_stream_registry(process_id)
            if registry is None:
                raise AssertionError("published stream registry is unavailable")
            descriptors += tuple(
                descriptor
                for descriptor in registry.inspect().descriptors
                if descriptor.key.process_id == process_id
            )
        roles = tuple((process_id, "rng_states") for process_id in published)
        return PublishedStreamManifest(
            StreamManifest(root_seed, logical_box_ids, lanes, descriptors),
            published,
            roles,
        )

    def initialize_published_streams(
        self,
        session: ResidentSession,
        *,
        process_ids: tuple[str, ...] | None = None,
        logical_box_ids: tuple[str, ...] | None = None,
    ) -> None:
        """Explicitly reinitialize selected currently published stream lanes."""
        self.validate_pinned_session(session)
        self.assert_step_closed()
        _, registered_ids, _ = self._stream_metadata()
        published = tuple(
            process_id
            for process_id in ("coagulation", "wall_loss")
            if self._published_stream_registry(process_id) is not None
        )
        requested = published if process_ids is None else process_ids
        selected_processes, selected_ids = _resolve_stream_selection(
            requested,
            logical_box_ids,
            registered_logical_box_ids=registered_ids,
        )
        for process_id in selected_processes:
            if process_id not in published:
                raise ValueError("Requested RNG stream has not been acquired.")
        selected_registries: list[tuple[str, _PublishedStreamRegistry]] = []
        for process_id in selected_processes:
            registry = self._published_stream_registry(process_id)
            if registry is None:
                raise AssertionError("published stream registry is unavailable")
            registry.preflight_selected(
                process_ids=(process_id,), logical_box_ids=selected_ids
            )
            selected_registries.append((process_id, registry))
        try:
            for process_id, registry in selected_registries:
                registry.initialize_selected(
                    process_ids=(process_id,), logical_box_ids=selected_ids
                )
        except _StreamWriterError as error:
            from particula.execution.gpu_session import _fault_resident_session

            _fault_resident_session(self._session)
            raise error.error from error

    def validate_diagnostic_outputs(
        self, session: ResidentSession, outputs: tuple[Any, ...]
    ) -> None:
        """Validate caller-owned diagnostic outputs without publishing them.

        The diagnostics boundary owns neither these arrays nor their lifetime.
        This metadata-only check rejects aliasing with resident primaries and
        acquired sidecars, while accepting canonical empty ``(B, S)`` arrays.
        It does not allocate, launch, synchronize, transfer, inspect payloads,
        acquire a sidecar, or mutate registry state.

        Args:
            session: Exact active session pinned by this registry.
            outputs: Exact tuple of caller-owned ``float64`` ``(B, S)`` Warp
                arrays to validate in registration order.

        Raises:
            TypeError: If ``outputs`` is not an exact tuple or an output is not
                a Warp array.
            ValueError: If session ownership, output schema, device, pointer,
                contiguity, or byte-range nonaliasing validation fails.
        """
        self.validate_pinned_session(session)
        if type(outputs) is not tuple:
            raise TypeError("outputs must be an exact tuple.")
        ranges: list[tuple[int, int] | None] = []
        values = list(outputs)
        for output in values:
            entry = ManifestEntry(
                "diagnostic output", "diagnostics", wp.float64, "bs"
            )
            byte_range = self._validate_array(entry, output, capacity=None)
            if byte_range is not None and byte_range[0] == 0:
                raise ValueError(
                    "Nonempty diagnostic outputs must have a valid pointer."
                )
            ranges.append(byte_range)
        protected = list(_primary_arrays(self._session)) + [
            value
            for bindings in self._bindings.values()
            for value in bindings.values()
        ]
        protected_ranges = [self._array_range(value) for value in protected]
        for index, (output, byte_range) in enumerate(
            zip(values, ranges, strict=True)
        ):
            if any(output is value for value in protected):
                raise ValueError(
                    "Diagnostic outputs must not alias resident resources."
                )
            if any(
                self._ranges_overlap(byte_range, item)
                for item in protected_ranges
            ):
                raise ValueError(
                    "Diagnostic output byte ranges must not overlap."
                )
            for other, other_range in zip(
                values[index + 1 :], ranges[index + 1 :], strict=True
            ):
                if output is other or self._ranges_overlap(
                    byte_range, other_range
                ):
                    raise ValueError(
                        "Diagnostic outputs must not overlap each other."
                    )

    def validate_diagnostic_registrations(
        self, session: ResidentSession, registrations: tuple[Any, ...]
    ) -> None:
        """Validate closed diagnostic bindings without acquiring resources.

        Outputs and accounting inputs are caller-owned, same-device contiguous
        arrays. Inputs may alias other inputs because they are read-only, but
        no input or output may overlap resident or acquired storage, and outputs
        may not overlap any diagnostic binding.
        """
        self.validate_pinned_session(session)
        if type(registrations) is not tuple:
            raise TypeError("registrations must be an exact tuple.")
        outputs, output_entries, inputs, input_entries = (
            self._diagnostic_binding_entries(registrations)
        )
        output_ranges = [
            self._validate_array(entry, value, None)
            for entry, value in zip(output_entries, outputs, strict=True)
        ]
        input_ranges = [
            self._validate_array(entry, value, None)
            for entry, value in zip(input_entries, inputs, strict=True)
        ]
        protected = list(_primary_arrays(self._session)) + [
            value
            for bindings in self._bindings.values()
            for value in bindings.values()
        ]
        protected_ranges = [self._array_range(value) for value in protected]
        self._validate_diagnostic_binding_nonalias(
            outputs,
            output_ranges,
            inputs,
            input_ranges,
            protected,
            protected_ranges,
        )
        self._validate_diagnostic_accounting_values(registrations)

    def _validate_diagnostic_accounting_values(
        self, registrations: tuple[Any, ...]
    ) -> None:
        """Reject invalid ledger payloads before any diagnostic writer
        launch.
        """
        dimensions = self._session.dimensions
        if not dimensions.n_boxes or not dimensions.n_species:
            return
        device = cast(Any, self._session.particles).masses.device
        for registration in registrations:
            for value, require_nonnegative in (
                (registration.energy_transfer, False),
                (registration.baseline_total_mass, False),
                (registration.source_ledger, True),
                (registration.sink_ledger, True),
            ):
                if value is None:
                    continue
                invalid = wp.zeros(1, dtype=wp.int32, device=device)
                wp.launch(
                    _scan_diagnostic_accounting,
                    dim=(dimensions.n_boxes, dimensions.n_species),
                    inputs=[value, require_nonnegative, invalid],
                    device=device,
                )
                if invalid.numpy()[0] != 0:
                    if require_nonnegative:
                        raise ValueError(
                            "Diagnostic source and sink ledgers must be finite "
                            "and nonnegative."
                        )
                    raise ValueError(
                        "Diagnostic accounting inputs must be finite."
                    )

    @staticmethod
    def _diagnostic_binding_entries(
        registrations: tuple[Any, ...],
    ) -> tuple[list[Any], list[ManifestEntry], list[Any], list[ManifestEntry]]:
        """Return caller-owned diagnostic bindings and their exact schemas."""
        outputs: list[Any] = []
        inputs: list[Any] = []
        output_entries: list[ManifestEntry] = []
        input_entries: list[ManifestEntry] = []
        for registration in registrations:
            shape_kind: Literal["b", "bs"] = (
                "b"
                if registration.operation.value
                == "particle_number_concentration"
                else "bs"
            )
            outputs.append(registration.output)
            output_entries.append(
                ManifestEntry(
                    "diagnostic output", "diagnostics", wp.float64, shape_kind
                )
            )
            for value in (
                registration.energy_transfer,
                registration.baseline_total_mass,
                registration.source_ledger,
                registration.sink_ledger,
            ):
                if value is not None:
                    inputs.append(value)
                    input_entries.append(
                        ManifestEntry(
                            "diagnostic accounting input",
                            "diagnostics",
                            wp.float64,
                            "bs",
                        )
                    )
        return outputs, output_entries, inputs, input_entries

    def _validate_diagnostic_binding_nonalias(
        self,
        outputs: list[Any],
        output_ranges: list[tuple[int, int] | None],
        inputs: list[Any],
        input_ranges: list[tuple[int, int] | None],
        protected: list[Any],
        protected_ranges: list[tuple[int, int] | None],
    ) -> None:
        """Reject diagnostic bindings that overlap protected/output storage."""
        bindings = outputs + inputs
        ranges = output_ranges + input_ranges
        for index, (value, byte_range) in enumerate(
            zip(bindings, ranges, strict=True)
        ):
            if any(value is protected_value for protected_value in protected):
                raise ValueError(
                    "Diagnostic bindings must not alias resident resources."
                )
            if any(
                self._ranges_overlap(byte_range, item)
                for item in protected_ranges
            ):
                raise ValueError(
                    "Diagnostic binding byte ranges must not overlap."
                )
            if index < len(outputs):
                for other, other_range in zip(
                    bindings[index + 1 :], ranges[index + 1 :], strict=True
                ):
                    if value is other or self._ranges_overlap(
                        byte_range, other_range
                    ):
                        raise ValueError(
                            "Diagnostic outputs must not overlap bindings."
                        )

    def validate_wall_loss_resources(
        self, session: ResidentSession, resources: WallLossResources
    ) -> None:
        """Validate one established wall-loss view without acquiring sidecars.

        This concrete-only adapter seam verifies the exact active session before
        checking that ``resources`` is the already-published wall-loss view. It
        retains the view and its RNG sidecar by identity, performs metadata and
        identity checks only, and does not allocate, acquire, inspect payloads,
        mutate registry state, transfer, synchronize, or recover failures.

        Args:
            session: Exact active session pinned by this registry.
            resources: Exact established wall-loss resource view.

        Raises:
            TypeError: If ``resources`` is not an exact wall-loss view.
            ValueError: If the family is unavailable, the view differs from the
                published view, or its pinned sidecar binding changed.
        """
        self.validate_pinned_session(session)
        if type(resources) is not WallLossResources:
            raise TypeError("resources must be an exact WallLossResources.")
        published = self._views.get("wall_loss")
        if published is None:
            raise ValueError("wall_loss resources have not been acquired.")
        if resources is not published:
            raise ValueError("resources must be the published wall_loss view.")
        bindings = self._bindings["wall_loss"]
        if resources.rng_states is not bindings["rng_states"]:
            raise ValueError("wall_loss resource bindings changed.")
        self._validate_array(
            _WALL_LOSS.entries[0], resources.rng_states, capacity=None
        )

    def validate_condensation_resources(
        self, session: ResidentSession, resources: CondensationResources
    ) -> None:
        """Require the exact established condensation view.

        This does not acquire a new resource binding.
        """
        self.validate_pinned_session(session)
        if type(resources) is not CondensationResources:
            raise TypeError("resources must be an exact CondensationResources.")
        if resources is not self._views.get("condensation"):
            raise ValueError(
                "resources must be the published condensation view."
            )
        for entry in _CONDENSATION.entries:
            value = getattr(resources.scratch_buffers, entry.role)
            if value is not self._bindings["condensation"][entry.role]:
                raise ValueError("condensation resource bindings changed.")
            self._validate_array(entry, value, capacity=None)

    def validate_coagulation_resources(
        self, session: ResidentSession, resources: CoagulationResources
    ) -> None:
        """Require the exact established coagulation view.

        This does not acquire a new resource binding.
        """
        self.validate_pinned_session(session)
        if type(resources) is not CoagulationResources:
            raise TypeError("resources must be an exact CoagulationResources.")
        if resources is not self._views.get("coagulation"):
            raise ValueError(
                "resources must be the published coagulation view."
            )
        if resources.collision_capacity != self._capacities.get("coagulation"):
            raise ValueError("coagulation resource capacity changed.")
        for entry in _COAGULATION.entries:
            value = getattr(resources, entry.role)
            if value is not self._bindings["coagulation"][entry.role]:
                raise ValueError("coagulation resource bindings changed.")
            self._validate_array(
                entry, value, capacity=resources.collision_capacity
            )

    def validate_nucleation_resources(
        self, session: ResidentSession, resources: NucleationResources
    ) -> None:
        """Validate one established nucleation view without acquiring sidecars.

        This concrete-only adapter seam verifies exact active-session ownership,
        the exact published view identity, and every pinned record binding. It
        retains all resource records and sidecars by identity and does not
        allocate, acquire, inspect payloads, mutate state, transfer,
        synchronize, or recover failures.

        Args:
            session: Exact active session pinned by this registry.
            resources: Exact established nucleation resource view.

        Raises:
            TypeError: If ``resources`` is not an exact nucleation view.
            ValueError: If the family is unavailable, the view differs from the
                published view, or a pinned sidecar binding changed.
        """
        self.validate_pinned_session(session)
        if type(resources) is not NucleationResources:
            raise TypeError("resources must be an exact NucleationResources.")
        published = self._views.get("nucleation")
        if published is None:
            raise ValueError("nucleation resources have not been acquired.")
        if resources is not published:
            raise ValueError("resources must be the published nucleation view.")
        published_records = self._nucleation_records
        if published_records is None:
            raise ValueError("nucleation resource records are unavailable.")
        resource_records = (
            resources.scratch,
            resources.finalized_demand,
            resources.diagnostics,
            resources.exhaustion,
            resources.exhaustion.resampling_buffers,
        )
        record_types = (
            NucleationScratchBuffers,
            NucleationFinalizedDemandBuffers,
            NucleationDiagnosticBuffers,
            NucleationExhaustionBuffers,
            ResamplingBuffers,
        )
        if any(
            type(record) is not record_type or record is not published_record
            for record, published_record, record_type in zip(
                resource_records, published_records, record_types, strict=True
            )
        ):
            raise ValueError("nucleation resource record bindings changed.")
        bindings = self._record_bindings(resources.scratch)
        bindings.update(self._record_bindings(resources.finalized_demand))
        bindings.update(self._record_bindings(resources.diagnostics))
        bindings.update(self._record_bindings(resources.exhaustion))
        bindings.update(
            self._record_bindings(resources.exhaustion.resampling_buffers)
        )
        for entry in _NUCLEATION.entries:
            if bindings.get(entry.role) is not self._bindings["nucleation"].get(
                entry.role
            ):
                raise ValueError("nucleation resource bindings changed.")
            self._validate_array(entry, bindings[entry.role], capacity=None)

    def _enumerate_resources(
        self,
    ) -> tuple[tuple[str, str, Any, int | None], ...]:
        """Return established sidecars in deterministic manifest order.

        This checkpoint-only seam validates the exact active pinned session but
        does not synchronize, copy, allocate, or inspect array payloads. Each
        item is ``(family, role, live_array, capacity)``. The returned arrays
        remain registry-owned live device arrays; a checkpoint controller alone
        decides whether and how to capture immutable host bytes.

        Returns:
            Established sidecars as deterministic family/role descriptors with
            live arrays and optional coagulation capacity.

        Raises:
            ValueError: If the exact pinned session is inactive or its protected
                metadata, containers, or primary-array identities drifted.
        """
        self.validate_pinned_session(self._session)
        entries: list[tuple[str, str, Any, int | None]] = []
        for manifest in self.manifests:
            bindings = self._bindings.get(manifest.family)
            if bindings is None:
                continue
            capacity = self._capacities.get(manifest.family)
            entries.extend(
                (manifest.family, entry.role, bindings[entry.role], capacity)
                for entry in manifest.entries
                if entry.role != "rng_states"
            )
        return tuple(entries)

    def _enumerate_published_rng_streams(
        self,
    ) -> tuple[tuple[str, StreamManifest, Any], ...]:
        """Return live published RNG bindings in canonical order.

        This checkpoint-private preflight performs only metadata and identity
        validation.  It deliberately does not read state words or synchronize.
        """
        self.validate_pinned_session(self._session)
        manifest = self.inspect_published_streams(self._session).stream
        result: list[tuple[str, StreamManifest, Any]] = []
        for process_id in ("coagulation", "wall_loss"):
            bindings = self._bindings.get(process_id)
            stream = self._published_stream_registry(process_id)
            if bindings is not None and stream is not None:
                state = bindings.get("rng_states")
                if (
                    state is None
                    or stream.state_array_for(process_id) is not state
                ):
                    raise ValueError(
                        "published RNG binding identity is invalid."
                    )
                entry = (
                    _COAGULATION.entries[2]
                    if process_id == "coagulation"
                    else _WALL_LOSS.entries[0]
                )
                self._validate_array(
                    entry,
                    state,
                    self._capacities.get(process_id),
                )
                self._reject_primary_aliases([state])
                expected_descriptors = tuple(
                    descriptor
                    for descriptor in manifest.descriptors
                    if descriptor.key.process_id == process_id
                )
                actual_descriptors = tuple(
                    descriptor
                    for descriptor in stream.inspect().descriptors
                    if descriptor.key.process_id == process_id
                )
                if actual_descriptors != expected_descriptors:
                    raise ValueError("published RNG stream schema is invalid.")
                state_range = self._array_range(state)
                for family, other_bindings in self._bindings.items():
                    for role, other in other_bindings.items():
                        if other is state:
                            if family != process_id or role != "rng_states":
                                raise ValueError(
                                    "published RNG sidecars alias."
                                )
                            continue
                        if self._ranges_overlap(
                            state_range, self._array_range(other)
                        ):
                            raise ValueError("published RNG sidecars alias.")
                result.append((process_id, manifest, bindings["rng_states"]))
        return tuple(result)

    def _has_resident_coagulation_stream(self) -> bool:
        """Return whether an initialized resident coagulation stream exists."""
        return self._coagulation_stream_registry is not None

    def _has_resident_rng_stream(self) -> bool:
        """Return whether any initialized resident RNG stream exists."""
        return (
            self._coagulation_stream_registry is not None
            or self._wall_loss_stream_registry is not None
        )

    def _restore_published_rng_views(
        self, process_ids: tuple[str, ...]
    ) -> None:
        """Publish prevalidated restored stream bindings without reseeding.

        Checkpoint restart calls this only after it has bulk-uploaded and bound
        the ordinary sidecars and current-word arrays.  It deliberately creates
        no initial words and invokes no acquisition API.
        """
        self.validate_pinned_session(self._session)
        root_seed, logical_box_ids, lanes = self._stream_metadata()
        if "coagulation" in process_ids:
            bindings = self._bindings["coagulation"]
            state = bindings.get("rng_states")
            if state is None:
                raise ValueError("restored coagulation RNG binding is missing.")
            self._views["coagulation"] = CoagulationResources(
                self._capacities["coagulation"], **bindings
            )
            self._coagulation_stream_registry = _RestoredStreamRegistry(
                root_seed, logical_box_ids, lanes, "coagulation", state
            )
        if "wall_loss" in process_ids:
            state = self._bindings["wall_loss"].get("rng_states")
            if state is None:
                raise ValueError("restored wall-loss RNG binding is missing.")
            self._views["wall_loss"] = WallLossResources(
                **self._bindings["wall_loss"]
            )
            self._wall_loss_stream_registry = _RestoredStreamRegistry(
                root_seed, logical_box_ids, lanes, "wall_loss", state
            )

    def reserve_open_step(self, token: Any) -> None:
        """Reserve the binding's sole open timestep token by identity.

        The resident step guard calls this only after validating the pinned
        session. This bookkeeping-only seam neither acquires resources nor
        performs runtime work.

        Args:
            token: Newly created opaque resident-step token.

        Raises:
            RuntimeError: If this exact registry binding already has a token.
        """
        if self._open_step_token is not None:
            raise RuntimeError("A resident timestep is already open.")
        self._open_step_token = token

    def release_open_step(self, token: Any) -> None:
        """Release the exact outstanding binding-level timestep token.

        Args:
            token: The exact token previously reserved for this binding.

        Raises:
            ValueError: If ``token`` is not the outstanding token by identity.
        """
        if token is not self._open_step_token:
            raise ValueError("token does not match the open resident timestep.")
        self._open_step_token = None

    def assert_step_closed(self) -> None:
        """Reject lifecycle work while this registry has an open timestep.

        This binding-wide check covers every guard sharing the registry rather
        than only the guard passed to a lifecycle boundary.

        Raises:
            RuntimeError: If a resident timestep remains open.
        """
        if self._open_step_token is not None:
            raise RuntimeError("A resident timestep is open.")

    def _validate_session_state(self) -> None:
        """Recheck the metadata-only invariants needed by this boundary."""
        self._validate_session_carriers()
        # This handles exact-but-fabricated frozen instances and verifies all
        # protected primary schemas without inspecting their payloads.
        self._session.__post_init__()

    def _validate_session_carriers(self) -> None:
        """Validate exact lifecycle carriers before inspecting schemas."""
        if type(self._session) is not ResidentSession:
            raise TypeError("session must remain an exact ResidentSession.")
        if type(self._session.dimensions) is not ResidentDimensions:
            raise TypeError("session.dimensions must remain exact.")
        if type(self._session.lifecycle) is not ResidentLifecycle:
            raise TypeError("session.lifecycle must remain exact.")
        if self._session.lifecycle is not ResidentLifecycle.ACTIVE:
            raise ValueError("session.lifecycle must be ACTIVE.")

    def _shape(
        self, entry: ManifestEntry, capacity: int | None = None
    ) -> tuple[int, ...]:
        """Resolve a manifest entry to its fixed resident-session shape.

        Args:
            entry: Manifest role whose shape formula is resolved.
            capacity: Required collision or communication-edge capacity for
                ``"bc2"``, ``"e"``, and ``"en"`` entries.

        Returns:
            Exact Warp-array shape for the entry.

        Raises:
            ValueError: If a collision-pair or communication-edge shape lacks
                its required capacity.
        """
        dimensions = self._session.dimensions
        shapes: dict[str, tuple[int, ...]] = {
            "b": (dimensions.n_boxes,),
            "bn": (dimensions.n_boxes, dimensions.n_particles),
            "bs": (dimensions.n_boxes, dimensions.n_species),
            "bns": (
                dimensions.n_boxes,
                dimensions.n_particles,
                dimensions.n_species,
            ),
            "status": (1,),
        }
        if entry.shape_kind == "bc2":
            if capacity is None:
                raise ValueError("collision capacity is required.")
            return (dimensions.n_boxes, capacity, 2)
        if entry.shape_kind == "e":
            if capacity is None:
                raise ValueError("communication edge capacity is required.")
            return (capacity,)
        if entry.shape_kind == "en":
            if capacity is None:
                raise ValueError("communication edge capacity is required.")
            return (capacity, dimensions.n_particles)
        return shapes[entry.shape_kind]

    def _validate_array(
        self, entry: ManifestEntry, value: Any, capacity: int | None
    ) -> tuple[int, int] | None:
        """Validate one sidecar's Warp metadata against its manifest role.

        Args:
            entry: Expected dtype and shape specification.
            value: Caller-supplied Warp array to inspect without reading.
            capacity: Collision or communication-edge capacity for entries
                whose schema requires it.

        Returns:
            Nonempty half-open byte range, or ``None`` for an empty array.

        Raises:
            TypeError: If ``value`` is not a Warp array.
            ValueError: If its schema, device, pointer, or strides are invalid.
        """
        array_type = getattr(wp, "array", None)
        if isinstance(array_type, type) and not isinstance(value, array_type):
            raise TypeError(f"{entry.role} must be a Warp array.")
        if not (
            type(value).__module__.startswith("warp")
            and type(value).__name__ == "array"
        ):
            raise TypeError(f"{entry.role} must be a Warp array.")
        shape = self._shape(entry, capacity)
        if value.dtype != entry.dtype or value.shape != shape:
            raise ValueError(f"{entry.role} has incompatible schema.")
        if value.device != self._signature[2]:
            raise ValueError(f"{entry.role} device must match session device.")
        return self._contiguous_range(value, shape, entry.dtype, entry.role)

    def _contiguous_range(
        self,
        value: Any,
        shape: tuple[int, ...],
        dtype: Any,
        role: str,
    ) -> tuple[int, int] | None:
        """Validate contiguous metadata and return a nonempty byte range."""
        strides = getattr(value, "strides", None)
        if not isinstance(strides, tuple) or len(strides) != len(shape):
            raise ValueError(f"{role} must have contiguous strides.")
        item_size = _item_size(dtype)
        expected: list[int] = []
        stride = item_size
        for length in reversed(shape):
            expected.insert(0, stride)
            stride *= length
        if strides != tuple(expected):
            raise ValueError(f"{role} must be contiguous.")
        count = 1
        for length in shape:
            count = self._checked_product(count, length)
        if count == 0:
            return None
        pointer = getattr(value, "ptr", None)
        if not isinstance(pointer, Integral) or pointer <= 0:
            raise ValueError(f"{role} must have a valid pointer.")
        if pointer % item_size:
            raise ValueError(
                f"{role} pointer must be {item_size}-byte aligned."
            )
        capacity = getattr(value, "capacity", None)
        required = count * item_size
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, Integral)
            or capacity < required
            or capacity % item_size
        ):
            raise ValueError(
                f"{role} must have sufficient integral storage capacity."
            )
        return int(pointer), int(pointer) + required

    @staticmethod
    def _checked_product(left: int, right: int) -> int:
        if right < 0 or left > _MAX_SIZE // max(right, 1):
            raise ValueError(
                "Resource allocation size exceeds supported range."
            )
        return left * right

    def _allocate(self, entry: ManifestEntry, capacity: int | None) -> Any:
        """Allocate one manifest-conforming sidecar on the pinned device.

        Args:
            entry: Manifest role to allocate.
            capacity: Collision or communication-edge capacity for entries
                whose schema requires it.

        Returns:
            Zero-filled Warp array matching the entry's fixed schema.

        Raises:
            ValueError: If the computed allocation element or byte count
                exceeds the supported range.
        """
        shape = self._shape(entry, capacity)
        count = 1
        for length in shape:
            count = self._checked_product(count, length)
        self._checked_product(count, _item_size(entry.dtype))
        return wp.zeros(shape, dtype=entry.dtype, device=self._signature[2])

    @staticmethod
    def _ranges_overlap(
        left: tuple[int, int] | None, right: tuple[int, int] | None
    ) -> bool:
        """Return whether two nonempty half-open byte ranges overlap."""
        return (
            left is not None
            and right is not None
            and left[0] < right[1]
            and right[0] < left[1]
        )

    def _protected_ranges(self) -> list[tuple[int, int] | None]:
        """Return metadata-only byte ranges for protected primary arrays."""
        return [
            self._array_range(array) for array in _primary_arrays(self._session)
        ]

    @staticmethod
    def _reject_shared_identities(values: list[Any], others: list[Any]) -> None:
        """Reject identity reuse within or across sidecar ownership sets."""
        for index, value in enumerate(values):
            if any(value is other for other in values[index + 1 :]):
                raise ValueError("Sidecar roles must not share identity.")
            if any(value is other for other in others):
                raise ValueError("Sidecar roles must not share identity.")

    def _reject_primary_aliases(self, values: list[Any]) -> None:
        """Reject sidecars that share identity or bytes with primaries."""
        primaries = list(_primary_arrays(self._session))
        if any(value is primary for value in values for primary in primaries):
            raise ValueError("Sidecars must not alias session primaries.")
        candidate_ranges = [self._array_range(value) for value in values]
        for candidate in candidate_ranges:
            if any(
                self._ranges_overlap(candidate, primary)
                for primary in self._protected_ranges()
            ):
                raise ValueError("Sidecar byte ranges must not overlap.")

    def _validate_nonalias(
        self,
        bindings: dict[str, Any],
        entries: tuple[ManifestEntry, ...],
        capacity: int | None,
    ) -> None:
        ranges = [
            self._validate_array(entry, bindings[entry.role], capacity)
            for entry in entries
        ]
        values = [bindings[entry.role] for entry in entries]
        registered = [
            value
            for family_bindings in self._bindings.values()
            for value in family_bindings.values()
        ]
        registered_ranges = [self._array_range(value) for value in registered]
        self._reject_shared_identities(values, registered)
        self._reject_primary_aliases(values)
        for index, byte_range in enumerate(ranges):
            if any(
                self._ranges_overlap(byte_range, other)
                for other in ranges[index + 1 :] + registered_ranges
            ):
                raise ValueError("Sidecar byte ranges must not overlap.")

    def _validate_supplied_nonalias(
        self,
        supplied: dict[str, Any],
        entries: tuple[ManifestEntry, ...],
    ) -> None:
        """Reject supplied aliases before allocating omitted sidecars."""
        values = [supplied[entry.role] for entry in entries]
        values = [value for value in values if value is not None]
        registered = [
            value
            for family_bindings in self._bindings.values()
            for value in family_bindings.values()
        ]
        registered_ranges = [self._array_range(value) for value in registered]
        self._reject_shared_identities(values, registered)
        self._reject_primary_aliases(values)
        ranges = [self._array_range(value) for value in values]
        for index, byte_range in enumerate(ranges):
            if any(
                self._ranges_overlap(byte_range, other)
                for other in ranges[index + 1 :] + registered_ranges
            ):
                raise ValueError("Sidecar byte ranges must not overlap.")

    @staticmethod
    def _array_range(array: Any) -> tuple[int, int] | None:
        """Return one validated registry array's nonempty byte range."""
        strides = getattr(array, "strides", None)
        if not isinstance(strides, tuple) or len(strides) != len(array.shape):
            raise ValueError("Registry arrays must have contiguous strides.")
        item_size = _item_size(array.dtype)
        expected: list[int] = []
        stride = item_size
        for length in reversed(array.shape):
            expected.insert(0, stride)
            stride *= length
        if strides != tuple(expected):
            raise ValueError("Registry arrays must be contiguous.")
        count = 1
        for length in array.shape:
            count *= length
        if count == 0:
            return None
        pointer = getattr(array, "ptr", None)
        if not isinstance(pointer, Integral) or pointer <= 0:
            raise ValueError("Registry arrays must have a valid pointer.")
        if pointer % item_size:
            raise ValueError(
                "Registry array pointers must be element-size aligned."
            )
        capacity = getattr(array, "capacity", None)
        required = count * item_size
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, Integral)
            or capacity < required
            or capacity % item_size
        ):
            raise ValueError(
                "Registry arrays must have sufficient integral "
                "storage capacity."
            )
        return int(pointer), int(pointer) + required

    def _acquire(  # noqa: C901
        self,
        manifest: ResourceManifest,
        supplied: dict[str, Any],
        capacity: int | None = None,
        *,
        publish: bool = True,
    ) -> dict[str, Any]:
        """Validate, allocate, and atomically publish one resource family.

        Args:
            manifest: Complete schema for the resource family.
            supplied: Role-to-array bindings; ``None`` requests allocation.
            capacity: Collision or communication-edge capacity for a manifest
                that requires it.
            publish: Whether to register a newly validated family immediately.

        Returns:
            Pinned role-to-array bindings for the established family.

        Raises:
            TypeError: If supplied arrays fail the required Warp type checks.
            ValueError: If the session drifted, bindings are incompatible,
                alias protected storage, or coagulation capacity changes.
        """
        self._validate_session_signature()
        if manifest.family in self._bindings:
            if (
                capacity is not None
                and self._capacities.get(manifest.family) != capacity
            ):
                raise ValueError(
                    "collision_capacity cannot change after acquisition."
                )
            existing = self._bindings[manifest.family]
            for role, value in supplied.items():
                if value is not None and value is not existing[role]:
                    raise ValueError("Established sidecars cannot be replaced.")
            return existing
        candidate = dict(supplied)
        for entry in manifest.entries:
            value = candidate[entry.role]
            if value is not None:
                self._validate_array(entry, value, capacity)
        self._validate_supplied_nonalias(candidate, manifest.entries)
        for entry in manifest.entries:
            if candidate[entry.role] is None:
                candidate[entry.role] = self._allocate(entry, capacity)
        self._validate_nonalias(candidate, manifest.entries, capacity)
        if publish:
            self._bindings[manifest.family] = candidate
            if capacity is not None:
                self._capacities[manifest.family] = capacity
        return candidate

    def acquire_condensation(
        self, *, buffers: CondensationScratchBuffers | None = None
    ) -> CondensationResources:
        """Acquire one complete pinned condensation scratch record.

        Args:
            buffers: Optional complete exact native scratch record. Missing
                records are allocated as a complete fixed-shape set.

        Returns:
            Stable view containing the native scratch record by identity.

        Raises:
            TypeError: If ``buffers`` is not an exact native record.
            ValueError: If it is incomplete, incompatible, aliases protected
                storage, or the pinned session has drifted.
        """
        if (
            buffers is not None
            and type(buffers) is not CondensationScratchBuffers
        ):
            raise TypeError(
                "buffers must be an exact CondensationScratchBuffers."
            )
        supplied = {
            entry.role: None
            if buffers is None
            else getattr(buffers, entry.role)
            for entry in _CONDENSATION.entries
        }
        if buffers is not None and any(
            value is None for value in supplied.values()
        ):
            raise ValueError("buffers must be complete.")
        bindings = self._acquire(_CONDENSATION, supplied)
        if "condensation" not in self._views:
            self._views["condensation"] = CondensationResources(
                CondensationScratchBuffers(**bindings)
            )
        return self._views["condensation"]

    def acquire_coagulation(
        self,
        collision_capacity: int,
        *,
        collision_pairs: Any | None = None,
        n_collisions: Any | None = None,
        rng_states: Any | None = None,
    ) -> CoagulationResources:
        """Acquire fixed-capacity coagulation outputs and one RNG sidecar.

        The first successful acquisition validates supplied sidecars and
        nonaliasing before allocating omitted arrays. It then initializes the
        single ``(n_boxes,)`` ``wp.uint32`` coagulation sidecar from immutable
        session stream metadata and publishes the view. Compatible later calls
        return that exact view without allocation or reseeding. This is not a
        wall-loss stream, reset or inspection API, hidden transfer,
        synchronization, or public checkpoint-persistence boundary. A
        schema-v3 restart privately restores captured current words without
        invoking this acquisition method or reseeding.

        Args:
            collision_capacity: Positive, non-boolean integral collision bound.
            collision_pairs: Optional ``int32`` collision-pair sidecar.
            n_collisions: Optional ``int32`` per-box count sidecar.
            rng_states: Optional ``uint32`` persistent per-box RNG sidecar.

        Returns:
            Stable view with the fixed capacity and pinned native sidecars.

        Raises:
            TypeError: If capacity is not a non-boolean integral or a supplied
                sidecar is not a Warp array.
            ValueError: If capacity, schema, aliasing, session signature, or a
                replacement request is incompatible with the established state.
        """
        if isinstance(collision_capacity, bool) or not isinstance(
            collision_capacity, Integral
        ):
            raise TypeError(
                "collision_capacity must be a non-boolean integral."
            )
        maximum_capacity = max(
            1,
            min(
                _INT32_MAX,
                _MAX_SIZE // max(1, self._session.dimensions.n_boxes * 2 * 4),
                self._session.dimensions.n_particles**2,
            ),
        )
        if collision_capacity <= 0 or collision_capacity > maximum_capacity:
            raise ValueError(
                "collision_capacity must be positive and within resident "
                "fixed-capacity bounds."
            )
        already_published = "coagulation" in self._bindings
        bindings = self._acquire(
            _COAGULATION,
            {
                "collision_pairs": collision_pairs,
                "n_collisions": n_collisions,
                "rng_states": rng_states,
            },
            int(collision_capacity),
            publish=already_published,
        )
        if not already_published:
            stream = self._session.metadata.stream
            if stream.n_boxes == 0 and self._session.dimensions.n_boxes:
                # Compatibility for direct legacy session construction. Factory
                # sessions always retain explicit stream metadata.
                logical_box_ids = tuple(
                    str(index)
                    for index in range(self._session.dimensions.n_boxes)
                )
                lanes = tuple(range(self._session.dimensions.n_boxes))
                root_seed = 0
            else:
                logical_box_ids = stream.logical_box_ids
                lanes = stream.lanes
                root_seed = stream.root_seed
            # P1 presently defines both process namespaces. The wall-loss array
            # is temporary P1 initialization storage, never a resident resource.
            temporary_wall_loss = wp.zeros(
                self._shape(_WALL_LOSS.entries[0]),
                dtype=wp.uint32,
                device=self._signature[2],
            )
            registry = StreamRegistry(
                root_seed,
                self._session.dimensions.n_boxes,
                logical_box_ids,
                lanes,
                (
                    ("coagulation", bindings["rng_states"]),
                    ("wall_loss", temporary_wall_loss),
                ),
            )
            registry.initialize()
            view = CoagulationResources(int(collision_capacity), **bindings)
            self._bindings["coagulation"] = bindings
            self._capacities["coagulation"] = int(collision_capacity)
            self._coagulation_stream_registry = registry
            self._views["coagulation"] = view
        return self._views["coagulation"]

    def acquire_wall_loss(
        self, *, rng_states: Any | None = None
    ) -> WallLossResources:
        """Acquire one initialized persistent wall-loss RNG sidecar.

        The first successful acquisition validates or allocates the single
        ``(n_boxes,)`` ``wp.uint32`` sidecar, initializes it from the wall-loss
        namespace, then publishes the view. Compatible later calls return the
        exact view without allocation or reseeding. Initializing this sidecar
        does not reseed a published coagulation stream. Schema-v3 checkpoint
        restart can privately restore captured current words without invoking
        this acquisition method or reseeding.

        Args:
            rng_states: Optional ``uint32`` per-box native RNG sidecar.

        Returns:
            Stable view containing the pinned RNG sidecar.

        Raises:
            TypeError: If a supplied sidecar is not a Warp array.
            ValueError: If the sidecar schema, aliasing, session signature, or
                an established binding is incompatible.
        """
        already_published = "wall_loss" in self._bindings
        bindings = self._acquire(
            _WALL_LOSS,
            {"rng_states": rng_states},
            publish=already_published,
        )
        if not already_published:
            stream = self._session.metadata.stream
            if stream.n_boxes == 0 and self._session.dimensions.n_boxes:
                logical_box_ids = tuple(
                    str(index)
                    for index in range(self._session.dimensions.n_boxes)
                )
                lanes = tuple(range(self._session.dimensions.n_boxes))
                root_seed = 0
            else:
                logical_box_ids = stream.logical_box_ids
                lanes = stream.lanes
                root_seed = stream.root_seed
            coagulation = self._bindings.get("coagulation", {}).get(
                "rng_states"
            )
            has_published_coagulation = coagulation is not None
            if coagulation is None:
                coagulation = wp.zeros(
                    self._shape(_COAGULATION.entries[2]),
                    dtype=wp.uint32,
                    device=self._signature[2],
                )
            registry = StreamRegistry(
                root_seed,
                self._session.dimensions.n_boxes,
                logical_box_ids,
                lanes,
                (
                    ("coagulation", coagulation),
                    ("wall_loss", bindings["rng_states"]),
                ),
            )
            if has_published_coagulation:
                # Do not reseed the existing resident coagulation stream while
                # initializing this newly acquired independent namespace.
                registry.initialize_process("wall_loss")
            else:
                registry.initialize()
            view = WallLossResources(**bindings)
            self._bindings["wall_loss"] = bindings
            self._wall_loss_stream_registry = registry
            self._views["wall_loss"] = view
        return self._views["wall_loss"]

    def acquire_communication(  # noqa: C901
        self,
        configuration: CommunicationConfiguration,
        *,
        buffers: GasCommunicationBuffers
        | ParticleCommunicationBuffers
        | None = None,
    ) -> CommunicationResources:
        """Pin one closed resident communication map and native work record.

        This is the sole P1 validation and optional-allocation boundary for this
        family. It accepts only GAS or PARTICLES closed maps, validates the
        configuration once, then pins maps, work arrays, and optional prescribed
        volumes by identity after schema and byte-range nonaliasing checks.
        Reacquisition may return the established binding but never replaces it.
        It does not execute a communication primitive, inspect payload values,
        transfer, synchronize, initialize RNG state, or recover a writer error.

        Args:
            configuration: Exact P1-validated closed resident map and optional
                prescribed-volume update.
            buffers: Optional complete native mode-matched work record. Omitted
                work arrays are allocated on the pinned device.

        Returns:
            The stable identity-bound published communication resource view.

        Raises:
            TypeError: If the configuration or supplied buffer record has an
                inexact or mode-incompatible type.
            ValueError: If P1 validation, session identity, resource schema,
                capacity, or nonaliasing checks fail, or a binding is replaced.
        """
        self._validate_session_signature()
        if type(configuration) is not CommunicationConfiguration:
            raise TypeError(
                "configuration must be an exact CommunicationConfiguration."
            )
        validated = validate_communication_configuration(
            configuration, self._session.dimensions, self._signature[2]
        )
        if validated is not configuration:
            raise ValueError("configuration validation must retain identity.")
        map_data = configuration.communication_map
        final_volumes = configuration.prescribed_volume.final_volumes
        if final_volumes is not None:
            volume_entry = ManifestEntry(
                "final_volumes", "communication", wp.float64, "b"
            )
            volume_range = self._validate_array(
                volume_entry, final_volumes, capacity=None
            )
            self._reject_primary_aliases([final_volumes])
            registered = [
                value
                for bindings in self._bindings.values()
                for value in bindings.values()
            ]
            if any(final_volumes is value for value in registered) or any(
                self._ranges_overlap(volume_range, self._array_range(value))
                for value in registered
            ):
                raise ValueError(
                    "final_volumes must not alias resident resources."
                )
        mode = map_data.transport_mode
        if mode not in (
            CommunicationTransportMode.GAS,
            CommunicationTransportMode.PARTICLES,
        ):
            raise ValueError(
                "resident communication supports GAS or PARTICLES only."
            )
        opposite_family = (
            "communication_particles"
            if mode is CommunicationTransportMode.GAS
            else "communication_gas"
        )
        if opposite_family in self._views:
            raise ValueError(
                "Only one resident communication family may be bound."
            )
        family = (
            "communication_gas"
            if mode is CommunicationTransportMode.GAS
            else "communication_particles"
        )
        manifest = (
            _GAS_COMMUNICATION
            if mode is CommunicationTransportMode.GAS
            else _PARTICLE_COMMUNICATION
        )
        expected = (
            GasCommunicationBuffers
            if mode is CommunicationTransportMode.GAS
            else ParticleCommunicationBuffers
        )
        if buffers is not None and type(buffers) is not expected:
            raise TypeError(
                "buffers must match the communication transport mode."
            )
        supplied = {
            "source_boxes": map_data.source_boxes,
            "destination_boxes": map_data.destination_boxes,
            "enabled": map_data.enabled,
            "rates": map_data.rates,
        }
        for entry in manifest.entries[4:]:
            supplied[entry.role] = (
                None
                if buffers is None or not hasattr(buffers, entry.role)
                else getattr(buffers, entry.role)
            )
        native_roles = {entry.role for entry in manifest.entries[4:]}
        if buffers is not None and any(
            supplied[role] is None
            for role in native_roles.intersection(
                self._record_bindings(buffers)
            )
        ):
            raise ValueError("communication buffers must be complete.")
        bindings = self._acquire(
            manifest, supplied, int(map_data.edge_capacity)
        )
        if family not in self._views:
            native: GasCommunicationBuffers | ParticleCommunicationBuffers
            if mode is CommunicationTransportMode.GAS:
                native = GasCommunicationBuffers(
                    bindings["amounts"],
                    bindings["amount_deltas"],
                    bindings["outbound_amounts"],
                )
            else:
                native = ParticleCommunicationBuffers(
                    bindings["source_debits"],
                    bindings["destination_credits"],
                    bindings["assignments"],
                    bindings["request_concentrations"],
                )
            execution_state = ResidentCommunicationState(
                bindings["invalid"],
                bindings["active_or_demand"],
                bindings["volume_invalid"],
                bindings["volume_changed"],
                bindings.get("initial_masses"),
                bindings.get("initial_concentration"),
                bindings.get("initial_charge"),
            )
            self._views[family] = CommunicationResources(
                configuration,
                native,
                configuration.prescribed_volume.final_volumes,
                execution_state,
            )
        view = self._views[family]
        if view.configuration is not configuration:
            raise ValueError(
                "Established communication configuration cannot change."
            )
        return view

    def validate_communication_resources(
        self, session: ResidentSession, resources: CommunicationResources
    ) -> None:
        """Metadata-validate an established communication resource view.

        This execution-time seam requires the exact active pinned session and
        published view, then rechecks mode, identities, shapes, device,
        contiguity, and nonaliasing metadata. It intentionally does not repeat
        P1 payload validation, allocate, acquire, inspect values, transfer,
        synchronize, mutate bindings, or invoke a native primitive.

        Args:
            session: Exact active session retained by this registry.
            resources: Exact published communication resource view.

        Raises:
            TypeError: If ``resources`` or its configuration has an inexact
                concrete type.
            ValueError: If the session, mode, view, sidecar identity, or schema
                no longer matches the pinned binding.
        """
        self.validate_pinned_session(session)
        if type(resources) is not CommunicationResources:
            raise TypeError(
                "resources must be an exact CommunicationResources."
            )
        configuration = resources.configuration
        if type(configuration) is not CommunicationConfiguration:
            raise TypeError(
                "configuration must be an exact CommunicationConfiguration."
            )
        mode = configuration.communication_map.transport_mode
        family = (
            "communication_gas"
            if mode is CommunicationTransportMode.GAS
            else "communication_particles"
        )
        manifest = (
            _GAS_COMMUNICATION
            if mode is CommunicationTransportMode.GAS
            else _PARTICLE_COMMUNICATION
        )
        if mode not in (
            CommunicationTransportMode.GAS,
            CommunicationTransportMode.PARTICLES,
        ):
            raise ValueError(
                "resident communication supports GAS or PARTICLES only."
            )
        if resources is not self._views.get(family):
            raise ValueError(
                "resources must be the published communication view."
            )
        bindings = self._bindings[family]
        if (
            resources.final_volumes
            is not configuration.prescribed_volume.final_volumes
        ):
            raise ValueError("communication final volumes binding changed.")
        if resources.final_volumes is not None:
            self._validate_array(
                ManifestEntry(
                    "final_volumes", "communication", wp.float64, "b"
                ),
                resources.final_volumes,
                capacity=None,
            )
        values = {
            "source_boxes": configuration.communication_map.source_boxes,
            "destination_boxes": (
                configuration.communication_map.destination_boxes
            ),
            "enabled": configuration.communication_map.enabled,
            "rates": configuration.communication_map.rates,
        }
        values.update(self._record_bindings(resources.buffers))
        values.update(
            {
                "invalid": resources.execution_state.invalid,
                "active_or_demand": resources.execution_state.active_or_demand,
                "volume_invalid": resources.execution_state.volume_invalid,
                "volume_changed": resources.execution_state.volume_changed,
                "initial_masses": resources.execution_state.initial_masses,
                "initial_concentration": (
                    resources.execution_state.initial_concentration
                ),
                "initial_charge": resources.execution_state.initial_charge,
            }
        )
        for entry in manifest.entries:
            if values.get(entry.role) is not bindings[entry.role]:
                raise ValueError("communication resource bindings changed.")
            self._validate_array(
                entry,
                bindings[entry.role],
                configuration.communication_map.edge_capacity,
            )

    @staticmethod
    def _record_bindings(record: Any) -> dict[str, Any]:
        """Return dataclass field bindings for one exact native record."""
        return {
            field.name: getattr(record, field.name) for field in fields(record)
        }

    def _nucleation_supplied_bindings(
        self,
        scratch: NucleationScratchBuffers | None,
        finalized_demand: NucleationFinalizedDemandBuffers | None,
        diagnostics: NucleationDiagnosticBuffers | None,
        exhaustion: NucleationExhaustionBuffers | None,
    ) -> dict[str, Any]:
        """Validate and flatten optional complete nucleation records."""
        records = (scratch, finalized_demand, diagnostics, exhaustion)
        record_types = (
            NucleationScratchBuffers,
            NucleationFinalizedDemandBuffers,
            NucleationDiagnosticBuffers,
            NucleationExhaustionBuffers,
        )
        supplied = {entry.role: None for entry in _NUCLEATION.entries}
        for record, record_type in zip(records, record_types, strict=True):
            if record is not None and type(record) is not record_type:
                raise TypeError(
                    "nucleation records must have exact native types."
                )
            if record is not None:
                supplied.update(self._record_bindings(record))
        if exhaustion is not None:
            resampling = exhaustion.resampling_buffers
            if type(resampling) is not ResamplingBuffers:
                raise TypeError(
                    "resampling_buffers must be exact ResamplingBuffers."
                )
            if any(
                value is None
                for value in self._record_bindings(resampling).values()
            ):
                raise ValueError(
                    "Supplied nucleation records must be complete."
                )
            supplied.update(self._record_bindings(resampling))
        if any(
            record is not None
            and any(
                value is None
                for value in self._record_bindings(record).values()
            )
            for record in records
        ):
            raise ValueError("Supplied nucleation records must be complete.")
        return supplied

    def _nucleation_view(self, bindings: dict[str, Any]) -> NucleationResources:
        """Construct complete native nucleation records from pinned bindings."""

        def build(record_type: Any) -> Any:
            return record_type(
                **{
                    field.name: bindings[field.name]
                    for field in fields(record_type)
                }
            )

        resampling = build(ResamplingBuffers)
        exhaustion = NucleationExhaustionBuffers(
            resampling,
            **{
                field.name: bindings[field.name]
                for field in fields(NucleationExhaustionBuffers)
                if field.name != "resampling_buffers"
            },
        )
        return NucleationResources(
            build(NucleationScratchBuffers),
            build(NucleationFinalizedDemandBuffers),
            build(NucleationDiagnosticBuffers),
            exhaustion,
        )

    def acquire_nucleation(
        self,
        *,
        scratch: NucleationScratchBuffers | None = None,
        finalized_demand: NucleationFinalizedDemandBuffers | None = None,
        diagnostics: NucleationDiagnosticBuffers | None = None,
        exhaustion: NucleationExhaustionBuffers | None = None,
    ) -> NucleationResources:
        """Acquire complete pinned native nucleation records and scratch.

        Args:
            scratch: Optional complete exact nucleation scratch record.
            finalized_demand: Optional complete exact finalized-demand record.
            diagnostics: Optional complete exact diagnostic record.
            exhaustion: Optional complete exact exhaustion record, including its
                complete nested resampling buffers.

        Returns:
            Stable view holding complete native records built from pinned
            arrays.

        Raises:
            TypeError: If supplied records are not exact native record types.
            ValueError: If records are incomplete, their sidecars are
                incompatible or aliasing, or the session signature drifted.
        """
        supplied = self._nucleation_supplied_bindings(
            scratch,
            finalized_demand,
            diagnostics,
            exhaustion,
        )
        bindings = self._acquire(_NUCLEATION, supplied)
        if "nucleation" not in self._views:
            self._views["nucleation"] = self._nucleation_view(bindings)
            view = self._views["nucleation"]
            self._nucleation_records = (
                view.scratch,
                view.finalized_demand,
                view.diagnostics,
                view.exhaustion,
                view.exhaustion.resampling_buffers,
            )
        return self._views["nucleation"]
