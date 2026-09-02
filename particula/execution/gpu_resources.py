"""Allocate concrete reusable Warp sidecars for one active resident session.

This direct-import-only, Warp-dependent boundary pins complete fixed-shape
native sidecar families and one capture resource selection to one exact
``ACTIVE`` :class:`ResidentSession`.
It allocates and validates resources only: it neither executes a process,
transfers, synchronizes, nor resizes. Coagulation and wall-loss acquisition
initialize distinct P1-derived persistent RNG streams exactly once before
publishing their resident resources. The checkpoint-private restoration seam
instead publishes prevalidated current stream words without reseeding. The
manifests and views here are concrete-only and are deliberately not exported
from :mod:`particula.execution`.

The registry retains array identities and performs metadata-only schema and
nonaliasing checks. It does not establish allocator provenance, execute a
kernel, or change session lifecycle. Its direct-module-only logical inventory
reports immutable manifest schema metadata and logical bytes, not
allocator-reserved bytes. Capture registration retains one exact closed
communication view, if selected, and an ordered diagnostic registration tuple.
It validates schemas and byte-range nonaliasing on the host without inspecting
payloads; reports reuse that retained metadata. Inventory reporting neither
inspects payloads nor acquires, allocates, binds, or mutates sidecars. Explicit
lifecycle methods may inspect frozen stream metadata or reset selected
published lanes without hidden transfer or synchronization.
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
from heapq import heappop, heappush
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
    "ResourceInventoryCapacities",
    "ResolvedResourceInventoryEntry",
    "LogicalResourceRoleReport",
    "LogicalResourceFamilyReport",
    "LogicalResourceReport",
    "SelectedResourceRole",
    "SelectedResourceFamilyReport",
    "SelectedResourceInventory",
    "CaptureResourceRequirements",
    "CaptureResourceSet",
    "PublishedStreamManifest",
    "GPUResourceRegistry",
    "CondensationResources",
    "CoagulationResources",
    "DilutionResources",
    "WallLossResources",
    "NucleationResources",
    "PreparedResourceViews",
    "CommunicationResources",
]

_INT32_MAX = 2**31 - 1
_MAX_SIZE = (1 << 63) - 1
_ShapeKind = Literal["b", "bn", "bs", "bns", "bc2", "e", "en", "status"]


@wp.kernel
def _scan_diagnostic_accounting(
    values: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
    require_nonnegative: bool,
    invalid: wp.array(dtype=wp.int32),  # type: ignore[valid-type]
) -> None:
    """Record invalid diagnostic accounting values in one device status lane."""
    box, species = wp.tid()  # type: ignore[misc]
    value = values[box, species]
    if not wp.isfinite(value) or (require_nonnegative and value < 0.0):
        wp.atomic_max(invalid, 0, 1)


@dataclass(frozen=True)
class ManifestEntry:
    """Describe one fixed-shape concrete sidecar role.

    Attributes:
        role: Name of the sidecar within its resource family.
        family: Canonical resource family that owns the role.
        dtype: Warp dtype declared for the sidecar.
        shape_kind: Symbolic shape formula resolved by the registry.
    """

    role: str
    family: str
    dtype: Any
    shape_kind: _ShapeKind


@dataclass(frozen=True)
class ResourceManifest:
    """Declare immutable sidecar schemas for one resource family.

    Attributes:
        family: Canonical name of the resource family.
        entries: Ordered role declarations comprising the family schema.
    """

    family: str
    entries: tuple[ManifestEntry, ...]


@dataclass(frozen=True)
class ResourceInventoryCapacities:
    """Hold frozen capacities for direct-module-only inventory reports.

    The carrier resolves declared manifest shapes only. It contains no device
    payload or allocator-capacity data, and using it neither acquires nor
    allocates resources.

    Attributes:
        collision_capacity: Logical collision-sidecar capacity.
        gas_edge_capacity: Logical gas-communication edge capacity.
        particle_edge_capacity: Logical particle-communication edge capacity.
    """

    collision_capacity: int
    gas_edge_capacity: int
    particle_edge_capacity: int


@dataclass(frozen=True)
class ResolvedResourceInventoryEntry:
    """Describe one frozen direct-module-only resolved manifest role.

    The pointer-free entry contains declaration-resolved schema metadata only.
    It has no device payload, acquisition state, or allocator-capacity
    information.

    Attributes:
        family: Canonical manifest family containing the role.
        role: Canonical role name within ``family``.
        dtype: Declared Warp dtype for the logical role.
        shape: Resolved logical shape.
        capacity_source: Input that resolved any dynamic shape extent.
        ownership: Declared owner category for the role.
    """

    family: str
    role: str
    dtype: Any
    shape: tuple[int, ...]
    capacity_source: Literal[
        "fixed",
        "collision_capacity",
        "gas_edge_capacity",
        "particle_edge_capacity",
    ]
    ownership: Literal["registry_or_caller_sidecar", "caller_configuration"]


@dataclass(frozen=True)
class LogicalResourceRoleReport:
    """Report frozen logical accounting for one direct-module-only role.

    Counts describe the resolved schema, not allocator-reserved bytes. Report
    construction is read-only: it neither inspects device payloads nor acquires,
    allocates, binds, or mutates resources.

    Attributes:
        entry: Pointer-free resolved metadata for the manifest role.
        element_count: Logical element count of ``entry.shape``.
        logical_byte_count: Logical schema bytes, excluding allocator overhead.
    """

    entry: ResolvedResourceInventoryEntry
    element_count: int
    logical_byte_count: int


@dataclass(frozen=True)
class LogicalResourceFamilyReport:
    """Report frozen logical accounting for one direct-module-only family.

    Counts cover manifest-defined roles only, not allocator-reserved bytes.
    Constructing the report does not inspect payloads or acquire resources.

    Attributes:
        family: Canonical manifest family name.
        roles: Ordered frozen reports for the family's declared roles.
        logical_byte_count: Sum of role logical schema bytes.
    """

    family: str
    roles: tuple[LogicalResourceRoleReport, ...]
    logical_byte_count: int


@dataclass(frozen=True)
class LogicalResourceReport:
    """Report frozen aggregate accounting for direct-module-only manifests.

    The pointer-free report covers manifest-defined roles only and reports
    logical schema bytes, not allocator-reserved bytes. Its read-only creation
    neither inspects payloads nor acquires, allocates, binds, or mutates
    sidecars.

    Attributes:
        families: Ordered frozen reports for all canonical manifest families.
        logical_byte_count: Sum of family logical schema bytes.
    """

    families: tuple[LogicalResourceFamilyReport, ...]
    logical_byte_count: int


@dataclass(frozen=True, eq=False)
class SelectedResourceRole:
    """Describe one identity-pinned capture resource role.

    This concrete-only carrier retains a caller- or registry-owned array by
    identity for the lifetime of one capture selection. It records schema and
    logical-byte metadata only; it neither transfers ownership nor becomes
    checkpoint authority.

    Attributes:
        canonical_name: Deterministic name for the selected role.
        family: Resource family containing the role.
        dtype: Warp dtype of the retained array.
        shape: Resolved fixed-shape schema.
        value: Retained array reference, preserved by identity.
        element_count: Number of logical elements in ``shape``.
        logical_byte_count: Logical bytes required by the role.
        read_only: Whether the role is a read-only accounting input.
    """

    canonical_name: str
    family: str
    dtype: Any
    shape: tuple[int, ...]
    value: Any
    element_count: int
    logical_byte_count: int
    read_only: bool = False


@dataclass(frozen=True, eq=False)
class SelectedResourceFamilyReport:
    """Report ordered metadata-only roles selected from one capture family.

    Reports retain zero-extent roles in deterministic order even though those
    roles do not participate in byte-range overlap checks.

    Attributes:
        family: Canonical name of the selected resource family.
        roles: Ordered identity-pinned role records in the family.
        logical_byte_count: Checked sum of the roles' logical byte counts.
    """

    family: str
    roles: tuple[SelectedResourceRole, ...]
    logical_byte_count: int


@dataclass(frozen=True, eq=False)
class SelectedResourceInventory:
    """Retain one exact concrete capture resource selection.

    The optional communication view is one already-published, closed GAS or
    PARTICLES view; diagnostic registrations remain caller-owned.
    Re-registration succeeds only for this exact retained identity selection.
    No device data is copied, inspected, or made checkpoint-visible.

    Attributes:
        communication_resources: Selected closed communication view, if present.
        registrations: Exact ordered diagnostic-registration tuple.
        families: Deterministic selected-family reports.
        logical_byte_count: Checked total of all selected role bytes.
    """

    communication_resources: "CommunicationResources | None"
    registrations: tuple[Any, ...]
    families: tuple[SelectedResourceFamilyReport, ...]
    logical_byte_count: int


@dataclass(frozen=True)
class PublishedStreamManifest:
    """Describe immutable identity metadata for currently published streams.

    No live device arrays, pointers, device values, or current stream words are
    exposed by this inspection carrier.

    Attributes:
        stream: Frozen stream identity and descriptor metadata.
        published_process_ids: Processes with currently published streams.
        sidecar_roles: Published process and sidecar-role pairs.
    """

    stream: StreamManifest
    published_process_ids: tuple[str, ...]
    sidecar_roles: tuple[tuple[str, str], ...]


@dataclass(frozen=True, eq=False)
class CondensationResources:
    """Expose a complete native condensation scratch record.

    Attributes:
        scratch_buffers: Pinned native scratch arrays for condensation steps.
    """

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

    Attributes:
        collision_capacity: Fixed maximum number of collision pairs per box.
        collision_pairs: Caller- or registry-owned ``int32`` pair storage.
        n_collisions: Per-box ``int32`` collision-count storage.
        rng_states: Persistent per-box ``uint32`` RNG state.
    """

    collision_capacity: int
    collision_pairs: Any
    n_collisions: Any
    rng_states: Any


@dataclass(frozen=True, eq=False)
class DilutionResources:
    """Expose complete caller-owned dilution preparation sidecars.

    This descriptor-only record is deliberately neither acquired nor published
    by the registry. Prepared callers may retain both validated sidecars by
    identity. The registry read-only validates their exact ``float64``
    ``(B,)`` schemas, device, contiguity, and nonaliasing before preparation;
    it does not allocate, bind, or mutate them. Direct dilution calls retain
    their existing fallback-allocation behavior when no record is supplied.

    Attributes:
        normalized_coefficient: Caller-owned normalized per-box dilution
            coefficient sidecar with shape ``(B,)``.
        factors: Caller-owned per-box dilution factor sidecar with shape
            ``(B,)``.
    """

    normalized_coefficient: Any
    factors: Any


@dataclass(frozen=True, eq=False)
class WallLossResources:
    """Expose one independently initialized wall-loss RNG sidecar.

    The registry publishes this identity-bound view only after initializing its
    ``rng_states`` once from the wall-loss process namespace. Compatible
    reacquisition returns the same view and sidecar by identity without
    allocation or reseeding. The sidecar is distinct from coagulation state and
    has no public reset or inspection API. Schema-v3 checkpoint restart can
    privately restore fresh bindings from captured current words.

    Attributes:
        rng_states: Persistent per-box ``uint32`` wall-loss RNG state.
    """

    rng_states: Any


@dataclass(frozen=True, eq=False)
class NucleationResources:
    """Expose complete native nucleation sidecar records.

    Attributes:
        scratch: Pinned working arrays for nucleation planning.
        finalized_demand: Pinned finalized-demand arrays.
        diagnostics: Pinned diagnostic arrays.
        exhaustion: Pinned exhaustion and resampling arrays.
    """

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
    """Expose registry-pinned status and snapshot storage for barriers.

    Attributes:
        invalid: Device status lane for invalid communication state.
        active_or_demand: Device status lane for active or demanded work.
        volume_invalid: Device status lane for invalid volume state.
        volume_changed: Device status lane for volume-change detection.
        initial_masses: Optional particle-mass snapshot storage.
        initial_concentration: Optional particle-concentration snapshot.
        initial_charge: Optional particle-charge snapshot.
    """

    invalid: Any
    active_or_demand: Any
    volume_invalid: Any
    volume_changed: Any
    initial_masses: Any | None = None
    initial_concentration: Any | None = None
    initial_charge: Any | None = None


@dataclass(frozen=True, eq=False)
class PreparedResourceViews:
    """Carry optional complete prepared-process views by exact identity.

    This private concrete carrier groups independently optional resource views
    for prepared process adapters. Validation requires exact view carriers and
    validates every supplied family read-only against the pinned active session;
    established families must be their published views, while dilution remains
    descriptor-only. Constructing or validating this carrier never acquires,
    publishes, allocates, initializes, or mutates resource state.

    Attributes:
        condensation: Optional established condensation resource view.
        coagulation: Optional established coagulation resource view.
        dilution: Optional complete caller-owned dilution descriptor view.
        wall_loss: Optional established wall-loss resource view.
        nucleation: Optional established nucleation resource view.
    """

    condensation: CondensationResources | None = None
    coagulation: CoagulationResources | None = None
    dilution: DilutionResources | None = None
    wall_loss: WallLossResources | None = None
    nucleation: NucleationResources | None = None


@dataclass(frozen=True, eq=False)
class CaptureResourceRequirements:
    """Describe the complete setup-only resource request for one capture set.

    The carrier retains all supplied objects by exact identity.  It is host
    metadata only: registry-dependent schema, alias, and publication checks are
    deliberately performed by
    :meth:`GPUResourceRegistry.prepare_capture_resources`.
    ``None`` for an enabled allocatable family requests private allocation.

    Attributes:
        session: Exact resident session pinned by the target registry.
        capacities: Exact immutable logical inventory capacities.
        inventory: Exact P3 selected resource inventory.
        prepared_views: Canonical process views required by preparation.
        communication_resources: Exact selected communication view, if present.
        condensation: Supplied condensation record or an allocation request.
        coagulation: Supplied coagulation view or an allocation request.
        wall_loss: Supplied wall-loss view or an allocation request.
        nucleation: Supplied nucleation view or an allocation request.
    """

    session: ResidentSession
    capacities: ResourceInventoryCapacities
    inventory: SelectedResourceInventory
    prepared_views: PreparedResourceViews
    communication_resources: CommunicationResources | None
    condensation: CondensationScratchBuffers | None = None
    coagulation: CoagulationResources | None = None
    wall_loss: WallLossResources | None = None
    nucleation: NucleationResources | None = None
    family_order: tuple[str, ...] = (
        "condensation",
        "coagulation",
        "wall_loss",
        "nucleation",
        "communication_gas",
        "communication_particles",
        "dilution",
    )

    def __post_init__(self) -> None:
        """Reject incomplete or noncanonical host-only request metadata."""
        exact = (
            (self.session, ResidentSession, "session"),
            (self.capacities, ResourceInventoryCapacities, "capacities"),
            (self.inventory, SelectedResourceInventory, "inventory"),
            (self.prepared_views, PreparedResourceViews, "prepared_views"),
        )
        for value, expected_type, name in exact:
            if type(value) is not expected_type:
                raise TypeError(
                    f"{name} must be an exact {expected_type.__name__}."
                )
        if (
            self.communication_resources is not None
            and type(self.communication_resources) is not CommunicationResources
        ):
            raise TypeError("communication_resources must be exact or None.")
        if (
            self.inventory.communication_resources
            is not self.communication_resources
        ):
            raise ValueError(
                "communication_resources must match the P3 inventory."
            )
        if type(self.family_order) is not tuple or self.family_order != (
            "condensation",
            "coagulation",
            "wall_loss",
            "nucleation",
            "communication_gas",
            "communication_particles",
            "dilution",
        ):
            raise ValueError("Capture resource family order is not canonical.")
        expected_resources = (
            (self.condensation, CondensationScratchBuffers, "condensation"),
            (self.coagulation, CoagulationResources, "coagulation"),
            (self.wall_loss, WallLossResources, "wall_loss"),
            (self.nucleation, NucleationResources, "nucleation"),
        )
        views = (
            self.prepared_views.condensation,
            self.prepared_views.coagulation,
            self.prepared_views.wall_loss,
            self.prepared_views.nucleation,
        )
        for (resource_value, resource_type, name), _view in zip(
            expected_resources, views, strict=True
        ):
            if (
                resource_value is not None
                and type(resource_value) is not resource_type
            ):
                raise TypeError(f"{name} request has an invalid exact type.")
        if self.condensation is not None and any(
            getattr(self.condensation, field.name) is None
            for field in fields(CondensationScratchBuffers)
        ):
            raise ValueError("condensation request must be complete.")
        if self.nucleation is not None and any(
            getattr(self.nucleation, name) is None
            for name in (
                "scratch",
                "finalized_demand",
                "diagnostics",
                "exhaustion",
            )
        ):
            raise ValueError("nucleation request must be complete.")


@dataclass(frozen=True, eq=False)
class CaptureResourceSet:
    """Retain one atomically published capture resource set by exact identity.

    This setup-only carrier owns no payload copies, bytes, RNG words, or
    accounting.  It exposes retained metadata and views solely for later
    identity validation; callers cannot use it to acquire, execute, transfer,
    synchronize, or mutate resource bindings.
    """

    requirements: CaptureResourceRequirements
    capacities: ResourceInventoryCapacities
    inventory: SelectedResourceInventory
    report: LogicalResourceReport
    prepared_views: PreparedResourceViews
    communication_resources: CommunicationResources | None
    condensation: CondensationResources | None
    coagulation: CoagulationResources | None
    wall_loss: WallLossResources | None
    nucleation: NucleationResources | None
    coagulation_stream_registry: _PublishedStreamRegistry | None
    wall_loss_stream_registry: _PublishedStreamRegistry | None


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
_DILUTION = ResourceManifest(
    "dilution",
    (
        _entry("normalized_coefficient", "dilution", wp.float64, "b"),
        _entry("factors", "dilution", wp.float64, "b"),
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


_CALLER_CONFIGURATION_ROLES = frozenset(
    (family, role)
    for family in ("communication_gas", "communication_particles")
    for role in ("source_boxes", "destination_boxes", "enabled", "rates")
)


def _inventory_metadata(
    entry: ManifestEntry,
) -> tuple[
    Literal[
        "fixed",
        "collision_capacity",
        "gas_edge_capacity",
        "particle_edge_capacity",
    ],
    Literal["registry_or_caller_sidecar", "caller_configuration"],
]:
    """Return declaration-only capacity and ownership metadata for one role."""
    capacity_source: Literal[
        "fixed",
        "collision_capacity",
        "gas_edge_capacity",
        "particle_edge_capacity",
    ]
    if entry.family == "coagulation" and entry.role == "collision_pairs":
        capacity_source = "collision_capacity"
    elif entry.family == "communication_gas" and entry.shape_kind == "e":
        capacity_source = "gas_edge_capacity"
    elif entry.family == "communication_particles" and entry.shape_kind in (
        "e",
        "en",
    ):
        capacity_source = "particle_edge_capacity"
    else:
        capacity_source = "fixed"
    ownership: Literal[
        "registry_or_caller_sidecar",
        "caller_configuration",
    ] = (
        "caller_configuration"
        if (entry.family, entry.role) in _CALLER_CONFIGURATION_ROLES
        else "registry_or_caller_sidecar"
    )
    return capacity_source, ownership


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


def _validated_extent(value: Any) -> int:
    """Return one nonnegative non-boolean integral shape extent."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(
            "Resource shape extents must be non-boolean nonnegative integers."
        )
    if value > _MAX_SIZE:
        raise ValueError("Resource allocation size exceeds supported range.")
    return int(value)


def _checked_product(left: int, right: Any) -> int:
    """Multiply validated resource extents without exceeding ``_MAX_SIZE``."""
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, Integral)
        or not isinstance(right, Integral)
        or left < 0
        or right < 0
    ):
        raise ValueError("Resource allocation size exceeds supported range.")
    left = int(left)
    right = int(right)
    if left > _MAX_SIZE // max(right, 1):
        raise ValueError("Resource allocation size exceeds supported range.")
    return left * right


def _shape_element_count(shape: tuple[int, ...]) -> int:
    """Return the checked logical element count for one resource shape."""
    count = 1
    for extent in shape:
        count = _checked_product(count, _validated_extent(extent))
    return count


def _logical_byte_count(shape: tuple[int, ...], dtype: Any) -> int:
    """Return checked logical schema bytes for one resource shape and dtype."""
    return _checked_product(_shape_element_count(shape), _item_size(dtype))


def _checked_sum(values: tuple[int, ...] | list[int]) -> int:
    """Return a checked sum of logical resource counts."""
    total = 0
    for value in values:
        total = _checked_product(1, total + _validated_extent(value))
    return total


def _contiguous_strides(shape: tuple[int, ...], dtype: Any) -> tuple[int, ...]:
    """Return checked canonical contiguous byte strides for one shape."""
    stride = _item_size(dtype)
    expected: list[int] = []
    for extent in reversed(shape):
        extent = _validated_extent(extent)
        expected.insert(0, stride)
        stride = _checked_product(stride, extent)
    return tuple(expected)


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

    Capture registration is a separate concrete-only host-metadata seam. It
    pins one optional published closed communication view and one ordered
    diagnostic registration tuple by identity after complete schema and
    nonaliasing validation. It neither changes normal resource enumeration nor
    participates in checkpoint or restart state.
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
        self._capture_inventory: SelectedResourceInventory | None = None
        self._capture_resource_set: CaptureResourceSet | None = None
        self._capture_resource_fingerprint: tuple[Any, ...] | None = None

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
            _DILUTION,
        )

    def logical_resource_report(
        self, capacities: ResourceInventoryCapacities
    ) -> LogicalResourceReport:
        """Return a frozen direct-module-only logical resource report.

        Resolves every canonical manifest role, including both communication
        families, from pinned dimensions and explicit capacities. The result
        reports logical schema bytes rather than allocator-reserved bytes. This
        read-only accessor does not inspect payloads or pointers, or acquire,
        allocate, bind, or mutate resources.

        Args:
            capacities: Exact frozen carrier of logical collision, gas-edge,
                and particle-edge capacities.

        Returns:
            Pointer-free frozen family and role reports in canonical declaration
            order.

        Raises:
            TypeError: If ``capacities`` is not an exact
                ``ResourceInventoryCapacities`` carrier.
            ValueError: If a capacity, checked logical-byte calculation, or
                pinned session metadata is invalid.
        """
        self.validate_pinned_session(self._session)
        if type(capacities) is not ResourceInventoryCapacities:
            raise TypeError(
                "capacities must be an exact ResourceInventoryCapacities."
            )
        collision_capacity = _validated_extent(capacities.collision_capacity)
        gas_edge_capacity = _validated_extent(capacities.gas_edge_capacity)
        particle_edge_capacity = _validated_extent(
            capacities.particle_edge_capacity
        )
        capacity_by_source = {
            "collision_capacity": collision_capacity,
            "gas_edge_capacity": gas_edge_capacity,
            "particle_edge_capacity": particle_edge_capacity,
        }
        family_reports: list[LogicalResourceFamilyReport] = []
        for manifest in self.manifests:
            roles: list[LogicalResourceRoleReport] = []
            for manifest_entry in manifest.entries:
                capacity_source, ownership = _inventory_metadata(manifest_entry)
                capacity = capacity_by_source.get(capacity_source)
                shape = self._shape(manifest_entry, capacity)
                resolved_entry = ResolvedResourceInventoryEntry(
                    manifest_entry.family,
                    manifest_entry.role,
                    manifest_entry.dtype,
                    shape,
                    capacity_source,
                    ownership,
                )
                roles.append(
                    LogicalResourceRoleReport(
                        resolved_entry,
                        _shape_element_count(shape),
                        _logical_byte_count(shape, manifest_entry.dtype),
                    )
                )
            family_reports.append(
                LogicalResourceFamilyReport(
                    manifest.family,
                    tuple(roles),
                    _checked_sum([role.logical_byte_count for role in roles]),
                )
            )
        families = tuple(family_reports)
        return LogicalResourceReport(
            families,
            _checked_sum([family.logical_byte_count for family in families]),
        )

    def selected_resource_report(self) -> SelectedResourceInventory:
        """Return the one previously registered concrete capture inventory.

        The retained report is returned by identity. This accessor validates
        only the pinned active session and does not rescan arrays, allocate,
        synchronize, transfer, inspect payloads, or rebuild role metadata.

        Returns:
            The immutable inventory retained during registration.

        Raises:
            ValueError: If no capture selection has been registered or the
                pinned session has drifted.
        """
        self.validate_pinned_session(self._session)
        if self._capture_inventory is None:
            raise ValueError("Capture resources have not been registered.")
        return self._capture_inventory

    def _capture_fingerprint(
        self, requirements: CaptureResourceRequirements
    ) -> tuple[Any, ...]:
        """Return immutable capture identity and capacity data."""
        views = requirements.prepared_views
        return (
            id(requirements.session),
            id(requirements.capacities),
            id(requirements.inventory),
            id(requirements.prepared_views),
            id(requirements.communication_resources),
            tuple(
                id(value)
                for value in (
                    requirements.condensation,
                    requirements.coagulation,
                    requirements.wall_loss,
                    requirements.nucleation,
                    views.condensation,
                    views.coagulation,
                    views.dilution,
                    views.wall_loss,
                    views.nucleation,
                )
            ),
            requirements.capacities.collision_capacity,
            requirements.capacities.gas_edge_capacity,
            requirements.capacities.particle_edge_capacity,
        )

    def _validate_capture_requirements(
        self, requirements: CaptureResourceRequirements
    ) -> None:
        """Validate capture request identities before allocation or staging."""
        if type(requirements) is not CaptureResourceRequirements:
            raise TypeError(
                "requirements must be an exact CaptureResourceRequirements."
            )
        self.validate_pinned_session(requirements.session)
        if requirements.inventory is not self.selected_resource_report():
            raise ValueError("requirements must retain the exact P3 inventory.")
        if (
            requirements.communication_resources
            is not self.get_communication_resources()
        ):
            raise ValueError("capture communication resource identity changed.")
        communication = requirements.communication_resources
        if communication is not None:
            map_data = communication.configuration.communication_map
            capacity = int(map_data.edge_capacity)
            supplied_capacity = (
                requirements.capacities.gas_edge_capacity
                if map_data.transport_mode is CommunicationTransportMode.GAS
                else requirements.capacities.particle_edge_capacity
            )
            if supplied_capacity != capacity:
                raise ValueError(
                    "capture communication capacity must match its P3 resource."
                )

    def _capture_set_matches(
        self, requirements: CaptureResourceRequirements
    ) -> bool:
        """Return whether a request retains every published capture identity."""
        capture_set = self._capture_resource_set
        if capture_set is None:
            return False
        return (
            requirements is capture_set.requirements
            and self._capture_resource_fingerprint
            == self._capture_fingerprint(requirements)
            and requirements.inventory is capture_set.inventory
            and requirements.capacities is capture_set.capacities
            and requirements.communication_resources
            is capture_set.communication_resources
        )

    def validate_capture_resource_set(
        self, requirements: CaptureResourceRequirements
    ) -> CaptureResourceSet:
        """Return the retained matching capture set without resource work.

        The accessor is metadata-only.  It neither constructs reports nor
        acquires, allocates, initializes, reads, transfers, synchronizes, or
        mutates any resource binding.
        """
        if type(requirements) is not CaptureResourceRequirements:
            raise TypeError(
                "requirements must be an exact CaptureResourceRequirements."
            )
        self.validate_pinned_session(requirements.session)
        if not self._capture_set_matches(requirements):
            raise ValueError(
                "Capture resource set identities are incompatible."
            )
        capture_set = self._capture_resource_set
        if capture_set is None:
            raise ValueError("Capture resources have not been prepared.")
        return capture_set

    def _validate_staged_nonalias(
        self,
        bindings: dict[str, Any],
        entries: tuple[ManifestEntry, ...],
        capacity: int | None,
        staged: tuple[dict[str, Any], ...],
    ) -> None:
        """Validate a private family against published and staged sidecars."""
        ranges = [
            self._validate_array(entry, bindings[entry.role], capacity)
            for entry in entries
        ]
        values = [bindings[entry.role] for entry in entries]
        other_values = [
            value for family in staged for value in family.values()
        ] + [
            value
            for family in self._bindings.values()
            for value in family.values()
        ]
        self._reject_shared_identities(values, other_values)
        self._reject_primary_aliases(values)
        other_ranges = [self._array_range(value) for value in other_values]
        for index, byte_range in enumerate(ranges):
            if any(
                self._ranges_overlap(byte_range, other)
                for other in ranges[index + 1 :] + other_ranges
            ):
                raise ValueError("Sidecar byte ranges must not overlap.")

    def _stage_family(
        self,
        manifest: ResourceManifest,
        supplied: dict[str, Any],
        capacity: int | None,
        staged: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build one unpublished family after complete supplied preflight."""
        existing = self._bindings.get(manifest.family)
        if existing is not None:
            if (
                capacity is not None
                and self._capacities.get(manifest.family) != capacity
            ):
                raise ValueError("Capture resource capacity changed.")
            if any(
                supplied[entry.role] is not None
                and supplied[entry.role] is not existing[entry.role]
                for entry in manifest.entries
            ):
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
        self._validate_staged_nonalias(
            candidate, manifest.entries, capacity, tuple(staged)
        )
        staged.append(candidate)
        return candidate

    def _new_stream_registry(
        self, process_id: str, state: Any, other_state: Any
    ) -> _PublishedStreamRegistry:
        """Create and initialize exactly one unpublished resident stream."""
        root_seed, logical_box_ids, lanes = self._stream_metadata()
        initialize_all = other_state is None
        if other_state is None:
            other_state = wp.zeros(
                self._shape(
                    _WALL_LOSS.entries[0]
                    if process_id == "coagulation"
                    else _COAGULATION.entries[2]
                ),
                dtype=wp.uint32,
                device=self._signature[2],
            )
        pairs = (
            (
                "coagulation",
                state if process_id == "coagulation" else other_state,
            ),
            ("wall_loss", state if process_id == "wall_loss" else other_state),
        )
        registry = StreamRegistry(
            root_seed,
            self._session.dimensions.n_boxes,
            logical_box_ids,
            lanes,
            pairs,
        )
        if initialize_all:
            registry.initialize()
        else:
            registry.initialize_process(process_id)
        return registry

    def prepare_capture_resources(  # noqa: C901
        self, requirements: CaptureResourceRequirements
    ) -> CaptureResourceSet:
        """Atomically stage and publish the complete frozen capture set.

        A compatible exact repeat returns the original outer set. All allocator,
        schema, alias, view, report, and stream-initialization work remains
        private until the final non-fallible publication assignments.
        """
        if type(requirements) is not CaptureResourceRequirements:
            raise TypeError(
                "requirements must be an exact CaptureResourceRequirements."
            )
        self.validate_pinned_session(requirements.session)
        if self._capture_resource_set is not None:
            if self._capture_set_matches(requirements):
                return self._capture_resource_set
            raise ValueError(
                "Capture resource set identities are incompatible."
            )
        self._validate_capture_requirements(requirements)
        report = self.logical_resource_report(requirements.capacities)
        views = requirements.prepared_views
        staged: list[dict[str, Any]] = []
        staged_families: dict[str, dict[str, Any]] = {}

        def stage(
            enabled: bool,
            manifest: ResourceManifest,
            supplied: dict[str, Any],
            capacity: int | None = None,
        ) -> dict[str, Any] | None:
            """Stage only requested process families in canonical order."""
            if not enabled:
                return None
            result = self._stage_family(manifest, supplied, capacity, staged)
            if manifest.family not in self._bindings:
                staged_families[manifest.family] = result
            return result

        condensation_bindings = stage(
            views.condensation is not None
            or requirements.condensation is not None,
            _CONDENSATION,
            {
                entry.role: None
                if requirements.condensation is None
                else getattr(requirements.condensation, entry.role)
                for entry in _CONDENSATION.entries
            },
        )
        collision_capacity = _validated_extent(
            requirements.capacities.collision_capacity
        )
        if (
            views.coagulation is not None
            or requirements.coagulation is not None
        ) and collision_capacity <= 0:
            raise ValueError(
                "collision_capacity must be positive for coagulation."
            )
        coagulation_bindings = stage(
            views.coagulation is not None
            or requirements.coagulation is not None,
            _COAGULATION,
            {
                entry.role: None
                if requirements.coagulation is None
                else getattr(requirements.coagulation, entry.role)
                for entry in _COAGULATION.entries
            },
            collision_capacity,
        )
        wall_loss_bindings = stage(
            views.wall_loss is not None or requirements.wall_loss is not None,
            _WALL_LOSS,
            {
                "rng_states": None
                if requirements.wall_loss is None
                else requirements.wall_loss.rng_states
            },
        )
        nucleation_bindings = stage(
            views.nucleation is not None or requirements.nucleation is not None,
            _NUCLEATION,
            (
                {entry.role: None for entry in _NUCLEATION.entries}
                if requirements.nucleation is None
                else self._nucleation_supplied_bindings(
                    requirements.nucleation.scratch,
                    requirements.nucleation.finalized_demand,
                    requirements.nucleation.diagnostics,
                    requirements.nucleation.exhaustion,
                )
            ),
        )
        if views.dilution is not None:
            self.validate_dilution_resources(
                requirements.session, views.dilution
            )

        condensation_view = (
            None
            if condensation_bindings is None
            else self._views.get(
                "condensation",
                CondensationResources(
                    CondensationScratchBuffers(**condensation_bindings)
                ),
            )
        )
        coagulation_view = (
            None
            if coagulation_bindings is None
            else self._views.get(
                "coagulation",
                CoagulationResources(
                    collision_capacity, **coagulation_bindings
                ),
            )
        )
        wall_loss_view = (
            None
            if wall_loss_bindings is None
            else self._views.get(
                "wall_loss", WallLossResources(**wall_loss_bindings)
            )
        )
        nucleation_view = (
            None
            if nucleation_bindings is None
            else self._views.get(
                "nucleation", self._nucleation_view(nucleation_bindings)
            )
        )
        for required, supplied, created, name in (
            (
                views.condensation,
                requirements.condensation,
                condensation_view,
                "condensation",
            ),
            (
                views.coagulation,
                requirements.coagulation,
                coagulation_view,
                "coagulation",
            ),
            (
                views.wall_loss,
                requirements.wall_loss,
                wall_loss_view,
                "wall_loss",
            ),
            (
                views.nucleation,
                requirements.nucleation,
                nucleation_view,
                "nucleation",
            ),
        ):
            if supplied is not None and required is not created:
                raise ValueError(f"prepared {name} view identity changed.")

        coagulation_stream = self._coagulation_stream_registry
        wall_loss_stream = self._wall_loss_stream_registry
        if "coagulation" in staged_families:
            if coagulation_bindings is None:
                raise RuntimeError("Coagulation bindings were not staged.")
            other = (
                wall_loss_bindings["rng_states"]
                if wall_loss_bindings is not None
                else None
            )
            coagulation_stream = self._new_stream_registry(
                "coagulation", coagulation_bindings["rng_states"], other
            )
        if "wall_loss" in staged_families:
            if wall_loss_bindings is None:
                raise RuntimeError("Wall-loss bindings were not staged.")
            other = (
                coagulation_bindings["rng_states"]
                if coagulation_bindings is not None
                else None
            )
            wall_loss_stream = self._new_stream_registry(
                "wall_loss", wall_loss_bindings["rng_states"], other
            )
        # Allocation requests have no established view in the requirements.
        # Retain one complete canonical carrier containing the resolved staged
        # or established views, rather than exposing an input placeholder.
        prepared = PreparedResourceViews(
            condensation_view,
            coagulation_view,
            views.dilution,
            wall_loss_view,
            nucleation_view,
        )
        candidate = CaptureResourceSet(
            requirements,
            requirements.capacities,
            requirements.inventory,
            report,
            prepared,
            requirements.communication_resources,
            condensation_view,
            coagulation_view,
            wall_loss_view,
            nucleation_view,
            coagulation_stream,
            wall_loss_stream,
        )
        # All preceding operations can fail. Publication is assignment-only.
        self._bindings.update(staged_families)
        if "coagulation" in staged_families:
            self._capacities["coagulation"] = collision_capacity
        if condensation_view is not None:
            self._views["condensation"] = condensation_view
        if coagulation_view is not None:
            self._views["coagulation"] = coagulation_view
        if wall_loss_view is not None:
            self._views["wall_loss"] = wall_loss_view
        if nucleation_view is not None:
            self._views["nucleation"] = nucleation_view
            self._nucleation_records = (
                nucleation_view.scratch,
                nucleation_view.finalized_demand,
                nucleation_view.diagnostics,
                nucleation_view.exhaustion,
                nucleation_view.exhaustion.resampling_buffers,
            )
        self._coagulation_stream_registry = coagulation_stream
        self._wall_loss_stream_registry = wall_loss_stream
        self._capture_resource_fingerprint = self._capture_fingerprint(
            requirements
        )
        self._capture_resource_set = candidate
        return candidate

    def _selected_role(
        self,
        canonical_name: str,
        family: str,
        entry: ManifestEntry,
        value: Any,
        capacity: int | None,
        *,
        read_only: bool = False,
    ) -> tuple[SelectedResourceRole, tuple[int, int] | None]:
        """Resolve selected-role metadata once without inspecting payloads."""
        byte_range = self._validate_array(entry, value, capacity)
        shape = self._shape(entry, capacity)
        return (
            SelectedResourceRole(
                canonical_name,
                family,
                entry.dtype,
                shape,
                value,
                _shape_element_count(shape),
                _logical_byte_count(shape, entry.dtype),
                read_only,
            ),
            byte_range,
        )

    def _capture_candidate_roles(
        self,
        communication_resources: CommunicationResources | None,
        registrations: tuple[Any, ...],
    ) -> tuple[
        tuple[SelectedResourceFamilyReport, ...],
        tuple[SelectedResourceRole, ...],
        tuple[tuple[int, int] | None, ...],
    ]:
        """Build deterministic selected-role metadata before publication.

        The caller has already validated the selection's exact session,
        communication view, and diagnostic registrations. This helper resolves
        each selected schema and byte range once; it does not perform device I/O
        or mutate the capture inventory.
        """
        grouped: dict[str, list[SelectedResourceRole]] = {}
        ranges: list[tuple[int, int] | None] = []

        def add(
            canonical_name: str,
            family: str,
            entry: ManifestEntry,
            value: Any,
            capacity: int | None = None,
            *,
            read_only: bool = False,
        ) -> None:
            role, byte_range = self._selected_role(
                canonical_name,
                family,
                entry,
                value,
                capacity,
                read_only=read_only,
            )
            grouped.setdefault(family, []).append(role)
            ranges.append(byte_range)

        if communication_resources is not None:
            configuration = communication_resources.configuration
            map_data = configuration.communication_map
            mode = map_data.transport_mode
            manifest = (
                _GAS_COMMUNICATION
                if mode is CommunicationTransportMode.GAS
                else _PARTICLE_COMMUNICATION
            )
            family = manifest.family
            values = {
                "source_boxes": map_data.source_boxes,
                "destination_boxes": map_data.destination_boxes,
                "enabled": map_data.enabled,
                "rates": map_data.rates,
                **self._record_bindings(communication_resources.buffers),
                "invalid": communication_resources.execution_state.invalid,
                "active_or_demand": (
                    communication_resources.execution_state.active_or_demand
                ),
                "volume_invalid": (
                    communication_resources.execution_state.volume_invalid
                ),
                "volume_changed": (
                    communication_resources.execution_state.volume_changed
                ),
                "initial_masses": (
                    communication_resources.execution_state.initial_masses
                ),
                "initial_concentration": (
                    communication_resources.execution_state.initial_concentration
                ),
                "initial_charge": (
                    communication_resources.execution_state.initial_charge
                ),
            }
            for entry in manifest.entries:
                add(
                    f"{family}:{entry.role}",
                    family,
                    entry,
                    values[entry.role],
                    int(map_data.edge_capacity),
                )
            if communication_resources.final_volumes is not None:
                entry = ManifestEntry("final_volumes", family, wp.float64, "b")
                add(
                    f"{family}:final_volumes",
                    family,
                    entry,
                    communication_resources.final_volumes,
                )

        from particula.execution.diagnostics import (
            ResidentDiagnosticOperation,
        )

        for index, registration in enumerate(registrations):
            operation = registration.operation
            shape_kind: Literal["b", "bs"] = (
                "b"
                if operation
                is ResidentDiagnosticOperation.PARTICLE_NUMBER_CONCENTRATION
                else "bs"
            )
            prefix = f"diagnostics:{index}:{operation.value}"
            add(
                f"{prefix}:output",
                "diagnostics",
                ManifestEntry(
                    "diagnostic output",
                    "diagnostics",
                    wp.float64,
                    shape_kind,
                ),
                registration.output,
            )
            for name in (
                "energy_transfer",
                "baseline_total_mass",
                "source_ledger",
                "sink_ledger",
            ):
                value = getattr(registration, name)
                if value is not None:
                    add(
                        f"{prefix}:{name}",
                        "diagnostics",
                        ManifestEntry(
                            "diagnostic accounting input",
                            "diagnostics",
                            wp.float64,
                            "bs",
                        ),
                        value,
                        read_only=True,
                    )
        families = tuple(
            SelectedResourceFamilyReport(
                family,
                tuple(roles),
                _checked_sum([role.logical_byte_count for role in roles]),
            )
            for family, roles in grouped.items()
        )
        roles = tuple(role for family in families for role in family.roles)
        if len(roles) != len(set(role.canonical_name for role in roles)):
            raise ValueError("Selected capture roles must be unique.")
        return families, roles, tuple(ranges)

    def _validate_capture_nonalias(  # noqa: C901
        self,
        roles: tuple[SelectedResourceRole, ...],
        ranges: tuple[tuple[int, int] | None, ...],
        selected_published: set[int],
    ) -> None:
        """Reject capture overlap using one host-only sorted interval sweep.

        Read-only diagnostic accounting inputs may share one another.  Every
        other selected, primary, or established-sidecar overlap is forbidden.
        """
        # A selected communication view necessarily repeats its already
        # published native sidecars in ``_bindings``.  Exclude only those
        # published values from the protected set.  Do not exclude every
        # selected value: a communication map or final-volume array may have
        # been illicitly reused as an unrelated established sidecar, and that
        # alias must still be rejected.
        protected = list(_primary_arrays(self._session)) + [
            value
            for bindings in self._bindings.values()
            for value in bindings.values()
            if id(value) not in selected_published
        ]
        protected.extend(
            view.final_volumes
            for view in self._views.values()
            if type(view) is CommunicationResources
            and view.final_volumes is not None
            and id(view.final_volumes) not in selected_published
        )
        intervals: list[tuple[int, int, int, bool, str]] = []
        published_seen: set[int] = set()
        for index, (role, byte_range) in enumerate(
            zip(roles, ranges, strict=True)
        ):
            if byte_range is not None:
                value_id = id(role.value)
                if (
                    value_id in selected_published
                    and role.family != "diagnostics"
                ):
                    if value_id in published_seen:
                        continue
                    published_seen.add(value_id)
                intervals.append(
                    (
                        byte_range[0],
                        byte_range[1],
                        index,
                        role.read_only,
                        role.canonical_name,
                    )
                )
        for index, value in enumerate(protected, start=len(roles)):
            byte_range = self._array_range(value)
            if byte_range is not None:
                intervals.append(
                    (byte_range[0], byte_range[1], index, False, "protected")
                )
        intervals.sort(key=lambda item: (item[0], item[1], item[2]))
        active: list[int] = []
        writable_active: list[int] = []
        for interval in intervals:
            start = interval[0]
            while active and active[0] <= start:
                heappop(active)
            while writable_active and writable_active[0] <= start:
                heappop(writable_active)
            if active and (not interval[3] or writable_active):
                raise ValueError(
                    "Selected capture resource byte ranges must not overlap."
                )
            heappush(active, interval[1])
            if not interval[3]:
                heappush(writable_active, interval[1])

    def register_capture_resources(
        self,
        session: ResidentSession,
        communication_resources: CommunicationResources | None,
        registrations: tuple[Any, ...],
    ) -> SelectedResourceInventory:
        """Pin one exact metadata-only communication and diagnostics selection.

        The first successful call retains an immutable identity inventory. A
        later exact repeat returns it; all other repeats reject without changing
        the retained inventory. A communication view, when supplied, must be
        an already-published closed GAS or PARTICLES view. Candidate metadata
        and nonaliasing are completely validated before first publication. This
        method performs no allocation, device dispatch, synchronization,
        transfer, or payload inspection.

        Args:
            session: Exact active session pinned by this registry.
            communication_resources: Optional published closed communication
                view to include in the selection.
            registrations: Exact ordered tuple of diagnostic registrations.

        Returns:
            The newly retained inventory, or the same inventory for an exact
            repeat registration.

        Raises:
            TypeError: If a carrier or registration tuple has the wrong exact
                type.
            ValueError: If session identity, publication identity, schemas,
                or nonaliasing constraints are invalid, or a different
                selection was already registered.
        """
        self.validate_pinned_session(session)
        inventory = self._capture_inventory
        if inventory is not None:
            if (
                communication_resources is inventory.communication_resources
                and registrations is inventory.registrations
            ):
                return inventory
            raise ValueError("Capture resources have already been registered.")
        if communication_resources is not None:
            if type(communication_resources) is not CommunicationResources:
                raise TypeError(
                    "communication_resources must be CommunicationResources "
                    "or None."
                )
            self.validate_communication_resources(
                session, communication_resources
            )
        if type(registrations) is not tuple:
            raise TypeError("registrations must be an exact tuple.")
        from particula.execution.diagnostics import (
            ResidentDiagnosticRegistration,
        )

        if not all(
            type(item) is ResidentDiagnosticRegistration
            for item in registrations
        ):
            raise TypeError(
                "registrations must be exact "
                "ResidentDiagnosticRegistration tuple."
            )
        self.validate_diagnostic_registrations(session, registrations)
        families, roles, ranges = self._capture_candidate_roles(
            communication_resources, registrations
        )
        selected_published: set[int] = set()
        if communication_resources is not None:
            selected_published.update(
                id(value)
                for value in (
                    communication_resources.configuration.communication_map.source_boxes,
                    communication_resources.configuration.communication_map.destination_boxes,
                    communication_resources.configuration.communication_map.enabled,
                    communication_resources.configuration.communication_map.rates,
                    *self._record_bindings(
                        communication_resources.buffers
                    ).values(),
                    communication_resources.execution_state.invalid,
                    communication_resources.execution_state.active_or_demand,
                    communication_resources.execution_state.volume_invalid,
                    communication_resources.execution_state.volume_changed,
                    communication_resources.execution_state.initial_masses,
                    communication_resources.execution_state.initial_concentration,
                    communication_resources.execution_state.initial_charge,
                    communication_resources.final_volumes,
                )
                if value is not None
            )
        self._validate_capture_nonalias(roles, ranges, selected_published)
        candidate = SelectedResourceInventory(
            communication_resources,
            registrations,
            families,
            _checked_sum([family.logical_byte_count for family in families]),
        )
        self._capture_inventory = candidate
        return candidate

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

    def _validate_diagnostic_binding_nonalias(  # noqa: C901
        self,
        outputs: list[Any],
        output_ranges: list[tuple[int, int] | None],
        inputs: list[Any],
        input_ranges: list[tuple[int, int] | None],
        protected: list[Any],
        protected_ranges: list[tuple[int, int] | None],
    ) -> None:
        """Reject diagnostic bindings that overlap protected/output storage.

        This metadata-only sorted sweep permits only accounting-input aliases;
        it avoids pairwise work for large diagnostic selections.
        """
        bindings = outputs + inputs
        ranges = output_ranges + input_ranges
        intervals: list[tuple[int, int, int, bool, bool]] = []
        for index, byte_range in enumerate(ranges):
            if byte_range is not None:
                intervals.append(
                    (
                        byte_range[0],
                        byte_range[1],
                        index,
                        index >= len(outputs),
                        False,
                    )
                )
        for index, byte_range in enumerate(
            protected_ranges, start=len(bindings)
        ):
            if byte_range is not None:
                intervals.append(
                    (byte_range[0], byte_range[1], index, False, True)
                )
        intervals.sort(key=lambda item: (item[0], item[1], item[2]))
        active: list[int] = []
        writable_active: list[int] = []
        protected_active: list[int] = []
        for start, end, _, read_only, is_protected in intervals:
            while active and active[0] <= start:
                heappop(active)
            while writable_active and writable_active[0] <= start:
                heappop(writable_active)
            while protected_active and protected_active[0] <= start:
                heappop(protected_active)
            if active and (not read_only or writable_active):
                if is_protected or protected_active:
                    raise ValueError(
                        "Diagnostic bindings must not alias resident resources."
                    )
                raise ValueError(
                    "Diagnostic outputs must not overlap bindings."
                )
            heappush(active, end)
            if not read_only:
                heappush(writable_active, end)
            if is_protected:
                heappush(protected_active, end)

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

        Args:
            session: Exact active session pinned by this registry.
            resources: Exact published condensation resource view.

        Raises:
            TypeError: If ``resources`` is not an exact condensation view.
            ValueError: If the view or any pinned sidecar binding has changed.

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

        Args:
            session: Exact active session pinned by this registry.
            resources: Exact published coagulation resource view.

        Raises:
            TypeError: If ``resources`` is not an exact coagulation view.
            ValueError: If capacity, view identity, or sidecar bindings changed.

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

    def validate_dilution_resources(
        self, session: ResidentSession, resources: DilutionResources
    ) -> None:
        """Validate one complete unbound dilution view without publishing it.

        Dilution is descriptor-only in this phase, so this validation accepts a
        caller-owned complete view but deliberately does not retain it in the
        registry. It performs metadata-only exact-schema, device, contiguity,
        primary-alias, role-reuse, and cross-family nonaliasing checks without
        allocating, publishing, initializing, or mutating sidecars.

        Args:
            session: Exact active session pinned by this registry.
            resources: Complete caller-owned dilution descriptor view.

        Raises:
            TypeError: If ``resources`` is not an exact ``DilutionResources``
                carrier or either sidecar is not a Warp array.
            ValueError: If the pinned session, sidecar schema, device,
                contiguity, identity, or byte-range nonaliasing is invalid.
        """
        self.validate_pinned_session(session)
        if type(resources) is not DilutionResources:
            raise TypeError("resources must be an exact DilutionResources.")
        bindings = {
            "normalized_coefficient": resources.normalized_coefficient,
            "factors": resources.factors,
        }
        self._validate_nonalias(bindings, _DILUTION.entries, capacity=None)

    def validate_prepared_resource_views(
        self, session: ResidentSession, views: PreparedResourceViews
    ) -> None:
        """Validate a complete private prepared-view carrier read-only.

        Bound family views must be the exact established publication. The
        descriptor-only dilution view is checked against the same protected and
        established storage, but is not acquired or retained. This method is a
        setup-only seam; enqueue paths retain the validated references and do
        not call it again. It does not inspect payloads, acquire, allocate,
        publish, initialize, transfer, synchronize, or mutate state.

        Args:
            session: Exact active session pinned by this registry.
            views: Exact optional-family carrier to validate.

        Raises:
            TypeError: If ``views`` or a supplied nested resource carrier has
                the wrong exact type.
            ValueError: If session, publication identity, sidecar schema, or
                nonaliasing validation fails for a supplied family.
        """
        self.validate_pinned_session(session)
        if type(views) is not PreparedResourceViews:
            raise TypeError("views must be an exact PreparedResourceViews.")
        if views.condensation is not None:
            self.validate_condensation_resources(session, views.condensation)
        if views.coagulation is not None:
            self.validate_coagulation_resources(session, views.coagulation)
        if views.dilution is not None:
            self.validate_dilution_resources(session, views.dilution)
        if views.wall_loss is not None:
            self.validate_wall_loss_resources(session, views.wall_loss)
        if views.nucleation is not None:
            self.validate_nucleation_resources(session, views.nucleation)

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
            return (dimensions.n_boxes, _validated_extent(capacity), 2)
        if entry.shape_kind == "e":
            if capacity is None:
                raise ValueError("communication edge capacity is required.")
            return (_validated_extent(capacity),)
        if entry.shape_kind == "en":
            if capacity is None:
                raise ValueError("communication edge capacity is required.")
            return (_validated_extent(capacity), dimensions.n_particles)
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
        """Validate contiguous metadata and return a nonempty byte range.

        Args:
            value: Warp array whose storage metadata is checked.
            shape: Expected logical shape of the array.
            dtype: Expected Warp dtype.
            role: Role name used in validation errors.

        Returns:
            Half-open byte range for nonempty storage, or ``None`` for an empty
            logical array.

        Raises:
            ValueError: If strides, pointer alignment, or storage capacity is
                invalid.
        """
        strides = getattr(value, "strides", None)
        if not isinstance(strides, tuple) or len(strides) != len(shape):
            raise ValueError(f"{role} must have contiguous strides.")
        item_size = _item_size(dtype)
        if strides != _contiguous_strides(shape, dtype):
            raise ValueError(f"{role} must be contiguous.")
        count = _shape_element_count(shape)
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
        required = _logical_byte_count(shape, dtype)
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, Integral)
            or capacity < required
            or capacity % item_size
        ):
            raise ValueError(
                f"{role} must have sufficient integral storage capacity."
            )
        pointer_value = int(pointer)
        if pointer_value > _MAX_SIZE - required:
            raise ValueError(f"{role} byte range exceeds supported range.")
        return pointer_value, pointer_value + required

    @staticmethod
    def _checked_product(left: int, right: int) -> int:
        """Multiply resource extents with central checked arithmetic."""
        return _checked_product(left, right)

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
        _logical_byte_count(shape, entry.dtype)
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
        """Reject overlaps among candidate, registered, and primary arrays.

        Args:
            bindings: Candidate role-to-array bindings to validate.
            entries: Manifest entries defining the candidate schemas.
            capacity: Dynamic capacity required by the manifest, if any.

        Raises:
            TypeError: If a candidate binding is not a Warp array.
            ValueError: If a binding has invalid metadata or overlaps protected
                storage.
        """
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
        shape = tuple(array.shape)
        if strides != _contiguous_strides(shape, array.dtype):
            raise ValueError("Registry arrays must be contiguous.")
        count = _shape_element_count(shape)
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
        required = _logical_byte_count(shape, array.dtype)
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
        pointer_value = int(pointer)
        if pointer_value > _MAX_SIZE - required:
            raise ValueError(
                "Registry array byte range exceeds supported range."
            )
        return pointer_value, pointer_value + required

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

    def get_communication_resources(self) -> CommunicationResources | None:
        """Return the established concrete communication view, if any.

        This metadata-only accessor exposes the single identity-pinned resident
        communication family so an explicit checkpoint restart can reuse its
        restored configuration without accessing registry internals.
        """
        self._validate_session_signature()
        gas = self._views.get("communication_gas")
        particles = self._views.get("communication_particles")
        if gas is not None and particles is not None:
            raise ValueError(
                "Only one resident communication family may be bound."
            )
        return gas if gas is not None else particles

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
            """Build one native record from its corresponding role bindings."""
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
