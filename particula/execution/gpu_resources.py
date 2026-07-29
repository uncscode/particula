"""Allocate concrete reusable Warp sidecars for one active resident session.

This direct-import-only, Warp-dependent boundary pins complete fixed-shape
native sidecar families to one exact ``ACTIVE`` :class:`ResidentSession`.
It allocates and validates resources only: it neither executes a process nor
transfers, synchronizes, restores, resizes, or initializes RNG state.  The
manifests and views here are concrete-only and are deliberately not exported
from :mod:`particula.execution`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral
from typing import Any, Literal, cast

import warp as wp

from particula.execution.gpu_session import (
    ResidentDimensions,
    ResidentLifecycle,
    ResidentSession,
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
    "GPUResourceRegistry",
    "CondensationResources",
    "CoagulationResources",
    "WallLossResources",
    "NucleationResources",
]

_INT32_MAX = 2**31 - 1
_MAX_SIZE = (1 << 63) - 1
_ShapeKind = Literal["b", "bn", "bs", "bns", "bc2"]


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


@dataclass(frozen=True, eq=False)
class CondensationResources:
    """Expose a complete native condensation scratch record."""

    scratch_buffers: CondensationScratchBuffers


@dataclass(frozen=True, eq=False)
class CoagulationResources:
    """Expose native coagulation output and persistent RNG sidecars."""

    collision_capacity: int
    collision_pairs: Any
    n_collisions: Any
    rng_states: Any


@dataclass(frozen=True, eq=False)
class WallLossResources:
    """Expose the native persistent wall-loss RNG sidecar."""

    rng_states: Any


@dataclass(frozen=True, eq=False)
class NucleationResources:
    """Expose complete native nucleation sidecar records."""

    scratch: NucleationScratchBuffers
    finalized_demand: NucleationFinalizedDemandBuffers
    diagnostics: NucleationDiagnosticBuffers
    exhaustion: NucleationExhaustionBuffers


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
    No payload is read, copied, synchronized, or mutated by acquisition.
    """

    def __init__(self, session: ResidentSession) -> None:
        """Create a registry for one exact, already validated active session."""
        if type(session) is not ResidentSession:
            raise TypeError("session must be an exact ResidentSession.")
        self._session = session
        self._validate_session_state()
        self._signature = self._session_signature()
        self._bindings: dict[str, dict[str, Any]] = {}
        self._views: dict[str, Any] = {}
        self._capacities: dict[str, int] = {}

    @property
    def manifests(self) -> tuple[ResourceManifest, ...]:
        """Return the canonical immutable direct-module manifest set."""
        return (_CONDENSATION, _COAGULATION, _WALL_LOSS, _NUCLEATION)

    def _session_signature(self) -> tuple[Any, ...]:
        particles = cast(Any, self._session.particles)
        return (
            self._session.lifecycle,
            self._session.dimensions,
            particles.masses.device,
            *(id(value) for value in _primary_arrays(self._session)),
        )

    def _validate_session_signature(self) -> None:
        """Reject a fabricated, inactive, or replaced resident session state."""
        self._validate_session_carriers()
        if self._session_signature() != self._signature:
            raise ValueError("ResidentSession signature changed.")

        self._session.__post_init__()

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
        }
        if entry.shape_kind == "bc2":
            if capacity is None:
                raise ValueError("collision capacity is required.")
            return (dimensions.n_boxes, capacity, 2)
        return shapes[entry.shape_kind]

    def _validate_array(
        self, entry: ManifestEntry, value: Any, capacity: int | None
    ) -> tuple[int, int] | None:
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
        if not isinstance(getattr(value, "ptr", None), Integral) or (
            value.ptr < 0
        ):
            raise ValueError(f"{role} must have a valid pointer.")
        return int(value.ptr), int(value.ptr) + count * item_size

    @staticmethod
    def _checked_product(left: int, right: int) -> int:
        if right < 0 or left > _MAX_SIZE // max(right, 1):
            raise ValueError(
                "Resource allocation size exceeds supported range."
            )
        return left * right

    def _allocate(self, entry: ManifestEntry, capacity: int | None) -> Any:
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

    @staticmethod
    def _array_range(array: Any) -> tuple[int, int] | None:
        """Return one validated registry array's nonempty byte range."""
        strides = getattr(array, "strides", None)
        if not isinstance(strides, tuple) or len(strides) != len(array.shape):
            raise ValueError("Registry arrays must have contiguous strides.")
        count = 1
        for length in array.shape:
            count *= length
        if count == 0:
            return None
        if not isinstance(getattr(array, "ptr", None), Integral) or (
            array.ptr < 0
        ):
            raise ValueError("Registry arrays must have a valid pointer.")
        size = _item_size(array.dtype)
        return int(array.ptr), int(array.ptr) + count * size

    def _acquire(
        self,
        manifest: ResourceManifest,
        supplied: dict[str, Any],
        capacity: int | None = None,
    ) -> dict[str, Any]:
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
        for entry in manifest.entries:
            if candidate[entry.role] is None:
                candidate[entry.role] = self._allocate(entry, capacity)
        self._validate_nonalias(candidate, manifest.entries, capacity)
        self._bindings[manifest.family] = candidate
        if capacity is not None:
            self._capacities[manifest.family] = capacity
        return candidate

    def acquire_condensation(
        self, *, buffers: CondensationScratchBuffers | None = None
    ) -> CondensationResources:
        """Acquire one complete pinned condensation scratch record."""
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
        """Acquire fixed-capacity coagulation output and RNG sidecars."""
        if isinstance(collision_capacity, bool) or not isinstance(
            collision_capacity, Integral
        ):
            raise TypeError(
                "collision_capacity must be a non-boolean integral."
            )
        if collision_capacity <= 0 or collision_capacity > _INT32_MAX:
            raise ValueError(
                "collision_capacity must be positive and int32-sized."
            )
        bindings = self._acquire(
            _COAGULATION,
            {
                "collision_pairs": collision_pairs,
                "n_collisions": n_collisions,
                "rng_states": rng_states,
            },
            int(collision_capacity),
        )
        if "coagulation" not in self._views:
            self._views["coagulation"] = CoagulationResources(
                int(collision_capacity), **bindings
            )
        return self._views["coagulation"]

    def acquire_wall_loss(
        self, *, rng_states: Any | None = None
    ) -> WallLossResources:
        """Acquire one persistent wall-loss RNG sidecar."""
        bindings = self._acquire(_WALL_LOSS, {"rng_states": rng_states})
        if "wall_loss" not in self._views:
            self._views["wall_loss"] = WallLossResources(**bindings)
        return self._views["wall_loss"]

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
        """Acquire complete pinned native nucleation records and scratch."""
        supplied = self._nucleation_supplied_bindings(
            scratch,
            finalized_demand,
            diagnostics,
            exhaustion,
        )
        bindings = self._acquire(_NUCLEATION, supplied)
        if "nucleation" not in self._views:
            self._views["nucleation"] = self._nucleation_view(bindings)
        return self._views["nucleation"]
