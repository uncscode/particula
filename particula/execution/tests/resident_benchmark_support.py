"""Provide concrete host-only support for resident benchmark evidence.

This test-support module defines the bounded artifact schema used by resident
benchmark tests. It validates immutable host records, produces deterministic
JSON, and atomically writes generic JSON below an explicit ``.artifacts``
root. It neither imports nor probes Warp or CUDA, allocates device resources,
changes resident execution, adds package exports, nor supplies a user-facing
API.
"""

# ruff: noqa: C901, E501

from __future__ import annotations

import json
import math
import os
import platform
import secrets
import stat
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

RESIDENT_BENCHMARK_SCHEMA_VERSION = 2
"""Version of the resident benchmark JSON envelope."""

MAX_TIMING_SAMPLES = 10_000
"""Maximum P1 timing samples accepted for one executed benchmark result."""

MAX_WARMUP_SAMPLES = 10_000
"""Maximum paired warmup samples accepted for resident timing evidence."""

MAX_ARTIFACT_PAYLOAD_BYTES = 1_048_576
"""Maximum UTF-8 size accepted for an untrusted artifact payload."""

MAX_ARTIFACT_NESTING_DEPTH = 32
"""Maximum JSON nesting depth accepted for an artifact payload."""

MAX_ARTIFACT_ROWS = 500
"""Maximum case or result rows accepted in one artifact."""

MAX_ARTIFACT_CONTAINER_ITEMS = 1_000
"""Maximum items in any decoded JSON mapping or list."""

PROCESS_ORDER = (
    "communication",
    "environment",
    "gas",
    "condensation",
    "coagulation",
    "dilution",
    "wall_loss",
    "nucleation",
    "diagnostics",
)
"""Canonical ordering for process combinations in resident evidence."""

SUPPORTED_TIMING_MODES = frozenset(
    {
        "wall_clock",
        "device_synchronized",
        "prepared_uncaptured_device_synchronized",
        "captured_replay_device_synchronized",
    }
)
_CAPTURE_COMPARISON_TIMING_MODES = frozenset(
    {
        "prepared_uncaptured_device_synchronized",
        "captured_replay_device_synchronized",
    }
)
RESIDENT_CAPTURE_COMPARISON_DESTINATION = (
    "benchmarks/resident_capture_comparison.json"
)
SUPPORTED_COMMUNICATION = frozenset({"none", "gas", "particles"})
RESIDENT_BOX_COUNTS = (1, 10, 100, 1000)
"""Canonical host-only resident scaling rows; requests are never downscaled."""
REQUIRED_METADATA_FIELDS = frozenset(
    {
        "timestamp_utc",
        "command",
        "python_version",
        "platform",
        "warp_version",
        "device",
        "synchronization_method",
        "warmup",
        "timestep_count",
        "seed",
        "prepared_signature_digest",
    }
)

_WARP_VERSION_FIELDS = frozenset({"status", "value"})
_WARP_VERSION_ERROR_FIELDS = frozenset({"status", "value", "error"})
_DEVICE_FIELDS = frozenset({"status", "identity", "memory"})
_DEVICE_ERROR_FIELDS = frozenset({"status", "identity", "memory", "error"})


class ResidentBenchmarkStatus(str, Enum):
    """Represent the outcome of one requested resident benchmark measurement.

    ``EXECUTED`` records timing evidence. ``UNAVAILABLE`` and
    ``SKIPPED_BUDGET`` record an explicit non-execution reason without timing
    samples.
    """

    EXECUTED = "executed"
    UNAVAILABLE = "unavailable"
    SKIPPED_BUDGET = "skipped_budget"


@dataclass(frozen=True, slots=True)
class ResidentBenchmarkAvailability:
    """Describe preconstruction CUDA/native-capture availability."""

    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate an availability carrier before it reaches fixture setup."""
        if not isinstance(self.available, bool):
            raise TypeError("availability must be a bool.")
        if self.available:
            if self.reason is not None:
                raise ValueError(
                    "available availability must not have a reason."
                )
        elif not isinstance(self.reason, str) or not self.reason:
            raise ValueError(
                "unavailable availability requires a nonempty reason."
            )


@dataclass(frozen=True, slots=True)
class ResidentBenchmarkPreflight:
    """Store a validated host-only matrix outcome before fixture construction."""

    case: "ResidentBenchmarkCase"
    status: ResidentBenchmarkStatus
    reason: str

    def __post_init__(self) -> None:
        """Require one allowed status and a nonempty deterministic reason."""
        if not isinstance(self.case, ResidentBenchmarkCase):
            raise TypeError("case must be a ResidentBenchmarkCase.")
        if self.status not in {
            ResidentBenchmarkStatus.EXECUTED,
            ResidentBenchmarkStatus.SKIPPED_BUDGET,
            ResidentBenchmarkStatus.UNAVAILABLE,
        }:
            raise ValueError("preflight status is invalid.")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("preflight reason must be a nonempty string.")


def _require_int(value: object, name: str, *, positive: bool = False) -> int:
    """Validate and return a non-bool integer field.

    Args:
        value: Candidate integer value.
        name: Field name used in error messages.
        positive: Require a value greater than zero when true; otherwise
            require a nonnegative value.

    Returns:
        The validated integer.

    Raises:
        TypeError: If ``value`` is not an integer or is a boolean.
        ValueError: If ``value`` violates the requested lower bound.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-bool integer.")
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


MAX_RESIDENT_MEMORY_BYTES = (1 << 63) - 1
"""Largest supported analytical resident-memory quantity in bytes."""

MEMORY_PROVENANCES = frozenset({"analytical", "registry_logical", "projected"})
MEMORY_SCENARIOS = frozenset({"steady_state", "checkpoint", "tape"})
RESIDENT_DIAGNOSTIC_OPERATIONS = (
    "gas_concentration_snapshot",
    "saturation_ratio_snapshot",
    "total_species_mass",
    "particle_number_concentration",
    "latent_heat_energy",
    "conservation_residual",
)
RESIDENT_MEMORY_COMMUNICATIONS = ("none", "gas", "particles")


def _require_memory_bytes(value: object, name: str) -> int:
    """Validate a bounded nonnegative analytical byte count."""
    value = _require_int(value, name)
    if value > MAX_RESIDENT_MEMORY_BYTES:
        raise ValueError(f"{name} exceeds MAX_RESIDENT_MEMORY_BYTES.")
    return value


def checked_dense_array_bytes(shape: object, itemsize: object) -> int:
    """Return bounded dense-array bytes without fixed-width arithmetic.

    Args:
        shape: Exact tuple of nonnegative Python integer extents.
        itemsize: Positive Python integer item width in bytes.

    Returns:
        The checked logical byte count.

    Raises:
        TypeError: If arguments are not exact required container/value types.
        ValueError: If an extent, item width, or product exceeds the limit.
    """
    if type(shape) is not tuple:
        raise TypeError("shape must be a tuple.")
    result = _require_memory_bytes(itemsize, "itemsize")
    if result == 0:
        raise ValueError("itemsize must be positive.")
    extents = tuple(
        _require_memory_bytes(extent, f"shape[{index}]")
        for index, extent in enumerate(shape)
    )
    if 0 in extents:
        return 0
    for extent in extents:
        if result > MAX_RESIDENT_MEMORY_BYTES // extent:
            raise ValueError("dense array byte count exceeds limit.")
        result *= extent
    return result


def _checked_sum(values: tuple[int, ...] | list[int]) -> int:
    """Return a bounded sum of already validated analytical byte counts."""
    total = 0
    for value in values:
        value = _require_memory_bytes(value, "byte count")
        if total > MAX_RESIDENT_MEMORY_BYTES - value:
            raise ValueError("analytical byte count exceeds limit.")
        total += value
    return total


def _checked_add(left: int, right: int) -> int:
    """Return a bounded sum of two validated analytical byte quantities."""
    return _checked_sum([left, right])


@dataclass(frozen=True, slots=True)
class ResidentMemoryCategory:
    """Name one analytical memory category in bytes and its scenario role."""

    name: str
    byte_count: int
    provenance: str
    included_in_steady_state: bool
    scenario: str

    def __post_init__(self) -> None:
        """Validate category units, provenance, and scenario exclusion rules."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("memory category name must be a nonempty string.")
        _require_memory_bytes(self.byte_count, "memory category byte_count")
        if (
            not isinstance(self.provenance, str)
            or self.provenance not in MEMORY_PROVENANCES
        ):
            raise ValueError("memory category provenance is invalid.")
        if (
            not isinstance(self.scenario, str)
            or self.scenario not in MEMORY_SCENARIOS
        ):
            raise ValueError("memory category scenario is invalid.")
        if not isinstance(self.included_in_steady_state, bool):
            raise TypeError("included_in_steady_state must be a bool.")
        if self.scenario != "steady_state" and self.included_in_steady_state:
            raise ValueError("checkpoint and tape categories are excluded.")


@dataclass(frozen=True, slots=True)
class ResidentMemoryModel:
    """Store validated resident, checkpoint, and tape byte scenarios.

    Totals are logical bytes; allocator reservations and unknown Epic I overhead
    are excluded.
    """

    categories: tuple[ResidentMemoryCategory, ...]
    excluded_epic_i_overhead: str = "unknown Epic I overhead excluded"
    steady_state_bytes: int = field(init=False)
    checkpoint_bytes: int = field(init=False)
    tape_bytes: int = field(init=False)
    inactive_particle_capacity_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        """Validate category reconciliation and store checked scenario totals."""
        if not isinstance(self.categories, tuple):
            raise TypeError("categories must be a tuple.")
        if self.excluded_epic_i_overhead != "unknown Epic I overhead excluded":
            raise ValueError("excluded_epic_i_overhead is invalid.")
        if not all(
            isinstance(item, ResidentMemoryCategory) for item in self.categories
        ):
            raise TypeError(
                "categories must contain ResidentMemoryCategory values."
            )
        names = tuple(item.name for item in self.categories)
        if len(names) != len(set(names)):
            raise ValueError("memory category names must be unique.")
        excluded = [
            item
            for item in self.categories
            if item.scenario == "steady_state"
            and not item.included_in_steady_state
        ]
        if any(
            item.name != "inactive_particle_capacity_attribution"
            and not item.name.startswith("communication.")
            for item in excluded
        ):
            raise ValueError("steady-state exclusion name is invalid.")
        communication = [
            item for item in excluded if item.name.startswith("communication.")
        ]
        if len(communication) > 1 or any(
            item.name.removeprefix("communication.")
            not in RESIDENT_MEMORY_COMMUNICATIONS
            or item.byte_count
            for item in communication
        ):
            raise ValueError("communication selection must have zero bytes.")
        inactive = [
            item
            for item in self.categories
            if item.name == "inactive_particle_capacity_attribution"
        ]
        if len(inactive) != 1 or inactive[0] not in excluded:
            raise ValueError(
                "inactive capacity attribution is required and excluded."
            )
        if inactive[0].provenance != "analytical":
            raise ValueError(
                "inactive capacity attribution must be analytical."
            )
        totals = {
            "steady_state": _checked_sum(
                [
                    item.byte_count
                    for item in self.categories
                    if item.scenario == "steady_state"
                    and item.included_in_steady_state
                ]
            ),
            "checkpoint": _checked_sum(
                [
                    item.byte_count
                    for item in self.categories
                    if item.scenario == "checkpoint"
                ]
            ),
            "tape": _checked_sum(
                [
                    item.byte_count
                    for item in self.categories
                    if item.scenario == "tape"
                ]
            ),
        }
        for scenario, value in totals.items():
            object.__setattr__(self, f"{scenario}_bytes", value)
        object.__setattr__(
            self, "inactive_particle_capacity_bytes", inactive[0].byte_count
        )


def _memory_category(
    name: str,
    byte_count: int,
    provenance: str = "analytical",
    included: bool = True,
    scenario: str = "steady_state",
) -> ResidentMemoryCategory:
    """Build one internal resident-memory category with explicit defaults."""
    return ResidentMemoryCategory(
        name, byte_count, provenance, included, scenario
    )


def build_resident_memory_model(
    *,
    n_boxes: object,
    n_particles: object,
    n_species: object,
    active_slots_per_box: object,
    registry_logical_byte_count: object,
    diagnostics: object,
    communication: object,
    checkpoint_sidecar_copy_bytes: object,
    checkpoint_inspection_copy_bytes: object,
) -> ResidentMemoryModel:
    """Build host-only resident logical-memory accounting scenarios.

    Primary and diagnostic storage are logical device bytes. Checkpoint values
    are separate host-copy scenarios; allocator reservations and tape storage
    are excluded until an explicit projection is attached.
    """
    boxes = _require_memory_bytes(n_boxes, "n_boxes")
    particles = _require_memory_bytes(n_particles, "n_particles")
    species = _require_memory_bytes(n_species, "n_species")
    active = _require_memory_bytes(active_slots_per_box, "active_slots_per_box")
    if active > particles:
        raise ValueError("active_slots_per_box must not exceed n_particles.")
    registry_bytes = _require_memory_bytes(
        registry_logical_byte_count, "registry_logical_byte_count"
    )
    sidecar_bytes = _require_memory_bytes(
        checkpoint_sidecar_copy_bytes, "checkpoint_sidecar_copy_bytes"
    )
    inspection_bytes = _require_memory_bytes(
        checkpoint_inspection_copy_bytes, "checkpoint_inspection_copy_bytes"
    )
    if not isinstance(diagnostics, tuple):
        raise TypeError("diagnostics must be a tuple.")
    if any(not isinstance(item, str) for item in diagnostics):
        raise TypeError("diagnostics must contain strings.")
    positions = tuple(
        RESIDENT_DIAGNOSTIC_OPERATIONS.index(item)
        if item in RESIDENT_DIAGNOSTIC_OPERATIONS
        else -1
        for item in diagnostics
    )
    if -1 in positions or len(set(diagnostics)) != len(diagnostics):
        raise ValueError("diagnostics are invalid.")
    if positions != tuple(sorted(positions)):
        raise ValueError("diagnostics must use canonical order.")
    if not isinstance(communication, str):
        raise TypeError("communication must be a string.")
    if communication not in RESIDENT_MEMORY_COMMUNICATIONS:
        raise ValueError("communication is invalid.")
    categories = [
        _memory_category(
            "primary.particles.masses",
            checked_dense_array_bytes((boxes, particles, species), 8),
        ),
        _memory_category(
            "primary.particles.concentration",
            checked_dense_array_bytes((boxes, particles), 8),
        ),
        _memory_category(
            "primary.particles.charge",
            checked_dense_array_bytes((boxes, particles), 8),
        ),
        _memory_category(
            "primary.particles.density",
            checked_dense_array_bytes((species,), 8),
        ),
        _memory_category(
            "primary.particles.volume", checked_dense_array_bytes((boxes,), 8)
        ),
        _memory_category(
            "primary.gas.molar_mass", checked_dense_array_bytes((species,), 8)
        ),
        _memory_category(
            "primary.gas.concentration",
            checked_dense_array_bytes((boxes, species), 8),
        ),
        _memory_category(
            "primary.gas.vapor_pressure",
            checked_dense_array_bytes((boxes, species), 8),
        ),
        _memory_category(
            "primary.gas.partitioning",
            checked_dense_array_bytes((boxes, species), 4),
        ),
        _memory_category(
            "primary.environment.temperature",
            checked_dense_array_bytes((boxes,), 8),
        ),
        _memory_category(
            "primary.environment.pressure",
            checked_dense_array_bytes((boxes,), 8),
        ),
        _memory_category(
            "primary.environment.saturation_ratio",
            checked_dense_array_bytes((boxes, species), 8),
        ),
    ]
    primary_bytes = _checked_sum([item.byte_count for item in categories])
    categories.append(
        _memory_category(
            "registry.resource_manifest", registry_bytes, "registry_logical"
        )
    )
    for operation in diagnostics:
        shape = (
            (boxes,)
            if operation == "particle_number_concentration"
            else (boxes, species)
        )
        categories.append(
            _memory_category(
                f"diagnostic.{operation}", checked_dense_array_bytes(shape, 8)
            )
        )
    categories.append(
        _memory_category(f"communication.{communication}", 0, included=False)
    )
    particle_slot_bytes = _checked_add(
        checked_dense_array_bytes((species,), 8), 16
    )
    inactive_bytes = checked_dense_array_bytes(
        (boxes, particles - active), particle_slot_bytes
    )
    categories.append(
        _memory_category(
            "inactive_particle_capacity_attribution",
            inactive_bytes,
            included=False,
        )
    )
    categories.extend(
        (
            _memory_category(
                "checkpoint.primary_copy",
                primary_bytes,
                included=False,
                scenario="checkpoint",
            ),
            _memory_category(
                "checkpoint.sidecar_copy",
                sidecar_bytes,
                included=False,
                scenario="checkpoint",
            ),
            _memory_category(
                "checkpoint.inspection_copy",
                inspection_bytes,
                included=False,
                scenario="checkpoint",
            ),
        )
    )
    return ResidentMemoryModel(tuple(categories))


def project_full_retention_tape_bytes(
    timesteps: object, state_bytes: object
) -> int:
    """Return checked full-retention tape bytes for timesteps and state bytes."""
    steps = _require_memory_bytes(timesteps, "timesteps")
    state = _require_memory_bytes(state_bytes, "state_bytes")
    return 0 if state == 0 else checked_dense_array_bytes((steps,), state)


def project_checkpointed_tape_bytes(
    timesteps: object,
    state_bytes: object,
    checkpoint_bytes: object,
    interval: object,
) -> int:
    """Return checked checkpointed-tape bytes using ceiling checkpoint count."""
    steps = _require_memory_bytes(timesteps, "timesteps")
    state = _require_memory_bytes(state_bytes, "state_bytes")
    checkpoint = _require_memory_bytes(checkpoint_bytes, "checkpoint_bytes")
    every = _require_memory_bytes(interval, "interval")
    if every == 0:
        raise ValueError("interval must be positive.")
    checkpoint_count = steps // every + int(steps % every != 0)
    return _checked_sum(
        [
            0
            if checkpoint == 0
            else checked_dense_array_bytes((checkpoint_count,), checkpoint),
            0 if state == 0 else checked_dense_array_bytes((every,), state),
        ]
    )


def with_tape_projection(
    model: object, tape_bytes: object
) -> ResidentMemoryModel:
    """Return a new model with one projected tape scenario in logical bytes."""
    if not isinstance(model, ResidentMemoryModel):
        raise TypeError("model must be a ResidentMemoryModel.")
    if any(item.scenario == "tape" for item in model.categories):
        raise ValueError("model already has a tape projection.")
    value = _require_memory_bytes(tape_bytes, "tape_bytes")
    return ResidentMemoryModel(
        model.categories
        + (
            _memory_category(
                "tape.projected", value, "projected", False, "tape"
            ),
        ),
        model.excluded_epic_i_overhead,
    )


def _require_float(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    """Validate and return a finite numeric field.

    Args:
        value: Candidate numeric value.
        name: Field name used in error messages.
        minimum: Optional inclusive lower bound.

    Returns:
        The value converted to ``float``.

    Raises:
        TypeError: If ``value`` is not a non-boolean number.
        ValueError: If the value is nonfinite or below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return converted


def _validate_shape(value: object, name: str) -> tuple[int, int, int]:
    """Validate a positive three-dimensional benchmark shape.

    Args:
        value: Candidate shape tuple in boxes, particles, and species order.
        name: Field name used in error messages.

    Returns:
        The validated shape as a three-item integer tuple.

    Raises:
        TypeError: If ``value`` is not a three-item tuple of integers.
        ValueError: If any dimension is not positive.
    """
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a three-item tuple.")
    return tuple(
        _require_int(item, f"{name}[{index}]", positive=True)
        for index, item in enumerate(value)
    )  # type: ignore[return-value]


def _freeze_mapping(
    value: object, name: str, *, nonempty: bool = False
) -> Mapping[str, Any]:
    """Normalize and recursively freeze a string-keyed mapping.

    Args:
        value: Candidate mapping to validate.
        name: Field name used in error messages.
        nonempty: Require at least one mapping entry when true.

    Returns:
        An immutable, normalized mapping suitable for a frozen record.

    Raises:
        TypeError: If ``value`` is not a mapping or contains unsupported data.
        ValueError: If the mapping is required to be nonempty but is empty.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    if nonempty and not value:
        raise ValueError(f"{name} must not be empty.")
    normalized = _normalize_json(value)
    if not isinstance(
        normalized, dict
    ):  # Defensive: mappings normalize to dicts.
        raise TypeError(f"{name} must be a mapping.")
    return _freeze_json_value(normalized)


def _freeze_json_value(value: Any) -> Any:
    """Recursively make a normalized JSON-compatible value immutable.

    Dictionaries become read-only mapping proxies and lists become tuples.
    Scalar JSON values are returned unchanged.

    Args:
        value: Normalized JSON-compatible value to freeze.

    Returns:
        An immutable value with the same JSON-compatible contents.
    """
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _validate_metadata(value: object) -> Mapping[str, Any]:
    """Validate the complete stable metadata schema for an artifact.

    Args:
        value: Candidate metadata mapping.

    Returns:
        An immutable metadata mapping with validated provenance fields.

    Raises:
        TypeError: If a field has an unsupported type.
        ValueError: If fields, timestamp, or provenance values are invalid.
    """
    metadata = _freeze_mapping(value, "metadata", nonempty=True)
    if set(metadata) != REQUIRED_METADATA_FIELDS:
        raise ValueError("metadata has invalid fields.")
    for name in (
        "timestamp_utc",
        "command",
        "python_version",
        "platform",
        "synchronization_method",
        "prepared_signature_digest",
    ):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ValueError(f"metadata {name} must be a nonempty string.")
    try:
        timestamp = datetime.fromisoformat(
            metadata["timestamp_utc"].replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError("metadata timestamp_utc must be valid UTC.") from error
    if (
        timestamp.tzinfo is None
        or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
        or not metadata["timestamp_utc"].endswith("Z")
    ):
        raise ValueError("metadata timestamp_utc must be valid UTC.")
    _require_int(metadata["warmup"], "metadata warmup")
    _require_int(
        metadata["timestep_count"], "metadata timestep_count", positive=True
    )
    _require_int(metadata["seed"], "metadata seed")
    _validate_warp_version(metadata["warp_version"])
    _validate_device_provenance(metadata["device"])
    return metadata


def _validate_status(value: object, name: str) -> str:
    """Validate and return one supported provenance status.

    Args:
        value: Candidate status value.
        name: Provenance field name used in error messages.

    Returns:
        The validated status string.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If the status is not supported.
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} status must be a string.")
    if value not in {"available", "unavailable", "error"}:
        raise ValueError(f"{name} status is invalid.")
    return value


def _validate_warp_version(value: object) -> None:
    """Validate exact status-qualified Warp-version provenance.

    Args:
        value: Candidate Warp-version provenance mapping.

    Raises:
        TypeError: If the mapping or one of its values has an invalid type.
        ValueError: If the mapping has an unsupported status or field set.
    """
    if not isinstance(value, Mapping):
        raise TypeError("metadata warp_version must be a mapping.")
    status = _validate_status(value.get("status"), "metadata warp_version")
    expected = (
        _WARP_VERSION_ERROR_FIELDS
        if status == "error"
        else _WARP_VERSION_FIELDS
    )
    if set(value) != expected:
        raise ValueError("metadata warp_version has invalid fields.")
    if status == "available":
        if not isinstance(value["value"], str) or not value["value"]:
            raise TypeError("available warp_version value must be a string.")
    elif value["value"] is not None:
        raise TypeError("unavailable Warp version value must be None.")
    if status == "error" and (
        not isinstance(value["error"], str) or not value["error"]
    ):
        raise TypeError(
            "error Warp version provenance requires an error string."
        )


def _validate_device_provenance(value: object) -> None:
    """Validate exact status-qualified device provenance.

    Args:
        value: Candidate device identity and memory mapping.

    Raises:
        TypeError: If the mapping or one of its values has an invalid type.
        ValueError: If the mapping has an unsupported status or field set.
    """
    if not isinstance(value, Mapping):
        raise TypeError("metadata device must be a mapping.")
    status = _validate_status(value.get("status"), "metadata device")
    expected = _DEVICE_ERROR_FIELDS if status == "error" else _DEVICE_FIELDS
    if set(value) != expected:
        raise ValueError("metadata device has invalid fields.")
    if status == "available":
        if not isinstance(value["identity"], str) or not value["identity"]:
            raise TypeError("available device identity must be a string.")
        _require_int(value["memory"], "available device memory")
    elif value["identity"] is not None or value["memory"] is not None:
        raise TypeError("unavailable device values must be None.")
    if status == "error" and (
        not isinstance(value["error"], str) or not value["error"]
    ):
        raise TypeError("error device provenance requires an error string.")


def _canonical_processes(value: object) -> tuple[str, ...]:
    """Validate the supported process names and their canonical ordering.

    Args:
        value: Candidate nonempty process-name tuple.

    Returns:
        The validated process tuple.

    Raises:
        TypeError: If the value is not a tuple of strings.
        ValueError: If names are unsupported, duplicated, or out of order.
    """
    if not isinstance(value, tuple) or not value:
        raise TypeError("processes must be a nonempty tuple.")
    if any(not isinstance(process, str) for process in value):
        raise TypeError("processes must contain strings.")
    if len(set(value)) != len(value) or any(
        process not in PROCESS_ORDER for process in value
    ):
        raise ValueError("processes must be unique supported process names.")
    expected = tuple(process for process in PROCESS_ORDER if process in value)
    if value != expected:
        raise ValueError("processes must use canonical process ordering.")
    return value


def build_resident_benchmark_case_id(
    *,
    requested_shape: tuple[int, int, int],
    actual_shape: tuple[int, int, int],
    active_fraction: float,
    processes: tuple[str, ...],
    communication: str,
    diagnostics: tuple[str, ...],
    warmup: int,
    timestep_count: int,
    seed: int,
) -> str:
    """Build the canonical identifier for a resident benchmark configuration.

    Args:
        requested_shape: Requested ``(boxes, particles, species)`` capacity.
        actual_shape: Realized ``(boxes, particles, species)`` capacity.
        active_fraction: Fraction of realized particle slots that are active.
        processes: Nonempty process tuple in canonical resident order.
        communication: Supported resident communication selection.
        diagnostics: Unique diagnostic selection names.
        warmup: Number of warmup timesteps.
        timestep_count: Positive number of measured timesteps.
        seed: Nonnegative deterministic benchmark seed.

    Returns:
        The exact configuration-derived case identifier.

    Raises:
        TypeError: If an input has an unsupported type or container shape.
        ValueError: If a value is out of bounds or not a canonical selection.
    """
    requested_shape = _validate_shape(requested_shape, "requested_shape")
    actual_shape = _validate_shape(actual_shape, "actual_shape")
    if any(
        actual_item > requested_item
        for actual_item, requested_item in zip(
            actual_shape, requested_shape, strict=True
        )
    ):
        raise ValueError("actual_shape must not exceed requested_shape.")
    active_fraction = _require_float(active_fraction, "active_fraction")
    if not 0.0 <= active_fraction <= 1.0:
        raise ValueError("active_fraction must be in [0, 1].")
    processes = _canonical_processes(processes)
    if not isinstance(communication, str):
        raise TypeError("communication must be a string.")
    if communication not in SUPPORTED_COMMUNICATION:
        raise ValueError("communication must be a supported selection.")
    if not isinstance(diagnostics, tuple):
        raise TypeError("diagnostics must be a tuple.")
    if any(not isinstance(item, str) for item in diagnostics):
        raise TypeError("diagnostics must contain strings.")
    if len(set(diagnostics)) != len(diagnostics):
        raise ValueError("diagnostics must not contain duplicates.")
    _require_int(warmup, "warmup")
    _require_int(timestep_count, "timestep_count", positive=True)
    _require_int(seed, "seed")
    return (
        f"r{requested_shape[0]}x{requested_shape[1]}x{requested_shape[2]}"
        f"-a{actual_shape[0]}x{actual_shape[1]}x{actual_shape[2]}"
        f"-f{active_fraction:.17g}-p{'-'.join(processes)}"
        f"-c{communication}-d{''.join(f'{len(item)}:{item}' for item in diagnostics) or 'none'}"
        f"-w{warmup}-t{timestep_count}-s{seed}"
    )


@dataclass(frozen=True, slots=True)
class ResidentBenchmarkCase:
    """Store one immutable, canonical resident benchmark configuration.

    This concrete test-support record validates all host configuration before
    benchmark execution. Its ``case_id`` must exactly reproduce the canonical
    identifier derived from the remaining fields.

    Attributes:
        case_id: Canonical configuration-derived identifier.
        requested_shape: Requested ``(boxes, particles, species)`` capacity.
        actual_shape: Realized capacity, bounded by ``requested_shape``.
        active_fraction: Fraction of realized particle slots that are active.
        processes: Nonempty canonical resident process selection.
        communication: Resident communication selection.
        diagnostics: Unique diagnostic selection names.
        warmup: Nonnegative count of warmup timesteps.
        timestep_count: Positive count of measured timesteps.
        seed: Nonnegative deterministic benchmark seed.
    """

    case_id: str
    requested_shape: tuple[int, int, int]
    actual_shape: tuple[int, int, int]
    active_fraction: float
    processes: tuple[str, ...]
    communication: str
    diagnostics: tuple[str, ...]
    warmup: int
    timestep_count: int
    seed: int

    def __post_init__(self) -> None:
        """Validate the canonical configuration after frozen construction.

        Raises:
            TypeError: If a field has an unsupported type or container shape.
            ValueError: If capacities, selections, or the identifier are invalid.
        """
        requested = _validate_shape(self.requested_shape, "requested_shape")
        actual = _validate_shape(self.actual_shape, "actual_shape")
        if any(
            actual_item > requested_item
            for actual_item, requested_item in zip(
                actual, requested, strict=True
            )
        ):
            raise ValueError("actual_shape must not exceed requested_shape.")
        fraction = _require_float(self.active_fraction, "active_fraction")
        if fraction > 1.0:
            raise ValueError("active_fraction must be in [0, 1].")
        processes = _canonical_processes(self.processes)
        canonical_id = build_resident_benchmark_case_id(
            requested_shape=requested,
            actual_shape=actual,
            active_fraction=fraction,
            processes=processes,
            communication=self.communication,
            diagnostics=self.diagnostics,
            warmup=self.warmup,
            timestep_count=self.timestep_count,
            seed=self.seed,
        )
        if not isinstance(self.case_id, str):
            raise TypeError("case_id must be a string.")
        if self.case_id != canonical_id:
            raise ValueError(
                "case_id must exactly match its canonical configuration."
            )


def build_default_resident_benchmark_matrix() -> tuple[
    ResidentBenchmarkCase, ...
]:
    """Build the fixed box-first P3 matrix without sizing or device work.

    The four rows use 1, 10, 100, and 1,000 boxes, each with 16 particles and
    2 species per box. They share explicit active-slot, process,
    communication, and diagnostics axes. Each row preserves its requested
    capacity as its actual capacity; P3 classifies budget and P2 availability
    only. P4--P5 own byte formulas and allocator analysis.

    Returns:
        Immutable canonical cases in ascending requested box-count order.
    """
    common: dict[str, Any] = {
        "active_fraction": 1.0,
        "processes": (
            "communication",
            "condensation",
            "coagulation",
            "dilution",
            "wall_loss",
            "nucleation",
            "diagnostics",
        ),
        "communication": "gas",
        "diagnostics": ("gas", "saturation"),
        "warmup": 2,
        "timestep_count": 3,
        "seed": 1582,
    }
    cases = []
    for n_boxes in RESIDENT_BOX_COUNTS:
        shape = (n_boxes, 16, 2)
        cases.append(
            ResidentBenchmarkCase(
                case_id=build_resident_benchmark_case_id(
                    requested_shape=shape, actual_shape=shape, **common
                ),
                requested_shape=shape,
                actual_shape=shape,
                **common,
            )
        )
    return tuple(cases)


def preflight_resident_benchmark_case(
    case: ResidentBenchmarkCase,
    *,
    budget_bytes: object,
    estimate_requested_bytes: Any,
    availability: Any,
) -> ResidentBenchmarkPreflight:
    """Classify one exact request before probing CUDA or allocating a fixture.

    P3 reuses the injected P1/P2 requested-case estimate and availability
    seams rather than deriving byte formulas or inspecting allocator state.
    Equality with the configured budget is eligible. No row is downscaled or
    redirected to CPU/Warp-CPU; availability is queried only after all host
    validation and budget classification complete.

    Args:
        case: Exact canonical matrix case to classify.
        budget_bytes: Positive configured allocation budget in bytes.
        estimate_requested_bytes: P1/P2 estimator for the requested case.
        availability: Zero-argument P2 CUDA/native-capture availability probe.

    Returns:
        An executed, budget-skipped, or unavailable preflight outcome that
        retains the case's requested and actual capacity metadata.

    Raises:
        TypeError: If a carrier, callback, or callback result has an invalid
            type.
        ValueError: If a case shape, budget, or requested-case estimate is
            invalid.
    """
    if not isinstance(case, ResidentBenchmarkCase):
        raise TypeError("case must be a ResidentBenchmarkCase.")
    _validate_shape(case.requested_shape, "requested_shape")
    _validate_shape(case.actual_shape, "actual_shape")
    if case.requested_shape != case.actual_shape:
        raise ValueError("matrix cases must retain their requested shape.")
    budget = _require_int(budget_bytes, "budget_bytes", positive=True)
    if not callable(estimate_requested_bytes):
        raise TypeError("estimate_requested_bytes must be callable.")
    if not callable(availability):
        raise TypeError("availability must be callable.")
    estimate = _require_int(
        estimate_requested_bytes(case), "requested-case estimate", positive=True
    )
    if estimate > budget:
        return ResidentBenchmarkPreflight(
            case,
            ResidentBenchmarkStatus.SKIPPED_BUDGET,
            f"requested estimate {estimate} exceeds budget {budget}",
        )
    result = availability()
    if not isinstance(result, ResidentBenchmarkAvailability):
        raise TypeError(
            "availability must return ResidentBenchmarkAvailability."
        )
    if not result.available:
        return ResidentBenchmarkPreflight(
            case,
            ResidentBenchmarkStatus.UNAVAILABLE,
            result.reason or "unavailable",
        )
    return ResidentBenchmarkPreflight(
        case, ResidentBenchmarkStatus.EXECUTED, "eligible exact requested shape"
    )


@dataclass(frozen=True, slots=True)
class ResidentTimingSummary:
    """Store deterministic summary statistics for nonnegative host timings.

    Attributes:
        count: Positive number of summarized samples.
        minimum: Smallest sample value.
        median: Median sample value.
        mean: Arithmetic mean of the samples.
        p95: Nearest-rank 95th-percentile sample value.
    """

    count: int
    minimum: float
    median: float
    mean: float
    p95: float

    def __post_init__(self) -> None:
        """Validate independently constructed nonnegative summary fields.

        Raises:
            TypeError: If a field has an unsupported numeric type.
            ValueError: If count is not positive or a statistic is invalid.
        """
        _require_int(self.count, "count", positive=True)
        for name in ("minimum", "median", "mean", "p95"):
            _require_float(getattr(self, name), name, minimum=0.0)


def summarize_timing_samples(samples: object) -> ResidentTimingSummary:
    """Summarize bounded, nonnegative timing samples deterministically.

    The input is capped at ``MAX_TIMING_SAMPLES`` before sorting. The p95 uses
    the nearest-rank index ``ceil(0.95 * count) - 1`` in sorted zero-based
    order.

    Args:
        samples: Nonempty tuple of at most ``MAX_TIMING_SAMPLES`` timings.

    Returns:
        Validated minimum, median, mean, and nearest-rank p95 summary.

    Raises:
        TypeError: If ``samples`` is not a tuple of numeric values.
        ValueError: If samples are empty, excessive, nonfinite, or negative.
    """
    if not isinstance(samples, tuple):
        raise TypeError("samples must be a tuple.")
    if not samples:
        raise ValueError("samples must not be empty.")
    if len(samples) > MAX_TIMING_SAMPLES:
        raise ValueError("samples exceeds MAX_TIMING_SAMPLES.")
    validated = tuple(
        _require_float(value, "sample", minimum=0.0) for value in samples
    )
    ordered = sorted(validated)
    return ResidentTimingSummary(
        count=len(ordered),
        minimum=ordered[0],
        median=statistics.median(ordered),
        mean=statistics.fmean(ordered),
        p95=ordered[math.ceil(0.95 * len(ordered)) - 1],
    )


@dataclass(frozen=True, slots=True)
class ResidentBenchmarkResult:
    """Store immutable evidence or a non-execution outcome for one case.

    Executed rows require supported timing data whose summary exactly matches
    the samples. Non-executed rows require a reason and prohibit timing data.

    Attributes:
        case_id: Identifier of the referenced benchmark case.
        timing_mode: Supported timing mode for executed rows, otherwise ``None``.
        requested_shape: Requested shape copied from the referenced case.
        status: Executed, unavailable, or budget-skipped outcome.
        reason: Non-execution reason, or ``None`` for executed rows.
        samples: Raw nonnegative timings for executed rows.
        summary: Deterministic summary for executed rows, otherwise ``None``.
        provenance: Nonempty immutable host provenance reference.
    """

    case_id: str
    timing_mode: str | None
    requested_shape: tuple[int, int, int]
    status: ResidentBenchmarkStatus
    reason: str | None
    samples: tuple[float, ...]
    summary: ResidentTimingSummary | None
    provenance: Mapping[str, Any]
    setup_elapsed_seconds: float | None = None
    capture_elapsed_seconds: float | None = None

    def __post_init__(self) -> None:
        """Validate status-specific evidence and provenance consistency.

        Raises:
            TypeError: If a field or provenance value has an unsupported type.
            ValueError: If timing evidence or a non-execution outcome is invalid.
        """
        if not isinstance(self.case_id, str) or not self.case_id:
            raise TypeError("case_id must be a nonempty string.")
        _validate_shape(self.requested_shape, "requested_shape")
        if not isinstance(self.status, ResidentBenchmarkStatus):
            raise TypeError("status must be a ResidentBenchmarkStatus.")
        if self.timing_mode is not None and not isinstance(
            self.timing_mode, str
        ):
            raise TypeError("timing_mode must be a string or None.")
        if not isinstance(self.samples, tuple):
            raise TypeError("samples must be a tuple.")
        provenance = _freeze_mapping(
            self.provenance, "provenance", nonempty=True
        )
        object.__setattr__(self, "provenance", provenance)
        for name in ("setup_elapsed_seconds", "capture_elapsed_seconds"):
            value = getattr(self, name)
            if value is not None:
                _require_float(value, name, minimum=0.0)
        if self.status is ResidentBenchmarkStatus.EXECUTED:
            if self.timing_mode not in SUPPORTED_TIMING_MODES:
                raise ValueError(
                    "executed results require a supported timing_mode."
                )
            expected = summarize_timing_samples(self.samples)
            if self.summary != expected:
                raise ValueError("summary must exactly match timing samples.")
            if self.reason is not None:
                if not isinstance(self.reason, str):
                    raise TypeError("reason must be a string or None.")
                raise ValueError("executed results must not have a reason.")
            if self.timing_mode not in _CAPTURE_COMPARISON_TIMING_MODES and (
                self.setup_elapsed_seconds is not None
                or self.capture_elapsed_seconds is not None
            ):
                raise ValueError(
                    "only capture comparison modes may contain timing provenance."
                )
            if self.timing_mode in _CAPTURE_COMPARISON_TIMING_MODES and (
                self.setup_elapsed_seconds is None
                or self.capture_elapsed_seconds is None
            ):
                raise ValueError(
                    "capture comparison modes require timing provenance."
                )
        else:
            if (
                self.timing_mode is not None
                or self.samples
                or self.summary is not None
            ):
                raise ValueError(
                    "non-executed results cannot contain timing data."
                )
            if self.reason is not None and not isinstance(self.reason, str):
                raise TypeError("reason must be a string or None.")
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError(
                    "non-executed results require a nonempty reason."
                )
            if (
                self.setup_elapsed_seconds is not None
                or self.capture_elapsed_seconds is not None
            ):
                raise ValueError(
                    "non-executed results cannot contain timing provenance."
                )


@dataclass(frozen=True, slots=True)
class ResidentBenchmarkArtifact:
    """Store one complete, immutable resident benchmark evidence artifact.

    Attributes:
        metadata: Complete stable host provenance metadata.
        cases: Canonical benchmark configurations with unique identifiers.
        results: Case-referencing results with unique case/mode identities.
    """

    metadata: Mapping[str, Any]
    cases: tuple[ResidentBenchmarkCase, ...]
    results: tuple[ResidentBenchmarkResult, ...]

    def __post_init__(self) -> None:
        """Validate complete metadata and cross-record artifact references.

        Raises:
            TypeError: If records are not tuples of the supported record types.
            ValueError: If metadata, identifiers, or references are inconsistent.
        """
        metadata = _validate_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        if not isinstance(self.cases, tuple) or not all(
            isinstance(case, ResidentBenchmarkCase) for case in self.cases
        ):
            raise TypeError(
                "cases must be a tuple of ResidentBenchmarkCase records."
            )
        if not isinstance(self.results, tuple) or not all(
            isinstance(result, ResidentBenchmarkResult)
            for result in self.results
        ):
            raise TypeError(
                "results must be a tuple of ResidentBenchmarkResult records."
            )
        case_map = {case.case_id: case for case in self.cases}
        if len(case_map) != len(self.cases):
            raise ValueError("cases must have unique case_id values.")
        identities: set[tuple[str, str | None]] = set()
        for result in self.results:
            case = case_map.get(result.case_id)
            if case is None:
                raise ValueError("result references an unknown case_id.")
            if result.requested_shape != case.requested_shape:
                raise ValueError("result requested_shape must match its case.")
            identity = (result.case_id, result.timing_mode)
            if identity in identities:
                raise ValueError(
                    "results must have unique case_id/timing_mode rows."
                )
            identities.add(identity)


def build_resident_benchmark_metadata(
    *,
    timestamp_utc: datetime,
    command: str,
    synchronization_method: str,
    warmup: int,
    timestep_count: int,
    seed: int,
    prepared_signature_digest: str,
    warp_version: Mapping[str, Any],
    device: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build complete host-only provenance without probing Warp or a device.

    The caller supplies Warp and device availability/value mappings explicitly;
    this helper only reads Python and platform metadata.

    Args:
        timestamp_utc: UTC-aware timestamp for the benchmark invocation.
        command: Reproduction command.
        synchronization_method: Timing synchronization method.
        warmup: Nonnegative number of warmup timesteps.
        timestep_count: Positive number of measured timesteps.
        seed: Nonnegative deterministic benchmark seed.
        prepared_signature_digest: Nonempty prepared-signature digest.
        warp_version: Explicit Warp version or availability mapping.
        device: Explicit device identity or availability mapping.

    Returns:
        Immutable, complete metadata matching the artifact schema.

    Raises:
        TypeError: If an input has an unsupported type.
        ValueError: If UTC, scalar, or mapping validation fails.
    """
    if not isinstance(timestamp_utc, datetime):
        raise TypeError("timestamp_utc must be a datetime.")
    if (
        timestamp_utc.tzinfo is None
        or timestamp_utc.utcoffset() != timezone.utc.utcoffset(timestamp_utc)
    ):
        raise ValueError("timestamp_utc must be UTC-aware.")
    for name, value in (
        ("command", command),
        ("synchronization_method", synchronization_method),
        ("prepared_signature_digest", prepared_signature_digest),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string.")
    metadata = {
        "timestamp_utc": timestamp_utc.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "command": command,
        "python_version": sys.version,
        "platform": platform.platform(),
        "warp_version": _freeze_mapping(warp_version, "warp_version"),
        "device": _freeze_mapping(device, "device"),
        "synchronization_method": synchronization_method,
        "warmup": _require_int(warmup, "warmup"),
        "timestep_count": _require_int(
            timestep_count, "timestep_count", positive=True
        ),
        "seed": _require_int(seed, "seed"),
        "prepared_signature_digest": prepared_signature_digest,
    }
    return _validate_metadata(metadata)


def _normalize_json(value: object) -> Any:
    """Convert supported values into deterministic JSON-compatible values.

    Args:
        value: Value to normalize, including supported frozen records.

    Returns:
        A recursively normalized value containing JSON-compatible types.

    Raises:
        TypeError: If a value or mapping key is unsupported.
        ValueError: If a floating-point value is nonfinite.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return _normalize_json(value.value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite floats cannot be serialized.")
        return value
    if isinstance(value, (tuple, list)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("mapping keys must be strings.")
        return {key: _normalize_json(item) for key, item in value.items()}
    if isinstance(value, ResidentTimingSummary):
        return {
            "count": value.count,
            "minimum": value.minimum,
            "median": value.median,
            "mean": value.mean,
            "p95": value.p95,
        }
    if isinstance(value, ResidentBenchmarkCase):
        return {
            name: _normalize_json(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, ResidentBenchmarkResult):
        return {
            name: _normalize_json(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, ResidentBenchmarkArtifact):
        return {
            name: _normalize_json(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}.")


def serialize_resident_benchmark_artifact(
    artifact: ResidentBenchmarkArtifact,
) -> str:
    """Serialize an artifact as deterministic schema-versioned JSON.

    The serialized UTF-8-compatible JSON uses sorted keys, two-space
    indentation, and exactly one trailing newline.

    Args:
        artifact: Fully validated resident benchmark evidence artifact.

    Returns:
        Schema-envelope JSON text with deterministic formatting.

    Raises:
        TypeError: If ``artifact`` is not a resident benchmark artifact.
        ValueError: If a nested value is not JSON-normalizable.
    """
    if not isinstance(artifact, ResidentBenchmarkArtifact):
        raise TypeError("artifact must be a ResidentBenchmarkArtifact.")
    return (
        json.dumps(
            {
                "schema_version": RESIDENT_BENCHMARK_SCHEMA_VERSION,
                "artifact": _normalize_json(artifact),
            },
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _require_fields(
    value: object, fields: set[str], name: str
) -> Mapping[str, Any]:
    """Require an object to be a dictionary with exactly the given fields.

    Args:
        value: Decoded object to validate.
        fields: Exact set of permitted keys.
        name: Object name used in error messages.

    Returns:
        The validated dictionary.

    Raises:
        ValueError: If the object is not a dictionary with the exact fields.
    """
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} has invalid fields.")
    return value


def collect_paired_device_timings(
    *,
    uncaptured_operation: Any,
    replay_operation: Any,
    synchronize: Any,
    clock: Any,
    warmup_count: int,
    sample_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Collect alternating paired completed-device timings.

    Warmups alternate uncaptured then replay without synchronization. Each
    measured operation follows ``clock, operation, synchronize, clock``;
    synchronization therefore measures completed device work rather than only
    host enqueue time. Setup and capture timing are intentionally outside this
    collector.

    Args:
        uncaptured_operation: Zero-argument prepared-operation callback.
        replay_operation: Zero-argument captured-replay callback.
        synchronize: Callback that waits for the device operation to complete.
        clock: Monotonic clock returning elapsed seconds.
        warmup_count: Number of unsynchronized paired warmup iterations.
        sample_count: Number of synchronized samples to collect per operation.

    Returns:
        Two immutable timing tuples in uncaptured-then-replay order.

    Raises:
        TypeError: If counts or callbacks have unsupported types.
        ValueError: If counts are outside their configured bounds or a measured
            elapsed time is negative or nonfinite.
    """
    warmup_count = _require_int(warmup_count, "warmup_count")
    sample_count = _require_int(sample_count, "sample_count", positive=True)
    if warmup_count > MAX_WARMUP_SAMPLES:
        raise ValueError("warmup_count exceeds MAX_WARMUP_SAMPLES.")
    if sample_count > MAX_TIMING_SAMPLES:
        raise ValueError("sample_count exceeds MAX_TIMING_SAMPLES.")
    callbacks = (uncaptured_operation, replay_operation, synchronize, clock)
    if not all(callable(callback) for callback in callbacks):
        raise TypeError("operations, synchronize, and clock must be callable.")
    for _ in range(warmup_count):
        uncaptured_operation()
        replay_operation()
    uncaptured_samples: list[float] = []
    replay_samples: list[float] = []
    for _ in range(sample_count):
        start = clock()
        uncaptured_operation()
        synchronize()
        uncaptured_samples.append(
            _require_float(clock() - start, "uncaptured elapsed", minimum=0.0)
        )
        start = clock()
        replay_operation()
        synchronize()
        replay_samples.append(
            _require_float(clock() - start, "replay elapsed", minimum=0.0)
        )
    return tuple(uncaptured_samples), tuple(replay_samples)


def write_resident_capture_comparison_artifact(
    artifact_root: str | os.PathLike[str], artifact: ResidentBenchmarkArtifact
) -> Path:
    """Atomically persist the isolated resident-comparison schema envelope.

    Args:
        artifact_root: Existing ``.artifacts`` directory used as the output
            root.
        artifact: Fully validated resident benchmark artifact to serialize.

    Returns:
        The fixed resident-comparison artifact path.

    Raises:
        OSError: If validation, serialization, or atomic persistence fails.
    """
    serialized = serialize_resident_benchmark_artifact(artifact)
    return write_json_artifact(
        artifact_root,
        RESIDENT_CAPTURE_COMPARISON_DESTINATION,
        json.loads(serialized),
    )


def _validate_payload_structure(value: object) -> None:
    """Reject decoded JSON structures exceeding bounded artifact limits.

    Args:
        value: Decoded JSON value to inspect.

    Raises:
        ValueError: If nesting depth or container size exceeds its limit.
    """
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_ARTIFACT_NESTING_DEPTH:
            raise ValueError("payload exceeds maximum nesting depth.")
        if isinstance(current, dict):
            if len(current) > MAX_ARTIFACT_CONTAINER_ITEMS:
                raise ValueError("payload mapping exceeds maximum item count.")
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            if len(current) > MAX_ARTIFACT_CONTAINER_ITEMS:
                raise ValueError("payload list exceeds maximum item count.")
            pending.extend((item, depth + 1) for item in current)


def _validate_json_nesting(payload: str) -> None:
    """Bound JSON nesting before the standard-library decoder recurses.

    Args:
        payload: JSON text whose bracket nesting should be bounded.

    Raises:
        ValueError: If the text exceeds the maximum nesting depth.
    """
    depth = 0
    quoted = False
    escaped = False
    for character in payload:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
        elif character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > MAX_ARTIFACT_NESTING_DEPTH:
                raise ValueError("payload exceeds maximum nesting depth.")
        elif character in "]}":
            depth -= 1


def deserialize_resident_benchmark_artifact(
    payload: str,
) -> ResidentBenchmarkArtifact:
    """Deserialize a schema envelope through normal record constructors.

    Args:
        payload: JSON text containing the exact supported schema envelope.

    Returns:
        A fully validated immutable resident benchmark artifact.

    Raises:
        TypeError: If ``payload`` is not text.
        ValueError: If JSON, the envelope, or any reconstructed record is invalid.
    """
    if not isinstance(payload, str):
        raise TypeError("payload must be a string.")
    if len(payload.encode("utf-8")) > MAX_ARTIFACT_PAYLOAD_BYTES:
        raise ValueError("payload exceeds maximum byte size.")
    _validate_json_nesting(payload)
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("payload is not valid JSON.") from error
    _validate_payload_structure(envelope)
    envelope = _require_fields(
        envelope, {"schema_version", "artifact"}, "envelope"
    )
    schema_version = envelope["schema_version"]
    if schema_version not in {1, RESIDENT_BENCHMARK_SCHEMA_VERSION}:
        raise ValueError("unsupported schema_version.")
    raw = _require_fields(
        envelope["artifact"], {"metadata", "cases", "results"}, "artifact"
    )
    for name in ("cases", "results"):
        if not isinstance(raw[name], list):
            raise ValueError(f"artifact {name} must be a list.")
        if len(raw[name]) > MAX_ARTIFACT_ROWS:
            raise ValueError(f"artifact {name} exceeds maximum row count.")
    cases = []
    for item in raw["cases"]:
        item = _require_fields(
            item, set(ResidentBenchmarkCase.__dataclass_fields__), "case"
        )
        cases.append(
            ResidentBenchmarkCase(
                case_id=item["case_id"],
                requested_shape=tuple(item["requested_shape"]),
                actual_shape=tuple(item["actual_shape"]),
                active_fraction=item["active_fraction"],
                processes=tuple(item["processes"]),
                communication=item["communication"],
                diagnostics=tuple(item["diagnostics"]),
                warmup=item["warmup"],
                timestep_count=item["timestep_count"],
                seed=item["seed"],
            )
        )
    results = []
    for item in raw["results"]:
        fields = set(ResidentBenchmarkResult.__dataclass_fields__)
        if schema_version == 1:
            fields -= {"setup_elapsed_seconds", "capture_elapsed_seconds"}
        item = _require_fields(item, fields, "result")
        summary = item["summary"]
        if summary is not None:
            summary = ResidentTimingSummary(
                **_require_fields(
                    summary,
                    set(ResidentTimingSummary.__dataclass_fields__),
                    "summary",
                )
            )
        results.append(
            ResidentBenchmarkResult(
                case_id=item["case_id"],
                timing_mode=item["timing_mode"],
                requested_shape=tuple(item["requested_shape"]),
                status=ResidentBenchmarkStatus(item["status"]),
                reason=item["reason"],
                samples=tuple(item["samples"]),
                summary=summary,
                provenance=item["provenance"],
                setup_elapsed_seconds=(
                    item.get("setup_elapsed_seconds")
                    if schema_version != 1
                    else None
                ),
                capture_elapsed_seconds=(
                    item.get("capture_elapsed_seconds")
                    if schema_version != 1
                    else None
                ),
            )
        )
    return ResidentBenchmarkArtifact(
        metadata=raw["metadata"], cases=tuple(cases), results=tuple(results)
    )


def write_json_artifact(
    artifact_root: str | os.PathLike[str],
    relative_destination: str | os.PathLike[str],
    payload: object,
) -> Path:
    """Atomically write normalized generic JSON below an existing artifacts root.

    The root must be a non-symlink directory literally named ``.artifacts``.
    Destinations must be relative, contain no parent traversal, and cannot
    resolve outside that root. The writer validates and serializes before
    creating directories or temporary files, then writes, fsyncs, and replaces
    a same-directory temporary file. It does not promise recovery after a
    successful replacement followed by an operating-system failure.

    Args:
        artifact_root: Existing directory named ``.artifacts``.
        relative_destination: Contained relative JSON destination path.
        payload: Supported JSON-normalizable value without an artifact envelope.

    Returns:
        Final destination path after successful atomic replacement.

    Raises:
        OSError: If serialization, path validation, writing, replacement, or
            temporary cleanup fails.
    """
    try:
        serialized = (
            json.dumps(
                _normalize_json(payload),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise OSError("artifact payload serialization failed.") from error
    root_path = Path(artifact_root)
    root_fd: int | None = None
    parent_fd: int | None = None
    temporary_name: str | None = None
    try:
        if root_path.name != ".artifacts" or not hasattr(os, "O_NOFOLLOW"):
            raise OSError(
                "artifact_root must be an existing .artifacts directory."
            )
        relative = Path(relative_destination)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative == Path(".")
        ):
            raise OSError(
                "relative_destination must be contained and relative."
            )
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            root_fd = os.open(root_path, flags)
        except OSError as error:
            raise OSError(
                f"artifact path validation failed: {error}"
            ) from error
        parent_fd = os.dup(root_fd)
        for part in relative.parts[:-1]:
            try:
                child_fd = os.open(part, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                os.mkdir(part, dir_fd=parent_fd)
                try:
                    child_fd = os.open(part, flags, dir_fd=parent_fd)
                except OSError as error:
                    raise OSError(
                        "artifact path escapes artifact_root."
                    ) from error
            except OSError as error:
                raise OSError("artifact path escapes artifact_root.") from error
            os.close(parent_fd)
            parent_fd = child_fd
        try:
            destination_status = os.stat(
                relative.name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            destination_status = None
        if destination_status is not None and stat.S_ISLNK(
            destination_status.st_mode
        ):
            raise OSError("artifact path escapes artifact_root.")
        for _ in range(10):
            temporary_name = f".resident-benchmark-{secrets.token_hex(16)}"
            try:
                temporary_fd = os.open(
                    temporary_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent_fd,
                )
                break
            except FileExistsError:
                temporary_name = None
        else:
            raise OSError("could not allocate artifact temporary file.")
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(
            temporary_name,
            relative.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temporary_name = None
        return root_path / relative
    except OSError as error:
        if (
            str(error).startswith("artifact path validation failed")
            or str(error).startswith("artifact path escapes")
            or str(error).startswith("artifact_root")
            or str(error).startswith("relative_destination")
        ):
            if not str(error).startswith("artifact path"):
                raise OSError(
                    f"artifact path validation failed: {error}"
                ) from error
            raise
        cleanup_error: OSError | None = None
        if temporary_name is not None and parent_fd is not None:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError as caught:
                cleanup_error = caught
        if cleanup_error is not None:
            raise OSError(
                "artifact write failed; temporary cleanup also failed."
            ) from cleanup_error
        raise OSError("artifact write failed.") from error
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
        if root_fd is not None:
            os.close(root_fd)
