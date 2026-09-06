"""Validate bounded, host-only profiling evidence for GPU test support.

This concrete test-support module defines strict evidence records, canonical
JSON, and contained raw-report provenance. It does not import Warp, probe
hardware, start a profiler, collect measurements, or form part of the public
``particula.gpu`` API. Its fixed workload matrix records requested evidence;
it does not claim that either workload ran or was feasible.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

PROFILING_SCHEMA_VERSION = 1
MAX_STRING_LENGTH = 256
MAX_ARTIFACT_PAYLOAD_BYTES = 1_000_000
MAX_ARTIFACT_NESTING_DEPTH = 16
MAX_ARTIFACT_CONTAINER_ITEMS = 256
MAX_ARTIFACT_ROWS = 64
MAX_RAW_REPORT_BYTES = 16_000_000
RAW_REPORT_HASH_CHUNK_BYTES = 65_536
REPLAY_COUNTS = (1, 10, 100, 1000)
EVIDENCE_STATUSES = frozenset(("executed", "unavailable"))
METHOD_SOURCES = frozenset(("host_launch", "synchronized_elapsed", "profiler"))
METRIC_UNITS = {
    "host_launch_duration": "ns",
    "synchronized_elapsed_duration": "ns",
    "profiler_gpu_duration": "ns",
    "profiler_gpu_memory": "bytes",
}
METHOD_METRICS = {
    "host_launch": ("host_launch_duration",),
    "synchronized_elapsed": ("synchronized_elapsed_duration",),
    "profiler": ("profiler_gpu_duration", "profiler_gpu_memory"),
}
FROZEN_WORKLOAD_FIELDS = {
    "small": ((1, 16, 2), 1.0),
    "medium": ((1000, 16, 2), 1.0),
}
_SAFE_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/+\-]*$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _require_text(value: object, name: str) -> str:
    """Return bounded, nonempty, printable evidence text."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if (
        not value
        or len(value) > MAX_STRING_LENGTH
        or not _SAFE_TEXT.fullmatch(value)
    ):
        raise ValueError(f"{name} must be bounded safe text.")
    return value


def _require_machine_text(value: object, name: str) -> str:
    """Validate machine metadata without permitting path-like values."""
    text = _require_text(value, name)
    if Path(text).is_absolute() or "/" in text or "\\" in text:
        raise ValueError(f"{name} must not contain a filesystem path.")
    return text


def _require_int(value: object, name: str, *, positive: bool = False) -> int:
    """Return an integer while rejecting Boolean values."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} has an invalid value.")
    return value


def _require_number(
    value: object, name: str, *, positive: bool = False
) -> float:
    """Return a finite scalar while rejecting Boolean and nonnumeric values."""
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a numeric scalar.")
    result = float(cast(int | float, value))
    if (
        not math.isfinite(result)
        or result < 0.0
        or (positive and result == 0.0)
    ):
        raise ValueError(f"{name} has an invalid value.")
    return result


def _require_exact_keys(
    value: object, keys: set[str], name: str
) -> dict[str, object]:
    """Require an exact mapping schema without accepting extra fields."""
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} has unexpected fields.")
    return value


def _unique_texts(values: object, name: str) -> tuple[str, ...]:
    """Validate an ordered, unique, bounded tuple of strings."""
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a nonempty tuple.")
    result = tuple(_require_text(value, name) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must be unique.")
    return result


@dataclass(frozen=True, slots=True)
class ProfilingWorkload:
    """Describe one fixed resident profiling workload.

    Attributes:
        workload_id: Canonical identifier derived from all remaining fields.
        label: Closed workload label, either ``"small"`` or ``"medium"``.
        shape: Positive ``(boxes, particles, species)`` dimensions.
        active_fraction: Fraction of fixed slots requested as active.
        processes: Ordered, unique resident process names.
        communication: Requested resident communication family.
        diagnostics: Ordered, unique requested diagnostic names.
        warmup: Requested warmup timestep count.
        sample_count: Requested measured sample count.
        seed: Fixed workload seed.
        duration_seconds: Requested duration in seconds.
        replay_counts: Fixed ordered replay-count matrix.
    """

    workload_id: str
    label: str
    shape: tuple[int, int, int]
    active_fraction: float
    processes: tuple[str, ...]
    communication: str
    diagnostics: tuple[str, ...]
    warmup: int
    sample_count: int
    seed: int
    duration_seconds: float
    replay_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the closed workload schema and canonical identifier.

        Raises:
            TypeError: If a numeric field has an unsupported type.
            ValueError: If a field is out of bounds or the identifier differs
                from its canonical value.
        """
        if self.label not in {"small", "medium"}:
            raise ValueError("label must be small or medium.")
        if (
            not isinstance(self.shape, tuple)
            or len(self.shape) != 3
            or any(type(item) is not int or item <= 0 for item in self.shape)
        ):
            raise ValueError("shape must be a positive three-integer tuple.")
        fraction = _require_number(self.active_fraction, "active_fraction")
        if fraction > 1.0:
            raise ValueError("active_fraction must not exceed one.")
        processes = _unique_texts(self.processes, "processes")
        diagnostics = _unique_texts(self.diagnostics, "diagnostics")
        _require_text(self.communication, "communication")
        _require_int(self.warmup, "warmup")
        _require_int(self.sample_count, "sample_count", positive=True)
        _require_int(self.seed, "seed")
        _require_number(
            self.duration_seconds, "duration_seconds", positive=True
        )
        if self.replay_counts != REPLAY_COUNTS:
            raise ValueError("replay_counts must be the fixed replay matrix.")
        canonical = build_profiling_workload_id(
            label=self.label,
            shape=self.shape,
            active_fraction=fraction,
            processes=processes,
            communication=self.communication,
            diagnostics=diagnostics,
            warmup=self.warmup,
            sample_count=self.sample_count,
            seed=self.seed,
            duration_seconds=self.duration_seconds,
            replay_counts=self.replay_counts,
        )
        if self.workload_id != canonical:
            raise ValueError("workload_id must match the canonical workload.")
        expected_shape, expected_fraction = FROZEN_WORKLOAD_FIELDS[self.label]
        if (
            self.shape != expected_shape
            or fraction != expected_fraction
            or processes
            != (
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
            or self.communication != "gas"
            or diagnostics != ("gas", "saturation")
            or self.warmup != 2
            or self.sample_count != 3
            or self.seed != 1582
            or self.duration_seconds != 0.5
        ):
            raise ValueError(
                "workload must be one of the frozen configurations."
            )


def build_profiling_workload_id(
    *,
    label: str,
    shape: tuple[int, int, int],
    active_fraction: float,
    processes: tuple[str, ...],
    communication: str,
    diagnostics: tuple[str, ...],
    warmup: int,
    sample_count: int,
    seed: int,
    duration_seconds: float,
    replay_counts: tuple[int, ...],
) -> str:
    """Build a stable identifier from every nonidentifier workload field.

    This helper hashes the supplied fields as canonical compact JSON. Callers
    must validate the fields separately; this function does not establish a
    workload or execute it.

    Args:
        label: Closed workload label.
        shape: Requested ``(boxes, particles, species)`` dimensions.
        active_fraction: Requested fraction of active slots.
        processes: Ordered process names.
        communication: Requested communication family.
        diagnostics: Ordered diagnostic names.
        warmup: Requested warmup timestep count.
        sample_count: Requested measured sample count.
        seed: Requested workload seed.
        duration_seconds: Requested duration in seconds.
        replay_counts: Ordered replay-count matrix.

    Returns:
        Canonical ``profiling-`` prefixed workload identifier.
    """
    value = {
        "active_fraction": active_fraction,
        "communication": communication,
        "diagnostics": diagnostics,
        "duration_seconds": duration_seconds,
        "label": label,
        "processes": processes,
        "replay_counts": replay_counts,
        "sample_count": sample_count,
        "seed": seed,
        "shape": shape,
        "warmup": warmup,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return f"profiling-{hashlib.sha256(encoded.encode()).hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class MachineProvenance:
    """Store bounded machine metadata without environment or payload data.

    Attributes:
        machine_id: Opaque bounded machine identifier.
        platform: Bounded operating-system platform label.
        python_version: Bounded Python version label.
        cuda_version: Bounded CUDA version label.
        driver_version: Bounded device-driver version label.
        device: Bounded device label.
        source_revision: Bounded source revision identifier.
    """

    machine_id: str
    platform: str
    python_version: str
    cuda_version: str
    driver_version: str
    device: str
    source_revision: str

    def __post_init__(self) -> None:
        """Validate bounded metadata that cannot carry paths or payloads.

        Raises:
            TypeError: If a metadata field is not text.
            ValueError: If metadata is empty, unsafe, oversized, or path-like.
        """
        for field in self.__dataclass_fields__:
            _require_machine_text(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class MeasurementMethod:
    """Describe one noncombined source of duration evidence.

    Attributes:
        method_id: Closed identifier matching ``source``.
        source: One host-launch, synchronized-elapsed, or profiler source.
        command: Bounded command provenance text.
        version: Bounded measurement-tool version text.
        duration_unit: Required raw-duration unit, ``"ns"``.
    """

    method_id: str
    source: str
    command: str
    version: str
    duration_unit: str

    def __post_init__(self) -> None:
        """Validate one closed, noncombined measurement source.

        Raises:
            TypeError: If command or version is not text.
            ValueError: If the source, identifier, unit, or text is invalid.
        """
        if self.source not in METHOD_SOURCES or self.method_id != self.source:
            raise ValueError(
                "method_id must identify its one supported source."
            )
        _require_text(self.command, "command")
        _require_text(self.version, "version")
        if self.duration_unit != "ns":
            raise ValueError("duration_unit must be ns.")


@dataclass(frozen=True, slots=True)
class RawDurationSample:
    """Store one absolute raw duration sample in nanoseconds.

    Attributes:
        replay_count: One configured replay count.
        duration_ns: Positive absolute duration in nanoseconds.
    """

    replay_count: int
    duration_ns: int

    def __post_init__(self) -> None:
        """Validate a configured replay count and positive duration.

        Raises:
            TypeError: If duration is not an integer.
            ValueError: If either field is outside its closed bounds.
        """
        if (
            self.replay_count not in REPLAY_COUNTS
            or type(self.replay_count) is not int
        ):
            raise ValueError("replay_count must be a configured integer.")
        _require_int(self.duration_ns, "duration_ns", positive=True)


@dataclass(frozen=True, slots=True)
class NormalizedMetric:
    """Store one closed-vocabulary absolute profiling metric.

    Attributes:
        name: Closed absolute-metric name.
        value: Finite nonnegative measured value, never an inferred substitute.
        unit: Unit required by ``name``.
    """

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        """Validate the closed absolute-metric vocabulary and value.

        Raises:
            TypeError: If the value is not a numeric scalar.
            ValueError: If the name, unit, or value is invalid.
        """
        if (
            self.name not in METRIC_UNITS
            or self.unit != METRIC_UNITS[self.name]
        ):
            raise ValueError("metric name and unit are not supported.")
        _require_number(self.value, "value")


@dataclass(frozen=True, slots=True)
class RawReportProvenance:
    """Store a contained raw-report filename, size, and streaming digest.

    Attributes:
        raw_filename: Plain filename under the injected raw-report root.
        byte_size: Positive bounded raw-report size in bytes.
        sha256: Lowercase SHA-256 digest of the report bytes.
    """

    raw_filename: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        """Validate root-independent, contained report provenance fields.

        Raises:
            TypeError: If filename or size has an unsupported type.
            ValueError: If a field is unsafe, out of bounds, or malformed.
        """
        _validate_raw_filename(self.raw_filename)
        size = _require_int(self.byte_size, "byte_size", positive=True)
        if size > MAX_RAW_REPORT_BYTES:
            raise ValueError("raw report exceeds the byte limit.")
        if not isinstance(self.sha256, str) or not _DIGEST.fullmatch(
            self.sha256
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest.")


@dataclass(frozen=True, slots=True)
class ExecutedEvidence:
    """Store complete executed evidence without inferred aggregate values.

    Attributes:
        status: Literal ``"executed"`` status.
        workload: Exact workload associated with the evidence.
        machine: Bounded machine provenance.
        method: One measurement source.
        raw_samples: Nonempty absolute raw-duration samples.
        metrics: Unique closed-vocabulary normalized metrics.
        raw_reports: Nonempty unique raw-report provenance records.
    """

    status: str
    workload: ProfilingWorkload
    machine: MachineProvenance
    method: MeasurementMethod
    raw_samples: tuple[RawDurationSample, ...]
    metrics: tuple[NormalizedMetric, ...]
    raw_reports: tuple[RawReportProvenance, ...]

    def __post_init__(self) -> None:
        """Validate that an executed row has complete nonfabricated context.

        Raises:
            TypeError: If context or evidence sequences have invalid types.
            ValueError: If status, sequences, or identities are invalid.
        """
        if self.status != "executed":
            raise ValueError("executed evidence must have executed status.")
        if not isinstance(self.workload, ProfilingWorkload):
            raise TypeError("workload must be a ProfilingWorkload.")
        if not isinstance(self.machine, MachineProvenance) or not isinstance(
            self.method, MeasurementMethod
        ):
            raise TypeError("executed context is invalid.")
        _validate_executed_sequences(
            self.raw_samples,
            self.metrics,
            self.raw_reports,
            self.method,
            self.workload,
        )


def _validate_executed_sequences(  # noqa: C901
    raw_samples: tuple[RawDurationSample, ...],
    metrics: tuple[NormalizedMetric, ...],
    raw_reports: tuple[RawReportProvenance, ...],
    method: MeasurementMethod,
    workload: ProfilingWorkload,
) -> None:
    """Validate bounded executed evidence sequences and their identities."""
    if (
        not isinstance(raw_samples, tuple)
        or not isinstance(metrics, tuple)
        or not isinstance(raw_reports, tuple)
    ):
        raise TypeError("executed evidence sequences must be tuples.")
    if not raw_samples or not raw_reports:
        raise ValueError("executed evidence needs samples and raw reports.")
    expected_samples = tuple(
        replay_count
        for replay_count in REPLAY_COUNTS
        for _ in range(workload.sample_count)
    )
    if tuple(item.replay_count for item in raw_samples) != expected_samples:
        raise ValueError("raw samples must follow the complete replay matrix.")
    if not all(isinstance(item, RawDurationSample) for item in raw_samples):
        raise TypeError("raw_samples are invalid.")
    if not all(isinstance(item, NormalizedMetric) for item in metrics):
        raise TypeError("metrics are invalid.")
    if not all(isinstance(item, RawReportProvenance) for item in raw_reports):
        raise TypeError("raw_reports are invalid.")
    if len({item.name for item in metrics}) != len(metrics):
        raise ValueError("metric names must be unique.")
    if tuple(item.name for item in metrics) != METHOD_METRICS[method.source]:
        raise ValueError("metrics must match the ordered measurement method.")
    if len({item.raw_filename for item in raw_reports}) != len(raw_reports):
        raise ValueError("raw filenames must be unique.")
    if (
        len(raw_samples) > MAX_ARTIFACT_CONTAINER_ITEMS
        or len(metrics) > MAX_ARTIFACT_CONTAINER_ITEMS
        or len(raw_reports) > MAX_ARTIFACT_CONTAINER_ITEMS
    ):
        raise ValueError("executed evidence sequence exceeds item limit.")


@dataclass(frozen=True, slots=True)
class UnavailableEvidence:
    """Store an exact workload and deterministic unavailability reason.

    Attributes:
        status: Literal ``"unavailable"`` status.
        workload: Original requested workload, without a smaller substitute.
        reason: Bounded deterministic reason evidence was unavailable.
    """

    status: str
    workload: ProfilingWorkload
    reason: str

    def __post_init__(self) -> None:
        """Validate an unavailable row without measurement context.

        Raises:
            TypeError: If the reason is not text.
            ValueError: If status, workload, or reason is invalid.
        """
        if self.status != "unavailable" or not isinstance(
            self.workload, ProfilingWorkload
        ):
            raise ValueError("unavailable evidence is invalid.")
        _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class ProfilingArtifact:
    """Store the ordered union of executed and unavailable evidence rows.

    Attributes:
        evidence: Nonempty, workload-ordered executed or unavailable rows.
    """

    evidence: tuple[ExecutedEvidence | UnavailableEvidence, ...]

    def __post_init__(self) -> None:
        """Validate bounded row types, identities, and workload ordering.

        Raises:
            TypeError: If a row is not an evidence record.
            ValueError: If rows are empty, oversized, duplicate, or unordered.
        """
        if (
            not isinstance(self.evidence, tuple)
            or not self.evidence
            or len(self.evidence) > MAX_ARTIFACT_ROWS
        ):
            raise ValueError("evidence must be a bounded nonempty tuple.")
        canonical_workloads = build_default_profiling_workload_matrix()
        if len(self.evidence) != len(canonical_workloads):
            raise ValueError("evidence must cover the frozen workload matrix.")
        for row in self.evidence:
            if not isinstance(row, (ExecutedEvidence, UnavailableEvidence)):
                raise TypeError("evidence rows are invalid.")
        if tuple(row.workload for row in self.evidence) != canonical_workloads:
            raise ValueError("evidence must follow workload ordering.")


def build_default_profiling_workload_matrix() -> tuple[ProfilingWorkload, ...]:
    """Build the fixed small and medium E8-F6 profiling workload matrix.

    The returned records request only ``(1, 16, 2)`` and ``(1000, 16, 2)``
    workloads. Building them neither probes hardware nor claims execution,
    feasibility, timing, or profiler evidence.

    Returns:
        Ordered canonical small and medium workload records.
    """
    common = {
        "active_fraction": 1.0,
        "processes": (
            "communication",
            "environment",
            "gas",
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
        "sample_count": 3,
        "seed": 1582,
        "duration_seconds": 0.5,
        "replay_counts": REPLAY_COUNTS,
    }
    rows = []
    for label, boxes in (("small", 1), ("medium", 1000)):
        shape = (boxes, 16, 2)
        rows.append(
            ProfilingWorkload(
                workload_id=build_profiling_workload_id(
                    label=label, shape=shape, **cast(dict[str, Any], common)
                ),
                label=label,
                shape=shape,
                **cast(dict[str, Any], common),
            )
        )
    return tuple(rows)


def ensure_profiling_raw_root(artifact_root: str | Path) -> Path:
    """Create and return the contained raw-report root below an injected root.

    Args:
        artifact_root: Existing nonsymlink directory named ``.artifacts``.

    Returns:
        Resolved ``benchmarks/profiling/raw`` directory below ``artifact_root``.

    Raises:
        ValueError: If the root or staging parents are symlinks, invalid, or
            escape the injected artifact root.
    """
    root = Path(artifact_root)
    if root.name != ".artifacts" or not root.is_dir() or root.is_symlink():
        raise ValueError(
            "artifact_root must be an existing nonsymlink .artifacts."
        )
    resolved_root = root.resolve(strict=True)
    raw_root = root / "benchmarks" / "profiling" / "raw"
    for parent in (root / "benchmarks", root / "benchmarks" / "profiling"):
        if parent.exists() and parent.is_symlink():
            raise ValueError("raw root parents must not be symlinks.")
    raw_root.mkdir(parents=True, exist_ok=True)
    if raw_root.is_symlink():
        raise ValueError("raw root must not be a symlink.")
    resolved_raw_root = raw_root.resolve(strict=True)
    try:
        resolved_raw_root.relative_to(resolved_root)
    except ValueError:
        raise ValueError("raw root must remain below artifact_root.") from None
    return resolved_raw_root


def _validate_raw_filename(raw_filename: object) -> str:
    """Require one plain filename rather than an injectable filesystem path."""
    if not isinstance(raw_filename, str):
        raise TypeError("raw_filename must be a string.")
    path = Path(raw_filename)
    if (
        not raw_filename
        or raw_filename in {".", ".."}
        or path.is_absolute()
        or len(path.parts) != 1
        or "/" in raw_filename
        or "\\" in raw_filename
        or not _SAFE_TEXT.fullmatch(raw_filename)
    ):
        raise ValueError("raw_filename must be one contained filename.")
    return raw_filename


def _hash_report_descriptor(directory: Path, filename: str) -> tuple[int, str]:
    """Hash one bounded regular no-follow descriptor beneath ``directory``."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(
            directory, os.O_RDONLY | os.O_DIRECTORY | nofollow
        )
        try:
            report_fd = os.open(
                filename, os.O_RDONLY | nofollow, dir_fd=directory_fd
            )
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise ValueError(
            "raw report must be a contained regular file."
        ) from error
    try:
        initial = os.fstat(report_fd)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("raw report must be a regular file.")
        size = initial.st_size
        if size <= 0 or size > MAX_RAW_REPORT_BYTES:
            raise ValueError("raw report has an invalid byte size.")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(report_fd, RAW_REPORT_HASH_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_RAW_REPORT_BYTES:
                raise ValueError("raw report has an invalid byte size.")
            digest.update(chunk)
        if os.fstat(report_fd) != initial or total != size:
            raise ValueError("raw report changed while it was hashed.")
        return size, digest.hexdigest()
    finally:
        os.close(report_fd)


def build_raw_report_provenance(
    artifact_root: str | Path, raw_filename: str
) -> RawReportProvenance:
    """Build safe provenance for one injected-root-contained raw report.

    Hashing streams bounded report bytes in fixed-size chunks. The returned
    record stores only a filename, size, and digest; it never stores a path.

    Args:
        artifact_root: Existing injected ``.artifacts`` directory.
        raw_filename: Plain contained raw-report filename.

    Returns:
        Validated size and SHA-256 provenance for the report.

    Raises:
        TypeError: If the filename has an unsupported type.
        ValueError: If the root, filename, file containment, or byte size is
            invalid.
    """
    filename = _validate_raw_filename(raw_filename)
    raw_root = ensure_profiling_raw_root(artifact_root)
    size, digest = _hash_report_descriptor(raw_root, filename)
    return RawReportProvenance(raw_filename, size, digest)


def verify_raw_report_provenance(
    artifact_root: str | Path, provenance: RawReportProvenance
) -> None:
    """Rehash a report and fail closed when its provenance has changed.

    Args:
        artifact_root: Existing injected ``.artifacts`` directory.
        provenance: Previously built filename, size, and digest record.

    Raises:
        TypeError: If ``provenance`` is not a raw-report provenance record.
        ValueError: If the report is unsafe, missing, oversized, or no longer
            matches its recorded size and digest.
    """
    if not isinstance(provenance, RawReportProvenance):
        raise TypeError("provenance must be RawReportProvenance.")
    current = build_raw_report_provenance(
        artifact_root, provenance.raw_filename
    )
    if current != provenance:
        raise ValueError("raw report provenance no longer matches.")


def to_json_value(artifact: ProfilingArtifact) -> dict[str, object]:
    """Convert validated evidence into the exact current schema envelope.

    Args:
        artifact: Validated ordered profiling evidence.

    Returns:
        JSON-compatible current-version envelope without writing or rehashing
        raw reports.

    Raises:
        TypeError: If ``artifact`` is not a profiling artifact.
    """
    if not isinstance(artifact, ProfilingArtifact):
        raise TypeError("artifact must be a ProfilingArtifact.")
    return {
        "schema_version": PROFILING_SCHEMA_VERSION,
        "artifact": {"evidence": [asdict(row) for row in artifact.evidence]},
    }


def serialize_profiling_artifact(artifact: ProfilingArtifact) -> str:
    """Serialize evidence as compact, canonical, finite JSON text.

    Args:
        artifact: Validated profiling artifact to serialize.

    Returns:
        Compact key-sorted current-schema JSON text.

    Raises:
        TypeError: If ``artifact`` is not a profiling artifact.
        ValueError: If JSON conversion would encounter a nonfinite value.
    """
    serialized = json.dumps(
        to_json_value(artifact),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    if len(serialized.encode("utf-8")) > MAX_ARTIFACT_PAYLOAD_BYTES:
        raise ValueError("serialized artifact exceeds maximum byte size.")
    return serialized


def _scan_json_character(
    character: str,
    *,
    depth: int,
    quoted: bool,
    escaped: bool,
) -> tuple[int, bool, bool]:
    """Advance JSON string and bracket state for one character."""
    if quoted:
        if escaped:
            return depth, quoted, False
        if character == "\\":
            return depth, quoted, True
        if character == '"':
            return depth, False, False
        return depth, quoted, False
    if character == '"':
        return depth, True, False
    if character in "[{":
        return depth + 1, False, False
    if character in "]}":
        return depth - 1, False, False
    return depth, False, False


def _validate_json_nesting(payload: str) -> None:
    """Make one predecode scan that bounds JSON bracket nesting."""
    depth = 0
    quoted = False
    escaped = False
    for character in payload:
        depth, quoted, escaped = _scan_json_character(
            character,
            depth=depth,
            quoted=quoted,
            escaped=escaped,
        )
        if depth > MAX_ARTIFACT_NESTING_DEPTH:
            raise ValueError("payload exceeds maximum nesting depth.")
        if depth < 0:
            raise ValueError("payload has invalid bracket nesting.")
    if depth or quoted:
        raise ValueError("payload has invalid JSON nesting.")


def _validate_structure(value: object) -> None:
    """Bound decoded container sizes with one iterative walk."""
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_ARTIFACT_NESTING_DEPTH:
            raise ValueError("payload exceeds maximum nesting depth.")
        if isinstance(current, dict):
            if len(current) > MAX_ARTIFACT_CONTAINER_ITEMS:
                raise ValueError("payload mapping exceeds item limit.")
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            if len(current) > MAX_ARTIFACT_CONTAINER_ITEMS:
                raise ValueError("payload list exceeds item limit.")
            pending.extend((item, depth + 1) for item in current)


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys at every depth."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("payload contains duplicate object keys.")
        result[key] = value
    return result


def _workload_from_json(value: object) -> ProfilingWorkload:
    """Reconstruct one workload with its exact schema."""
    raw = _require_exact_keys(
        value, set(ProfilingWorkload.__dataclass_fields__), "workload"
    )
    return ProfilingWorkload(
        **cast(dict[str, Any], raw)
        | {
            "shape": tuple(cast(Any, raw["shape"])),
            "processes": tuple(cast(Any, raw["processes"])),
            "diagnostics": tuple(cast(Any, raw["diagnostics"])),
            "replay_counts": tuple(cast(Any, raw["replay_counts"])),
        }
    )


def deserialize_profiling_artifact(payload: str) -> ProfilingArtifact:
    """Deserialize bounded current-version JSON through strict constructors.

    The host-only decoder bounds UTF-8 size and bracket nesting before JSON
    decoding, then validates exact record keys and reconstructs records through
    their fail-closed constructors. It performs no filesystem or hardware work.

    Args:
        payload: Current-schema JSON text.

    Returns:
        Validated profiling artifact.

    Raises:
        TypeError: If ``payload`` is not text.
        ValueError: If payload bounds, JSON, schema, record keys, or evidence
            values are invalid.
    """
    if not isinstance(payload, str):
        raise TypeError("payload must be a string.")
    if len(payload.encode("utf-8")) > MAX_ARTIFACT_PAYLOAD_BYTES:
        raise ValueError("payload exceeds maximum byte size.")
    _validate_json_nesting(payload)
    try:
        envelope = json.loads(
            payload, object_pairs_hook=_reject_duplicate_object_keys
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("payload is not valid JSON.") from error
    _validate_structure(envelope)
    envelope = _require_exact_keys(
        envelope, {"schema_version", "artifact"}, "envelope"
    )
    if envelope["schema_version"] != PROFILING_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version.")
    artifact = _require_exact_keys(
        envelope["artifact"], {"evidence"}, "artifact"
    )
    rows = artifact["evidence"]
    if not isinstance(rows, list) or not rows or len(rows) > MAX_ARTIFACT_ROWS:
        raise ValueError("artifact evidence is invalid.")
    evidence: list[ExecutedEvidence | UnavailableEvidence] = []
    for row in rows:
        if not isinstance(row, dict) or "status" not in row:
            raise ValueError("evidence row is invalid.")
        status = row["status"]
        if status == "executed":
            raw = cast(
                dict[str, Any],
                _require_exact_keys(
                    row,
                    set(ExecutedEvidence.__dataclass_fields__),
                    "executed evidence",
                ),
            )
            machine = MachineProvenance(
                **cast(
                    dict[str, Any],
                    _require_exact_keys(
                        raw["machine"],
                        set(MachineProvenance.__dataclass_fields__),
                        "machine",
                    ),
                )
            )
            method = MeasurementMethod(
                **cast(
                    dict[str, Any],
                    _require_exact_keys(
                        raw["method"],
                        set(MeasurementMethod.__dataclass_fields__),
                        "method",
                    ),
                )
            )
            samples = tuple(
                RawDurationSample(
                    **cast(
                        dict[str, Any],
                        _require_exact_keys(
                            item,
                            set(RawDurationSample.__dataclass_fields__),
                            "sample",
                        ),
                    )
                )
                for item in raw["raw_samples"]
            )
            metrics = tuple(
                NormalizedMetric(
                    **cast(
                        dict[str, Any],
                        _require_exact_keys(
                            item,
                            set(NormalizedMetric.__dataclass_fields__),
                            "metric",
                        ),
                    )
                )
                for item in raw["metrics"]
            )
            reports = tuple(
                RawReportProvenance(
                    **cast(
                        dict[str, Any],
                        _require_exact_keys(
                            item,
                            set(RawReportProvenance.__dataclass_fields__),
                            "report",
                        ),
                    )
                )
                for item in raw["raw_reports"]
            )
            evidence.append(
                ExecutedEvidence(
                    raw["status"],
                    _workload_from_json(raw["workload"]),
                    machine,
                    method,
                    samples,
                    metrics,
                    reports,
                )
            )
        elif status == "unavailable":
            raw = cast(
                dict[str, Any],
                _require_exact_keys(
                    row,
                    set(UnavailableEvidence.__dataclass_fields__),
                    "unavailable evidence",
                ),
            )
            evidence.append(
                UnavailableEvidence(
                    raw["status"],
                    _workload_from_json(raw["workload"]),
                    raw["reason"],
                )
            )
        else:
            raise ValueError("evidence status is not supported.")
    return ProfilingArtifact(tuple(evidence))
