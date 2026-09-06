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
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, cast

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

# E8-F7-P3 intentionally uses a schema independent of the P1 timing artifact.
NSIGHT_EVIDENCE_SCHEMA_VERSION = 1
PROCESS_TIMEOUT_SECONDS = 30
MAX_PROCESS_OUTPUT_BYTES = 16_384
MAX_EXPORT_ROWS = 4_096
NSYS_BANNER = "2026.1.3.425-1"
NCU_BANNER = "2026.2.1.5-1"
NSYS_COLUMNS = ("kernel_name", "start_ns", "duration_ns", "correlation_id")
NCU_COLUMNS = (
    "kernel_name",
    "invocations",
    "metric_name",
    "metric_value",
    "unit",
    "correlation_id",
)
NSIGHT_METRICS = frozenset(
    (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "dram__throughput.avg",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    )
)
NSIGHT_METRIC_UNITS = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
    "dram__throughput.avg": "GB/s",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "transactions",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "%",
}
WORKER_COMMAND = (
    sys.executable,
    "-m",
    "particula.gpu.tests.profiling_workload_runner",
    "--workload",
    "small",
    "--mode",
    "captured-replay",
)


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
    """Validate the complete sample, metric, and report sequences.

    Args:
        raw_samples: Ordered duration samples for every replay count.
        metrics: Ordered absolute metrics required by the measurement method.
        raw_reports: Unique raw-report provenance records.
        method: Measurement method that determines the metric sequence.
        workload: Workload that determines the required sample count.

    Raises:
        TypeError: If a sequence or sequence item has the wrong type.
        ValueError: If a sequence is empty, duplicated, incomplete, or
            inconsistent with the workload or method.
    """
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
    """Advance JSON quoting and bracket state for one character.

    Args:
        character: JSON character to inspect.
        depth: Current unmatched opening-bracket depth.
        quoted: Whether the scanner is inside a JSON string.
        escaped: Whether the previous string character was a backslash.

    Returns:
        Updated ``(depth, quoted, escaped)`` scanner state.
    """
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
    """Make one predecode scan that bounds JSON bracket nesting.

    Args:
        payload: JSON text to scan before decoding.

    Raises:
        ValueError: If the payload exceeds the depth bound or has unbalanced
            brackets or quotes.
    """
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
    """Bound decoded container sizes with one iterative walk.

    Args:
        value: Decoded JSON value whose nested mappings and lists are checked.

    Raises:
        ValueError: If a nested container or value depth exceeds its bound.
    """
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
    """Build a JSON object while rejecting duplicate keys at every depth.

    Args:
        pairs: Object key-value pairs supplied by ``json.loads``.

    Returns:
        A dictionary containing the unique object members.

    Raises:
        ValueError: If an object contains a duplicate key.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("payload contains duplicate object keys.")
        result[key] = value
    return result


def _workload_from_json(value: object) -> ProfilingWorkload:
    """Reconstruct one workload with its exact schema.

    Args:
        value: Decoded workload mapping from an artifact payload.

    Returns:
        A validated profiling workload.

    Raises:
        ValueError: If the mapping has unexpected fields or invalid values.
    """
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


@dataclass(frozen=True, slots=True)
class NsightToolQualification:
    """Store one exact Nsight executable qualification result.

    Attributes:
        tool: Qualified executable name, either ``"nsys"`` or ``"ncu"``.
        banner: Exact allow-listed version banner returned by the executable.
    """

    tool: str
    banner: str

    def __post_init__(self) -> None:
        """Validate the closed tool and literal version banner."""
        expected = {"nsys": NSYS_BANNER, "ncu": NCU_BANNER}
        if self.tool not in expected or self.banner != expected[self.tool]:
            raise ValueError("Nsight tool qualification is not supported.")


@dataclass(frozen=True, slots=True)
class NsightCommandOutcome:
    """Store a bounded command result without retaining command paths.

    Attributes:
        tool: Command owner, such as ``"nsys"``, ``"ncu"``, or ``"worker"``.
        stage: Process stage that produced the result.
        returncode: Native process return code.
        stdout: Bounded standard output captured from the process.
        stderr: Bounded standard error captured from the process.
    """

    tool: str
    stage: str
    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        """Validate the bounded command-result fields."""
        if self.tool not in {"nsys", "ncu", "worker"}:
            raise ValueError("unknown command tool.")
        if self.stage not in {"probe", "worker", "collect", "export"}:
            raise ValueError("unknown command stage.")
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an integer.")
        for value in (self.stdout, self.stderr):
            if (
                not isinstance(value, str)
                or len(value.encode()) > MAX_PROCESS_OUTPUT_BYTES
            ):
                raise ValueError("command diagnostics exceed the byte limit.")


@dataclass(frozen=True, slots=True)
class NsightUnavailable:
    """Represent unavailable profiling without fabricated metrics.

    Attributes:
        tool: Tool or worker that could not provide evidence.
        reason: Bounded explanation for the unavailable result.
    """

    tool: str
    reason: str

    def __post_init__(self) -> None:
        """Validate the unavailable tool and bounded reason."""
        if self.tool not in {"nsys", "ncu", "worker"}:
            raise ValueError("unknown unavailable tool.")
        _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class NsightFailed:
    """Represent one bounded failed process stage.

    Attributes:
        outcome: Captured command result with a nonzero return code.
    """

    outcome: NsightCommandOutcome

    def __post_init__(self) -> None:
        """Require an outcome representing a failed process."""
        if self.outcome.returncode == 0:
            raise ValueError("failed outcome must have a nonzero return code.")


@dataclass(frozen=True, slots=True)
class NsightKernelRow:
    """Store one strict attributed or unattributed native profiler row.

    Attributes:
        tool: Profiler that produced the row.
        kernel_name: Exact native kernel name reported by the profiler.
        correlation_id: Positive process-to-kernel correlation identifier.
        metric_name: Allow-listed Compute metric, or ``None`` for Systems rows.
        value: Normalized duration or metric value.
        unit: Unit associated with ``value``.
        invocations: Positive invocation count represented by the row.
        attribution: Exact mapping status for the resident process.
        provenance: Contained raw report that supplied the row.
    """

    tool: str
    kernel_name: str
    correlation_id: int
    metric_name: str | None
    value: float | int
    unit: str
    invocations: int
    attribution: str
    provenance: RawReportProvenance

    def __post_init__(self) -> None:
        """Validate the profiler-specific metric row schema."""
        if self.tool not in {"nsys", "ncu"}:
            raise ValueError("unknown profiler tool.")
        _require_text(self.kernel_name, "kernel_name")
        _require_int(self.correlation_id, "correlation_id", positive=True)
        _require_int(self.invocations, "invocations", positive=True)
        if self.attribution not in {"attributed", "unattributed", "ambiguous"}:
            raise ValueError("invalid row attribution.")
        if not isinstance(self.provenance, RawReportProvenance):
            raise TypeError("provenance must be raw report provenance.")
        if self.tool == "nsys":
            if self.metric_name is not None or self.unit != "ns":
                raise ValueError("timeline rows require duration nanoseconds.")
            _require_int(self.value, "duration_ns", positive=True)
            return
        if self.metric_name not in NSIGHT_METRICS:
            raise ValueError("metric is not allow-listed.")
        if self.unit != NSIGHT_METRIC_UNITS[self.metric_name]:
            raise ValueError("metric unit is not supported.")
        _require_number(self.value, "metric_value")


@dataclass(frozen=True, slots=True)
class NsightEvidence:
    """Store ordered Nsight rows for one exact workload and process.

    Attributes:
        qualification: Exact version qualification for the source tool.
        workload_id: Canonical workload identifier associated with the rows.
        process: Resident process represented by the evidence.
        rows: Nonempty bounded profiler rows in source order.
    """

    qualification: NsightToolQualification
    workload_id: str
    process: str
    rows: tuple[NsightKernelRow, ...]

    def __post_init__(self) -> None:
        """Validate the bounded evidence rows for one qualified tool."""
        _require_text(self.workload_id, "workload_id")
        _require_text(self.process, "process")
        if not self.rows or len(self.rows) > MAX_EXPORT_ROWS:
            raise ValueError("Nsight evidence rows are invalid.")
        if any(row.tool != self.qualification.tool for row in self.rows):
            raise ValueError("rows must match evidence tool.")


def _parse_csv(text: str, columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Parse an exact comma-only CSV schema with bounded rows.

    Args:
        text: Exported CSV text, including its required final newline.
        columns: Exact ordered header names required by the export schema.

    Returns:
        Parsed rows represented as string-valued dictionaries.

    Raises:
        ValueError: If the export is oversized, malformed, or has the wrong
            header or row shape.
    """
    import csv

    if len(text.encode()) > MAX_RAW_REPORT_BYTES or not text.endswith("\n"):
        raise ValueError("CSV export has invalid bounds or line ending.")
    try:
        rows = list(csv.reader(text.splitlines(), strict=True))
    except csv.Error as error:
        raise ValueError("CSV export is malformed.") from error
    if (
        not rows
        or tuple(rows[0]) != columns
        or len(set(rows[0])) != len(columns)
    ):
        raise ValueError("CSV export schema is not supported.")
    if len(rows) - 1 > MAX_EXPORT_ROWS or any(
        len(row) != len(columns) for row in rows[1:]
    ):
        raise ValueError("CSV export rows are invalid.")
    return [dict(zip(columns, row, strict=True)) for row in rows[1:]]


def parse_nsys_timeline_csv(
    text: str, provenance: RawReportProvenance, process_ids: dict[int, str]
) -> tuple[NsightKernelRow, ...]:
    """Parse strict Systems timeline CSV rows with exact ID attribution.

    Args:
        text: Bounded Systems CSV export.
        provenance: Contained raw report that produced the export.
        process_ids: Exact correlation-to-kernel mapping evidence.

    Returns:
        Parsed duration rows with attributed or unattributed status.

    Raises:
        ValueError: If the CSV schema or a duration or identifier is invalid.
    """
    rows = []
    for row in _parse_csv(text, NSYS_COLUMNS):
        correlation = _require_int(
            int(row["correlation_id"]), "correlation_id", positive=True
        )
        name = row["kernel_name"]
        attribution = (
            "attributed"
            if process_ids.get(correlation) == name
            else "unattributed"
        )
        rows.append(
            NsightKernelRow(
                "nsys",
                name,
                correlation,
                None,
                _require_int(
                    int(row["duration_ns"]), "duration_ns", positive=True
                ),
                "ns",
                1,
                attribution,
                provenance,
            )
        )
    return tuple(rows)


def parse_ncu_metrics_csv(
    text: str, provenance: RawReportProvenance, process_ids: dict[int, str]
) -> tuple[NsightKernelRow, ...]:
    """Parse strict Compute CSV rows with exact ID attribution.

    Args:
        text: Bounded Compute CSV export.
        provenance: Contained raw report that produced the export.
        process_ids: Exact correlation-to-kernel mapping evidence.

    Returns:
        Parsed metric rows with attributed or unattributed status.

    Raises:
        ValueError: If the CSV schema, metric, unit, or numeric field is
            invalid.
    """
    rows = []
    for row in _parse_csv(text, NCU_COLUMNS):
        try:
            value = float(row["metric_value"])
        except ValueError as error:
            raise ValueError("metric value is invalid.") from error
        correlation = _require_int(
            int(row["correlation_id"]), "correlation_id", positive=True
        )
        name = row["kernel_name"]
        attribution = (
            "attributed"
            if process_ids.get(correlation) == name
            else "unattributed"
        )
        rows.append(
            NsightKernelRow(
                "ncu",
                name,
                correlation,
                row["metric_name"],
                value,
                row["unit"],
                _require_int(
                    int(row["invocations"]), "invocations", positive=True
                ),
                attribution,
                provenance,
            )
        )
    return tuple(rows)


# Injected test runners may accept the optional ``cwd`` keyword used for
# profiler collection, while probe and worker fakes commonly accept only the
# command tuple.
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _bounded_text(value: str | bytes | None) -> str:
    """Decode and bound process diagnostics before records are constructed.

    Args:
        value: Text, bytes, or ``None`` returned by a process runner.

    Returns:
        UTF-8-decoded bounded diagnostic text.

    Raises:
        ValueError: If the decoded diagnostics exceed the byte limit.
    """
    if value is None:
        return ""
    text = value.decode(errors="replace") if isinstance(value, bytes) else value
    if len(text.encode()) > MAX_PROCESS_OUTPUT_BYTES:
        raise ValueError("process diagnostics exceed the byte limit.")
    return text


def run_profile_command(
    command: tuple[str, ...],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an internally-created immutable argument tuple without a shell.

    Args:
        command: Nonempty tuple of internally constructed process arguments.

    Returns:
        Completed subprocess result with bounded text output.

    Raises:
        TypeError: If ``command`` is not a nonempty string tuple.
        OSError: If the requested executable cannot be launched.
        subprocess.TimeoutExpired: If the fixed process timeout is exceeded.
    """
    if (
        not isinstance(command, tuple)
        or not command
        or not all(isinstance(item, str) for item in command)
    ):
        raise TypeError("command must be a nonempty immutable string tuple.")
    process = subprocess.Popen(  # noqa: S603 - command is internally constructed.
        command,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )
    outputs: dict[str, bytes] = {"stdout": b"", "stderr": b""}
    overflowed = threading.Event()

    def read_stream(name: str, stream: Any) -> None:
        """Read a stream and terminate at the output cap."""
        chunks: list[bytes] = []
        size = 0
        while chunk := stream.read(4096):
            size += len(chunk)
            if size > MAX_PROCESS_OUTPUT_BYTES:
                overflowed.set()
                process.terminate()
                break
            chunks.append(chunk)
        outputs[name] = b"".join(chunks)

    threads = [
        threading.Thread(target=read_stream, args=("stdout", process.stdout)),
        threading.Thread(target=read_stream, args=("stderr", process.stderr)),
    ]
    for thread in threads:
        thread.start()
    try:
        process.wait(timeout=PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        for thread in threads:
            thread.join()
    if overflowed.is_set():
        raise ValueError("process diagnostics exceed the byte limit.")
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        outputs["stdout"].decode(errors="replace"),
        outputs["stderr"].decode(errors="replace"),
    )


def qualify_nsight_tool(
    tool: str, runner: Runner = run_profile_command
) -> NsightToolQualification | NsightUnavailable | NsightFailed:
    """Probe one allow-listed tool and accept only its exact banner.

    Args:
        tool: Executable name, either ``"nsys"`` or ``"ncu"``.
        runner: Injectable process runner used for the version probe.

    Returns:
        Exact qualification, unavailable outcome, or failed outcome.

    Raises:
        ValueError: If ``tool`` is not allow-listed.
    """
    if tool not in {"nsys", "ncu"}:
        raise ValueError("tool must be nsys or ncu.")
    command = (tool, "--version")
    try:
        result = runner(command)
        outcome = NsightCommandOutcome(
            tool,
            "probe",
            result.returncode,
            _bounded_text(result.stdout),
            _bounded_text(result.stderr),
        )
    except OSError:
        return NsightUnavailable(tool, "binary not found")
    except subprocess.TimeoutExpired:
        return NsightUnavailable(tool, "probe timed out")
    except ValueError:
        return NsightUnavailable(tool, "probe diagnostics rejected")
    if outcome.returncode:
        return NsightFailed(outcome)
    expected = NSYS_BANNER if tool == "nsys" else NCU_BANNER
    if outcome.stdout != expected + "\n" or outcome.stderr:
        return NsightUnavailable(tool, "unexpected version banner")
    return NsightToolQualification(tool, expected)


def _plain_report_path(raw_root: Path, filename: str) -> Path:
    """Return one plain contained report path before a profiler can run.

    Args:
        raw_root: Nonsymlink directory reserved for raw reports.
        filename: Plain report filename supplied by the collector.

    Returns:
        Contained report path beneath ``raw_root``.

    Raises:
        ValueError: If the filename or destination is unsafe or symlinked.
    """
    filename = _validate_raw_filename(filename)
    path = raw_root / filename
    if path.parent != raw_root or raw_root.is_symlink() or path.is_symlink():
        raise ValueError("report destination is not contained.")
    return path


def _run_outcome(
    tool: str,
    stage: str,
    command: tuple[str, ...],
    runner: Runner,
    cwd: Path | None = None,
) -> NsightCommandOutcome | NsightUnavailable:
    """Execute a bounded command and normalize expected process absence.

    Args:
        tool: Logical tool or worker name for the result.
        stage: Orchestration stage being executed.
        command: Immutable internally-created process arguments.
        runner: Injectable process runner.

    Returns:
        Bounded command outcome or explicit unavailability result.
    """
    try:
        result = runner(command) if cwd is None else runner(command, cwd=cwd)
        return NsightCommandOutcome(
            tool,
            stage,
            result.returncode,
            _bounded_text(result.stdout),
            _bounded_text(result.stderr),
        )
    except OSError:
        return NsightUnavailable(tool, f"{stage} binary not found")
    except subprocess.TimeoutExpired:
        return NsightUnavailable(tool, f"{stage} timed out")
    except ValueError:
        return NsightUnavailable(tool, f"{stage} diagnostics rejected")


def collect_nsight_evidence(
    *,
    tool: str,
    artifact_root: str | Path,
    report_filename: str,
    process_ids: dict[int, str],
    runner: Runner = run_profile_command,
) -> NsightEvidence | NsightUnavailable | NsightFailed:
    """Collect one closed worker report with bounded per-tool orchestration.

    The caller can inject only the runner.  Tool paths, worker arguments and
    profiler templates remain fixed here; arbitrary environment and arguments
    never cross this boundary.
    """
    raw_root = ensure_profiling_raw_root(artifact_root)
    report = _plain_report_path(raw_root, report_filename)
    qualification = qualify_nsight_tool(tool, runner)
    if not isinstance(qualification, NsightToolQualification):
        return qualification
    worker = _run_outcome("worker", "worker", WORKER_COMMAND, runner)
    if isinstance(worker, NsightUnavailable):
        return worker
    if worker.returncode == 3 and worker.stdout.startswith(
        "PROFILING_WORKLOAD_UNAVAILABLE: "
    ):
        return NsightUnavailable("worker", worker.stdout.rstrip())
    if worker.returncode:
        return NsightFailed(worker)
    # The tools receive only a fixed collection mode and the contained basename.
    collection = (
        ("nsys", "profile", "--force-overwrite=true", "--output", report.name)
        + WORKER_COMMAND
        if tool == "nsys"
        else (
            "ncu",
            "--csv",
            "--log-file",
            report.name,
            "--metrics",
            ",".join(sorted(NSIGHT_METRICS)),
        )
        + WORKER_COMMAND
    )
    collected = _run_outcome(tool, "collect", collection, runner, raw_root)
    if isinstance(collected, NsightUnavailable):
        return collected
    if collected.returncode:
        return NsightFailed(collected)
    export = (
        ("nsys", "export", "--type", "csv", report.name)
        if tool == "nsys"
        else ("ncu", "--import", report.name, "--csv")
    )
    exported = _run_outcome(tool, "export", export, runner, raw_root)
    if isinstance(exported, NsightUnavailable):
        return exported
    if exported.returncode:
        return NsightFailed(exported)
    # Export stdout is intentionally bounded before parsing and provenance is
    # created only after the collector's contained file exists successfully.
    report.write_text(exported.stdout, encoding="ascii")
    provenance = build_raw_report_provenance(artifact_root, report.name)
    parser = (
        parse_nsys_timeline_csv if tool == "nsys" else parse_ncu_metrics_csv
    )
    rows = parser(exported.stdout, provenance, process_ids)
    if not rows:
        return NsightUnavailable(tool, "export contains no kernel rows")
    workload = build_default_profiling_workload_matrix()[0]
    return NsightEvidence(qualification, workload.workload_id, "resident", rows)


# Analysis-only P4 records: they intentionally are not serialized artifacts.
ANALYSIS_MODES = frozenset(("prepared_uncaptured", "captured_replay"))
ANALYSIS_STATUSES = frozenset(("reconciled", "insufficient", "unavailable"))
ANALYSIS_CONFIDENCE = frozenset(("sufficient", "low", "none"))


def _analysis_tuple(value: object, name: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple.")
    return value


def _require_analysis_mode(value: object) -> str:
    """Return one supported analysis mode without coercion."""
    if not isinstance(value, str):
        raise TypeError("mode must be a string.")
    if value not in ANALYSIS_MODES:
        raise ValueError("mode is not supported.")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Explicit metadata-only artifact and raw-report binding."""

    artifact_key: str
    generation: str
    raw_reports: tuple[RawReportProvenance, ...]

    def __post_init__(self) -> None:
        """Validate metadata and immutable raw-report provenance.

        Raises:
            TypeError: If raw-report provenance is not a tuple of records.
            ValueError: If metadata is unsafe or report provenance is empty or
                duplicated.
        """
        _require_text(self.artifact_key, "artifact_key")
        _require_text(self.generation, "generation")
        reports = _analysis_tuple(self.raw_reports, "raw_reports")
        if not reports or not all(
            isinstance(x, RawReportProvenance) for x in reports
        ):
            raise TypeError("raw_reports must contain RawReportProvenance.")
        if len(reports) != len(set(reports)):
            raise ValueError("raw_reports must be unique.")


def _artifact_key(reference: ArtifactReference, mode: str, method: str) -> None:
    if reference.artifact_key != f"{mode}_{method}.json":
        raise ValueError(
            "artifact_key must match the explicit mode and method."
        )


@dataclass(frozen=True, slots=True)
class HostEvidenceBinding:
    """One explicitly mode-bound P2 measurement."""

    evidence: ExecutedEvidence
    mode: str
    reference: ArtifactReference

    def __post_init__(self) -> None:
        """Validate the explicit P2 mode, method, and provenance binding.

        Raises:
            TypeError: If evidence or its reference has an invalid type.
            ValueError: If the mode, method, artifact key, or provenance does
                not match.
        """
        if not isinstance(self.evidence, ExecutedEvidence):
            raise TypeError("evidence must be ExecutedEvidence.")
        _require_analysis_mode(self.mode)
        if not isinstance(self.reference, ArtifactReference):
            raise TypeError("reference must be ArtifactReference.")
        if self.evidence.method.source not in {
            "host_launch",
            "synchronized_elapsed",
        }:
            raise ValueError("host evidence method is not supported.")
        _artifact_key(self.reference, self.mode, self.evidence.method.source)
        if self.reference.raw_reports != self.evidence.raw_reports:
            raise ValueError(
                "reference provenance must equal evidence provenance."
            )


@dataclass(frozen=True, slots=True)
class MachineBoundKernelEvidence:
    """One explicitly mode- and machine-bound P3 export."""

    evidence: NsightEvidence
    mode: str
    machine: MachineProvenance
    reference: ArtifactReference

    def __post_init__(self) -> None:
        """Validate the explicit P3 mode, machine, and provenance binding.

        Raises:
            TypeError: If evidence, machine, or reference has an invalid type.
            ValueError: If the workload, resident process, row identities, or
                ordered provenance is invalid.
        """
        if not isinstance(self.evidence, NsightEvidence):
            raise TypeError("evidence must be NsightEvidence.")
        _require_analysis_mode(self.mode)
        if self.evidence.process != "resident":
            raise ValueError("kernel evidence mode or process is invalid.")
        if not isinstance(self.machine, MachineProvenance) or not isinstance(
            self.reference, ArtifactReference
        ):
            raise TypeError("kernel evidence context is invalid.")
        if (
            self.evidence.workload_id
            != build_default_profiling_workload_matrix()[0].workload_id
        ):
            raise ValueError(
                "kernel evidence must describe the frozen small workload."
            )
        reports = tuple(row.provenance for row in self.evidence.rows)
        if len(reports) != len(set(reports)):
            raise ValueError("row identities must be unique.")
        if reports != self.reference.raw_reports:
            raise ValueError(
                "reference provenance must equal ordered row provenance."
            )
        identities = {
            (
                row.tool,
                row.kernel_name,
                row.correlation_id,
                row.metric_name,
                row.provenance,
            )
            for row in self.evidence.rows
        }
        if len(identities) != len(self.evidence.rows):
            raise ValueError("Nsight row identities must be unique.")


@dataclass(frozen=True, slots=True)
class EvidenceUnavailable:
    """Unavailable P2 or P3 evidence without a fabricated measurement."""

    workload: ProfilingWorkload
    mode: str
    machine: MachineProvenance | None
    reference: ArtifactReference | None
    reason: str

    def __post_init__(self) -> None:
        """Validate unavailable evidence without fabricating measurements.

        Raises:
            TypeError: If workload, machine, or reference has an invalid type.
            ValueError: If the mode or bounded unavailability reason is
                invalid.
        """
        if not isinstance(self.workload, ProfilingWorkload):
            raise TypeError("workload must be ProfilingWorkload.")
        _require_analysis_mode(self.mode)
        if self.machine is not None and not isinstance(
            self.machine, MachineProvenance
        ):
            raise TypeError("machine is invalid.")
        if self.reference is not None and not isinstance(
            self.reference, ArtifactReference
        ):
            raise TypeError("reference is invalid.")
        _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class KernelContribution:
    """One ranked attributed Nsight duration contribution."""

    process: str
    kernel_name: str
    value: float
    provenance: RawReportProvenance
    row_position: int
    metric: str = "profiler_gpu_duration"
    unit: str = "ns"

    def __post_init__(self) -> None:
        """Validate one ranked, attributed profiler-duration contribution.

        Raises:
            TypeError: If provenance has an invalid type.
            ValueError: If text, value, position, metric, or unit is invalid.
        """
        _require_text(self.process, "process")
        _require_text(self.kernel_name, "kernel_name")
        _require_number(self.value, "value")
        if self.metric != "profiler_gpu_duration" or self.unit != "ns":
            raise ValueError("contribution metric or unit is invalid.")
        if not isinstance(self.provenance, RawReportProvenance):
            raise TypeError("provenance must be RawReportProvenance.")
        _require_int(self.row_position, "row_position")


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """Synchronized-elapsed versus attributed-Nsight duration comparison."""

    status: str
    host_total_ns: float
    profiler_total_ns: float
    signed_difference_ns: float
    absolute_difference_ns: float

    def __post_init__(self) -> None:
        """Validate finite host/profiler totals and their reconciliation state.

        Raises:
            ValueError: If a status or total is invalid or nonfinite.
        """
        if self.status not in {"reconciled", "non_reconcilable"}:
            raise ValueError("reconciliation status is invalid.")
        for field in (
            "host_total_ns",
            "profiler_total_ns",
            "absolute_difference_ns",
        ):
            _require_number(getattr(self, field), field)
        if type(self.signed_difference_ns) not in (
            int,
            float,
        ) or not math.isfinite(self.signed_difference_ns):
            raise ValueError("signed_difference_ns must be finite.")
        expected_difference = self.profiler_total_ns - self.host_total_ns
        if self.signed_difference_ns != expected_difference:
            raise ValueError("signed_difference_ns is inconsistent.")
        if self.absolute_difference_ns != abs(expected_difference):
            raise ValueError("absolute_difference_ns is inconsistent.")


@dataclass(frozen=True, slots=True)
class PerformanceDecision:
    """Immutable machine- and workload-bounded analysis outcome."""

    status: str
    confidence: str
    workload: ProfilingWorkload
    mode: str
    machine: MachineProvenance | None
    contributions: tuple[KernelContribution, ...]
    reconciliation: Reconciliation | None
    limitations: tuple[str, ...]
    host_references: tuple[ArtifactReference, ...] = ()
    kernel_references: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate an immutable bounded performance-analysis decision.

        Raises:
            TypeError: If contribution or limitation sequences are invalid.
            ValueError: If decision fields, mode, or workload are invalid.
        """
        if (
            self.status not in ANALYSIS_STATUSES
            or self.confidence not in ANALYSIS_CONFIDENCE
        ):
            raise ValueError("decision status or confidence is invalid.")
        _require_analysis_mode(self.mode)
        if not isinstance(self.workload, ProfilingWorkload):
            raise ValueError("decision workload is invalid.")
        if not all(
            isinstance(x, KernelContribution)
            for x in _analysis_tuple(self.contributions, "contributions")
        ):
            raise TypeError("contributions are invalid.")
        for limitation in _analysis_tuple(self.limitations, "limitations"):
            _require_text(limitation, "limitation")
        for name, references in (
            ("host_references", self.host_references),
            ("kernel_references", self.kernel_references),
        ):
            values = _analysis_tuple(references, name)
            if not all(isinstance(x, ArtifactReference) for x in values):
                raise TypeError(f"{name} are invalid.")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique.")


def analyze_machine_bounded_performance(
    host_inputs: tuple[HostEvidenceBinding | EvidenceUnavailable, ...],
    kernel_inputs: tuple[MachineBoundKernelEvidence | EvidenceUnavailable, ...],
) -> PerformanceDecision:
    """Pure fail-closed analysis of explicit captured-replay evidence."""
    hosts = _analysis_tuple(host_inputs, "host_inputs")
    kernels = _analysis_tuple(kernel_inputs, "kernel_inputs")
    if not hosts or not kernels:
        raise ValueError("both evidence families are required.")
    if not all(
        isinstance(x, (HostEvidenceBinding, EvidenceUnavailable)) for x in hosts
    ) or not all(
        isinstance(x, (MachineBoundKernelEvidence, EvidenceUnavailable))
        for x in kernels
    ):
        raise TypeError("analysis inputs are invalid.")
    unavailable = next(
        (x for x in (*hosts, *kernels) if isinstance(x, EvidenceUnavailable)),
        None,
    )
    workload = (
        unavailable.workload
        if unavailable
        else cast(HostEvidenceBinding, hosts[0]).evidence.workload
    )
    if unavailable:
        if unavailable.mode != "captured_replay":
            return PerformanceDecision(
                "insufficient",
                "low",
                workload,
                unavailable.mode,
                unavailable.machine,
                (),
                None,
                ("unavailable evidence is not captured replay",),
            )
        return PerformanceDecision(
            "unavailable",
            "none",
            workload,
            unavailable.mode,
            unavailable.machine,
            (),
            None,
            (unavailable.reason,),
        )
    bound_hosts = cast(tuple[HostEvidenceBinding, ...], hosts)
    bound_kernels = cast(tuple[MachineBoundKernelEvidence, ...], kernels)
    if (
        any(x.mode != "captured_replay" for x in bound_hosts)
        or any(x.mode != "captured_replay" for x in bound_kernels)
        or workload.label != "small"
    ):
        return PerformanceDecision(
            "insufficient",
            "low",
            workload,
            "captured_replay",
            None,
            (),
            None,
            ("mode or workload mismatch",),
        )
    machine = bound_hosts[0].evidence.machine
    methods = tuple(x.evidence.method.source for x in bound_hosts)
    if len(methods) != len(set(methods)):
        raise ValueError("duplicate host evidence method.")
    if any(
        x.evidence.workload != workload or x.evidence.machine != machine
        for x in bound_hosts
    ) or any(
        x.evidence.workload_id != workload.workload_id or x.machine != machine
        for x in bound_kernels
    ):
        return PerformanceDecision(
            "insufficient",
            "low",
            workload,
            "captured_replay",
            machine,
            (),
            None,
            ("workload or machine mismatch",),
        )
    kernel_identities = tuple(
        (
            binding.evidence.qualification,
            binding.evidence.workload_id,
            binding.evidence.process,
            binding.evidence.rows,
        )
        for binding in bound_kernels
    )
    if len(kernel_identities) != len(set(kernel_identities)):
        raise ValueError("duplicate kernel evidence.")
    row_identities = tuple(
        (
            binding.evidence.process,
            row.tool,
            row.kernel_name,
            row.correlation_id,
            row.metric_name,
            row.provenance,
        )
        for binding in bound_kernels
        for row in binding.evidence.rows
    )
    if len(row_identities) != len(set(row_identities)):
        raise ValueError("duplicate kernel row identity.")
    return _analyze_bound_machine_performance(
        workload,
        machine,
        bound_hosts,
        bound_kernels,
    )


def _analyze_bound_machine_performance(
    workload: ProfilingWorkload,
    machine: MachineProvenance,
    bound_hosts: tuple[HostEvidenceBinding, ...],
    bound_kernels: tuple[MachineBoundKernelEvidence, ...],
) -> PerformanceDecision:
    """Analyze already validated, machine-matched captured-replay evidence."""
    synchronized = [
        x
        for x in bound_hosts
        if x.evidence.method.source == "synchronized_elapsed"
    ]
    if not synchronized:
        return PerformanceDecision(
            "insufficient",
            "low",
            workload,
            "captured_replay",
            machine,
            (),
            None,
            ("missing synchronized elapsed",),
        )
    rows = tuple(
        (binding, pos, row)
        for binding in bound_kernels
        for pos, row in enumerate(binding.evidence.rows)
    )
    if any(
        row.tool != "nsys"
        or row.metric_name is not None
        or row.unit != "ns"
        or row.attribution != "attributed"
        for _, _, row in rows
    ):
        return PerformanceDecision(
            "insufficient",
            "low",
            workload,
            "captured_replay",
            machine,
            (),
            None,
            ("incomplete attribution",),
        )
    contributions = tuple(
        sorted(
            (
                KernelContribution(
                    binding.evidence.process,
                    row.kernel_name,
                    float(row.value),
                    row.provenance,
                    pos,
                )
                for binding, pos, row in rows
            ),
            key=lambda x: (
                -x.value,
                x.process,
                x.kernel_name,
                x.provenance.raw_filename,
                x.provenance.sha256,
                x.row_position,
            ),
        )
    )
    host_total = sum(
        s.duration_ns / s.replay_count
        for s in synchronized[0].evidence.raw_samples
    ) / len(synchronized[0].evidence.raw_samples)
    profiler_total = sum(x.value for x in contributions)
    if not profiler_total or not host_total:
        return PerformanceDecision(
            "insufficient",
            "low",
            workload,
            "captured_replay",
            machine,
            (),
            None,
            ("zero total has no percentage",),
        )
    difference = profiler_total - host_total
    reconciliation = Reconciliation(
        "reconciled"
        if abs(difference) <= max(1.0, 0.05 * host_total)
        else "non_reconcilable",
        host_total,
        profiler_total,
        difference,
        abs(difference),
    )
    if reconciliation.status != "reconciled":
        return PerformanceDecision(
            "insufficient",
            "low",
            workload,
            "captured_replay",
            machine,
            (),
            reconciliation,
            ("non reconcilable totals",),
        )
    return PerformanceDecision(
        "reconciled",
        "sufficient",
        workload,
        "captured_replay",
        machine,
        contributions,
        reconciliation,
        (
            "not portable bounded to workload CUDA machine software source "
            "metric and artifacts",
        ),
        tuple(binding.reference for binding in bound_hosts),
        tuple(binding.reference for binding in bound_kernels),
    )


GUARDED_PROPOSAL_CATEGORIES = frozenset(
    (
        "kernel",
        "host_launch",
        "memory",
        "scientific",
        "ownership",
        "order",
        "rng",
    )
)
_CORRECTNESS_CATEGORIES = frozenset(("scientific", "ownership", "order", "rng"))
_PORTABLE_WORDING = re.compile(
    r"\b(portable|universal|all machines|always)\b", re.I
)
_CORRECTNESS_TEXT = re.compile(
    r"\b(scientific|equation|numerical\s+tolerance|ownership|transfer|process\s+order|rng)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class PerformanceProposal:
    """Evidence-linked, machine- and workload-bounded proposed change."""

    category: str
    text: str
    correctness_plan_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate bounded proposal wording and correctness-plan guardrails.

        Raises:
            ValueError: If the category, wording, or correctness plan is
                invalid or insufficient.
        """
        if self.category not in GUARDED_PROPOSAL_CATEGORIES:
            raise ValueError("proposal category is invalid.")
        text = _require_text(self.text, "text")
        if _PORTABLE_WORDING.search(text):
            raise ValueError("proposal text must not make portable claims.")
        if "machine" not in text.lower() or "workload" not in text.lower():
            raise ValueError(
                "proposal text must be machine- and workload-bounded."
            )
        plan = self.correctness_plan_reference
        if plan is not None:
            _require_text(plan, "correctness_plan_reference")
        if (
            self.category in _CORRECTNESS_CATEGORIES
            or _CORRECTNESS_TEXT.search(text)
        ) and not plan:
            raise ValueError(
                "guarded proposal requires a correctness plan reference."
            )


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A retained proposal emitted only from reconciled sufficient evidence."""

    decision: PerformanceDecision
    proposal: PerformanceProposal
    contribution: KernelContribution

    def __post_init__(self) -> None:
        """Validate that a recommendation retains reconciled ranked evidence.

        Raises:
            TypeError: If a decision, proposal, or contribution is invalid.
            ValueError: If the decision is not sufficient or does not retain
                the contribution and non-portability limitation.
        """
        if not isinstance(self.decision, PerformanceDecision):
            raise TypeError("decision must be PerformanceDecision.")
        if not isinstance(self.proposal, PerformanceProposal):
            raise TypeError("proposal must be PerformanceProposal.")
        if not isinstance(self.contribution, KernelContribution):
            raise TypeError("contribution must be KernelContribution.")
        if (
            self.decision.status != "reconciled"
            or self.decision.confidence != "sufficient"
        ):
            raise ValueError(
                "recommendations require reconciled sufficient evidence."
            )
        if (
            self.decision.reconciliation is None
            or self.contribution not in self.decision.contributions
        ):
            raise ValueError(
                "recommendation must retain ranked decision evidence."
            )
        if not any(
            "not portable" in item for item in self.decision.limitations
        ):
            raise ValueError(
                "recommendation requires a non-portability limitation."
            )
        required_limit_terms = (
            "workload",
            "CUDA machine",
            "software",
            "source",
            "metric",
            "artifacts",
        )
        limitation_text = " ".join(self.decision.limitations)
        if not all(
            term.lower() in limitation_text.lower()
            for term in required_limit_terms
        ):
            raise ValueError(
                "recommendation requires a complete bounded limitation."
            )


def build_machine_bounded_recommendation(
    decision: PerformanceDecision,
    proposal: PerformanceProposal,
) -> Recommendation:
    """Emit one guarded recommendation from the top retained contribution."""
    if not isinstance(decision, PerformanceDecision):
        raise TypeError("decision must be PerformanceDecision.")
    if not isinstance(proposal, PerformanceProposal):
        raise TypeError("proposal must be PerformanceProposal.")
    if not decision.contributions:
        raise ValueError("recommendation requires a ranked contribution.")
    return Recommendation(decision, proposal, decision.contributions[0])
