"""Host-only profiling evidence records for GPU test support.

This concrete test-support module defines evidence schemas and filesystem
provenance only.  It does not import Warp, probe hardware, collect timings, or
form part of the public :mod:`particula.gpu` API.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

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
    result = float(value)
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
    """Describe one fixed resident profiling workload."""

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
        """Validate a canonical workload configuration."""
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
    """Build the stable identifier from every nonidentifier workload field."""
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
    """Store bounded machine metadata without environment or payload data."""

    machine_id: str
    platform: str
    python_version: str
    cuda_version: str
    driver_version: str
    device: str
    source_revision: str

    def __post_init__(self) -> None:
        """Validate all bounded metadata fields."""
        for field in self.__dataclass_fields__:
            _require_text(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class MeasurementMethod:
    """Describe one noncombined source of duration evidence."""

    method_id: str
    source: str
    command: str
    version: str
    duration_unit: str

    def __post_init__(self) -> None:
        """Validate the closed method source and its canonical identifier."""
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
    """Store one absolute raw duration sample in nanoseconds."""

    replay_count: int
    duration_ns: int

    def __post_init__(self) -> None:
        """Validate a configured replay count and positive duration."""
        if (
            self.replay_count not in REPLAY_COUNTS
            or type(self.replay_count) is not int
        ):
            raise ValueError("replay_count must be a configured integer.")
        _require_int(self.duration_ns, "duration_ns", positive=True)


@dataclass(frozen=True, slots=True)
class NormalizedMetric:
    """Store one closed-vocabulary absolute profiling metric."""

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        """Validate metric vocabulary, unit, and nonnegative finite value."""
        if (
            self.name not in METRIC_UNITS
            or self.unit != METRIC_UNITS[self.name]
        ):
            raise ValueError("metric name and unit are not supported.")
        _require_number(self.value, "value")


@dataclass(frozen=True, slots=True)
class RawReportProvenance:
    """Store a contained raw-report filename, size, and streaming digest."""

    raw_filename: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        """Validate injected-root-safe report provenance fields."""
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
    """Store complete executed evidence without inferred aggregate values."""

    status: str
    workload: ProfilingWorkload
    machine: MachineProvenance
    method: MeasurementMethod
    raw_samples: tuple[RawDurationSample, ...]
    metrics: tuple[NormalizedMetric, ...]
    raw_reports: tuple[RawReportProvenance, ...]

    def __post_init__(self) -> None:
        """Validate complete, nonempty executed evidence."""
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
        )


def _validate_executed_sequences(
    raw_samples: tuple[RawDurationSample, ...],
    metrics: tuple[NormalizedMetric, ...],
    raw_reports: tuple[RawReportProvenance, ...],
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
    if not all(isinstance(item, RawDurationSample) for item in raw_samples):
        raise TypeError("raw_samples are invalid.")
    if not all(isinstance(item, NormalizedMetric) for item in metrics):
        raise TypeError("metrics are invalid.")
    if not all(isinstance(item, RawReportProvenance) for item in raw_reports):
        raise TypeError("raw_reports are invalid.")
    if len({item.name for item in metrics}) != len(metrics):
        raise ValueError("metric names must be unique.")
    if len({item.raw_filename for item in raw_reports}) != len(raw_reports):
        raise ValueError("raw filenames must be unique.")


@dataclass(frozen=True, slots=True)
class UnavailableEvidence:
    """Store an exact workload and deterministic unavailability reason."""

    status: str
    workload: ProfilingWorkload
    reason: str

    def __post_init__(self) -> None:
        """Validate a context-free unavailable evidence row."""
        if self.status != "unavailable" or not isinstance(
            self.workload, ProfilingWorkload
        ):
            raise ValueError("unavailable evidence is invalid.")
        _require_text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class ProfilingArtifact:
    """Store the ordered union of executed and unavailable evidence rows."""

    evidence: tuple[ExecutedEvidence | UnavailableEvidence, ...]

    def __post_init__(self) -> None:
        """Validate row types, uniqueness, and workload-matrix ordering."""
        if (
            not isinstance(self.evidence, tuple)
            or not self.evidence
            or len(self.evidence) > MAX_ARTIFACT_ROWS
        ):
            raise ValueError("evidence must be a bounded nonempty tuple.")
        executed: set[tuple[str, str, str]] = set()
        unavailable: set[str] = set()
        labels: list[str] = []
        for row in self.evidence:
            if isinstance(row, ExecutedEvidence):
                identity = (
                    row.workload.workload_id,
                    row.status,
                    row.method.method_id,
                )
                if identity in executed:
                    raise ValueError("duplicate executed evidence identity.")
                executed.add(identity)
            elif isinstance(row, UnavailableEvidence):
                if row.workload.workload_id in unavailable:
                    raise ValueError("duplicate unavailable workload.")
                unavailable.add(row.workload.workload_id)
            else:
                raise TypeError("evidence rows are invalid.")
            labels.append(row.workload.label)
        if labels != sorted(labels, key=("small", "medium").index):
            raise ValueError("evidence must follow workload ordering.")


def build_default_profiling_workload_matrix() -> tuple[ProfilingWorkload, ...]:
    """Build the fixed small and medium E8-F6 profiling workload matrix."""
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
                    label=label, shape=shape, **common
                ),
                label=label,
                shape=shape,
                **common,
            )
        )
    return tuple(rows)


def ensure_profiling_raw_root(artifact_root: str | Path) -> Path:
    """Create the contained raw-report root below an injected root."""
    root = Path(artifact_root)
    if root.name != ".artifacts" or not root.is_dir() or root.is_symlink():
        raise ValueError(
            "artifact_root must be an existing nonsymlink .artifacts."
        )
    raw_root = root / "benchmarks" / "profiling" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    if raw_root.is_symlink():
        raise ValueError("raw root must not be a symlink.")
    return raw_root.resolve(strict=True)


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
    ):
        raise ValueError("raw_filename must be one contained filename.")
    return raw_filename


def _raw_report_path(artifact_root: str | Path, raw_filename: object) -> Path:
    """Resolve a regular report beneath the explicit raw root."""
    filename = _validate_raw_filename(raw_filename)
    raw_root = ensure_profiling_raw_root(artifact_root)
    candidate = raw_root / filename
    if not candidate.is_file():
        raise ValueError("raw report must be a regular file.")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != raw_root or not resolved.is_file():
        raise ValueError("raw report escapes its raw root.")
    return resolved


def _hash_report(path: Path) -> tuple[int, str]:
    """Return bounded report size and a fixed-chunk SHA-256 digest."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_RAW_REPORT_BYTES:
        raise ValueError("raw report has an invalid byte size.")
    digest = hashlib.sha256()
    with path.open("rb") as report:
        for chunk in iter(
            lambda: report.read(RAW_REPORT_HASH_CHUNK_BYTES), b""
        ):
            digest.update(chunk)
    return size, digest.hexdigest()


def build_raw_report_provenance(
    artifact_root: str | Path, raw_filename: str
) -> RawReportProvenance:
    """Build safe provenance for one injected-root-contained raw report."""
    path = _raw_report_path(artifact_root, raw_filename)
    size, digest = _hash_report(path)
    return RawReportProvenance(raw_filename, size, digest)


def verify_raw_report_provenance(
    artifact_root: str | Path, provenance: RawReportProvenance
) -> None:
    """Rehash a raw report and fail closed if its provenance changed."""
    if not isinstance(provenance, RawReportProvenance):
        raise TypeError("provenance must be RawReportProvenance.")
    current = build_raw_report_provenance(
        artifact_root, provenance.raw_filename
    )
    if current != provenance:
        raise ValueError("raw report provenance no longer matches.")


def to_json_value(artifact: ProfilingArtifact) -> dict[str, object]:
    """Convert validated evidence into the exact JSON schema envelope value."""
    if not isinstance(artifact, ProfilingArtifact):
        raise TypeError("artifact must be a ProfilingArtifact.")
    return {
        "schema_version": PROFILING_SCHEMA_VERSION,
        "artifact": {"evidence": [asdict(row) for row in artifact.evidence]},
    }


def serialize_profiling_artifact(artifact: ProfilingArtifact) -> str:
    """Serialize evidence as compact canonical UTF-8-safe JSON text."""
    return json.dumps(
        to_json_value(artifact),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


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


def _workload_from_json(value: object) -> ProfilingWorkload:
    """Reconstruct one workload with its exact schema."""
    raw = _require_exact_keys(
        value, set(ProfilingWorkload.__dataclass_fields__), "workload"
    )
    return ProfilingWorkload(
        **raw
        | {
            "shape": tuple(raw["shape"]),
            "processes": tuple(raw["processes"]),
            "diagnostics": tuple(raw["diagnostics"]),
            "replay_counts": tuple(raw["replay_counts"]),
        }
    )


def deserialize_profiling_artifact(payload: str) -> ProfilingArtifact:
    """Deserialize current-version JSON through strict constructors."""
    if not isinstance(payload, str):
        raise TypeError("payload must be a string.")
    if len(payload.encode("utf-8")) > MAX_ARTIFACT_PAYLOAD_BYTES:
        raise ValueError("payload exceeds maximum byte size.")
    _validate_json_nesting(payload)
    try:
        envelope = json.loads(payload)
    except json.JSONDecodeError as error:
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
            raw = _require_exact_keys(
                row,
                set(ExecutedEvidence.__dataclass_fields__),
                "executed evidence",
            )
            machine = MachineProvenance(
                **_require_exact_keys(
                    raw["machine"],
                    set(MachineProvenance.__dataclass_fields__),
                    "machine",
                )
            )
            method = MeasurementMethod(
                **_require_exact_keys(
                    raw["method"],
                    set(MeasurementMethod.__dataclass_fields__),
                    "method",
                )
            )
            samples = tuple(
                RawDurationSample(
                    **_require_exact_keys(
                        item,
                        set(RawDurationSample.__dataclass_fields__),
                        "sample",
                    )
                )
                for item in raw["raw_samples"]
            )
            metrics = tuple(
                NormalizedMetric(
                    **_require_exact_keys(
                        item,
                        set(NormalizedMetric.__dataclass_fields__),
                        "metric",
                    )
                )
                for item in raw["metrics"]
            )
            reports = tuple(
                RawReportProvenance(
                    **_require_exact_keys(
                        item,
                        set(RawReportProvenance.__dataclass_fields__),
                        "report",
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
            raw = _require_exact_keys(
                row,
                set(UnavailableEvidence.__dataclass_fields__),
                "unavailable evidence",
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
