"""Host-only tests for resident benchmark evidence support."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from particula.execution.tests import resident_benchmark_support
from particula.execution.tests.resident_benchmark_support import (
    MAX_ARTIFACT_CONTAINER_ITEMS,
    MAX_ARTIFACT_NESTING_DEPTH,
    MAX_ARTIFACT_PAYLOAD_BYTES,
    MAX_ARTIFACT_ROWS,
    MAX_RESIDENT_MEMORY_BYTES,
    MAX_TIMING_SAMPLES,
    MAX_WARMUP_SAMPLES,
    RESIDENT_BOX_COUNTS,
    RESIDENT_CAPTURE_COMPARISON_DESTINATION,
    RESIDENT_DIAGNOSTIC_OPERATIONS,
    ResidentBenchmarkArtifact,
    ResidentBenchmarkAvailability,
    ResidentBenchmarkCase,
    ResidentBenchmarkPreflight,
    ResidentBenchmarkResult,
    ResidentBenchmarkStatus,
    ResidentMemoryCategory,
    ResidentMemoryModel,
    build_default_resident_benchmark_matrix,
    build_resident_benchmark_case_id,
    build_resident_benchmark_metadata,
    build_resident_memory_model,
    checked_dense_array_bytes,
    collect_paired_device_timings,
    deserialize_resident_benchmark_artifact,
    preflight_resident_benchmark_case,
    project_checkpointed_tape_bytes,
    project_full_retention_tape_bytes,
    serialize_resident_benchmark_artifact,
    summarize_timing_samples,
    with_tape_projection,
    write_json_artifact,
    write_resident_capture_comparison_artifact,
)


def _case() -> ResidentBenchmarkCase:
    """Return a valid canonical host-only benchmark case."""
    return ResidentBenchmarkCase(
        case_id=build_resident_benchmark_case_id(
            requested_shape=(2, 8, 3),
            actual_shape=(1, 4, 2),
            active_fraction=0.5,
            processes=("condensation", "dilution"),
            communication="none",
            diagnostics=("gas",),
            warmup=2,
            timestep_count=5,
            seed=7,
        ),
        requested_shape=(2, 8, 3),
        actual_shape=(1, 4, 2),
        active_fraction=0.5,
        processes=("condensation", "dilution"),
        communication="none",
        diagnostics=("gas",),
        warmup=2,
        timestep_count=5,
        seed=7,
    )


def _metadata():
    """Return complete injected metadata without a device probe."""
    return build_resident_benchmark_metadata(
        timestamp_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        command="pytest resident",
        synchronization_method="explicit",
        warmup=2,
        timestep_count=5,
        seed=7,
        prepared_signature_digest="abc",
        warp_version={"status": "unavailable", "value": None},
        device={"status": "unavailable", "identity": None, "memory": None},
    )


def _artifact() -> ResidentBenchmarkArtifact:
    """Return an artifact with executed and non-executed evidence."""
    case = _case()
    samples = (3.0, 1.0, 2.0)
    return ResidentBenchmarkArtifact(
        metadata=_metadata(),
        cases=(case,),
        results=(
            ResidentBenchmarkResult(
                case_id=case.case_id,
                timing_mode="wall_clock",
                requested_shape=case.requested_shape,
                status=ResidentBenchmarkStatus.EXECUTED,
                reason=None,
                samples=samples,
                summary=summarize_timing_samples(samples),
                provenance={"run": "one"},
            ),
            ResidentBenchmarkResult(
                case_id=case.case_id,
                timing_mode=None,
                requested_shape=case.requested_shape,
                status=ResidentBenchmarkStatus.UNAVAILABLE,
                reason="no device",
                samples=(),
                summary=None,
                provenance={"run": "two"},
            ),
        ),
    )


def test_default_matrix_preserves_all_exact_box_first_requests() -> None:
    """Keep the four canonical rows and every requested capacity unchanged."""
    cases = build_default_resident_benchmark_matrix()

    assert RESIDENT_BOX_COUNTS == (1, 10, 100, 1000)
    assert (
        tuple(case.requested_shape[0] for case in cases) == RESIDENT_BOX_COUNTS
    )
    assert all(case.requested_shape == case.actual_shape for case in cases)
    assert cases[-1].requested_shape == (1000, 16, 2)
    assert all(case.active_fraction == 1.0 for case in cases)
    assert all(
        case.processes
        == (
            "communication",
            "condensation",
            "coagulation",
            "dilution",
            "wall_loss",
            "nucleation",
            "diagnostics",
        )
        for case in cases
    )
    assert all(case.communication == "gas" for case in cases)
    assert all(case.diagnostics == ("gas", "saturation") for case in cases)
    assert all(
        case.case_id.startswith(f"r{case.requested_shape[0]}x")
        for case in cases
    )


def test_preflight_classifies_budget_before_availability_probe() -> None:
    """Reject over-budget rows without a device probe, allocation, or timing."""
    case = build_default_resident_benchmark_matrix()[0]
    probes: list[str] = []

    outcome = preflight_resident_benchmark_case(
        case,
        budget_bytes=10,
        estimate_requested_bytes=lambda _: 11,
        availability=lambda: probes.append("probe"),
    )

    assert outcome.status is ResidentBenchmarkStatus.SKIPPED_BUDGET
    assert outcome.case.requested_shape == outcome.case.actual_shape
    assert outcome.reason
    assert probes == []

    outcome = preflight_resident_benchmark_case(
        case,
        budget_bytes=11,
        estimate_requested_bytes=lambda _: 11,
        availability=lambda: ResidentBenchmarkAvailability(True),
    )
    assert outcome.status is ResidentBenchmarkStatus.EXECUTED


@pytest.mark.parametrize("budget, estimate", [(0, 1), (1, 0), (1, True)])
def test_preflight_rejects_invalid_budget_or_estimate_before_probe(
    budget: object, estimate: object
) -> None:
    """Fail closed before availability for invalid dimensions or estimates."""
    probes: list[str] = []
    with pytest.raises((TypeError, ValueError)):
        preflight_resident_benchmark_case(
            build_default_resident_benchmark_matrix()[0],
            budget_bytes=budget,
            estimate_requested_bytes=lambda _: estimate,
            availability=lambda: probes.append("probe"),
        )
    assert probes == []


def test_preflight_emits_structured_unavailable_without_fallback() -> None:
    """Preserve exact shape when native CUDA capture is unavailable."""
    case = build_default_resident_benchmark_matrix()[1]
    outcome = preflight_resident_benchmark_case(
        case,
        budget_bytes=1,
        estimate_requested_bytes=lambda _: 1,
        availability=lambda: ResidentBenchmarkAvailability(False, "no CUDA"),
    )
    assert isinstance(outcome, ResidentBenchmarkPreflight)
    assert outcome.status is ResidentBenchmarkStatus.UNAVAILABLE
    assert outcome.reason == "no CUDA"
    assert outcome.case.requested_shape == (10, 16, 2)


def test_records_summaries_metadata_and_round_trip_are_deterministic():
    """Validate records and byte-stable serialization with provenance."""
    artifact = _artifact()
    summary = summarize_timing_samples((3.0, 1.0, 2.0, 4.0))
    assert (
        summary.count,
        summary.minimum,
        summary.median,
        summary.mean,
        summary.p95,
    ) == (
        4,
        1.0,
        2.5,
        2.5,
        4.0,
    )
    assert _metadata()["timestamp_utc"] == "2026-01-02T00:00:00Z"
    with pytest.raises(FrozenInstanceError):
        artifact.cases[0].seed = 8
    first = serialize_resident_benchmark_artifact(artifact)
    assert first.endswith("\n")
    assert json.loads(first)["schema_version"] == 2
    assert deserialize_resident_benchmark_artifact(first) == artifact
    assert (
        serialize_resident_benchmark_artifact(
            deserialize_resident_benchmark_artifact(first)
        )
        == first
    )
    isolated_import = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys; import particula; sys.modules.pop('numpy', None); "
            "import particula.execution.tests.resident_benchmark_support; "
            "assert {'warp', 'numpy', 'particula.execution.gpu_resources'}.isdisjoint(sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert isolated_import.returncode == 0, isolated_import.stderr


def test_metadata_contains_complete_injected_host_provenance():
    """Keep unavailable device provenance complete without probing a device."""
    metadata = _metadata()

    assert set(metadata) == {
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
    assert metadata["warp_version"] == {"status": "unavailable", "value": None}
    assert metadata["device"] == {
        "status": "unavailable",
        "identity": None,
        "memory": None,
    }
    with pytest.raises(TypeError):
        metadata["command"] = "changed"
    with pytest.raises(TypeError):
        metadata["device"]["status"] = "changed"


@pytest.mark.parametrize(
    "warp_version, device",
    [
        (
            {"status": "available", "value": "1.9"},
            {"status": "available", "identity": "cuda:0", "memory": 8},
        ),
        (
            {"status": "unavailable", "value": None},
            {"status": "unavailable", "identity": None, "memory": None},
        ),
        (
            {"status": "error", "value": None, "error": "import failed"},
            {
                "status": "error",
                "identity": None,
                "memory": None,
                "error": "probe failed",
            },
        ),
    ],
)
def test_metadata_accepts_exact_status_qualified_provenance(
    warp_version, device
):
    """Accept every documented status-qualified provenance schema."""
    metadata = build_resident_benchmark_metadata(
        timestamp_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
        command="pytest resident",
        synchronization_method="explicit",
        warmup=2,
        timestep_count=5,
        seed=7,
        prepared_signature_digest="abc",
        warp_version=warp_version,
        device=device,
    )
    assert metadata["warp_version"] == warp_version
    assert metadata["device"] == device


@pytest.mark.parametrize(
    "warp_version, device, error",
    [
        (
            {"status": "unknown", "value": None},
            {"status": "unavailable", "identity": None, "memory": None},
            ValueError,
        ),
        (
            {"status": "available", "value": None},
            {"status": "unavailable", "identity": None, "memory": None},
            TypeError,
        ),
        (
            {"status": "unavailable", "value": None, "extra": None},
            {"status": "unavailable", "identity": None, "memory": None},
            ValueError,
        ),
        (
            {"status": "unavailable", "value": None},
            {"status": "available", "identity": "cuda:0", "memory": True},
            TypeError,
        ),
        (
            {"status": "unavailable", "value": None},
            {"status": "error", "identity": None, "memory": None},
            ValueError,
        ),
    ],
)
def test_metadata_rejects_malformed_status_qualified_provenance(
    warp_version, device, error
):
    """Reject extra, missing, and mistyped provenance fields deterministically."""
    with pytest.raises(error):
        build_resident_benchmark_metadata(
            timestamp_utc=datetime(2026, 1, 2, tzinfo=timezone.utc),
            command="pytest resident",
            synchronization_method="explicit",
            warmup=2,
            timestep_count=5,
            seed=7,
            prepared_signature_digest="abc",
            warp_version=warp_version,
            device=device,
        )


def test_result_statuses_require_their_respective_evidence():
    """Validate executed and explicit non-executed result contracts."""
    case = _case()
    samples = (0.0, 2.0)
    summary = summarize_timing_samples(samples)

    for status in (
        ResidentBenchmarkStatus.UNAVAILABLE,
        ResidentBenchmarkStatus.SKIPPED_BUDGET,
    ):
        result = ResidentBenchmarkResult(
            case_id=case.case_id,
            timing_mode=None,
            requested_shape=case.requested_shape,
            status=status,
            reason="explicit host-only outcome",
            samples=(),
            summary=None,
            provenance={"status": status.value},
        )
        assert result.status is status

    with pytest.raises(ValueError, match="exactly match"):
        ResidentBenchmarkResult(
            case_id=case.case_id,
            timing_mode="wall_clock",
            requested_shape=case.requested_shape,
            status=ResidentBenchmarkStatus.EXECUTED,
            reason=None,
            samples=samples,
            summary=summarize_timing_samples((1.0, 2.0)),
            provenance={"run": "bad-summary"},
        )
    with pytest.raises(ValueError, match="nonempty reason"):
        ResidentBenchmarkResult(
            case_id=case.case_id,
            timing_mode=None,
            requested_shape=case.requested_shape,
            status=ResidentBenchmarkStatus.UNAVAILABLE,
            reason="",
            samples=(),
            summary=None,
            provenance={"run": "missing-reason"},
        )
    with pytest.raises(ValueError, match="cannot contain timing data"):
        ResidentBenchmarkResult(
            case_id=case.case_id,
            timing_mode=None,
            requested_shape=case.requested_shape,
            status=ResidentBenchmarkStatus.SKIPPED_BUDGET,
            reason="budget",
            samples=(),
            summary=summary,
            provenance={"run": "timing-data"},
        )


@pytest.mark.parametrize(
    "samples, error",
    [
        ((), ValueError),
        ((-1.0,), ValueError),
        ((float("nan"),), ValueError),
        ((1.0,) * (MAX_TIMING_SAMPLES + 1), ValueError),
        ([1.0], TypeError),
    ],
)
def test_timing_samples_reject_invalid_values(samples, error):
    """Reject invalid sample containers and values before work."""
    with pytest.raises(error):
        summarize_timing_samples(samples)


def test_timing_summary_uses_nearest_rank_at_boundary_counts():
    """Use the documented nearest-rank p95 for one and twenty samples."""
    assert summarize_timing_samples((4.0,)).p95 == 4.0
    summary = summarize_timing_samples(
        tuple(float(value) for value in range(20))
    )
    assert summary.p95 == 18.0


def test_schema_rejects_noncanonical_and_invalid_result_references():
    """Reject invalid dimensions, case IDs, and artifact result references."""
    with pytest.raises(TypeError, match="requested_shape"):
        ResidentBenchmarkCase(
            "bad",
            (True, 2, 3),
            (1, 1, 1),
            0.0,
            ("condensation",),
            "none",
            (),
            0,
            1,
            0,
        )


def test_canonical_case_id_uses_injective_diagnostic_encoding():
    """Keep distinct hyphen-containing diagnostic selections distinct."""
    common = {
        "requested_shape": (1, 1, 1),
        "actual_shape": (1, 1, 1),
        "active_fraction": 1.0,
        "processes": ("condensation",),
        "communication": "none",
        "warmup": 0,
        "timestep_count": 1,
        "seed": 0,
    }
    first = build_resident_benchmark_case_id(diagnostics=("a-b", "c"), **common)
    second = build_resident_benchmark_case_id(
        diagnostics=("a", "b-c"), **common
    )
    assert first != second
    with pytest.raises(ValueError, match="canonical"):
        ResidentBenchmarkCase(
            "bad",
            (2, 2, 2),
            (1, 1, 1),
            0.0,
            ("dilution", "condensation"),
            "none",
            (),
            0,
            1,
            0,
        )
    case = _case()
    result = ResidentBenchmarkResult(
        case_id="unknown",
        timing_mode=None,
        requested_shape=case.requested_shape,
        status=ResidentBenchmarkStatus.SKIPPED_BUDGET,
        reason="limit",
        samples=(),
        summary=None,
        provenance={"source": "test"},
    )
    with pytest.raises(ValueError, match="unknown"):
        ResidentBenchmarkArtifact(_metadata(), (case,), (result,))

    with pytest.raises(ValueError, match="metadata has invalid fields"):
        ResidentBenchmarkArtifact({"command": "missing fields"}, (case,), ())
    with pytest.raises(ValueError, match="must not exceed"):
        ResidentBenchmarkCase(
            "bad",
            (2, 2, 2),
            (3, 1, 1),
            0.0,
            ("condensation",),
            "none",
            (),
            0,
            1,
            0,
        )


def test_artifact_rejects_duplicate_rows_and_malformed_deserialization():
    """Route duplicate and malformed JSON rows through normal validation."""
    artifact = _artifact()
    executed = artifact.results[0]
    with pytest.raises(ValueError, match="unique case_id/timing_mode"):
        ResidentBenchmarkArtifact(
            artifact.metadata,
            artifact.cases,
            (executed, executed),
        )

    payload = json.loads(serialize_resident_benchmark_artifact(artifact))
    payload["artifact"]["cases"][0]["actual_shape"] = [3, 4, 2]
    with pytest.raises(ValueError, match="must not exceed"):
        deserialize_resident_benchmark_artifact(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid fields"):
        deserialize_resident_benchmark_artifact('{"schema_version": 1}')


def test_artifact_deserialization_enforces_bounded_untrusted_input():
    """Reject excessive bytes, depth, rows, and nested metadata containers."""
    artifact = json.loads(serialize_resident_benchmark_artifact(_artifact()))
    boundary = json.dumps(artifact)
    assert deserialize_resident_benchmark_artifact(boundary)
    with pytest.raises(ValueError, match="byte size"):
        deserialize_resident_benchmark_artifact(
            " " * (MAX_ARTIFACT_PAYLOAD_BYTES + 1)
        )
    with pytest.raises(ValueError, match="nesting depth"):
        deserialize_resident_benchmark_artifact(
            "[" * (MAX_ARTIFACT_NESTING_DEPTH + 1)
        )
    artifact["artifact"]["cases"] = [{}] * (MAX_ARTIFACT_ROWS + 1)
    with pytest.raises(ValueError, match="row count"):
        deserialize_resident_benchmark_artifact(json.dumps(artifact))
    artifact = json.loads(serialize_resident_benchmark_artifact(_artifact()))
    artifact["artifact"]["metadata"]["nested"] = list(
        range(MAX_ARTIFACT_CONTAINER_ITEMS + 1)
    )
    with pytest.raises(ValueError, match="item count"):
        deserialize_resident_benchmark_artifact(json.dumps(artifact))


def test_write_json_artifact_is_contained_deterministic_and_atomic(
    tmp_path: Path,
):
    """Write generic normalized JSON only under a trusted artifact root."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    destination = write_json_artifact(
        root, "nested/result.json", {"b": 2, "a": 1}
    )
    assert destination == root / "nested/result.json"
    assert destination.read_text() == '{\n  "a": 1,\n  "b": 2\n}\n'
    write_json_artifact(root, "nested/result.json", {"a": 3})
    assert destination.read_text() == '{\n  "a": 3\n}\n'
    with pytest.raises(OSError, match="validation"):
        write_json_artifact(root, "../outside.json", {"a": 1})
    with pytest.raises(OSError, match="serialization"):
        write_json_artifact(root, "new/value.json", {"bad": object()})
    assert not (root / "new").exists()


def test_write_json_artifact_preserves_completed_file_on_write_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Retain completed bytes and clean temporary files after write failures."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    destination = write_json_artifact(root, "result.json", {"version": 1})
    original = destination.read_bytes()
    native_fsync = os.fsync

    def raise_fsync(_: int) -> None:
        """Simulate a failed durable temporary-file write."""
        raise OSError("fsync failed")

    monkeypatch.setattr(resident_benchmark_support.os, "fsync", raise_fsync)
    with pytest.raises(OSError, match="(validation|write)"):
        write_json_artifact(root, "result.json", {"version": 2})
    assert destination.read_bytes() == original
    assert not list(root.glob(".resident-benchmark-*"))

    monkeypatch.setattr(resident_benchmark_support.os, "fsync", native_fsync)

    def raise_replace(*_: object, **__: object) -> None:
        """Simulate replacement failure after a complete temporary write."""
        raise OSError("replace failed")

    monkeypatch.setattr(resident_benchmark_support.os, "replace", raise_replace)
    with pytest.raises(OSError, match="artifact write"):
        write_json_artifact(root, "result.json", {"version": 3})
    assert destination.read_bytes() == original
    assert not list(root.glob(".resident-benchmark-*"))


def test_write_json_artifact_rejects_symlink_escapes(tmp_path: Path):
    """Reject untrusted root, parent, and leaf symlinks before writing."""
    root = tmp_path / ".artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError, match="escapes"):
        write_json_artifact(root, "linked/value.json", {"safe": True})
    assert not (outside / "value.json").exists()
    root_link = tmp_path / "root-link" / ".artifacts"
    root_link.parent.mkdir()
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError, match="validation"):
        write_json_artifact(root_link, "value.json", {"safe": True})

    external_file = outside / "existing.json"
    external_file.write_text("outside\n")
    (root / "leaf.json").symlink_to(external_file)
    with pytest.raises(OSError, match="escapes"):
        write_json_artifact(root, "leaf.json", {"safe": True})
    assert external_file.read_text() == "outside\n"


def test_write_json_artifact_rejects_invalid_roots_and_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Reject invalid roots and retain failure context when cleanup also fails."""
    with pytest.raises(OSError, match="validation"):
        write_json_artifact(
            tmp_path / ".artifacts", "result.json", {"safe": True}
        )

    wrong_root = tmp_path / "artifacts"
    wrong_root.mkdir()
    with pytest.raises(OSError, match="validation"):
        write_json_artifact(wrong_root, "result.json", {"safe": True})

    root = tmp_path / ".artifacts"
    root.mkdir()
    destination = write_json_artifact(root, "result.json", {"version": 1})
    original = destination.read_bytes()

    def raise_replace(*_: object, **__: object) -> None:
        """Simulate replacement failure after writing a temporary file."""
        raise OSError("replace failed")

    def raise_unlink(*_: object, **__: object) -> None:
        """Simulate temporary cleanup failure."""
        raise OSError("unlink failed")

    monkeypatch.setattr(resident_benchmark_support.os, "replace", raise_replace)
    monkeypatch.setattr(resident_benchmark_support.os, "unlink", raise_unlink)
    with pytest.raises(OSError, match="cleanup also failed"):
        write_json_artifact(root, "result.json", {"version": 2})
    assert destination.read_bytes() == original

    monkeypatch.undo()
    for temporary in root.glob(".resident-benchmark-*"):
        temporary.unlink()


def test_write_json_artifact_rejects_directory_swap_to_external_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Keep descriptor-relative writes inside root during a directory swap."""
    root = tmp_path / ".artifacts"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    native_mkdir = os.mkdir

    def swap_after_mkdir(
        path: str, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        """Replace the newly made destination directory before it is reopened."""
        native_mkdir(path, mode, dir_fd=dir_fd)
        if path == "nested":
            nested = root / path
            nested.rename(root / "nested-original")
            nested.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(
        resident_benchmark_support.os, "mkdir", swap_after_mkdir
    )
    with pytest.raises(OSError, match="escapes"):
        write_json_artifact(root, "nested/result.json", {"safe": True})
    assert not (outside / "result.json").exists()


def test_paired_collector_alternates_without_warmup_synchronization() -> None:
    """Measure paired operations with exactly one sync per measured path."""
    calls: list[str] = []
    ticks = iter((0.0, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 14.0))
    uncaptured, replay = collect_paired_device_timings(
        uncaptured_operation=lambda: calls.append("uncaptured"),
        replay_operation=lambda: calls.append("replay"),
        synchronize=lambda: calls.append("sync"),
        clock=lambda: next(ticks),
        warmup_count=1,
        sample_count=2,
    )
    assert uncaptured == (1.0, 3.0)
    assert replay == (2.0, 4.0)
    assert calls == [
        "uncaptured",
        "replay",
        "uncaptured",
        "sync",
        "replay",
        "sync",
        "uncaptured",
        "sync",
        "replay",
        "sync",
    ]


@pytest.mark.parametrize(
    "warmup_count,sample_count",
    [
        (MAX_WARMUP_SAMPLES + 1, 1),
        (0, MAX_TIMING_SAMPLES + 1),
        (0, 0),
    ],
)
def test_paired_collector_rejects_counts_before_callbacks(
    warmup_count: int, sample_count: int
) -> None:
    """Reject bounded count failures before an operation, clock, or sync call."""
    calls: list[str] = []
    with pytest.raises(ValueError):
        collect_paired_device_timings(
            uncaptured_operation=lambda: calls.append("uncaptured"),
            replay_operation=lambda: calls.append("replay"),
            synchronize=lambda: calls.append("sync"),
            clock=lambda: calls.append("clock"),
            warmup_count=warmup_count,
            sample_count=sample_count,
        )
    assert calls == []


def test_comparison_writer_emits_one_fixed_schema_envelope(
    tmp_path: Path,
) -> None:
    """Persist only the fixed resident artifact without generic output state."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    artifact = _artifact()
    destination = write_resident_capture_comparison_artifact(root, artifact)
    assert destination == root / RESIDENT_CAPTURE_COMPARISON_DESTINATION
    assert (
        deserialize_resident_benchmark_artifact(destination.read_text())
        == artifact
    )
    assert list(root.rglob("*.json")) == [destination]


def test_schema_v1_decode_populates_absent_timing_provenance_with_none() -> (
    None
):
    """Retain backwards decoding while v2 remains the only emitted schema."""
    payload = json.loads(serialize_resident_benchmark_artifact(_artifact()))
    payload["schema_version"] = 1
    for result in payload["artifact"]["results"]:
        result.pop("setup_elapsed_seconds")
        result.pop("capture_elapsed_seconds")
    decoded = deserialize_resident_benchmark_artifact(json.dumps(payload))
    assert all(
        result.setup_elapsed_seconds is None for result in decoded.results
    )
    assert all(
        result.capture_elapsed_seconds is None for result in decoded.results
    )


def _memory_model() -> ResidentMemoryModel:
    """Return a representative host-only resident-memory model."""
    return build_resident_memory_model(
        n_boxes=2,
        n_particles=3,
        n_species=4,
        active_slots_per_box=1,
        registry_logical_byte_count=101,
        diagnostics=(
            "gas_concentration_snapshot",
            "particle_number_concentration",
        ),
        communication="gas",
        checkpoint_sidecar_copy_bytes=11,
        checkpoint_inspection_copy_bytes=13,
    )


def test_memory_model_accounts_primary_diagnostics_and_scenarios() -> None:
    """Account exact primary fields without double-counting attribution."""
    model = _memory_model()
    values = {item.name: item.byte_count for item in model.categories}
    primary = {
        "primary.particles.masses": 2 * 3 * 4 * 8,
        "primary.particles.concentration": 2 * 3 * 8,
        "primary.particles.charge": 2 * 3 * 8,
        "primary.particles.density": 4 * 8,
        "primary.particles.volume": 2 * 8,
        "primary.gas.molar_mass": 4 * 8,
        "primary.gas.concentration": 2 * 4 * 8,
        "primary.gas.vapor_pressure": 2 * 4 * 8,
        "primary.gas.partitioning": 2 * 4 * 4,
        "primary.environment.temperature": 2 * 8,
        "primary.environment.pressure": 2 * 8,
        "primary.environment.saturation_ratio": 2 * 4 * 8,
    }
    assert {key: values[key] for key in primary} == primary
    assert model.steady_state_bytes == sum(primary.values()) + 101 + 64 + 16
    assert values["inactive_particle_capacity_attribution"] == 2 * 2 * (
        4 * 8 + 16
    )
    assert values["communication.gas"] == 0
    assert model.checkpoint_bytes == sum(primary.values()) + 11 + 13
    assert model.inactive_particle_capacity_bytes == 2 * 2 * (4 * 8 + 16)
    assert [item.name for item in model.categories].count(
        "registry.resource_manifest"
    ) == 1


@pytest.mark.parametrize(
    "shape, itemsize, expected",
    [((2, 3), 8, 48), ((0, 3), 8, 0), ((), 8, 8)],
)
def test_checked_dense_array_bytes(shape, itemsize, expected) -> None:
    """Accept valid exact tuple shapes including zero extents."""
    assert checked_dense_array_bytes(shape, itemsize) == expected


@pytest.mark.parametrize(
    "shape, itemsize",
    [
        ([1], 8),
        ((True,), 8),
        ((-1,), 8),
        ((0, True), 8),
        ((0, MAX_RESIDENT_MEMORY_BYTES + 1), 8),
        ((1,), True),
        ((1,), 0),
        ((MAX_RESIDENT_MEMORY_BYTES, 2), 1),
    ],
)
def test_checked_dense_array_bytes_rejects_invalid_or_overflow(
    shape, itemsize
) -> None:
    """Reject non-exact, invalid, and over-limit dense schemas."""
    with pytest.raises((TypeError, ValueError)):
        checked_dense_array_bytes(shape, itemsize)


@pytest.mark.parametrize(
    "diagnostics",
    [
        (),
        *[(item,) for item in RESIDENT_DIAGNOSTIC_OPERATIONS],
        RESIDENT_DIAGNOSTIC_OPERATIONS,
    ],
)
def test_memory_model_accepts_canonical_diagnostic_subsets(diagnostics) -> None:
    """Size scalar and matrix diagnostic outputs in canonical order only."""
    model = build_resident_memory_model(
        n_boxes=2,
        n_particles=0,
        n_species=3,
        active_slots_per_box=0,
        registry_logical_byte_count=0,
        diagnostics=diagnostics,
        communication="none",
        checkpoint_sidecar_copy_bytes=0,
        checkpoint_inspection_copy_bytes=0,
    )
    values = {item.name: item.byte_count for item in model.categories}
    for operation in diagnostics:
        expected = 16 if operation == "particle_number_concentration" else 48
        assert values[f"diagnostic.{operation}"] == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diagnostics": ["gas_concentration_snapshot"]},
        {"diagnostics": ("unknown",)},
        {
            "diagnostics": (
                "saturation_ratio_snapshot",
                "gas_concentration_snapshot",
            )
        },
        {"communication": object()},
        {"n_boxes": True},
        {"active_slots_per_box": 4},
        {"registry_logical_byte_count": MAX_RESIDENT_MEMORY_BYTES + 1},
    ],
)
def test_memory_model_rejects_invalid_public_inputs(kwargs) -> None:
    """Reject invalid memory-model inputs before returning a model."""
    arguments = dict(
        n_boxes=2,
        n_particles=3,
        n_species=1,
        active_slots_per_box=1,
        registry_logical_byte_count=0,
        diagnostics=(),
        communication="none",
        checkpoint_sidecar_copy_bytes=0,
        checkpoint_inspection_copy_bytes=0,
    )
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        build_resident_memory_model(**arguments)


def test_memory_records_enforce_reconciliation_and_immutability() -> None:
    """Require unique categories and a sole nonadditive inactive attribution."""
    category = ResidentMemoryCategory(
        "inactive_particle_capacity_attribution",
        0,
        "analytical",
        False,
        "steady_state",
    )
    model = ResidentMemoryModel((category,))
    assert model.categories == (category,)
    with pytest.raises(FrozenInstanceError):
        category.name = "changed"
    with pytest.raises(ValueError):
        ResidentMemoryModel((category, category))
    with pytest.raises(ValueError):
        ResidentMemoryModel(())
    with pytest.raises(ValueError):
        ResidentMemoryCategory(
            "checkpoint.bad", 0, "analytical", True, "checkpoint"
        )


@pytest.mark.parametrize("communication", ("none", "gas", "particles"))
def test_memory_model_records_each_communication_selection_without_bytes(
    communication: str,
) -> None:
    """Keep each selected communication alternative visible but nonadditive."""
    model = build_resident_memory_model(
        n_boxes=0,
        n_particles=0,
        n_species=0,
        active_slots_per_box=0,
        registry_logical_byte_count=0,
        diagnostics=(),
        communication=communication,
        checkpoint_sidecar_copy_bytes=0,
        checkpoint_inspection_copy_bytes=0,
    )

    selection = next(
        item
        for item in model.categories
        if item.name == f"communication.{communication}"
    )
    assert selection.byte_count == 0
    assert not selection.included_in_steady_state
    assert model.steady_state_bytes == 0
    assert model.checkpoint_bytes == 0


@pytest.mark.parametrize(
    "diagnostics, communication",
    [
        (("gas_concentration_snapshot", "gas_concentration_snapshot"), "none"),
        (("gas_concentration_snapshot", 1), "none"),
        (("gas_concentration_snapshot",), "invalid"),
    ],
)
def test_memory_model_rejects_duplicate_and_invalid_selections(
    diagnostics: tuple[object, ...], communication: object
) -> None:
    """Reject duplicate diagnostic and unsupported communication selections."""
    with pytest.raises((TypeError, ValueError)):
        build_resident_memory_model(
            n_boxes=1,
            n_particles=1,
            n_species=1,
            active_slots_per_box=0,
            registry_logical_byte_count=0,
            diagnostics=diagnostics,
            communication=communication,
            checkpoint_sidecar_copy_bytes=0,
            checkpoint_inspection_copy_bytes=0,
        )


def test_memory_model_retains_zero_dimension_categories_and_full_activity() -> (
    None
):
    """Retain zero-byte fields and a zero inactive attribution at boundaries."""
    model = build_resident_memory_model(
        n_boxes=0,
        n_particles=4,
        n_species=0,
        active_slots_per_box=4,
        registry_logical_byte_count=0,
        diagnostics=RESIDENT_DIAGNOSTIC_OPERATIONS,
        communication="none",
        checkpoint_sidecar_copy_bytes=0,
        checkpoint_inspection_copy_bytes=0,
    )

    values = {item.name: item.byte_count for item in model.categories}
    assert values["inactive_particle_capacity_attribution"] == 0
    assert values["primary.particles.masses"] == 0
    assert all(
        values[f"diagnostic.{name}"] == 0
        for name in RESIDENT_DIAGNOSTIC_OPERATIONS
    )


def test_memory_model_requires_one_inactive_attribution() -> None:
    """Reject models that omit the required nonadditive inactive record."""
    model = _memory_model()
    categories = model.categories
    inactive_index = next(
        index
        for index, category in enumerate(categories)
        if category.name == "inactive_particle_capacity_attribution"
    )
    without_inactive = (
        categories[:inactive_index] + categories[inactive_index + 1 :]
    )
    with pytest.raises(ValueError):
        ResidentMemoryModel(without_inactive)

    with pytest.raises(ValueError):
        ResidentMemoryModel(categories, "unexpected overhead")


def test_tape_projections_are_checked_and_nonadditive() -> None:
    """Keep tape projections separate from resident and checkpoint totals."""
    model = _memory_model()
    assert project_full_retention_tape_bytes(3, 4) == 12
    assert project_full_retention_tape_bytes(0, 4) == 0
    assert project_full_retention_tape_bytes(3, 0) == 0
    assert project_checkpointed_tape_bytes(5, 4, 10, 2) == 38
    assert project_checkpointed_tape_bytes(0, 0, 0, 2) == 0
    projected = with_tape_projection(model, 19)
    assert projected.tape_bytes == 19
    assert projected.steady_state_bytes == model.steady_state_bytes
    assert projected.checkpoint_bytes == model.checkpoint_bytes
    with pytest.raises(ValueError):
        with_tape_projection(projected, 1)
    with pytest.raises((TypeError, ValueError)):
        project_checkpointed_tape_bytes(1, 1, 1, 0)


@pytest.mark.parametrize(
    "timesteps, state_bytes, checkpoint_bytes, interval, expected",
    [
        (4, 3, 10, 2, 26),
        (5, 4, 10, 2, 38),
        (0, 4, 10, 2, 8),
        (5, 0, 10, 2, 30),
        (5, 4, 0, 2, 8),
    ],
)
def test_checkpointed_tape_projection_uses_ceiling_count(
    timesteps: int,
    state_bytes: int,
    checkpoint_bytes: int,
    interval: int,
    expected: int,
) -> None:
    """Keep checkpoint and retained-window tape terms independently checked."""
    assert (
        project_checkpointed_tape_bytes(
            timesteps, state_bytes, checkpoint_bytes, interval
        )
        == expected
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (True, 1),
        (-1, 1),
        (1, -1),
        (MAX_RESIDENT_MEMORY_BYTES, 2),
    ],
)
def test_full_retention_tape_projection_rejects_invalid_or_overflow(
    arguments: tuple[object, object],
) -> None:
    """Reject invalid full-retention inputs before returning a byte count."""
    with pytest.raises((TypeError, ValueError)):
        project_full_retention_tape_bytes(*arguments)
