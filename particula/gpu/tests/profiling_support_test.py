"""Hardware-free tests for strict profiling evidence support."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from particula.gpu.tests import profiling_support as support


def _workloads() -> tuple[support.ProfilingWorkload, ...]:
    """Return the fixed local profiling matrix."""
    return support.build_default_profiling_workload_matrix()


def _machine() -> support.MachineProvenance:
    """Build valid synthetic machine provenance."""
    return support.MachineProvenance(
        "ci-machine", "linux", "3.12", "12.0", "550", "gpu-0", "abc123"
    )


def _method() -> support.MeasurementMethod:
    """Build one valid synthetic measurement method."""
    return support.MeasurementMethod(
        "profiler", "profiler", "tool --run", "1", "ns"
    )


def _executed() -> support.ExecutedEvidence:
    """Build a complete synthetic executed row."""
    return support.ExecutedEvidence(
        "executed",
        _workloads()[0],
        _machine(),
        _method(),
        tuple(
            support.RawDurationSample(replay_count, 10)
            for replay_count in support.REPLAY_COUNTS
            for _ in range(_workloads()[0].sample_count)
        ),
        (
            support.NormalizedMetric("profiler_gpu_duration", 10.0, "ns"),
            support.NormalizedMetric("profiler_gpu_memory", 20.0, "bytes"),
        ),
        (support.RawReportProvenance("trace.json", 5, "a" * 64),),
    )


def test_default_matrix_preserves_frozen_e8_f6_settings() -> None:
    """Test the exact two-row matrix and all shared settings."""
    workloads = _workloads()
    assert [item.label for item in workloads] == ["small", "medium"]
    assert [item.shape for item in workloads] == [(1, 16, 2), (1000, 16, 2)]
    assert [item.workload_id for item in workloads] == [
        item.workload_id for item in _workloads()
    ]
    for item in workloads:
        assert item.active_fraction == 1.0
        assert item.communication == "gas"
        assert item.diagnostics == ("gas", "saturation")
        assert item.warmup == 2
        assert item.sample_count == 3
        assert item.seed == 1582
        assert item.duration_seconds == 0.5
        assert item.replay_counts == (1, 10, 100, 1000)


def test_host_only_import_does_not_load_warp() -> None:
    """Test isolated profiling-support import leaves Warp unloaded."""
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import sys; import particula.gpu.tests.profiling_support; "
            "assert 'warp' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_analysis_records_reject_mutable_provenance_and_portable_proposals() -> (
    None
):
    """Test P4 analysis records retain immutable bounded inputs."""
    report = support.RawReportProvenance("trace.json", 5, "a" * 64)
    with pytest.raises(TypeError, match="raw_reports must be a tuple"):
        support.ArtifactReference(
            "captured_replay_host_launch.json",
            "one",
            [report],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="portable claims"):
        support.PerformanceProposal(
            "kernel", "portable improvement for this machine workload"
        )
    with pytest.raises(ValueError, match="correctness plan"):
        support.PerformanceProposal(
            "rng", "change bounded to this machine workload"
        )


def test_analysis_records_reject_wrong_type_modes() -> None:
    """Test all analysis records reject non-text modes without coercion."""
    host = _analysis_host_evidence("host_launch")
    kernel = _analysis_kernel_evidence("attributed")
    unavailable = support.EvidenceUnavailable(
        _workloads()[0], "captured_replay", _machine(), None, "unavailable"
    )
    decision = support.PerformanceDecision(
        "insufficient",
        "low",
        _workloads()[0],
        "captured_replay",
        None,
        (),
        None,
        ("limited",),
    )
    for factory in (
        lambda: support.HostEvidenceBinding(
            host.evidence,
            cast(str, 1),
            host.reference,
        ),
        lambda: support.MachineBoundKernelEvidence(
            kernel.evidence,
            1,  # type: ignore[arg-type]
            kernel.machine,
            kernel.reference,
        ),
        lambda: support.EvidenceUnavailable(
            unavailable.workload,
            1,  # type: ignore[arg-type]
            unavailable.machine,
            None,
            unavailable.reason,
        ),
        lambda: support.PerformanceDecision(
            decision.status,
            decision.confidence,
            decision.workload,
            1,  # type: ignore[arg-type]
            decision.machine,
            decision.contributions,
            decision.reconciliation,
            decision.limitations,
        ),
    ):
        with pytest.raises(TypeError, match="mode must be a string"):
            factory()


def test_analysis_rejects_duplicate_nsight_row_identity() -> None:
    """Test machine-bound evidence rejects repeated native row identities."""
    kernel = _analysis_kernel_evidence("attributed")
    duplicated = support.NsightEvidence(
        kernel.evidence.qualification,
        kernel.evidence.workload_id,
        kernel.evidence.process,
        (kernel.evidence.rows[0], kernel.evidence.rows[0]),
    )
    with pytest.raises(ValueError, match="row identities must be unique"):
        support.MachineBoundKernelEvidence(
            duplicated, "captured_replay", _machine(), kernel.reference
        )


def test_proposal_textual_numerical_tolerance_requires_correctness_plan() -> (
    None
):
    """Test numerical-tolerance wording is guarded regardless of category."""
    with pytest.raises(ValueError, match="correctness plan"):
        support.PerformanceProposal(
            "kernel", "adjust numerical tolerance for this machine workload"
        )


def test_analysis_unavailable_returns_no_fabricated_measurements() -> None:
    """Test unavailable inputs remain an explicit non-actionable decision."""
    unavailable = support.EvidenceUnavailable(
        _workloads()[0], "captured_replay", _machine(), None, "tool unavailable"
    )
    result = support.analyze_machine_bounded_performance(
        (unavailable,), (unavailable,)
    )
    assert result.status == "unavailable"
    assert result.confidence == "none"
    assert result.contributions == ()
    assert result.reconciliation is None


def test_artifact_round_trips_to_byte_identical_canonical_json() -> None:
    """Test the executed/unavailable union round-trips canonically."""
    unavailable = support.UnavailableEvidence(
        "unavailable", _workloads()[1], "native profiler unavailable"
    )
    artifact = support.ProfilingArtifact((_executed(), unavailable))
    serialized = support.serialize_profiling_artifact(artifact)
    assert (
        support.serialize_profiling_artifact(
            support.deserialize_profiling_artifact(serialized)
        )
        == serialized
    )


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        '{"schema_version":2,"artifact":{"evidence":[]}}',
        '{"schema_version":1,"artifact":{"rows":[]}}',
        "[" * (support.MAX_ARTIFACT_NESTING_DEPTH + 1),
    ],
)
def test_deserialization_rejects_invalid_envelopes(payload: str) -> None:
    """Test malformed, unsupported, and bounded JSON rejection."""
    with pytest.raises(ValueError):
        support.deserialize_profiling_artifact(payload)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: support.RawDurationSample(True, 1),
        lambda: support.RawDurationSample(1, 0),
        lambda: support.NormalizedMetric("profiler_gpu_duration", -1.0, "ns"),
        lambda: support.NormalizedMetric("percentage", 1.0, "%"),
        lambda: support.RawReportProvenance("report", True, "a" * 64),
        lambda: support.RawReportProvenance("report", 1, "A" * 64),
        lambda: support.MeasurementMethod(
            "profiler", "profiler", "run", "1", "ms"
        ),
    ],
)
def test_records_reject_invalid_scalar_and_closed_vocabularies(
    factory: object,
) -> None:
    """Test Boolean numerics and invalid closed-vocabulary record values."""
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_machine_provenance_rejects_path_like_metadata() -> None:
    """Test machine metadata cannot carry absolute or nested paths."""
    values = ["/etc/hostname", "device/name"]
    for value in values:
        with pytest.raises(ValueError):
            support.MachineProvenance(
                value, "linux", "3.12", "12.0", "550", "gpu-0", "abc123"
            )


def test_workloads_and_executed_rows_fail_closed() -> None:
    """Test invalid IDs, shapes, duplicate metrics, and missing samples."""
    workload = _workloads()[0]
    with pytest.raises(ValueError):
        support.ProfilingWorkload(
            "wrong",
            "small",
            (1, 16, 2),
            1.0,
            workload.processes,
            "gas",
            workload.diagnostics,
            2,
            3,
            1582,
            0.5,
            support.REPLAY_COUNTS,
        )
    with pytest.raises(ValueError):
        support.ProfilingWorkload(
            workload.workload_id,
            "small",
            (True, 16, 2),
            1.0,
            workload.processes,
            "gas",
            workload.diagnostics,
            2,
            3,
            1582,
            0.5,
            support.REPLAY_COUNTS,
        )
    metric = support.NormalizedMetric("profiler_gpu_duration", 1.0, "ns")
    with pytest.raises(ValueError):
        support.ExecutedEvidence(
            "executed",
            workload,
            _machine(),
            _method(),
            (),
            (metric,),
            (support.RawReportProvenance("trace", 1, "a" * 64),),
        )
    with pytest.raises(ValueError):
        support.ExecutedEvidence(
            "executed",
            workload,
            _machine(),
            _method(),
            _executed().raw_samples,
            (metric, metric),
            (support.RawReportProvenance("trace", 1, "a" * 64),),
        )


def test_fixed_rows_reject_altered_workloads_and_incomplete_evidence() -> None:
    """Test workload settings and replay samples remain frozen and complete."""
    workload = _workloads()[0]
    with pytest.raises(ValueError):
        support.ProfilingWorkload(
            "profiling-any",
            "small",
            workload.shape,
            0.5,
            workload.processes,
            workload.communication,
            workload.diagnostics,
            workload.warmup,
            workload.sample_count,
            workload.seed,
            workload.duration_seconds,
            workload.replay_counts,
        )
    with pytest.raises(ValueError):
        support.ExecutedEvidence(
            "executed",
            workload,
            _machine(),
            _method(),
            _executed().raw_samples[:-1],
            _executed().metrics,
            _executed().raw_reports,
        )
    with pytest.raises(ValueError):
        support.ProfilingArtifact((_executed(),))


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":1,"schema_version":1,"artifact":{"evidence":[]}}',
        '{"schema_version":1,"artifact":{"evidence":[],"evidence":[]}}',
    ],
)
def test_deserialization_rejects_duplicate_json_keys(payload: str) -> None:
    """Test duplicate JSON object keys fail before schema interpretation."""
    with pytest.raises(ValueError, match="valid JSON"):
        support.deserialize_profiling_artifact(payload)


def test_raw_provenance_is_contained_streamed_and_reverified(
    tmp_path: Path,
) -> None:
    """Test explicit root creation, digest validation, and changed report failure."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    raw_root = support.ensure_profiling_raw_root(root)
    report = raw_root / "trace.json"
    report.write_bytes(b"hello")
    provenance = support.build_raw_report_provenance(root, "trace.json")
    assert provenance.byte_size == 5
    assert provenance.sha256 == hashlib.sha256(b"hello").hexdigest()
    support.verify_raw_report_provenance(root, provenance)
    report.write_bytes(b"changed")
    with pytest.raises(ValueError):
        support.verify_raw_report_provenance(root, provenance)


@pytest.mark.parametrize(
    "filename", ["", ".", "..", "../trace", Path.cwd().anchor + "report", "a/b"]
)
def test_raw_provenance_rejects_unsafe_names(
    tmp_path: Path, filename: str
) -> None:
    """Test report provenance accepts only a single contained filename."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    with pytest.raises((TypeError, ValueError)):
        support.build_raw_report_provenance(root, filename)


def test_raw_provenance_rejects_oversized_and_external_symlink(
    tmp_path: Path,
) -> None:
    """Test raw-report bounds and symlink containment without hardware access."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    raw_root = support.ensure_profiling_raw_root(root)
    large = raw_root / "large"
    large.write_bytes(b"x" * (support.MAX_RAW_REPORT_BYTES + 1))
    with pytest.raises(ValueError):
        support.build_raw_report_provenance(root, "large")
    external = tmp_path / "external"
    external.write_bytes(b"x")
    try:
        (raw_root / "outside").symlink_to(external)
    except OSError:
        pytest.skip("platform does not permit symlink creation")
    with pytest.raises(ValueError):
        support.build_raw_report_provenance(root, "outside")


def test_raw_root_rejects_symlinked_parent(tmp_path: Path) -> None:
    """Test raw staging cannot escape through a symlinked directory parent."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (root / "benchmarks").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError):
        support.ensure_profiling_raw_root(root)


def test_unavailable_evidence_retains_requested_workload_only() -> None:
    """Test unavailable rows retain no fabricated measurement context."""
    unavailable = support.UnavailableEvidence(
        "unavailable", _workloads()[1], "native profiler unavailable"
    )
    assert unavailable.workload.shape == (1000, 16, 2)
    assert set(unavailable.__dataclass_fields__) == {
        "status",
        "workload",
        "reason",
    }
    with pytest.raises(ValueError):
        support.ProfilingArtifact((unavailable, _executed()))


def test_nsight_qualification_requires_exact_literal_banner() -> None:
    """Test a suffix or stderr prevents tool qualification."""
    calls: list[tuple[str, ...]] = []

    def fake_runner(
        command: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, "2026.1.3.425-1 extra\n", ""
        )

    result = support.qualify_nsight_tool("nsys", fake_runner)

    assert isinstance(result, support.NsightUnavailable)
    assert calls == [("nsys", "--version")]


@pytest.mark.parametrize("error", (PermissionError(), OSError("blocked")))
def test_nsight_qualification_normalizes_launch_os_errors(
    error: OSError,
) -> None:
    """Test launch failures become unavailable probe outcomes."""

    def failing_runner(
        command: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        raise error

    result = support.qualify_nsight_tool("nsys", failing_runner)

    assert result == support.NsightUnavailable("nsys", "binary not found")


@pytest.mark.parametrize("error", (PermissionError(), OSError("blocked")))
def test_command_outcome_normalizes_launch_os_errors(error: OSError) -> None:
    """Test collection-stage launch failures become unavailable outcomes."""

    def failing_runner(
        command: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        raise error

    outcome = support._run_outcome(
        "nsys", "collect", ("nsys", "profile"), failing_runner
    )

    assert outcome == support.NsightUnavailable(
        "nsys", "collect binary not found"
    )


def test_profile_runner_terminates_when_one_stream_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test streaming diagnostics terminate and reap an overflowing process."""

    class FakeStream:
        """Return one fixed payload and then EOF."""

        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def read(self, _: int) -> bytes:
            payload, self.payload = self.payload, b""
            return payload

    class FakeProcess:
        """Expose the minimal process API used by the streaming runner."""

        def __init__(self) -> None:
            self.stdout = FakeStream(b"ok")
            self.stderr = FakeStream(
                b"x" * (support.MAX_PROCESS_OUTPUT_BYTES + 1)
            )
            self.returncode = -15
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return self.returncode

        def kill(self) -> None:
            self.terminated = True

    process = FakeProcess()
    monkeypatch.setattr(
        support.subprocess, "Popen", lambda *args, **kwargs: process
    )

    with pytest.raises(ValueError, match="diagnostics exceed"):
        support.run_profile_command(("nsys", "--version"))

    assert process.terminated


def test_collect_nsight_evidence_uses_closed_commands_and_parser(
    tmp_path: Path,
) -> None:
    """Test collection has fixed stages, bounded commands, and parser handoff."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_runner(
        command: tuple[str, ...], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        if command == ("nsys", "--version"):
            return subprocess.CompletedProcess(
                command, 0, "2026.1.3.425-1\n", ""
            )
        if command[0] == "worker":
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ("nsys", "export"):
            return subprocess.CompletedProcess(
                command,
                0,
                "kernel_name,start_ns,duration_ns,correlation_id\nresident,1,5,1\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    # The fixed worker invocation begins with the Python interpreter, so map it
    # to a successful child process without allowing arbitrary arguments.
    result = support.collect_nsight_evidence(
        tool="nsys",
        artifact_root=root,
        report_filename="systems.csv",
        process_ids={1: "resident"},
        runner=fake_runner,
    )

    assert isinstance(result, support.NsightEvidence)
    assert result.rows[0].attribution == "attributed"
    assert calls[0] == (("nsys", "--version"), None)
    assert calls[1] == (support.WORKER_COMMAND, None)
    assert calls[2][1] == root / "benchmarks" / "profiling" / "raw"
    assert calls[3][1] == root / "benchmarks" / "profiling" / "raw"
    assert all(isinstance(command, tuple) for command, _ in calls)


def test_collect_ncu_evidence_uses_closed_metrics_and_parser(
    tmp_path: Path,
) -> None:
    """Test Compute collection exports one allow-listed attributed metric."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    def fake_runner(
        command: tuple[str, ...], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        if command == ("ncu", "--version"):
            return subprocess.CompletedProcess(command, 0, "2026.2.1.5-1\n", "")
        if command[:2] == ("ncu", "--import"):
            return subprocess.CompletedProcess(
                command,
                0,
                "kernel_name,invocations,metric_name,metric_value,unit,correlation_id\n"
                "resident,1,dram__throughput.avg,2.5,GB/s,1\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    result = support.collect_nsight_evidence(
        tool="ncu",
        artifact_root=root,
        report_filename="compute.csv",
        process_ids={1: "resident"},
        runner=fake_runner,
    )

    assert isinstance(result, support.NsightEvidence)
    assert result.rows[0].attribution == "attributed"
    assert result.rows[0].metric_name == "dram__throughput.avg"
    assert calls[0] == (("ncu", "--version"), None)
    assert calls[1] == (support.WORKER_COMMAND, None)
    assert calls[2][0][:2] == ("ncu", "--csv")
    assert calls[2][1] == root / "benchmarks" / "profiling" / "raw"
    assert calls[3] == (
        ("ncu", "--import", "compute.csv", "--csv"),
        root / "benchmarks" / "profiling" / "raw",
    )


def test_collect_nsight_normalizes_documented_worker_unavailable(
    tmp_path: Path,
) -> None:
    """Test the closed worker's unavailable status does not become a failure."""
    root = tmp_path / ".artifacts"
    root.mkdir()

    def fake_runner(
        command: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        if command == ("nsys", "--version"):
            return subprocess.CompletedProcess(
                command, 0, "2026.1.3.425-1\n", ""
            )
        return subprocess.CompletedProcess(
            command, 3, "PROFILING_WORKLOAD_UNAVAILABLE: no CUDA\n", ""
        )

    result = support.collect_nsight_evidence(
        tool="nsys",
        artifact_root=root,
        report_filename="systems.csv",
        process_ids={},
        runner=fake_runner,
    )

    assert result == support.NsightUnavailable(
        "worker", "PROFILING_WORKLOAD_UNAVAILABLE: no CUDA"
    )


def test_collect_nsight_rejects_unsafe_report_before_worker(
    tmp_path: Path,
) -> None:
    """Test report paths fail closed before a worker or profiler launches."""
    root = tmp_path / ".artifacts"
    root.mkdir()
    calls: list[tuple[str, ...]] = []

    def fake_runner(
        command: tuple[str, ...],
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "2026.1.3.425-1\n", "")

    with pytest.raises(ValueError):
        support.collect_nsight_evidence(
            tool="nsys",
            artifact_root=root,
            report_filename="../outside.csv",
            process_ids={},
            runner=fake_runner,
        )
    assert calls == []


def _analysis_host_evidence(method: str) -> support.HostEvidenceBinding:
    """Build valid captured-replay host evidence with 100 ns per replay."""
    report = support.RawReportProvenance("trace.json", 5, "a" * 64)
    metric_name = f"{method}_duration"
    evidence = support.ExecutedEvidence(
        "executed",
        _workloads()[0],
        _machine(),
        support.MeasurementMethod(method, method, "tool --run", "1", "ns"),
        tuple(
            support.RawDurationSample(replay_count, 100 * replay_count)
            for replay_count in support.REPLAY_COUNTS
            for _ in range(_workloads()[0].sample_count)
        ),
        (support.NormalizedMetric(metric_name, 100.0, "ns"),),
        (report,),
    )
    return support.HostEvidenceBinding(
        evidence,
        "captured_replay",
        support.ArtifactReference(
            f"captured_replay_{method}.json", "one", (report,)
        ),
    )


def _analysis_kernel_evidence(
    attribution: str,
) -> support.MachineBoundKernelEvidence:
    """Build one synthetic Systems duration row for analysis tests."""
    report = support.RawReportProvenance("systems.csv", 5, "b" * 64)
    row = support.NsightKernelRow(
        "nsys", "resident_kernel", 1, None, 100, "ns", 1, attribution, report
    )
    evidence = support.NsightEvidence(
        support.NsightToolQualification("nsys", support.NSYS_BANNER),
        _workloads()[0].workload_id,
        "resident",
        (row,),
    )
    return support.MachineBoundKernelEvidence(
        evidence,
        "captured_replay",
        _machine(),
        support.ArtifactReference("systems.json", "one", (report,)),
    )


def test_analysis_reconciles_explicit_evidence_and_ranks_recommendation() -> (
    None
):
    """Test analysis retains explicit matched evidence for one recommendation."""
    decision = support.analyze_machine_bounded_performance(
        (
            _analysis_host_evidence("host_launch"),
            _analysis_host_evidence("synchronized_elapsed"),
        ),
        (_analysis_kernel_evidence("attributed"),),
    )

    assert decision.status == "reconciled"
    assert decision.confidence == "sufficient"
    assert decision.reconciliation is not None
    assert decision.reconciliation.host_total_ns == 100.0
    assert decision.reconciliation.profiler_total_ns == 100.0
    assert [item.kernel_name for item in decision.contributions] == [
        "resident_kernel"
    ]
    recommendation = support.build_machine_bounded_recommendation(
        decision,
        support.PerformanceProposal(
            "kernel", "consider this change for the measured machine workload"
        ),
    )
    assert recommendation.contribution is decision.contributions[0]


def test_analysis_rejects_unattributed_kernel_evidence() -> None:
    """Test analysis does not reconcile profiler rows without attribution."""
    decision = support.analyze_machine_bounded_performance(
        (_analysis_host_evidence("synchronized_elapsed"),),
        (_analysis_kernel_evidence("unattributed"),),
    )

    assert decision.status == "insufficient"
    assert decision.confidence == "low"
    assert decision.contributions == ()
    assert decision.reconciliation is None
    assert decision.limitations == ("incomplete attribution",)
