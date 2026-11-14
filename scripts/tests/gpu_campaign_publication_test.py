"""Keep incomplete or inconsistent timing evidence out of published results."""

import json
from dataclasses import asdict

import pytest
from scripts import publish_gpu_campaign as publication
from scripts.gpu_studies.cases import StudyCase


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Build a tiny complete campaign with exact comparable continuations."""
    case = StudyCase("brownian-128", "brownian", 1, 128)
    modes = ("direct", "prepared", "captured", "legacy")
    monkeypatch.setattr(
        publication, "all_jobs", lambda: tuple((case, m) for m in modes)
    )
    monkeypatch.setattr(publication, "executable_hash", lambda: "current")
    manifest = {
        "qualified": True,
        "started_at": "2026-11-13T00:00:00+00:00",
        "jobs": {},
    }
    for mode in modes:
        key = f"brownian-128-{mode}"
        manifest["jobs"][key] = {"status": "passed"}
        report = {
            "qualified": True,
            "executable_sha256": "current",
            "case": asdict(case),
            "mode": mode,
            "warmups": 3,
            "samples": 7,
            "input_sha256": "same-input",
            "result_hash": "same-result",
            "timings": {
                stage: {
                    "samples_s": [0.1] * 7,
                    "median_s": 0.1,
                    "minimum_s": 0.1,
                    "maximum_s": 0.1,
                }
                for stage in (
                    ("resident", "download")
                    if mode == "legacy"
                    else (
                        "setup",
                        "resident",
                        "download",
                        "end_to_end",
                        "capture",
                    )
                )
            },
            "memory": {"mempool_peak_bytes": 1024, "method": "pool peak"},
            "gpu_name": "Test CUDA GPU",
            "warp": "1.11.1",
            "cpu_model": "Test CPU",
            "source_revision": "current",
            "science": {"large_unused_array": [1] * 1000},
            "new_future_state_field": [2] * 1000,
        }
        (tmp_path / f"{key}.json").write_text(json.dumps(report))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def mutate(directory, name, field, value):
    """Change one saved observation as an external corruption would."""
    path = directory / name
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))


def test_complete_evidence_keeps_only_plotted_values(campaign):
    """Validate repetitions locally but publish only displayed numeric summaries."""
    destination = campaign / "public.json"
    result = publication.export_campaign(campaign, destination)
    assert result["job_count"] == 4
    assert result["cross_mode_cuda_exact"] is True
    assert result["schema_version"] == 2
    assert result["reports"][0] == {
        "case": {"name": "brownian-128", "steps": 3, "slots": 128},
        "mode": "direct",
        "timings": {
            "resident": {"median_s": 0.1, "minimum_s": 0.1, "maximum_s": 0.1}
        },
    }
    encoded = destination.read_text()
    for omitted in (
        "samples_s",
        "science",
        "new_future_state_field",
        "result_hash",
    ):
        assert omitted not in encoded
    assert json.loads(destination.read_text()) == result


@pytest.mark.parametrize(
    ("file", "field", "value", "message"),
    [
        ("manifest.json", "qualified", False, "incomplete"),
        ("brownian-128-captured.json", "executable_sha256", "old", "stale"),
        (
            "brownian-128-captured.json",
            "result_hash",
            "different",
            "continuation",
        ),
        (
            "brownian-128-captured.json",
            "input_sha256",
            "different",
            "initial states",
        ),
        ("brownian-128-captured.json", "samples", 6, "repetitions"),
        (
            "brownian-128-captured.json",
            "timings",
            {
                stage: {"samples_s": [0.1] * (6 if stage == "resident" else 7)}
                for stage in (
                    "setup",
                    "resident",
                    "download",
                    "end_to_end",
                    "capture",
                )
            },
            "timing samples",
        ),
    ],
)
def test_bad_evidence_never_replaces_publication(
    campaign, file, field, value, message
):
    """Reject before writing so existing published evidence remains intact."""
    destination = campaign / "public.json"
    destination.write_text("original publication")
    mutate(campaign, file, field, value)
    with pytest.raises(ValueError, match=message):
        publication.export_campaign(campaign, destination)
    assert destination.read_text() == "original publication"


def test_unplotted_workloads_are_not_exported():
    """Scientific histories and unused timing cases remain local artifacts."""
    for name, mode in (
        ("heterogeneous-evolution", "evolution"),
        ("full-million", "prepared"),
        ("species-condensation-25", "direct"),
    ):
        assert (
            publication.plot_row({"case": {"name": name}, "mode": mode}) is None
        )


@pytest.mark.parametrize(
    "change", [None, "missing-case", "old-source", "failed"]
)
def test_cpu_ratios_require_complete_array_parity(tmp_path, change):
    """CPU performance cannot publish without matching physical-array checks."""
    case = asdict(StudyCase("condensation", "condensation", 1, 1024))
    report = {
        "qualified": True,
        "executable_sha256": "current",
        "cases": [{"case": case}],
    }
    if change == "missing-case":
        report["cases"] = []
    elif change == "old-source":
        report["executable_sha256"] = "old"
    elif change == "failed":
        report["qualified"] = False
    (tmp_path / "cpu-parity.json").write_text(json.dumps(report))
    rows = [{"case": case, "mode": "numpy"}]
    if change:
        with pytest.raises(ValueError, match="parity"):
            publication.parity_evidence(tmp_path, rows, "current")
    else:
        assert publication.parity_evidence(tmp_path, rows, "current") == report


@pytest.mark.parametrize(
    "mode", ["numpy", "direct", "prepared", "captured", "legacy"]
)
def test_each_required_timing_stage_is_present(mode):
    """Every expected stage is mandatory even when other samples are complete."""
    stages = {"resident", "download"}
    if mode != "legacy":
        stages.update(("setup", "end_to_end"))
    if mode not in {"numpy", "legacy"}:
        stages.add("capture")
    report = {
        "mode": mode,
        "warmups": 3,
        "samples": 7,
        "timings": {stage: {"samples_s": [0.1] * 7} for stage in stages},
    }
    publication.validate_samples(report, "complete")
    for omitted in stages:
        incomplete = dict(
            report,
            timings={
                stage: value
                for stage, value in report["timings"].items()
                if stage != omitted
            },
        )
        with pytest.raises(ValueError, match="Missing timing stages"):
            publication.validate_samples(incomplete, omitted)
    with pytest.raises(ValueError, match="Missing timing stages"):
        publication.validate_samples(dict(report, timings={}), "empty")


def test_missing_stage_preserves_existing_publication(campaign):
    """Missing capture timing cannot silently disappear from a published plot."""
    path = campaign / "brownian-128-captured.json"
    report = json.loads(path.read_text())
    del report["timings"]["capture"]
    path.write_text(json.dumps(report))
    destination = campaign / "public.json"
    destination.write_text("previous successful publication")
    with pytest.raises(ValueError, match="Missing timing stages"):
        publication.export_campaign(campaign, destination)
    assert destination.read_text() == "previous successful publication"
