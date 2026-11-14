"""Keep incomplete or inconsistent timing evidence out of published results."""

import json
from dataclasses import asdict

import pytest
from scripts import publish_gpu_campaign as publication
from scripts.gpu_studies.cases import StudyCase


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    """Build a tiny complete campaign with exact comparable continuations."""
    case = StudyCase("test", "brownian", 1, 128)
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
        key = f"test-{mode}"
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
            "timings": {"resident": {"samples_s": [0.1] * 7}},
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


def test_complete_evidence_keeps_samples_and_exactness(campaign):
    """Publication retains repetitions and exposes cross-mode validation."""
    destination = campaign / "public.json"
    result = publication.export_campaign(campaign, destination)
    assert result["job_count"] == 4
    assert result["cross_mode_cuda_exact"] is True
    assert result["reports"][0]["timings"]["resident"]["samples_s"] == [0.1] * 7
    assert json.loads(destination.read_text()) == result


@pytest.mark.parametrize(
    ("file", "field", "value", "message"),
    [
        ("manifest.json", "qualified", False, "incomplete"),
        ("test-captured.json", "executable_sha256", "old", "stale"),
        ("test-captured.json", "result_hash", "different", "continuation"),
        ("test-captured.json", "input_sha256", "different", "initial states"),
        ("test-captured.json", "samples", 6, "repetitions"),
        (
            "test-captured.json",
            "timings",
            {"resident": {"samples_s": [0.1] * 6}},
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


def test_compact_preserves_measurements_and_scientific_gates():
    """Only large raw histories are excluded from downloadable evidence."""
    result = publication.compact(
        {
            "timings": {"samples_s": [1, 2, 3]},
            "experiment": {"frames": ["large"], "restart_exact": True},
        }
    )
    assert result == {
        "timings": {"samples_s": [1, 2, 3]},
        "experiment": {"restart_exact": True},
    }


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
