"""Campaign recovery must never promote stale or incomplete GPU evidence."""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from scripts.gpu_studies import campaign as module
from scripts.gpu_studies.cases import StudyCase


@pytest.fixture
def tiny_campaign(monkeypatch):
    """Replace only the process executor and matrix; retain recovery logic."""
    case = StudyCase("control", "dilution", 1, 8)
    monkeypatch.setattr(module, "all_jobs", lambda: ((case, "prepared"),))
    monkeypatch.setattr(module, "executable_hash", lambda: "current")
    return case


def result(study_case, **changes):
    """Build provenance equivalent to a successful worker payload."""
    return {
        "qualified": True,
        "case": asdict(study_case),
        "mode": "prepared",
        "executable_sha256": "current",
        **changes,
    }


@pytest.mark.parametrize(
    "payload",
    ["{broken", "[]", "null", '{"qualified":true}'],
)
def test_zero_exit_with_bad_result_is_not_success(
    tmp_path, monkeypatch, tiny_campaign, payload
):
    """A process exit cannot substitute for valid scientific provenance."""

    def execute(command, **kwargs):
        (tmp_path / "control-prepared.json").write_text(payload)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", execute)
    report = module.campaign(tmp_path)
    assert not report["qualified"]
    assert report["jobs"]["control-prepared"]["status"] == "failed"


def test_resume_checks_case_and_source_and_preserves_deadline(
    tmp_path, monkeypatch, tiny_campaign
):
    """Changed workloads and code rerun; successful identical evidence reuses."""
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        module.save(tmp_path / "control-prepared.json", result(tiny_campaign))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", execute)
    first = module.campaign(tmp_path, hours=1)
    second = module.campaign(tmp_path, hours=8)
    assert first["qualified"] and second["qualified"]
    assert len(calls) == 1
    assert first["started_at"] == second["started_at"]
    assert second["hours"] == 1
    path = tmp_path / "control-prepared.json"
    for change in (
        {"case": {**asdict(tiny_campaign), "slots": 16}},
        {"executable_sha256": "old"},
        {"mode": "numpy"},
    ):
        module.save(path, result(tiny_campaign, **change))
        assert module.campaign(tmp_path)["qualified"]
    assert len(calls) == 4


def test_timeout_retains_required_case(tmp_path, monkeypatch, tiny_campaign):
    """A timeout is visible and never replaced by a smaller successful case."""

    def execute(command, **kwargs):
        assert kwargs["timeout"] <= 7
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(module.subprocess, "run", execute)
    report = module.campaign(tmp_path, case_seconds=7)
    row = report["jobs"]["control-prepared"]
    assert row["status"] == "timed_out"
    assert row["case"] == asdict(tiny_campaign)
    assert not report["qualified"]


def test_expired_budget_does_not_launch(tmp_path, monkeypatch, tiny_campaign):
    """Resuming an expired campaign cannot silently renew its eight hours."""
    module.save(
        tmp_path / "manifest.json",
        {
            "started_at": (
                datetime.now(timezone.utc) - timedelta(hours=2)
            ).isoformat(),
            "hours": 1,
            "jobs": {},
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("expired campaign launched a worker")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    report = module.campaign(tmp_path, hours=8)
    assert report["jobs"]["control-prepared"]["status"] == "budget_exhausted"
    assert not report["qualified"]
    assert json.loads((tmp_path / "manifest.json").read_text())["hours"] == 1
