"""Ensure downloadable execution reports reject incomplete evidence."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docs_report as report  # noqa: E402


@pytest.fixture
def evidence(monkeypatch):
    """Supply one complete successful CUDA publication bundle."""
    manifest = {
        "figure_dpi": 300,
        "elapsed_seconds": 1.0,
        "package_source_sha256": "a" * 64,
        "input_sha256": {},
        "revision": "b" * 40,
        "executed_at": "2026-11-13T12:00:00Z",
        "device": "cuda:0: NVIDIA RTX 5060",
        "python": "3.12.12",
        "versions": {"numpy": "2.3.4"},
        "outputs": [[{"image": "figure.png"}]],
        "files": {"source.py": "hash", "figure.png": "hash"},
    }
    monkeypatch.setattr(
        report,
        "registry",
        lambda: {
            "example": {
                "source": "docs/example.py",
                "policy": "cuda",
                "device": "cuda",
            }
        },
    )
    monkeypatch.setattr(
        report,
        "bundle",
        lambda _: (report.ROOT / "website/results/example/run", manifest),
    )
    monkeypatch.setattr(report, "result_state", lambda *_: "current")
    return manifest


def test_execution_report_preserves_provenance(evidence):
    """Report exact device, revision, elapsed time, and output counts."""
    result = report.execution_inventory()
    assert result["successful_count"] == 1
    row = result["examples"][0]
    assert row["device"] == evidence["device"]
    assert row["revision"] == evidence["revision"]
    assert row["figures"] == 1


@pytest.mark.parametrize(
    "defect", ["cpu", "duration", "source", "figure", "metadata"]
)
def test_execution_report_rejects_invalid_evidence(evidence, defect):
    """Missing provenance and CPU substitutions cannot qualify a CUDA run."""
    if defect == "cpu":
        evidence["device"] = "CPU"
    elif defect == "duration":
        evidence["elapsed_seconds"] = 0
    elif defect == "source":
        evidence["files"].pop("source.py")
    elif defect == "figure":
        evidence["files"].pop("figure.png")
    else:
        evidence.pop("package_source_sha256")
    with pytest.raises(ValueError):
        report.execution_inventory()
