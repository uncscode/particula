"""Regressions for the external recipe mismatch behind feedstock PR #54."""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_feedstock_contract.py"
MIRROR = ROOT / "conda/recipe/meta.yaml"
FAILED_RECIPE = Path(__file__).parent / "fixtures/feedstock_pr54_meta.yaml"


@pytest.fixture
def checker():
    """Load the data-only checker without importing application code."""
    spec = importlib.util.spec_from_file_location("feedstock_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_pr54_recipe_is_rejected_with_missing_inputs(checker, tmp_path):
    """The actual old recipe cannot pass against our corrected local mirror."""
    assert not checker.check_contract(MIRROR, FAILED_RECIPE, tmp_path)
    report = json.loads((tmp_path / "contract.json").read_text())
    assert set(report["differences"]) == {"test.source_files", "test.commands"}
    assert report["contracts"]["external"]["source_files"] == ["particula"]
    assert "pyproject.toml" in report["contracts"]["mirror"]["source_files"]
    assert (
        tmp_path / "external-meta.yaml"
    ).read_bytes() == FAILED_RECIPE.read_bytes()


def test_corrected_external_handoff_passes_despite_release_metadata(
    checker, tmp_path
):
    """Source/version/maintainer differences do not hide or fabricate drift."""
    recipe = FAILED_RECIPE.read_text()
    start = recipe.index("test:\n")
    end = recipe.index("about:\n")
    mirror = MIRROR.read_text()
    contract = mirror[mirror.index("test:\n") : mirror.index("about:\n")]
    external = tmp_path / "external.yaml"
    external.write_text(recipe[:start] + contract + recipe[end:])
    assert checker.check_contract(MIRROR, external, tmp_path / "evidence")


@pytest.mark.parametrize(
    "missing",
    [
        "pyproject.toml",
        "conftest.py",
        "docs/Examples/cpu_dilution.py",
        "examples_tests/dilution_example_test.py",
        "examples_tests/nucleation_example_test.py",
        "examples_tests/condensation_latent_heat_example_test.py",
    ],
)
def test_individual_missing_input_is_rejected(checker, tmp_path, missing):
    """Correct commands alone cannot certify an incomplete source_files list."""
    external = tmp_path / "external.yaml"
    external.write_text(MIRROR.read_text().replace(f"    - {missing}\n", ""))
    assert not checker.check_contract(MIRROR, external, tmp_path / "evidence")


def test_missing_examples_command_is_rejected(checker, tmp_path):
    """Both release suites must execute even if their inputs are present."""
    external = tmp_path / "external.yaml"
    external.write_text(
        MIRROR.read_text().replace(
            "    - python scripts/run_release_tests.py --installed --suite examples\n",
            "",
        )
    )
    assert not checker.check_contract(MIRROR, external, tmp_path / "evidence")


@pytest.mark.parametrize(
    "directive", ["if false", "for item in []", "macro hidden()"]
)
def test_enclosing_template_cannot_certify_absent_tests(checker, directive):
    """Reject control flow even when it closes after the next YAML section."""
    block = (
        "test:\n  requires: [pytest]\n  source_files: [particula]\n"
        "  commands: [pytest]\nabout:\n  summary: fixture\n"
    )
    ending = directive.split()[0]
    recipe = "{% " + directive + " %}\n" + block + "{% end" + ending + " %}\n"
    with pytest.raises(ValueError, match="Templated test contracts"):
        checker.test_contract(recipe)


def test_commented_out_contract_is_rejected(checker):
    """A Jinja comment cannot publish a contract that will not execute."""
    with pytest.raises(ValueError, match="Templated test contracts"):
        checker.test_contract("{#\n" + MIRROR.read_text() + "\n#}")


def test_requires_and_input_order_are_immaterial_but_command_order_is_not(
    checker,
):
    """Compare test semantics without depending on harmless YAML list order."""
    original = {
        "test": {
            "requires": ["pytest", "pip"],
            "source_files": ["particula", "conftest.py"],
            "commands": ["pip check", "pytest"],
        }
    }
    expected = checker.test_contract(yaml.safe_dump(original))
    original["test"]["requires"].reverse()
    original["test"]["source_files"].reverse()
    assert checker.test_contract(yaml.safe_dump(original)) == expected
    original["test"]["commands"].reverse()
    assert checker.test_contract(yaml.safe_dump(original)) != expected


@pytest.mark.parametrize(
    "recipe",
    [
        "package:\n  name: particula\n",
        "test:\n  commands: [pytest]\n",
        "test:\n  requires: [pytest]\n  source_files: [particula]\n  commands: pytest\n",
        "test:\n  requires: [pytest]\n  source_files: [particula]\n  commands: []\n",
        "test:\n  requires: [pytest]\n  source_files: [particula]\n  commands: [pytest] # [linux]\n",
        "test:\n  requires: [pytest]\n  source_files: [particula]\n  commands: ['{{ run_tests }}']\n",
        "test:\n  requires: [pytest]\n  source_files: [particula]\n  commands: [pytest]\n  commands: [true]\n",
        "test:\n  requires: &shared [pytest]\n  source_files: [particula]\n  commands: *shared\n",
    ],
)
def test_unsupported_or_incomplete_contract_fails_closed(checker, recipe):
    """Never report parity after silently dropping unsupported recipe logic."""
    with pytest.raises((ValueError, yaml.YAMLError)):
        checker.test_contract(recipe)


@pytest.mark.parametrize("use_failed_recipe", [True, False])
def test_cli_failure_and_evidence(tmp_path, use_failed_recipe):
    """Old or unavailable external recipes produce a failing CI exit code."""
    external = FAILED_RECIPE if use_failed_recipe else tmp_path / "absent.yaml"
    result = subprocess.run(  # noqa: S603 - fixed repository script
        [
            sys.executable,
            str(SCRIPT),
            "--external-recipe",
            str(external),
            "--mirror",
            str(MIRROR),
            "--artifacts",
            str(tmp_path / "evidence"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    report = json.loads((tmp_path / "evidence/contract.json").read_text())
    assert not report["passed"]
    if use_failed_recipe:
        assert set(report["differences"]) == {
            "test.source_files",
            "test.commands",
        }
    else:
        assert report["errors"]


@pytest.mark.parametrize("allow_drift", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "matching",
        "old_recipe",
        "legacy",
        "missing",
        "malformed",
        "template",
        "bad_mirror",
    ],
)
def test_cli_policy_keeps_parity_distinct_from_advisory_success(
    tmp_path, monkeypatch, allow_drift, case
):
    """Only valid drift may pass advisory checks; invalid evidence never does."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    external = tmp_path / "external.yaml"
    mirror = MIRROR
    recipe = MIRROR.read_text()
    if case == "old_recipe":
        recipe = FAILED_RECIPE.read_text()
    elif case == "legacy":
        # The legitimate v0.2.14 seven-input/one-command handoff also differs
        # from an unreleased two-suite mirror; source PRs must remain mergeable.
        recipe = "\n".join(
            line
            for line in recipe.splitlines()
            if not line.startswith("    - examples_tests/")
            and "--suite examples" not in line
        )
    elif case == "malformed":
        recipe = "test:\n  commands: [unterminated\n"
    elif case == "template":
        recipe = recipe.replace("    - pytest", "    - '{{ test_dependency }}'")
    elif case == "bad_mirror":
        mirror = tmp_path / "invalid-mirror.yaml"
        mirror.write_text("test:\n  commands: []\n")
    if case != "missing":
        external.write_text(recipe)
    artifacts = tmp_path / "evidence"
    result = subprocess.run(  # noqa: S603 - fixed repository CLI
        [
            sys.executable,
            str(SCRIPT),
            "--mirror",
            str(mirror),
            "--external-recipe",
            str(external),
            "--artifacts",
            str(artifacts),
            *(["--allow-drift"] if allow_drift else []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    matching = case == "matching"
    drift = case in {"old_recipe", "legacy"}
    accepted = matching or (drift and allow_drift)
    assert result.returncode == (0 if accepted else 1), result.stderr
    report = json.loads((artifacts / "contract.json").read_text())
    assert report["passed"] is matching
    assert report["check_passed"] is accepted
    assert report["mode"] == ("advisory" if allow_drift else "strict")
    assert bool(report["errors"]) is (not matching and not drift)
    if drift:
        assert set(report["differences"]) == {
            "test.source_files",
            "test.commands",
        }
    assert ("::warning::" in result.stdout) is (drift and allow_drift)
    if external.exists():
        assert (
            artifacts / "external-meta.yaml"
        ).read_bytes() == external.read_bytes()


@pytest.mark.parametrize(
    ("event", "exit_code"), [("pull_request", 0), ("workflow_dispatch", 1)]
)
def test_workflow_uses_advisory_only_for_source_prs(tmp_path, event, exit_code):
    """Execute the configured comparison command against the real drift case."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/conda-feedstock.yml").read_text()
    )
    job = workflow["jobs"]["feedstock-contract"]
    comparisons = [
        step
        for step in job["steps"]
        if "scripts/check_feedstock_contract.py" in step.get("run", "")
    ]
    assert len(comparisons) == 2
    (selected,) = [
        step
        for step in comparisons
        if step.get("if") == f"github.event_name == '{event}'"
    ]
    assert not job.get("continue-on-error", False)
    assert not workflow["jobs"]["conda-build"].get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in comparisons)
    command = shlex.split(selected["run"])
    command[0] = sys.executable
    index = command.index("--external-recipe") + 1
    command[index] = str(FAILED_RECIPE)
    command.extend(["--artifacts", str(tmp_path)])
    result = subprocess.run(  # noqa: S603 - repository workflow command, no shell
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    report = json.loads((tmp_path / "contract.json").read_text())
    assert report["passed"] is False
    assert report["check_passed"] is (event == "pull_request")
