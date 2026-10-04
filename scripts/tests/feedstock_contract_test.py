"""Regressions for the external recipe mismatch behind feedstock PR #54."""

from __future__ import annotations

import importlib.util
import json
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
