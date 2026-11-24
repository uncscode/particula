"""Execute the first CPU workflow and its published reload/restart snippets."""

import re
from pathlib import Path

import numpy as np
import pytest
from docs.studies.tutorials.first_cpu_workflow import simulate


@pytest.fixture(scope="module")
def workflow(tmp_path_factory):
    """Run the complete published workloads, including both exact restarts."""
    directory = tmp_path_factory.mktemp("first-cpu")
    report = simulate(directory / "first-cpu-results")
    return directory, report


def test_first_workflow_has_physical_budget_and_controlled_input(workflow):
    """Both cases retain initial number/mass and respond to added vapor."""
    directory, report = workflow
    for case in report["cases"]:
        vapor = case["initial_vapor_ug_m3"]
        total = np.asarray(case["gas_ug_m3"]) + case["particle_ug_m3"]
        initial_particle = 4 * np.pi * 1400 * (100e-9) ** 3 / 3 * 1e9 * 1e9
        np.testing.assert_allclose(total, vapor + initial_particle, rtol=2e-13)
        np.testing.assert_array_equal(case["time_s"], np.linspace(0, 600, 61))
        assert np.all(np.diff(case["radius_nm"]) > 0)
        assert np.all(np.asarray(case["gas_ug_m3"]) >= 0)
        assert (
            directory / f"first-cpu-results/organic-{vapor:g}.analysis"
        ).is_file()
        assert (
            directory / f"first-cpu-results/organic-{vapor:g}.restart"
        ).is_file()
    assert (
        report["cases"][1]["radius_nm"][-1]
        > report["cases"][0]["radius_nm"][-1]
    )


def test_published_reload_and_resume_snippets(workflow, monkeypatch):
    """The guide reads saved files and resumes with exact future gas history."""
    directory, _ = workflow
    root = Path(__file__).resolve().parents[1]
    page = root / "docs/guide/getting-started/first-simulation.md"
    guide = page.read_text()
    monkeypatch.chdir(directory)
    monkeypatch.syspath_prepend(str(root / "docs/studies/tutorials"))
    for heading in (
        "Reload and inspect results",
        "Resume the five-minute checkpoint",
    ):
        section = guide.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]
        (snippet,) = re.findall(r"```python\n(.*?)\n```", section, re.DOTALL)
        namespace = {}
        exec(compile(snippet, str(page), "exec"), namespace)  # noqa: S102
    assert namespace["resumed"].time == 600
