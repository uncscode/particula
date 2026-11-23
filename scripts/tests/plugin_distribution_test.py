"""Distribution contracts, including relocated archive execution."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "build_particula_plugins.py"
SPEC = importlib.util.spec_from_file_location("plugin_builder", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def example():
    """Load the standalone example without importing optional plotting."""
    path = builder.PLUGIN / "scripts/run_example.py"
    spec = importlib.util.spec_from_file_location("plugin_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mass_fraction", [0.99, 1.01])
def test_coagulation_rejects_particle_drift_hidden_by_gas(
    example, mass_fraction
):
    """A gas-dominated budget cannot hide a 1% particle loss or gain."""
    particle_mass = 4.864826562917495e-13
    gas_mass = 1e-8
    rows = [
        {
            "particle_mass_kg_m-3": mass,
            "total_mass_kg_m-3": mass + gas_mass,
        }
        for mass in (
            particle_mass,
            particle_mass * mass_fraction,
            particle_mass,
        )
    ]
    checks = example.mass_balance_checks(rows, "coagulation")
    assert checks["max_relative_total_mass_error"] < 1e-6
    assert checks["max_relative_particle_mass_error"] == pytest.approx(0.01)
    assert not checks["mass_balance_pass"]


def test_coagulation_accepts_conserved_particle_mass(example):
    """A closed coagulation budget with unchanged particle mass passes."""
    row = {"particle_mass_kg_m-3": 1e-13, "total_mass_kg_m-3": 1e-8}
    checks = example.mass_balance_checks([row, dict(row)], "coagulation")
    assert checks["mass_balance_pass"]
    assert checks["max_relative_particle_mass_error"] == 0.0


@pytest.mark.parametrize("total_fraction, passed", [(1.0, True), (0.99, False)])
def test_condensation_checks_combined_budget(example, total_fraction, passed):
    """Vapor uptake may change particle mass but must conserve total mass."""
    rows = [
        {"particle_mass_kg_m-3": 1e-10, "total_mass_kg_m-3": 1e-8},
        {
            "particle_mass_kg_m-3": 2e-10,
            "total_mass_kg_m-3": 1e-8 * total_fraction,
        },
    ]
    checks = example.mass_balance_checks(rows, "condensation")
    assert checks["mass_balance_pass"] is passed
    assert "max_relative_particle_mass_error" not in checks


def test_generated_resources_and_manifests():
    """Repository installation uses the current canonical payload."""
    _, catalog = builder.validate()
    builder.sync_or_check(catalog)


def test_generated_drift_rejected(tmp_path, monkeypatch):
    """A stale generated payload cannot be shipped silently."""
    import shutil

    target = tmp_path / "plugin"
    shutil.copytree(
        builder.PLUGIN, target, ignore=shutil.ignore_patterns("node_modules")
    )
    monkeypatch.setattr(builder, "PLUGIN", target)
    _, catalog = builder.validate()
    (target / "opencode/payload/skills/particula-run/SKILL.md").write_text(
        "stale"
    )
    with pytest.raises(ValueError, match="drift"):
        builder.sync_or_check(catalog)


def test_github_packages_contain_metadata_and_provenance(tmp_path):
    """GitHub distributions include only the supported host packages."""
    import hashlib

    manifest, _ = builder.validate()
    output = tmp_path / "github"
    builder.build(output, manifest)
    assert {path.name for path in output.iterdir()} == {
        "particula-helper-codex.zip",
        "particula-helper-claude.zip",
        "particula-helper-opencode.zip",
        "SHA256SUMS",
        "GITHUB-RELEASE.md",
    }
    path = output / "particula-helper-codex.zip"
    with zipfile.ZipFile(path) as archive:
        portable = json.loads(archive.read("plugin.json"))
        extension = portable["extensions"]["com.openai"]
        assert extension["requires_local_executor"] is True
        assert extension["interface"]["developerName"] == "Particula Developers"
        provenance = json.loads(archive.read("BUILD.json"))
        assert provenance["plugin_version"] == portable["version"]
        for name, digest in provenance["sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert (
            hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
        )
    github_notes = (output / "GITHUB-RELEASE.md").read_text()
    assert provenance["source_revision"] in github_notes
    assert manifest["version"] in github_notes
    assert "does not require OpenAI store verification" in github_notes


@pytest.mark.parametrize(
    "event_name, previous, current, expected",
    [
        ("push", "0.3.1", "0.3.2", "true"),
        ("push", "0.3.2", "0.3.2", "false"),
        ("push", "0.3.9", "0.3.10", "true"),
        ("push", "0.3.2", "0.3.1", None),
        ("pull_request", "0.3.1", "0.3.2", "false"),
        ("workflow_dispatch", "0.3.1", "0.3.2", "false"),
        ("release", "0.3.1", "0.3.2", "false"),
        ("merge_group", "0.3.1", "0.3.2", "false"),
    ],
)
def test_workflow_release_version_gate(
    tmp_path, monkeypatch, event_name, previous, current, expected
):
    """Run CI's actual gate against version bumps, reruns and non-push events."""
    workflow = yaml.safe_load(
        (builder.ROOT / ".github/workflows/pypi.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["validate"]["steps"]
        if step.get("id") == "package"
    )
    code = step["run"].split("\n", 1)[1].rsplit("\nPY", 1)[0]
    manifest = tmp_path / "plugins/particula-helper/plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"version": current}))
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"before": "a" * 40}))
    output = tmp_path / "output"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event_name)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_SHA", "b" * 40)

    def previous_manifest(command, **kwargs):
        assert event_name == "push"
        assert command == [
            "git",
            "show",
            f"{'a' * 40}:plugins/particula-helper/plugin.json",
        ]
        return json.dumps({"version": previous})

    monkeypatch.setattr(subprocess, "check_output", previous_manifest)
    if expected is None:
        with pytest.raises(SystemExit, match="must not decrease"):
            exec(compile(code, "workflow-version-gate", "exec"), {})  # noqa: S102 - trusted repository workflow
        assert not output.exists()
    else:
        exec(compile(code, "workflow-version-gate", "exec"), {})  # noqa: S102 - trusted repository workflow
        values = dict(
            line.split("=", 1) for line in output.read_text().splitlines()
        )
        assert values["version-increased"] == expected
        assert values["version"] == current
        assert values["artifact"] == f"particula-helper-{current}-{'b' * 40}"


def test_plugin_release_waits_for_all_contracts_on_version_bumps():
    """A successful smoke alone cannot authorize plugin-only publication."""
    workflow = yaml.safe_load(
        (builder.ROOT / ".github/workflows/pypi.yml").read_text()
    )
    jobs = workflow["jobs"]
    assert jobs["plugin-release"]["needs"] == "validate"
    steps = jobs["validate"]["steps"]
    gate = next(
        i for i, step in enumerate(steps) if step.get("id") == "package"
    )
    smoke = next(
        i
        for i, step in enumerate(steps)
        if "::test_relocated_archives_smoke" in step.get("run", "")
    )
    contracts = next(
        i
        for i, step in enumerate(steps)
        if '-k "not relocated_archives_smoke"' in step.get("run", "")
    )
    upload = next(
        i for i, step in enumerate(steps) if step.get("id") == "upload"
    )
    assert gate < contracts < upload
    assert smoke < upload and "if" not in steps[smoke]
    assert steps[contracts]["if"] == (
        "github.event_name == 'workflow_dispatch' || "
        "steps.package.outputs.version-increased == 'true'"
    )
    assert not steps[contracts].get("continue-on-error")


def test_workflow_builds_every_main_merge_without_path_filters():
    """Library-only merges and merge queues must also check plugin packages."""
    workflow = yaml.safe_load(
        (builder.ROOT / ".github/workflows/pypi.yml").read_text(),
    )
    # PyYAML's YAML 1.1 parser treats the unquoted GitHub key `on` as True.
    triggers = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request", "merge_group"):
        trigger = triggers[event]
        assert trigger["branches"] == (
            ["main", "codex/stage-*"] if event == "pull_request" else ["main"]
        )
        assert "paths" not in trigger
        assert "paths-ignore" not in trigger


@pytest.mark.parametrize(
    "exists, create_status", [(False, 0), (True, 0), (False, 9)]
)
def test_workflow_publishes_or_preserves_release(
    tmp_path, exists, create_status
):
    """Execute the release shell with a fake CLI, never contacting GitHub."""
    workflow = yaml.safe_load(
        (builder.ROOT / ".github/workflows/pypi.yml").read_text()
    )
    job = workflow["jobs"]["plugin-release"]
    step = next(
        step for step in job["steps"] if "gh release" in step.get("run", "")
    )
    assets = tmp_path / "plugin-dist"
    assets.mkdir()
    names = [
        "particula-helper-codex.zip",
        "particula-helper-claude.zip",
        "particula-helper-opencode.zip",
        "SHA256SUMS",
        "GITHUB-RELEASE.md",
    ]
    for name in names:
        (assets / name).write_text(name)
    log = tmp_path / "calls.jsonl"
    cli = tmp_path / "gh"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CALL_LOG'], 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1:3] == ['release', 'view']:\n"
        "    sys.exit(0 if os.environ['EXISTS'] == 'true' else 1)\n"
        "if sys.argv[1:3] == ['release', 'create']:\n"
        "    sys.exit(int(os.environ['CREATE_STATUS']))\n"
        "sys.exit(99)\n"
    )
    cli.chmod(0o755)
    bash = shutil.which("bash")
    assert bash is not None
    result = subprocess.run(  # noqa: S603 - trusted workflow with a local fake CLI
        [
            bash,
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            "-c",
            step["run"],
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "CALL_LOG": str(log),
            "EXISTS": str(exists).lower(),
            "CREATE_STATUS": str(create_status),
            "PLUGIN_VERSION": "0.3.4",
            "SOURCE_COMMIT": "b" * 40,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    tag = "particula-helper-v0.3.4"
    assert calls[0] == ["release", "view", tag]
    assert result.returncode == (0 if exists else create_status)
    if exists:
        # The existing release can be published or a historical draft.
        assert len(calls) == 1
    else:
        assert len(calls) == 2
        creation = calls[1]
        assert creation[:3] == ["release", "create", tag]
        assert "--draft" not in creation
        assert "--latest=false" in creation
        assert creation[creation.index("--target") + 1] == "b" * 40
        assert (
            creation[creation.index("--title") + 1] == "Particula Helper 0.3.4"
        )
        assert (
            creation[creation.index("--notes-file") + 1]
            == "plugin-dist/GITHUB-RELEASE.md"
        )
        assert all(f"plugin-dist/{name}" in creation for name in names)


def _workflow_condition(condition, event, repo, ref, increased, tag):
    """Evaluate the small trusted expression subset used by release guards."""
    # GitHub permits hyphens in property names; Python uses a local binding.
    expression = condition.replace(
        "needs.validate.outputs.version-increased", "increased"
    )
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", "not ", expression).strip()
    return eval(  # noqa: S307 - trusted repository condition, no builtins
        expression,
        {"__builtins__": {}},
        {
            "github": SimpleNamespace(
                event_name=event,
                repository=repo,
                ref=ref,
                event=SimpleNamespace(release=SimpleNamespace(tag_name=tag)),
            ),
            "increased": increased,
            "startsWith": lambda value, prefix: value.startswith(prefix),
        },
    )


@pytest.mark.parametrize(
    "event, repo, ref, increased, expected",
    [
        ("push", "uncscode/particula", "refs/heads/main", "true", True),
        ("push", "Gorkowski/particula", "refs/heads/main", "true", False),
        ("push", "uncscode/particula", "refs/heads/topic", "true", False),
        ("push", "uncscode/particula", "refs/heads/main", "false", False),
        (
            "pull_request",
            "uncscode/particula",
            "refs/heads/main",
            "true",
            False,
        ),
        ("merge_group", "uncscode/particula", "refs/heads/main", "true", False),
        (
            "workflow_dispatch",
            "uncscode/particula",
            "refs/heads/main",
            "true",
            False,
        ),
        ("release", "uncscode/particula", "refs/heads/main", "true", False),
    ],
)
def test_publication_requires_validated_upstream_version_bump(
    event, repo, ref, increased, expected
):
    """Forks, other events/branches and same-version pushes cannot publish."""
    workflow = yaml.safe_load(
        (builder.ROOT / ".github/workflows/pypi.yml").read_text()
    )
    job = workflow["jobs"]["plugin-release"]
    assert job["needs"] == "validate"
    assert (
        _workflow_condition(job["if"], event, repo, ref, increased, "")
        is expected
    )


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("particula-helper-v0.3.4", False),
        ("particula-aiomfac-v0.1.4", False),
        ("v0.3.1", True),
    ],
)
def test_plugin_release_does_not_publish_python_or_reattach_assets(
    tag, expected
):
    """Published plugin tags skip PyPI and the library asset attachment job."""
    workflow = yaml.safe_load(
        (builder.ROOT / ".github/workflows/pypi.yml").read_text()
    )
    assert "release-assets" not in workflow["jobs"]
    for job_name in ("release-plan", "validate", "testing"):
        assert (
            _workflow_condition(
                workflow["jobs"][job_name]["if"],
                "release",
                "uncscode/particula",
                "refs/heads/main",
                "false",
                tag,
            )
            is expected
        )


def test_report_format_matches_issue_template():
    """Researchers and packaged agents submit the same triage fields."""
    template = (
        builder.ROOT / ".github/ISSUE_TEMPLATE/bug_report.md"
    ).read_text()
    metadata = yaml.safe_load(template.split("---", 2)[1])
    assert not metadata.get("labels")
    reference = (builder.PLUGIN / "references/issue-report.md").read_text()
    assert (
        template[template.index("## Summary and impact") :]
        == reference[reference.index("## Summary and impact") :]
    )


def test_skills_relocate_with_their_references(tmp_path):
    """Every host carries all skills and their contained supporting references."""
    manifest, _ = builder.validate()
    output = tmp_path / "archives"
    builder.build(output, manifest)
    for archive_path in output.glob("*.zip"):
        destination = tmp_path / archive_path.stem
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        roots = (
            [
                destination / "particula-helper/payload/skills",
                destination / ".opencode/skills",
            ]
            if "opencode" in archive_path.name
            else [destination / "skills"]
        )
        for root in roots:
            pending = [root / name / "SKILL.md" for name in builder.SKILLS]
            visited = set()
            while pending:
                source = pending.pop().resolve()
                if source in visited:
                    continue
                visited.add(source)
                assert source.is_relative_to(destination)
                assert source.is_file()
                content = source.read_text()
                relatives = re.findall(r"`((?:\.\./|\./)[^`\s]+)", content)
                relatives += re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", content)
                for relative in relatives:
                    if "://" in relative:
                        continue
                    target = (source.parent / relative).resolve()
                    assert target.is_relative_to(destination)
                    assert target.exists(), (source, relative)
                    if target.suffix == ".md":
                        pending.append(target)


def test_archives_reproducible(tmp_path):
    """Two builds from the same source have identical archives."""
    manifest, _ = builder.validate()
    first, second = tmp_path / "first", tmp_path / "second"
    builder.build(first, manifest)
    builder.build(second, manifest)
    for path in first.glob("*.zip"):
        assert path.read_bytes() == (second / path.name).read_bytes()


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="The archive smoke test requires Matplotlib",
)
def test_relocated_archives_smoke(tmp_path):
    """Exercise the actual release ZIPs when supplied by the packaging job."""
    directory = os.environ.get("PARTICULA_PLUGIN_DIST")
    if directory:
        archives = Path(directory)
    else:
        archives = tmp_path / "archives"
        manifest, _ = builder.validate()
        builder.build(archives, manifest)
    for host in ("codex", "claude", "opencode"):
        archive_path = archives / f"particula-helper-{host}.zip"
        destination = tmp_path / archive_path.stem
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert archive.testzip() is None
            assert not any(
                Path(n).is_absolute() or ".." in Path(n).parts for n in names
            )
            assert not any(
                "node_modules" in n or "__pycache__" in n or n.endswith(".pyc")
                for n in names
            )
            archive.extractall(destination)
        if "opencode" in archive_path.name:
            roots = [
                destination / "particula-helper/payload",
                destination / ".opencode/skills/particula-run",
            ]
        else:
            roots = [destination]
        for index, root in enumerate(roots):
            doctor = subprocess.run(  # noqa: S603 - extracted, locally built test helpers
                [sys.executable, str(root / "scripts/doctor.py")],
                cwd=tmp_path,
                capture_output=True,
                text=True,
            )
            assert doctor.returncode == 0, doctor.stdout + doctor.stderr
            assert json.loads(doctor.stdout)["starter_ready"]
            output = tmp_path / f"{archive_path.stem}-{index}-run"
            command = [
                sys.executable,
                str(root / "scripts/run_example.py"),
                "--output",
                str(output),
            ]
            result = subprocess.run(  # noqa: S603 - extracted, locally built test helpers
                command,
                cwd=tmp_path,
                env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            report = json.loads((output / "run.json").read_text())
            assert report["checks"]["mass_balance_pass"]
            assert report["checks"]["number_conserved"]
            for name in ("time_series.csv", "distribution.csv", "result.png"):
                assert (output / name).stat().st_size > 0
            # Existing scientific results must never be silently overwritten.
            retry = subprocess.run(  # noqa: S603 - extracted, locally built test helpers
                command, cwd=tmp_path, capture_output=True, text=True
            )
            assert retry.returncode != 0
            assert "already exists" in retry.stderr
