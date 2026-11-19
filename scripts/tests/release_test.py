"""Exercise release gates and retry behavior without external publication."""

import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml
from scripts import release

SHA = "b" * 40


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Supply package metadata without requiring package imports."""
    (tmp_path / "particula").mkdir()
    (tmp_path / release.VERSION_PATH).write_text('__version__ = "0.3.2"\n')
    monkeypatch.setattr(release, "ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "source",
    [
        '__version__ = "0.3.2"\nraise RuntimeError("do not import")',
        '__version__ = "0.3.2"\n# unrelated change',
    ],
)
def test_literal_version_never_imports_package(source):
    """Version discovery must not run package initialization."""
    assert release.read_version(source) == "0.3.2"


@pytest.mark.parametrize(
    "source",
    [
        '__version__ = "0.3.2rc1"',
        '__version__ = "03.2.0"',
        "__version__ = 3",
        "__version__ = get_version()",
        '__version__ = "0.3.1"\n__version__ = "0.3.2"',
        'other = "0.3.2"',
    ],
)
def test_invalid_versions_rejected(source):
    """Ambiguous or nonstable release metadata cannot produce tags."""
    with pytest.raises((ValueError, TypeError)):
        release.read_version(source)


@pytest.mark.parametrize(
    "previous,current,publish",
    [
        ("0.3.1", "0.3.2", "true"),
        ("0.3.2", "0.3.2", "false"),
        ("0.3.9", "0.3.10", "true"),
        ("0.3.2", "0.4.0", "true"),
        ("0.3.2", "1.0.0", "true"),
    ],
)
def test_upstream_version_increase(
    source, monkeypatch, previous, current, publish
):
    """Compare numeric versions at the push boundary, not lexically."""
    (source / release.VERSION_PATH).write_text(f'__version__ = "{current}"\n')
    calls = []

    def git(*args):
        calls.append(args)
        return f'__version__ = "{previous}"\n'

    monkeypatch.setattr(release, "command", git)
    result = release.plan(
        "push", {"before": "a" * 40}, release.REPOSITORY, "refs/heads/main", SHA
    )
    assert result == {
        "version": current,
        "tag": f"v{current}",
        "build": publish,
        "publish": publish,
    }
    assert calls == [("git", "show", f"{'a' * 40}:{release.VERSION_PATH}")]


@pytest.mark.parametrize(
    "event,repo,ref,build",
    [
        ("push", "someone/particula", "refs/heads/main", "false"),
        ("push", release.REPOSITORY, "refs/heads/feature", "false"),
        ("pull_request", release.REPOSITORY, "refs/pull/1/merge", "true"),
        ("merge_group", release.REPOSITORY, "refs/heads/queue", "true"),
        ("workflow_dispatch", release.REPOSITORY, "refs/heads/main", "false"),
        ("release", "someone/particula", "refs/tags/v0.3.2", "false"),
    ],
)
def test_nonpublication_events(source, monkeypatch, event, repo, ref, build):
    """PRs build, but forks, branches and unrelated events never publish."""
    monkeypatch.setattr(
        release, "command", lambda *args: pytest.fail("Unexpected Git access")
    )
    result = release.plan(event, {}, repo, ref, SHA)
    assert result["build"] == build
    assert result["publish"] == "false"


def test_unknown_before_and_downgrades_fail(source, monkeypatch):
    """A missing comparison or downgrade cannot silently ship."""
    for before in ("", "0" * 40, "not-a-sha"):
        with pytest.raises(ValueError, match="previous main"):
            release.plan(
                "push",
                {"before": before},
                release.REPOSITORY,
                "refs/heads/main",
                SHA,
            )
    monkeypatch.setattr(
        release, "command", lambda *args: '__version__ = "0.4.0"'
    )
    with pytest.raises(ValueError, match="must not decrease"):
        release.plan(
            "push",
            {"before": "a" * 40},
            release.REPOSITORY,
            "refs/heads/main",
            SHA,
        )


@pytest.mark.parametrize(
    "tag,sha,prerelease,expected",
    [
        ("v0.3.2", SHA, False, "true"),
        ("particula-helper-v0.3.4", SHA, False, "false"),
        ("v0.3.1", SHA, False, None),
        ("v0.3.2", "c" * 40, False, None),
        ("v0.3.2", SHA, True, None),
    ],
)
def test_manual_release_must_match_source(
    source, monkeypatch, tag, sha, prerelease, expected
):
    """Keep manual library publication, excluding plugins and mismatched tags."""
    monkeypatch.setattr(release, "command", lambda *args: sha)
    event = {
        "release": {"tag_name": tag, "prerelease": prerelease, "draft": False}
    }
    if expected is None:
        with pytest.raises(ValueError):
            release.plan(
                "release", event, release.REPOSITORY, f"refs/tags/{tag}", SHA
            )
    else:
        assert (
            release.plan(
                "release", event, release.REPOSITORY, f"refs/tags/{tag}", SHA
            )["publish"]
            == expected
        )


@pytest.fixture
def repository(source, tmp_path, monkeypatch):
    """Use a real local Git remote; stub only GitHub API and CLI operations."""
    git = shutil.which("git")
    assert git

    def run(*args):
        return subprocess.run(  # noqa: S603 - local disposable repositories
            [git, *args], cwd=source, check=True, capture_output=True, text=True
        ).stdout.strip()

    remote = tmp_path / "remote.git"
    run("init", "--bare", str(remote))
    run("init", "-b", "main")
    run("config", "user.name", "Release test")
    run("config", "user.email", "test@example.invalid")
    run("add", "particula")
    run("commit", "-m", "Test source")
    run("remote", "add", "origin", str(remote))
    run("push", "origin", "main")
    sha = run("rev-parse", "HEAD")
    directory = tmp_path / "assets"
    directory.mkdir()
    for name in ("particula-0.3.2-py3-none-any.whl", "particula-0.3.2.tar.gz"):
        (directory / name).write_bytes(name.encode())
    monkeypatch.setenv("GITHUB_REPOSITORY", release.REPOSITORY)
    actual_command = release.command
    calls = []

    def command(*args):
        if args[0] == "gh":
            calls.append(args)
            return "https://example.invalid/release"
        return actual_command(*args)

    monkeypatch.setattr(release, "command", command)
    monkeypatch.setattr(release, "fetch_json", lambda *args, **kwargs: None)
    return directory, sha, calls, run


def test_exact_annotated_tag_and_published_assets(repository, monkeypatch):
    """Create a real annotated tag, push that tag only and attach artifacts."""
    directory, sha, calls, git = repository
    git("tag", "v0.3.1", sha)
    git("tag", "particula-helper-v0.3.4", sha)
    release.publish_github(directory, "0.3.2", sha)
    assert git("cat-file", "-t", "refs/tags/v0.3.2") == "tag"
    assert git("rev-parse", "refs/tags/v0.3.2^{commit}") == sha
    assert sha in git("ls-remote", "origin", "refs/tags/v0.3.2^{}")
    assert calls[0][:4] == ("gh", "release", "create", "v0.3.2")
    assert "--verify-tag" in calls[0]
    assert "--draft" not in calls[0]
    assert calls[0][calls[0].index("--notes-start-tag") + 1] == "v0.3.1"
    assert str(directory / "PYTHON-SHA256SUMS") in calls[0]
    existing = {
        "draft": False,
        "prerelease": False,
        "assets": [
            {"name": p.name, "digest": f"sha256:{release.digest(p)}"}
            for p in directory.iterdir()
        ],
    }
    monkeypatch.setattr(release, "fetch_json", lambda *args, **kwargs: existing)
    calls.clear()
    release.publish_github(directory, "0.3.2", sha)
    assert calls == []
    existing["assets"].pop()
    release.publish_github(directory, "0.3.2", sha)
    assert calls[0][:4] == ("gh", "release", "upload", "v0.3.2")
    assert "--clobber" not in calls[0]


def test_library_release_notes_use_numeric_stable_tag_order(monkeypatch):
    """Do not derive library notes from plugins, prereleases or lexical order."""
    monkeypatch.setattr(
        release,
        "command",
        lambda *args: "v0.3.9\nv0.3.10\nv0.3.11rc1\nv0.4.0\nvbad",
    )
    assert release.previous_library_tag("0.3.11") == "v0.3.10"
    assert release.previous_library_tag("0.3.1") is None


def test_tag_collision_does_not_move_tag(repository, source):
    """An older same-named tag is never replaced by a different commit."""
    directory, sha, calls, git = repository
    git("tag", "v0.3.2", sha)
    (source / "note").write_text("new commit")
    git("add", "note")
    git("commit", "-m", "Later source")
    with pytest.raises(ValueError, match="Refusing to move"):
        release.publish_github(directory, "0.3.2", git("rev-parse", "HEAD"))
    assert git("rev-parse", "refs/tags/v0.3.2") == sha
    assert not calls


@pytest.mark.parametrize("draft,wrong_digest", [(True, False), (False, True)])
def test_release_conflicts_fail_without_asset_writes(
    repository, monkeypatch, draft, wrong_digest
):
    """Preserve pre-existing drafts and differing files even on retries."""
    directory, sha, calls, _ = repository
    existing = {
        "draft": draft,
        "prerelease": False,
        "assets": [
            {
                "name": "particula-0.3.2.tar.gz",
                "digest": "sha256:wrong" if wrong_digest else None,
            }
        ],
    }
    monkeypatch.setattr(release, "fetch_json", lambda *args, **kwargs: existing)
    with pytest.raises(ValueError, match="draft|differs"):
        release.publish_github(directory, "0.3.2", sha)
    assert not calls


def test_publish_rejects_fork_and_wrong_checkout(repository, monkeypatch):
    """Publication preflight rejects unrelated repositories and source SHAs."""
    directory, sha, calls, _ = repository
    monkeypatch.setenv("GITHUB_REPOSITORY", "someone/particula")
    with pytest.raises(ValueError, match="upstream"):
        release.publish_github(directory, "0.3.2", sha)
    monkeypatch.setenv("GITHUB_REPOSITORY", release.REPOSITORY)
    with pytest.raises(ValueError, match="tested source"):
        release.publish_github(directory, "0.3.2", "a" * 40)
    assert not calls


def test_pypi_retry_checks_digests(repository, monkeypatch):
    """Skip-existing publication is safe only for the same distribution bytes."""
    directory, _, _, _ = repository
    release.check_pypi(directory, "0.3.2")  # No existing release.
    path = directory / "particula-0.3.2.tar.gz"
    existing = {
        "urls": [
            {"filename": path.name, "digests": {"sha256": release.digest(path)}}
        ]
    }
    monkeypatch.setattr(release, "fetch_json", lambda *args, **kwargs: existing)
    release.check_pypi(directory, "0.3.2")
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="different bytes"):
        release.check_pypi(directory, "0.3.2")


@pytest.mark.parametrize(
    "status,absent", [(404, True), (403, False), (500, False)]
)
def test_only_http_404_means_absence(monkeypatch, status, absent):
    """Authentication/server errors must not masquerade as missing releases."""
    error = HTTPError(
        "https://pypi.org/pypi/particula/0.3.2/json", status, "test", {}, None
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(release, "urlopen", fail)
    if absent:
        assert (
            release.fetch_json("https://pypi.org/pypi/particula/0.3.2/json")
            is None
        )
    else:
        with pytest.raises(HTTPError) as caught:
            release.fetch_json("https://pypi.org/pypi/particula/0.3.2/json")
        assert caught.value is error
    assert error.closed


def test_workflow_requires_same_run_tests_and_verified_artifacts():
    """Validate before any writes; run existing PyPI action in the same run."""
    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/pypi.yml"
        ).read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert jobs["pyping"]["needs"] == "release-plan"
    assert jobs["pyping"]["if"] == "needs.release-plan.outputs.build == 'true'"
    assert jobs["testing"]["strategy"]["matrix"]["python-version"] == [
        "3.12",
        "3.13",
        "3.14",
    ]
    commands = [step.get("run", "") for step in jobs["testing"]["steps"]]
    for suite in ("pytest -Werror", "pytest examples_tests"):
        assert any(
            suite in cmd and "not slow and not performance" in cmd
            for cmd in commands
        )
    assert any("pytest scripts/tests -Werror" in cmd for cmd in commands)
    slow = jobs["slow-examples"]
    assert slow["needs"] == "release-plan"
    assert slow["if"] == "needs.release-plan.outputs.publish == 'true'"
    assert any(
        'pytest examples_tests -Werror -m "slow and not performance"'
        in step.get("run", "")
        for step in slow["steps"]
    )
    # GitHub's default success() gate requires every dependency to succeed;
    # failed, cancelled and skipped prerequisites must all block publication.
    prerequisites = {
        "release-plan",
        "testing",
        "slow-examples",
        "pyping",
        "validate",
    }
    assert set(jobs["publish"]["needs"]) == prerequisites
    assert all(not jobs[job].get("continue-on-error") for job in prerequisites)
    assert (
        jobs["publish"]["if"]
        == "github.repository == 'uncscode/particula' && needs.release-plan.outputs.publish == 'true'"
    )
    assert jobs["publish"]["permissions"] == {"contents": "write"}
    publish = jobs["publish"]["steps"]
    names = [s.get("name") for s in publish]
    assert (
        names.index("Verify any previously uploaded PyPI files")
        < names.index("Tag the tested commit and publish the GitHub release")
        < names.index("Publish package")
    )
    producer = jobs["validate"]
    assert any(
        step.get("with", {}).get("name")
        == "${{ steps.package.outputs.artifact }}"
        for step in producer["steps"]
    )
    for job in (jobs["publish"], jobs["plugin-release"]):
        assert any(
            step.get("with", {}).get("name")
            == "${{ needs.validate.outputs.artifact }}"
            for step in job["steps"]
        )
    verify = names.index("Verify downloaded release files")
    assert verify < names.index("Verify any previously uploaded PyPI files")
    assert "check PYTHON-SHA256SUMS" in publish[verify]["run"]
    assert "check SHA256SUMS" in publish[verify]["run"]
    action = next(s for s in publish if "pypi-publish" in s.get("uses", ""))
    assert action["with"]["skip-existing"] is True
    assert action["with"]["password"] == "${{ secrets.PYPI_API_TOKEN }}"  # noqa: S105 - workflow secret reference, not a credential


@pytest.mark.parametrize("asset", ["particula-1.2.3.tar.gz", "helper.zip"])
def test_downloaded_assets_are_verified_before_publication(tmp_path, asset):
    """The publisher accepts both producers' bytes and rejects a changed file."""
    import hashlib

    workflow = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/pypi.yml"
        ).read_text()
    )
    verify = next(
        step["run"]
        for step in workflow["jobs"]["publish"]["steps"]
        if step.get("name") == "Verify downloaded release files"
    )
    for filename, manifest in (
        ("particula-1.2.3.tar.gz", "PYTHON-SHA256SUMS"),
        ("helper.zip", "SHA256SUMS"),
    ):
        data = filename.encode()
        (tmp_path / filename).write_bytes(data)
        (tmp_path / manifest).write_text(
            f"{hashlib.sha256(data).hexdigest()}  {filename}\n"
        )
    bash = shutil.which("bash")
    assert bash
    command = [bash, "-e", "-c", verify]
    subprocess.run(command, cwd=tmp_path, check=True)  # noqa: S603
    (tmp_path / asset).write_bytes(b"changed")
    result = subprocess.run(command, cwd=tmp_path, check=False)  # noqa: S603
    assert result.returncode != 0
