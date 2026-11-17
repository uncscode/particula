"""Plan version-triggered releases and publish verified GitHub artifacts.

All network writes live in the explicit ``github`` command. The ``plan`` command
reads Git history without importing Particula; ``pypi-check`` only reads PyPI.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = "particula/__init__.py"
REPOSITORY = "uncscode/particula"


def command(*args: str) -> str:
    """Run an explicit executable and arguments without a shell."""
    executable = shutil.which(args[0])
    if executable is None:
        raise RuntimeError(f"Missing executable: {args[0]}")
    return subprocess.run(  # noqa: S603 - explicit Git/GitHub CLI arguments
        [executable, *args[1:]],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def version_parts(version: str) -> tuple[int, ...]:
    """Accept canonical stable three-part versions for automatic releases."""
    if not re.fullmatch(
        r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", version
    ):
        raise ValueError(
            f"Expected a stable MAJOR.MINOR.PATCH version: {version}"
        )
    return tuple(map(int, version.split(".")))


def read_version(source: str) -> str:
    """Read one literal version assignment without running package code."""
    values = [
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "__version__"
            for t in node.targets
        )
    ]
    if len(values) != 1:
        raise ValueError("Expected exactly one __version__ assignment")
    version = ast.literal_eval(values[0])
    if not isinstance(version, str):
        raise ValueError("Version must be a string literal")
    version_parts(version)
    return version


def plan(
    event_name: str, event: dict, repository: str, ref: str, sha: str
) -> dict:
    """Gate publication on an upstream version increase or matching release."""
    version = read_version((ROOT / VERSION_PATH).read_text())
    result = {
        "version": version,
        "tag": f"v{version}",
        "build": "false",
        "publish": "false",
    }
    if event_name in {"pull_request", "merge_group"}:
        result["build"] = "true"
    elif (
        repository == REPOSITORY
        and event_name == "push"
        and ref == "refs/heads/main"
    ):
        before = event.get("before", "")
        if not re.fullmatch(r"[0-9a-f]{40}", before) or set(before) == {"0"}:
            raise ValueError(
                "A version release requires a known previous main commit"
            )
        previous = read_version(
            command("git", "show", f"{before}:{VERSION_PATH}")
        )
        if version_parts(version) < version_parts(previous):
            raise ValueError("Particula version must not decrease")
        if version_parts(version) > version_parts(previous):
            result.update(build="true", publish="true")
    elif repository == REPOSITORY and event_name == "release":
        release = event["release"]
        if release["tag_name"].startswith("particula-helper-v"):
            return result
        if (
            release["tag_name"] != result["tag"]
            or release.get("prerelease")
            or release.get("draft")
        ):
            raise ValueError(
                "Release tag must match the stable package version"
            )
        if (
            command("git", "rev-parse", f"refs/tags/{result['tag']}^{{commit}}")
            != sha
        ):
            raise ValueError(
                "Release tag does not point to the checked-out commit"
            )
        result.update(build="true", publish="true")
    return result


def fetch_json(url: str, *, github: bool = False) -> dict | None:
    """Read a fixed-service API; only a real 404 means absence."""
    allowed = "https://api.github.com/" if github else "https://pypi.org/"
    if not url.startswith(allowed):
        raise ValueError("Unexpected release API host")
    headers = {"Accept": "application/json"}
    if github:
        headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:  # noqa: S310 - fixed HTTPS services
            return json.load(response)
    except HTTPError as error:
        # HTTPError owns a response stream, including on the missing path.
        error.close()
        if error.code == 404:
            return None
        raise


def distributions(directory: Path, version: str) -> list[Path]:
    """Require exactly the versioned universal wheel and source archive."""
    version_parts(version)
    expected = [
        directory / f"particula-{version}-py3-none-any.whl",
        directory / f"particula-{version}.tar.gz",
    ]
    actual = set(directory.glob("*.whl")) | set(directory.glob("*.tar.gz"))
    if actual != set(expected) or any(
        not p.is_file() or p.is_symlink() for p in expected
    ):
        raise ValueError(
            "Expected exactly the release wheel and source archive"
        )
    return expected


def digest(path: Path) -> str:
    """Return an artifact's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pypi(directory: Path, version: str) -> None:
    """Allow upload retries only when any existing PyPI files are identical."""
    files = distributions(directory, version)
    release = fetch_json(f"https://pypi.org/pypi/particula/{version}/json")
    existing = (
        {item["filename"]: item for item in release["urls"]} if release else {}
    )
    for path in files:
        if path.name in existing and existing[path.name]["digests"][
            "sha256"
        ] != digest(path):
            raise ValueError(
                f"PyPI already has different bytes for {path.name}"
            )


def ensure_tag(version: str, sha: str) -> str:
    """Create an annotated tag or verify an existing tag without moving it."""
    tag = f"v{version}"
    # Fetching first makes annotated and lightweight existing tags comparable.
    command("git", "fetch", "origin", "--tags")
    tags = command("git", "tag", "--list", tag)
    if tags:
        if command("git", "rev-parse", f"refs/tags/{tag}^{{commit}}") != sha:
            raise ValueError(f"Refusing to move existing tag {tag}")
    else:
        command(
            "git",
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=41898282+github-actions[bot]@users.noreply.github.com",
            "tag",
            "-a",
            tag,
            sha,
            "-m",
            f"Particula {version}",
        )
        command("git", "push", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
    return tag


def upload_missing_assets(tag: str, assets: list[Path], release: dict) -> None:
    """Preserve existing assets and attach only missing build files."""
    if release["draft"] or release["prerelease"]:
        raise ValueError("Refusing to alter an existing draft or prerelease")
    existing = {asset["name"]: asset for asset in release["assets"]}
    # Check all conflicts before attaching any missing files. Never clobber.
    for path in assets:
        if (
            path.name in existing
            and existing[path.name].get("digest") != f"sha256:{digest(path)}"
        ):
            raise ValueError(
                f"Existing GitHub asset differs or lacks a digest: {path.name}"
            )
    missing = [str(path) for path in assets if path.name not in existing]
    if missing:
        command("gh", "release", "upload", tag, *missing, "--repo", REPOSITORY)


def previous_library_tag(version: str) -> str | None:
    """Select the previous stable library tag, excluding plugin releases."""
    candidates = []
    for tag in command("git", "tag", "--list", "v*").splitlines():
        try:
            parts = version_parts(tag[1:])
        except ValueError:
            continue
        if parts < version_parts(version):
            candidates.append((parts, tag))
    return max(candidates)[1] if candidates else None


def publish_github(directory: Path, version: str, sha: str) -> None:
    """Create an annotated exact-commit tag and release without overwrites."""
    version_parts(version)
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise ValueError("Only the upstream repository may publish")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Expected a full source commit SHA")
    if command("git", "rev-parse", "HEAD") != sha:
        raise ValueError("Checkout must match the tested source commit")
    if read_version((ROOT / VERSION_PATH).read_text()) != version:
        raise ValueError("Checkout version differs from release version")
    files = distributions(directory, version)
    sums = directory / "PYTHON-SHA256SUMS"
    sums.write_text("".join(f"{digest(path)}  {path.name}\n" for path in files))
    assets = sorted(path for path in directory.iterdir() if path.is_file())
    if any(path.is_symlink() for path in assets):
        raise ValueError("Release assets must not be symlinks")
    tag = ensure_tag(version, sha)
    release = fetch_json(
        f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{tag}",
        github=True,
    )
    if release is None:
        previous = previous_library_tag(version)
        notes = ["--notes-start-tag", previous] if previous else []
        command(
            "gh",
            "release",
            "create",
            tag,
            *(str(path) for path in assets),
            "--repo",
            REPOSITORY,
            "--verify-tag",
            "--title",
            f"Particula {version}",
            "--generate-notes",
            *notes,
        )
        return
    upload_missing_assets(tag, assets, release)


def main() -> None:
    """Run read-only planning/checks or explicit GitHub publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["plan", "github", "pypi-check"])
    parser.add_argument(
        "--directory", type=Path, default=Path("release-assets")
    )
    parser.add_argument("--version")
    parser.add_argument("--sha")
    args = parser.parse_args()
    if args.operation == "plan":
        result = plan(
            os.environ["GITHUB_EVENT_NAME"],
            json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
            os.environ["GITHUB_REPOSITORY"],
            os.environ["GITHUB_REF"],
            os.environ["GITHUB_SHA"],
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
            output.write(
                "".join(f"{key}={value}\n" for key, value in result.items())
            )
    elif args.operation == "github":
        publish_github(args.directory, args.version, args.sha)
    else:
        check_pypi(args.directory, args.version)


if __name__ == "__main__":
    main()
