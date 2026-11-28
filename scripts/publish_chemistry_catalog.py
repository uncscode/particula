"""Attach the tested chemistry catalog to its own immutable data version."""

import json
import os
import zipfile

from scripts import release


def publish():
    """Verify the target commit and data version before publishing the ZIP."""
    if os.environ.get("GITHUB_REPOSITORY") != release.REPOSITORY:
        raise ValueError("Only upstream may publish a catalog release")
    sha = release.command("git", "rev-parse", "HEAD")
    if sha != os.environ.get("GITHUB_SHA"):
        raise ValueError("Catalog checkout differs from the tested commit")
    version = (release.ROOT / "chemistry_catalog/VERSION").read_text().strip()
    release.version_parts(version)
    assets = release.ROOT / "dist/catalog"
    archive = assets / "particula-chemistry-catalog.zip"
    checksum = archive.with_suffix(".zip.sha256")
    if checksum.read_text().split() != [release.digest(archive), archive.name]:
        raise ValueError("Catalog asset checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
    project = "chemistry-catalog"
    tag = release.release_tag(project, version)
    if manifest["release"] != tag:
        raise ValueError("Archive version differs from catalog VERSION")
    release.ensure_tag(version, sha, project)
    existing = release.fetch_json(
        f"https://api.github.com/repos/{release.REPOSITORY}/releases/tags/{tag}",
        github=True,
    )
    if existing is not None:
        release.upload_missing_assets(tag, [archive, checksum], existing)
        return
    snapshots = manifest["snapshots"].values()
    count = sum(s["count"] for s in snapshots)
    native = sum(s["native_count"] for s in snapshots)
    release.command(
        "gh",
        "release",
        "create",
        tag,
        str(archive),
        str(checksum),
        "--repo",
        release.REPOSITORY,
        "--verify-tag",
        "--latest=false",
        "--title",
        f"Particula chemistry catalog {version}",
        "--notes",
        f"{count:,} reaction entries across {len(snapshots)} "
        f"configurations, including {native:,} native conversions.",
    )


if __name__ == "__main__":
    publish()
