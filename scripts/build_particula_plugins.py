"""Validate and package Particula Helper (requires PyYAML).

Use --sync after editing canonical resources, --check in CI, and --output
to build deterministic release ZIPs from an explicitly allowed file set.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "particula-helper"
RESOURCES = ("skills", "scripts", "references", "assets")
FILES = ("plugin.json", "requirements.txt", "LICENSE", "README.md")
SKILLS = {
    "particula-setup",
    "particula-guide",
    "particula-build",
    "particula-run",
    "particula-analyze",
    "particula-debug",
    "particula-plan",
    "particula-develop",
    "particula-test",
    "particula-docs",
    "particula-review",
}


def json_bytes(value):
    """Encode stable, readable JSON."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def tree(path):
    """Read resource files without following symlinks or including caches."""
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"Symlink is not a package resource: {item}")
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        if item.is_file():
            result[item.relative_to(path).as_posix()] = item.read_bytes()
    return result


def canonical():
    """Select only maintained plugin resources."""
    files = {name: (PLUGIN / name).read_bytes() for name in FILES}
    for folder in RESOURCES:
        files.update(
            {f"{folder}/{k}": v for k, v in tree(PLUGIN / folder).items()}
        )
    return files


def validate():
    """Validate host identities, skill schemas, and contained resource paths."""
    manifest = json.loads((PLUGIN / "plugin.json").read_text())
    if (
        manifest.get("$schema")
        != "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
    ):
        raise ValueError("Missing portable plugin schema")
    version = manifest["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Plugin version must be stable SemVer")
    validate_host_versions(version)
    for name in (".claude-plugin", ".agents/plugins"):
        market = json.loads((ROOT / name / "marketplace.json").read_text())
        entry = market["plugins"][0]
        source = entry["source"]
        path = source if isinstance(source, str) else source["path"]
        if (
            market["name"] != "particula"
            or entry["name"] != "particula-helper"
            or path != "./plugins/particula-helper"
        ):
            raise ValueError(f"Marketplace drift: {name}")
    catalog = validate_skills()
    interface = manifest["extensions"]["com.openai"]["interface"]
    if len(interface["shortDescription"]) > 30:
        raise ValueError("Plugin subtitle exceeds 30 characters")
    for key in ("logo", "logoDark", "composerIcon", "composerIconDark"):
        target = (PLUGIN / interface[key]).resolve()
        if not target.is_relative_to(PLUGIN) or not target.is_file():
            raise ValueError(f"Invalid icon: {key}")
    return manifest, catalog


def validate_host_versions(version):
    """Keep all host manifests and the OpenCode lockfile in step."""
    for name in (
        ".claude-plugin/plugin.json",
        ".codex-plugin/plugin.json",
        "opencode/package.json",
        "opencode/npm-shrinkwrap.json",
    ):
        host = json.loads((PLUGIN / name).read_text())
        expected = (
            "particula-helper-opencode"
            if name.startswith("opencode")
            else "particula-helper"
        )
        if host["name"] != expected or host["version"] != version:
            raise ValueError(f"Host identity/version drift: {name}")
        if name == "opencode/npm-shrinkwrap.json":
            if host["packages"][""]["version"] != version:
                raise ValueError("OpenCode lockfile root version drift")


def validate_skills():
    """Validate the skill catalog and its contained resource references."""
    catalog = []
    for skill in sorted((PLUGIN / "skills").glob("*/SKILL.md")):
        text = skill.read_text()
        match = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
        if not match:
            raise ValueError(f"Missing frontmatter: {skill}")
        meta = yaml.safe_load(match[1])
        if (
            meta["name"] != skill.parent.name
            or not isinstance(meta.get("description"), str)
            or not meta["description"].strip()
        ):
            raise ValueError(f"Invalid skill metadata: {skill}")
        for relative in re.findall(r"`((?:\.\./|\./)[^`\s]+)", text):
            target = (skill.parent / relative).resolve()
            if not target.is_relative_to(PLUGIN) or not target.exists():
                raise ValueError(
                    f"Missing/escaping skill reference: {relative}"
                )
        catalog.append(
            {"name": meta["name"], "description": meta["description"]}
        )
    if {item["name"] for item in catalog} != SKILLS:
        raise ValueError(
            "Particula skill catalog does not match expected names"
        )
    return catalog


def generated(catalog):
    """Return the complete standalone OpenCode payload."""
    result = canonical()
    result.pop("plugin.json")
    result["catalog.json"] = json_bytes(catalog)
    return result


def sync_or_check(catalog, sync=False):
    """Regenerate only the owned payload, or reject stale/extra files."""
    payload = PLUGIN / "opencode" / "payload"
    expected = generated(catalog)
    if sync:
        if payload.exists():
            shutil.rmtree(payload)
        for name, content in expected.items():
            path = payload / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        for name in ("LICENSE", "README.md"):
            shutil.copyfile(PLUGIN / name, PLUGIN / "opencode" / name)
    if tree(payload) != expected:
        raise ValueError("OpenCode payload drift; run with --sync")
    for name in ("LICENSE", "README.md"):
        if (PLUGIN / name).read_bytes() != (
            PLUGIN / "opencode" / name
        ).read_bytes():
            raise ValueError(f"OpenCode {name} drift; run with --sync")


def write_zip(path, files):
    """Write reproducible ZIP bytes, independent of working-tree mtimes."""
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)


def check_public_archive(path):
    """Inspect the skills-only upload without claiming portal approval."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Duplicate or corrupt public archive entries")
        for name in names:
            parts = Path(name).parts
            if (
                Path(name).is_absolute()
                or ".." in parts
                or "\\" in name
                or any(part.startswith(".") for part in parts)
                or "hooks" in parts
                or name in ("mcp.json", "mcp_servers.json")
            ):
                raise ValueError(f"Unsupported public archive entry: {name}")
        manifest = json.loads(archive.read("plugin.json"))
        extension = manifest["extensions"]["com.openai"]
        validate_public_components(manifest)
        if extension.get("requires_local_executor") is not True:
            raise ValueError("Particula public upload requires local execution")
        skills = {
            name.split("/")[1]
            for name in names
            if re.fullmatch(r"skills/[^/]+/SKILL.md", name)
        }
        if skills != SKILLS:
            raise ValueError("Public archive is missing expected skills")
        for key in ("logo", "composerIcon", "logoDark", "composerIconDark"):
            icon = extension["interface"].get(key)
            if icon is not None:
                if not icon.startswith("./") or icon[2:] not in names:
                    raise ValueError(f"Missing public icon: {key}")
        return manifest


def validate_public_components(manifest):
    """Reject integrations that do not belong in this skills-only upload."""
    extension = manifest["extensions"]["com.openai"]
    for key in ("apps", "hooks", "mcpServers"):
        if manifest.get(key) is not None or extension.get(key) is not None:
            raise ValueError(f"Public upload cannot declare {key}")


def write_public_submission(output, files, provenance):
    """Build a portable upload candidate and an explicit publisher handoff."""
    manifest = json.loads(files["plugin.json"])
    version = manifest["version"]
    archive = output / f"particula-helper-openai-{version}.zip"
    write_zip(archive, {**files, "BUILD.json": provenance})
    check_public_archive(archive)
    extension = manifest["extensions"]["com.openai"]
    source = json.loads(provenance)
    notes = extension.get("publication", {}).get("release_notes", "")
    (output / "OPENAI-SUBMISSION.md").write_text(
        f"# Particula Helper {version}\n\n"
        f"Upload candidate: `{archive.name}`\n\n"
        f"Source commit: `{source['source_revision']}`\n\n"
        f"Package sources modified locally: {source['working_tree_dirty']}\n\n"
        f"{notes}\n\n"
        "## Publication handoff\n\n"
        "This is a locally validated package candidate, not an approved or "
        "published OpenAI release. GitHub Actions does not submit it.\n\n"
        "- Verify the requested Particula Developers identity in the intended "
        "OpenAI organization; the portal controls the displayed publisher.\n"
        "- Confirm all supported countries and no required payments or "
        "purchases in the imported metadata.\n"
        "- Finish and verify product, support, privacy and terms pages for "
        "the listing; no policy URLs or legal commitments are invented here.\n"
        "- Test a clean desktop installation and the main workflows. "
        "Archive smoke tests do not prove live host discovery.\n"
        "- Upload to the existing public plugin when one exists. Keep the "
        "same identity; do not create a duplicate on each release.\n"
        "- Inspect metadata and skill scan results, complete applicable "
        "attestations, submit for review, then publish after approval.\n\n"
        "Skills-only packages do not need MCP review cases, reviewer "
        "credentials or an MCP demo recording.\n\n"
        "[OpenAI submission process]"
        "(https://developers.openai.com/plugins/deploy/submission)\n"
    )


def write_github_release_notes(output, manifest, provenance):
    """Describe the tested GitHub packages independently of store submission."""
    source = json.loads(provenance)
    version = manifest["version"]
    notes = (
        manifest["extensions"]["com.openai"]
        .get("publication", {})
        .get("release_notes", "")
    )
    (output / "GITHUB-RELEASE.md").write_text(
        f"# Particula Helper {version}\n\n"
        f"{notes}\n\n"
        f"Source commit: `{source['source_revision']}`\n\n"
        f"Package sources modified locally: {source['working_tree_dirty']}\n\n"
        "Packages: `particula-helper-codex.zip`, "
        "`particula-helper-claude.zip`, `particula-helper-opencode.zip`.\n\n"
        "Download all ZIPs listed in `SHA256SUMS` and run "
        "`sha256sum -c SHA256SUMS` to verify them.\n\n"
        "## Install or refresh\n\n"
        "Use the [assistant installation guides]"
        "(https://uncscode.github.io/particula/guide/ai-assistants/) "
        "for Codex, Claude and OpenCode. Repository marketplace installs "
        "track their configured branch or tag. Refresh/reinstall the plugin "
        "and start a new session to load updates. Publishing an archive "
        "does not update an existing installation automatically.\n\n"
        "Every successful main build has a version-and-commit-named "
        "download under [Assistant plugins workflow runs]"
        "(https://github.com/uncscode/particula/actions/workflows/"
        "assistant-plugins.yml?query=branch%3Amain+event%3Apush). "
        "Workflow downloads expire after 30 days. Version bumps create "
        "draft GitHub releases for longer-term distribution.\n\n"
        "This distribution does not require OpenAI store verification. "
        "The optional OpenAI ZIP and OPENAI-SUBMISSION.md are retained "
        "for a possible future submission, not a published store listing.\n"
    )


def build(output, manifest):
    """Create host archives and checksums in a fresh output directory."""
    if output.resolve().is_relative_to(PLUGIN):
        raise ValueError("Output must be outside the plugin source")
    output.mkdir(parents=True, exist_ok=False)
    git = shutil.which("git")
    if git is None:
        raise ValueError("Git is required for source provenance")
    revision = subprocess.check_output(  # noqa: S603 - fixed Git provenance commands
        [git, "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(  # noqa: S603 - fixed Git provenance commands
            [
                git,
                "status",
                "--porcelain",
                "--",
                "plugins",
                "scripts/build_particula_plugins.py",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    files = canonical()
    provenance = json_bytes(
        {
            "plugin_version": manifest["version"],
            "source_revision": revision,
            "working_tree_dirty": dirty,
            "compatibility": "references/compatibility.md",
            "sha256": {
                k: hashlib.sha256(v).hexdigest() for k, v in files.items()
            },
        }
    )
    for host, overlay in (
        ("codex", ".codex-plugin"),
        ("claude", ".claude-plugin"),
    ):
        selected = dict(files)
        selected[f"{overlay}/plugin.json"] = (
            PLUGIN / overlay / "plugin.json"
        ).read_bytes()
        selected["BUILD.json"] = provenance
        write_zip(output / f"particula-helper-{host}.zip", selected)
    write_public_submission(output, files, provenance)
    write_github_release_notes(output, manifest, provenance)
    # Native package plus a manual skills alternative in one OpenCode ZIP.
    native = {
        k: (PLUGIN / "opencode" / k).read_bytes()
        for k in (
            "index.js",
            "package.json",
            "npm-shrinkwrap.json",
            "LICENSE",
            "README.md",
        )
    }
    native.update(
        {
            f"payload/{k}": v
            for k, v in tree(PLUGIN / "opencode/payload").items()
        }
    )
    selected = {f"particula-helper/{k}": v for k, v in native.items()}
    selected["particula-helper/BUILD.json"] = provenance
    for skill in sorted(SKILLS):
        prefix = f".opencode/skills/{skill}/"
        for name, content in files.items():
            if name.startswith(f"skills/{skill}/"):
                local = name.removeprefix(f"skills/{skill}/")
                if local == "SKILL.md":
                    content = content.replace(b"../../", b"./")
                selected[prefix + local] = content
            elif name.startswith(("scripts/", "references/")) or name in (
                "requirements.txt",
                "LICENSE",
            ):
                selected[prefix + name] = content
    write_zip(output / "particula-helper-opencode.zip", selected)
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(output.glob("*.zip"))
        )
    )


def main():
    """Run the explicit sync/check/build operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest, catalog = validate()
    sync_or_check(catalog, sync=args.sync)
    if args.output:
        build(args.output, manifest)
    print(
        f"Particula Helper: {len(catalog)} skills and host packages validated"
    )


if __name__ == "__main__":
    main()
