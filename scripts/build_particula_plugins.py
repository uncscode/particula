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
    for name in (
        ".claude-plugin/plugin.json",
        ".codex-plugin/plugin.json",
        "opencode/package.json",
    ):
        host = json.loads((PLUGIN / name).read_text())
        expected = (
            "particula-helper-opencode"
            if name.startswith("opencode")
            else "particula-helper"
        )
        if host["name"] != expected or host["version"] != version:
            raise ValueError(f"Host identity/version drift: {name}")
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


def validate_skills():
    """Validate the six skill definitions and their resource references."""
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
        raise ValueError("Expected exactly six Particula skills")
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
    print("Particula Helper: six skills and host packages validated")


if __name__ == "__main__":
    main()
