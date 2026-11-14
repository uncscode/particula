"""Execute script examples explicitly; render saved results without execution.

Python percent-format scripts are the source of truth. Notebook objects exist
only in memory while nbclient captures rich output. Published bundles are
a single successful generation selected by an atomic pointer replacement.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

GIT = shutil.which("git") or "git"
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "website/examples.json"
RESULTS = ROOT / "website/results"
FIGURE_DPI = 300


def registry():
    """Return the explicit example inventory."""
    return json.loads(CONFIG.read_text())


def sections(source):
    """Read percent sections without creating a notebook file."""
    import jupytext

    return jupytext.reads(source, fmt="py:percent").cells


def executable_signature(source):
    """Preserve code-section boundaries so saved outputs stay attached."""
    return [
        ast.dump(ast.parse(cell.source))
        for cell in sections(source)
        if cell.cell_type == "code"
    ]


def fingerprint(config, root=ROOT):
    """Hash executable semantics, settings, and declared input bytes."""
    payload = {
        "figure_dpi": FIGURE_DPI,
        "sections": executable_signature((root / config["source"]).read_text()),
        "invoke": config.get("invoke", ""),
        "device": config["device"],
        "input_destinations": config.get("input_destinations", {}),
        "inputs": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in config.get("inputs", [])
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def bundle(example_id, results=RESULTS):
    """Load the last successful result generation and its metadata."""
    pointer = results / example_id / "current.json"
    if not pointer.exists():
        return None, None
    generation = json.loads(pointer.read_text())["generation"]
    if Path(generation).name != generation:
        raise ValueError("Invalid result generation")
    directory = pointer.parent / generation
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        path = directory / name
        if (
            Path(name).name != name
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(f"Missing or corrupt saved output: {name}")
    return directory, manifest


def result_state(config, manifest, root=ROOT):
    """Distinguish absent, matching, and stale output without executing."""
    if manifest is None:
        return "missing"
    return (
        "current"
        if fingerprint(config, root) == manifest["fingerprint"]
        else "stale"
    )


def execute(config, root=ROOT):
    """Execute one script in a fresh kernel and temporary working directory."""
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    if config["device"] == "cuda":
        import warp as wp

        if not wp.is_cuda_available():
            raise RuntimeError(
                "CUDA unavailable; no CPU fallback or result update"
            )
        device = wp.get_device("cuda")
        device_name = f"{device.alias}: {device.name}"
    else:
        device_name = "CPU"
    source_path = root / config["source"]
    source = source_path.read_text()
    cells = [c for c in sections(source) if c.cell_type == "code"]
    module_name = (
        "particula_docs_example" if config.get("invoke") else "__main__"
    )
    bootstrap = (
        "import sys\n"
        f"sys.path[:0] = {json.dumps([str(root), str(source_path.parent)])}\n"
        f"__file__ = {str(source_path)!r}\n"
        "sys.modules['particula_docs_example'] = sys.modules['__main__']\n"
        f"__name__ = {module_name!r}\n"
        "%matplotlib inline\n"
        "from matplotlib_inline.backend_inline import InlineBackend\n"
        "InlineBackend.instance().figure_formats = {'png'}\n"
        "InlineBackend.instance().print_figure_kwargs = "
        f"{{'bbox_inches': 'tight', 'dpi': {FIGURE_DPI}}}\n"
    )
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell(bootstrap), *cells]
    )
    if config.get("invoke"):
        notebook.cells.append(nbformat.v4.new_code_cell(config["invoke"]))
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    with tempfile.TemporaryDirectory(prefix="particula-example-") as temporary:
        for name in config.get("inputs", []):
            destination = Path(temporary) / config.get(
                "input_destinations", {}
            ).get(name, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination)
        try:
            NotebookClient(
                notebook,
                km=km,
                timeout=config["timeout"],
                resources={"metadata": {"path": temporary}},
                allow_errors=False,
            ).execute()
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
    return notebook.cells[1:], device_name


def sanitize_execution_output(value, root=ROOT):
    """Remove local account and checkout paths without hiding diagnostics."""
    if isinstance(value, dict):
        return {
            key: sanitize_execution_output(item, root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_execution_output(item, root) for item in value]
    if not isinstance(value, str):
        return value
    # Preserve useful repository-relative paths in warnings and tracebacks.
    value = value.replace(str(root) + "/", "")
    value = value.replace(str(root), "<checkout>")
    value = re.sub(r"/(?:home|Users)/[^/\s\"'<>]+", "~", value)
    value = re.sub(r"[A-Za-z]:\\Users\\[^\\\s\"'<>]+", "~", value)
    value = re.sub(
        r"/tmp/(?:particula-example-|ipykernel_)[^/\s\"'<>]+",  # noqa: S108 -- redaction pattern, not file access
        "<execution-temp>",
        value,
    )
    return value


def package_versions():
    """Record installed versions relevant to scientific execution."""
    versions = {}
    for package in (
        "particula",
        "numpy",
        "scipy",
        "matplotlib",
        "nbclient",
        "jupytext",
        "warp-lang",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def refresh(example_id, config, root=ROOT, results=RESULTS):
    """Publish a complete result generation only after successful execution."""
    source = (root / config["source"]).read_bytes()
    initial_fingerprint = fingerprint(config, root)
    started = time.monotonic()
    cells, device = execute(config, root)
    elapsed = time.monotonic() - started
    if (root / config["source"]).read_bytes() != source or fingerprint(
        config, root
    ) != initial_fingerprint:
        raise RuntimeError(
            "Example changed during execution; refusing to save results"
        )
    parent = results / example_id
    parent.mkdir(parents=True, exist_ok=True)
    generation = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        (pending / "source.py").write_bytes(source)
        outputs = []
        for index, cell in enumerate(cells):
            rendered = []
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "image/png" in data or "image/svg+xml" in data:
                    mime = (
                        "image/png" if "image/png" in data else "image/svg+xml"
                    )
                    ext = "png" if mime == "image/png" else "svg"
                    filename = f"figure-{index + 1}-{len(rendered) + 1}.{ext}"
                    value = (
                        base64.b64decode(data[mime])
                        if ext == "png"
                        else data[mime].encode()
                    )
                    (pending / filename).write_bytes(value)
                    rendered.append({"image": filename})
                elif "text/html" in data:
                    rendered.append({"html": data["text/html"]})
                elif "text/plain" in data:
                    rendered.append({"text": data["text/plain"]})
                elif output.get("output_type") == "stream":
                    rendered.append({"text": output["text"]})
            outputs.append(rendered)
        revision = subprocess.check_output(  # noqa: S603 -- Git argv without a shell
            [GIT, "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(  # noqa: S603 -- Git argv without a shell
                [GIT, "status", "--porcelain", "--", "particula"],
                cwd=root,
                text=True,
            )
        )
        manifest = {
            "schema": 1,
            "figure_dpi": FIGURE_DPI,
            "elapsed_seconds": elapsed,
            "package_source_sha256": hashlib.sha256(
                b"".join(
                    str(p.relative_to(root)).encode() + p.read_bytes()
                    for p in sorted((root / "particula").rglob("*.py"))
                )
            ).hexdigest(),
            "input_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in config.get("inputs", [])
            },
            "example": example_id,
            "fingerprint": initial_fingerprint,
            "source": config["source"],
            "revision": revision,
            "package_dirty": dirty,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "python": sys.version.split()[0],
            "versions": package_versions(),
            "outputs": sanitize_execution_output(outputs, root),
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in pending.iterdir()
            },
        }
        (pending / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        pending.rename(parent / generation)
        pointer = parent / ".current.tmp"
        pointer.write_text(json.dumps({"generation": generation}) + "\n")
        os.replace(pointer, parent / "current.json")
        prune_superseded_results(example_id, parent, generation)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    print(
        f"REFRESHED {example_id}: {device}; "
        f"{sum(len(x) for x in outputs)} outputs"
    )


def prune_superseded_results(example_id, parent, current):
    """Remove only identified superseded bundles after publishing a successor.

    Failed refreshes leave the previous successful result untouched. Unknown
    directories and symlinks are never treated as disposable result bundles.
    Git history retains the provenance of previously committed generations.
    """
    for candidate in parent.glob("run-*"):
        if candidate.name == current or candidate.is_symlink():
            continue
        manifest = candidate / "manifest.json"
        if (
            not candidate.is_dir()
            or not manifest.is_file()
            or manifest.is_symlink()
        ):
            continue
        try:
            metadata = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict) and metadata.get("example") == example_id:
            shutil.rmtree(candidate)


def selected_changed(configs, base, root=ROOT):
    """Compare against a base revision, including staged and unstaged edits."""
    changed = set(
        subprocess.check_output(  # noqa: S603 -- Git argv without a shell
            [GIT, "diff", "--name-only", "-z", base, "--"], cwd=root
        )
        .decode()
        .strip("\0")
        .split("\0")
    )
    changed.update(
        subprocess.check_output(  # noqa: S603 -- Git argv without a shell
            [GIT, "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode()
        .strip("\0")
        .split("\0")
    )
    all_changed = bool(
        changed & {"website/examples.json", "scripts/docs_examples.py"}
    )
    selected = []
    for key, config in configs.items():
        if all_changed or changed.intersection(config.get("inputs", [])):
            selected.append(key)
        elif config["source"] in changed:
            previous = subprocess.run(  # noqa: S603 -- Git argv without a shell
                [GIT, "show", f"{base}:{config['source']}"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            current = (root / config["source"]).read_text()

            if previous.returncode or executable_signature(
                current
            ) != executable_signature(previous.stdout):
                selected.append(key)
    return selected


def main():
    """Run explicit refreshes or read-only validation of changed examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("refresh")
    run.add_argument("example")
    check = sub.add_parser("check")
    check.add_argument("--changed", action="store_true")
    check.add_argument("--base", default="upstream/main")
    args = parser.parse_args()
    configs = registry()
    if args.command == "refresh":
        if args.example not in configs:
            parser.error(f"Unknown example: {args.example}")
        refresh(args.example, configs[args.example])
        return
    selected = (
        selected_changed(configs, args.base) if args.changed else list(configs)
    )
    for key, config in configs.items():
        fingerprint(config)
        _, manifest = bundle(key)
        state = result_state(config, manifest)
        print(f"RESULTS {key}: {state}")
        if state != "current":
            raise RuntimeError(
                f"{key}: {state} results; explicit refresh required"
            )
    for key in selected:
        config = configs[key]
        if config["policy"] != "cpu":
            print(
                f"SKIP {key}: {config['policy']} execution "
                "requires explicit refresh"
            )
            continue
        execute(config)
        print(f"PASS {key}: validation only; publication results unchanged")
    if not selected:
        print("No executable changes selected")


if __name__ == "__main__":
    main()
