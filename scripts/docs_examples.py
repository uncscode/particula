"""Execute script examples explicitly; render saved results without execution.

Python percent-format scripts are the source of truth. Notebook objects exist
only in memory while nbclient captures rich output. Published bundles are
one stable output folder per example, with Git preserving earlier versions.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.metadata
import json
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
MAX_PLOT_DATA_BYTES = 10 * 1024 * 1024


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
    for key in (
        "data_expression",
        "replot_invoke",
        "render_functions",
        "plot_inputs",
    ):
        if key in config:
            payload[key] = config[key]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def bundle(example_id, results=RESULTS):
    """Load the last successful result generation and its metadata."""
    directory = results / example_id
    pointer = directory / "current.json"
    if not (directory / "manifest.json").exists():
        if not pointer.exists():
            return None, None
        generation = json.loads(pointer.read_text())["generation"]
        if Path(generation).name != generation:
            raise ValueError("Invalid result generation")
        directory = directory / generation
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


def execute(config, root=ROOT, *, saved_data=None):
    """Execute one script in a fresh kernel and temporary working directory."""
    import nbformat
    from jupyter_client import KernelManager
    from nbclient import NotebookClient

    if config["device"] == "cuda" and saved_data is None:
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
        invocation = (
            config["invoke"] if saved_data is None else config["replot_invoke"]
        )
        notebook.cells.append(nbformat.v4.new_code_cell(invocation))
    if config.get("data_expression"):
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                "from scripts.docs_examples import write_plot_data\n"
                "write_plot_data('_plot_data.json', "
                + config["data_expression"]
                + ")"
            )
        )
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
        if saved_data is not None:
            (Path(temporary) / "_saved_data.json").write_text(
                json.dumps(saved_data, allow_nan=False)
            )
            notebook.cells[0].source += (
                "import json\nfrom pathlib import Path\n"
                "saved_data = json.loads(\n"
                "    Path('_saved_data.json').read_text())\n"
            )
        try:
            NotebookClient(
                notebook,
                km=km,
                timeout=config["timeout"],
                resources={"metadata": {"path": temporary}},
                allow_errors=False,
            ).execute()
            if config.get("data_expression"):
                data = json.loads(
                    (Path(temporary) / "_plot_data.json").read_text()
                )
                notebook.cells.pop()
                notebook.cells[1].metadata["plot_data"] = data
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


def write_plot_data(path, data):
    """Stream JSON summaries with a hard byte cap and no silent truncation."""
    path = Path(path)
    size = 0
    try:
        with path.open("wb") as output:
            for chunk in json.JSONEncoder(allow_nan=False).iterencode(data):
                encoded = chunk.encode("utf-8")
                size += len(encoded)
                if size > MAX_PLOT_DATA_BYTES:
                    raise ValueError("Plot data exceeds the 10 MiB limit")
                output.write(encoded)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_replot_data(example_id, config, root, results):
    """Reject missing, altered or scientifically stale numerical data."""
    directory, original = bundle(example_id, results)
    if not original or "data.json" not in original["files"]:
        raise ValueError(
            "No saved numerical data; a data-saving refresh is required"
        )
    if original.get("simulation_fingerprint") != simulation_fingerprint(
        config, root
    ):
        raise ValueError(
            "Simulation inputs changed; refresh required before replot"
        )
    if original["package_source_sha256"] != package_source_hash(root):
        raise ValueError(
            "Library source changed; refresh required before replot"
        )
    if (directory / "data.json").stat().st_size > MAX_PLOT_DATA_BYTES:
        raise ValueError("Saved plot data exceeds the 10 MiB limit")
    return (
        directory,
        original,
        json.loads((directory / "data.json").read_text()),
    )


def save_cell_outputs(cells, pending):
    """Capture displayed figures and text from executed cells."""
    outputs = []
    for index, cell in enumerate(cells):
        rendered = []
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data or "image/svg+xml" in data:
                mime = "image/png" if "image/png" in data else "image/svg+xml"
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
    return outputs


def validate_render(config, root, source, initial_fingerprint, cells, data):
    """Reject source drift or renderer mutation before touching published output."""
    if data is not None and cells[0]["metadata"]["plot_data"] != data:
        raise ValueError(
            "Renderer changed saved numerical data; refusing publication"
        )
    if (root / config["source"]).read_bytes() != source or fingerprint(
        config, root
    ) != initial_fingerprint:
        raise RuntimeError(
            "Example changed during execution; refusing to save results"
        )


def refresh(example_id, config, root=ROOT, results=RESULTS, *, replot=False):
    """Publish a complete result generation only after successful execution."""
    source = (root / config["source"]).read_bytes()
    initial_fingerprint = fingerprint(config, root)
    original = None
    directory = None
    data = None
    if replot:
        directory, original, data = load_replot_data(
            example_id, config, root, results
        )
    started = time.monotonic()
    cells, device = (
        execute(config, root, saved_data=data)
        if replot
        else execute(config, root)
    )
    elapsed = time.monotonic() - started
    validate_render(config, root, source, initial_fingerprint, cells, data)
    parent = results / example_id
    results.mkdir(parents=True, exist_ok=True)
    pending = Path(
        tempfile.mkdtemp(prefix=f".pending-{example_id}-", dir=results)
    )
    try:
        (pending / "source.py").write_bytes(source)
        if config.get("data_expression"):
            write_plot_data(
                pending / "data.json", cells[0]["metadata"]["plot_data"]
            )
        if replot:
            executed_source = directory / "executed_source.py"
            if not executed_source.exists():
                executed_source = directory / "source.py"
            shutil.copy2(executed_source, pending / "executed_source.py")
        outputs = save_cell_outputs(cells, pending)
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
            "package_source_sha256": package_source_hash(root),
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
        if config.get("data_expression"):
            manifest["simulation_fingerprint"] = simulation_fingerprint(
                config, root
            )
        if replot:
            for key in (
                "elapsed_seconds",
                "package_source_sha256",
                "revision",
                "package_dirty",
                "executed_at",
                "device",
                "python",
                "versions",
            ):
                manifest[key] = original[key]
            manifest["execution_input_sha256"] = original.get(
                "execution_input_sha256", original["input_sha256"]
            )
            manifest["rendered_at"] = datetime.now(timezone.utc).isoformat()
            manifest["render_revision"] = revision
            manifest["render_elapsed_seconds"] = elapsed
        (pending / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        publish_bundle(pending, parent)
    finally:
        if pending.exists():
            shutil.rmtree(pending)
    print(
        f"{'REPLOTTED' if replot else 'REFRESHED'} {example_id}: {device}; "
        f"{sum(len(x) for x in outputs)} outputs"
    )


def package_source_hash(root=ROOT):
    """Identify the scientific library used to generate saved numerical data."""
    return hashlib.sha256(
        b"".join(
            str(p.relative_to(root)).encode() + p.read_bytes()
            for p in sorted((root / "particula").rglob("*.py"))
        )
    ).hexdigest()


def simulation_fingerprint(config, root=ROOT):
    """Hash source and inputs excluding declared rendering-only components."""
    tree = ast.parse((root / config["source"]).read_text())
    render_functions = config.get("render_functions", [])
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in render_functions
        )
    ]
    payload = {
        "source": ast.dump(tree),
        "invoke": config.get("invoke"),
        "device": config["device"],
        "input_destinations": config.get("input_destinations", {}),
        "inputs": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in config.get("inputs", [])
            if name not in config.get("plot_inputs", [])
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def publish_bundle(pending, destination):
    """Replace a stable example folder, restoring the old bundle on failure."""
    backup = destination.with_name(".backup-" + destination.name)
    if backup.exists():
        raise RuntimeError(
            f"Unfinished publication backup requires recovery: {backup}"
        )
    existed = destination.exists()
    if existed:
        destination.rename(backup)
    try:
        pending.rename(destination)
    except BaseException:
        if existed:
            backup.rename(destination)
        raise
    if existed:
        shutil.rmtree(backup)


def migrate_results(example_id, results=RESULTS):
    """Flatten a verified legacy bundle, preserving execution provenance."""
    directory, manifest = bundle(example_id, results)
    destination = results / example_id
    if manifest is None or directory == destination:
        return
    pending = Path(
        tempfile.mkdtemp(prefix=f".pending-{example_id}-", dir=results)
    )
    try:
        for name in [*manifest["files"], "manifest.json"]:
            shutil.copy2(directory / name, pending / name)
        # Refuse to discard unrelated content during this one-time migration.
        for entry in destination.iterdir():
            if entry.name == "current.json":
                continue
            metadata = entry / "manifest.json"
            if entry.is_symlink() or not metadata.is_file():
                raise ValueError(
                    f"Unexpected content in legacy bundle: {entry}"
                )
            if json.loads(metadata.read_text()).get("example") != example_id:
                raise ValueError(f"Unrelated legacy bundle: {entry}")
        publish_bundle(pending, destination)
    finally:
        if pending.exists():
            shutil.rmtree(pending)


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
    replot = sub.add_parser("replot")
    replot.add_argument("example")
    sub.add_parser("migrate-results")
    check = sub.add_parser("check")
    check.add_argument("--changed", action="store_true")
    check.add_argument("--base", default="upstream/main")
    args = parser.parse_args()
    configs = registry()
    if args.command == "migrate-results":
        for key in configs:
            migrate_results(key)
        return
    if args.command in ("refresh", "replot"):
        if args.example not in configs:
            parser.error(f"Unknown example: {args.example}")
        refresh(
            args.example, configs[args.example], replot=args.command == "replot"
        )
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
