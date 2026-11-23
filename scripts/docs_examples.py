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
STYLE_INPUTS = ("particula/default.mplstyle", "particula/plotting/__init__.py")
MAX_PLOT_DATA_BYTES = 10 * 1024 * 1024


def registry():
    """Return the explicit example inventory."""
    return json.loads(CONFIG.read_text())


def import_study(key):
    """Import a published Python study using its stable execution key."""
    source = Path(registry()[key]["source"])
    return importlib.import_module(".".join(source.with_suffix("").parts))


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


def scientific_inputs(config):
    """Include installed science code when hashing study dependencies."""
    return [*config.get("inputs", []), *config.get("science_inputs", [])]


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
            for name in scientific_inputs(config)
        },
    }
    if config.get("full_canvas"):
        payload["full_canvas"] = True
        payload["style_files"] = {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in STYLE_INPUTS
        }
    for key in (
        "data_expression",
        "replot_invoke",
        "render_functions",
        "plot_inputs",
        "historical_run",
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


def use_historical_package(science_root, renderer_root):
    """Load the pinned science package and its separately versioned renderer."""
    from importlib.util import module_from_spec, spec_from_file_location

    import matplotlib as mpl
    from matplotlib import style

    sys.path.insert(0, science_root)
    try:
        import particula
    finally:
        # The package retains its own archive path. Other example imports
        # must use the same checkout whose inputs are hashed by the runner.
        sys.path.remove(science_root)

    if (
        Path(particula.__file__).resolve().parent
        != Path(science_root) / "particula"
    ):
        raise RuntimeError(
            "The kernel imported a different scientific checkout"
        )
    root = Path(renderer_root)
    spec = spec_from_file_location("particula.plotting", root / STYLE_INPUTS[1])
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    particula.plotting = module
    style.library["particula.default"] = mpl.rc_params_from_file(
        root / STYLE_INPUTS[0], use_default_template=False
    )


def execute(config, root=ROOT, *, saved_data=None, science_root=None):
    """Execute one script in a fresh kernel and temporary working directory."""
    import nbformat
    from jupyter_client import AsyncKernelManager
    from jupyter_core.utils import run_sync
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
    bbox = None if config.get("full_canvas") else "tight"
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
        f"{{'bbox_inches': {bbox!r}, 'dpi': {FIGURE_DPI}}}\n"
    )
    if science_root is not None:
        bootstrap += (
            "import runpy\n"
            f"tool = runpy.run_path({__file__!r})\n"
            "tool['use_historical_package']("
            f"{str(science_root)!r}, {str(root)!r})\n"
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
    km = AsyncKernelManager(kernel_name="python3")
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
            ).execute(cleanup_kc=True)
            if config.get("data_expression"):
                data = json.loads(
                    (Path(temporary) / "_plot_data.json").read_text()
                )
                notebook.cells.pop()
                notebook.cells[1].metadata["plot_data"] = data
        finally:
            if km.has_kernel:
                run_sync(km.shutdown_kernel)(now=True)
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


def package_versions(root=ROOT):
    """Record checkout identity separately from stale installed metadata."""
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
    if (root / "particula" / "__init__.py").is_file():
        from particula_docs import source_version

        version = source_version(root / "particula")
        installed = versions.get("particula")
        if installed is not None and installed != version:
            versions["particula-distribution"] = installed
        versions["particula"] = version
    return versions


def write_plot_data(path, data):
    """Stream JSON summaries with a hard byte cap and no silent truncation."""
    if not isinstance(data, (dict, list)):
        raise ValueError("Plot data must be a JSON object or array")
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


def load_replot_data(example_id, config, root, results, *, science_root=None):
    """Reject missing, altered or scientifically stale numerical data."""
    directory, original = bundle(example_id, results)
    if not original or "data.json" not in original["files"]:
        raise ValueError(
            "No saved numerical data; a data-saving refresh is required"
        )
    if original.get("historical_run") and science_root is None:
        raise ValueError(
            "Historical replot requires the pinned science checkout"
        )
    if science_root is not None:
        config, root = historical_study_inputs(
            example_id, config, root, science_root, original
        )
    if original.get("simulation_fingerprint") != simulation_fingerprint(
        config, root
    ):
        raise ValueError(
            "Simulation inputs changed; refresh required before replot"
        )
    if not science_matches(original, science_root or root):
        raise ValueError(
            "Library source changed; refresh required before replot"
        )
    if (directory / "data.json").stat().st_size > MAX_PLOT_DATA_BYTES:
        raise ValueError("Saved plot data exceeds the 10 MiB limit")
    data = json.loads((directory / "data.json").read_text())
    if not isinstance(data, (dict, list)):
        raise ValueError("Plot data must be a JSON object or array")
    return directory, original, data


def historical_study_inputs(example_id, config, root, science_root, original):
    """Locate the study that executed against the pinned scientific library."""
    pin = config.get("historical_run")
    if not pin:
        raise ValueError("This example has no pinned historical run")
    verify_execution_checkout(
        config,
        science_root,
        recorded_package_hash=original["package_source_sha256"],
    )
    if original["revision"] != pin["revision"]:
        # A simulation run before committing records its parent revision.
        # Accept the immediately following commit only with the observed bytes.
        if not original.get("package_dirty") or original[
            "package_source_sha256"
        ] != pin.get("package_source_sha256"):
            raise ValueError(
                "Saved data does not match the pinned historical run"
            )
        parents = subprocess.check_output(  # noqa: S603 -- pinned Git checkout
            [GIT, "show", "-s", "--format=%P", "HEAD"],
            cwd=science_root,
            text=True,
        ).split()
        if parents != [original["revision"]]:
            raise ValueError(
                "Saved data does not match the pinned historical run"
            )

    # A fresh archived-library run executes the study in the author checkout.
    # Its observed data may include new timings. Keep its execution digest
    # across redraws, independently of the reference archive's data digest.
    if "plot_source_revision" in original:
        recorded_pin = original["historical_run"]
        if recorded_pin["data_sha256"] != pin["data_sha256"]:
            raise ValueError("Historical reference data pin changed")
        expected = original.get("execution_data_sha256", pin["data_sha256"])
        study_config, study_root = config, root
    else:
        expected = pin["data_sha256"]
        study_config = json.loads(
            (science_root / "website/examples.json").read_text()
        )[example_id]
        study_root = science_root
    if original["files"]["data.json"] != expected:
        raise ValueError("Saved data does not match the pinned historical run")
    return study_config, study_root


def save_cell_outputs(cells, pending):
    """Capture displayed figures and text from executed cells."""
    outputs = []
    for index, cell in enumerate(cells):
        rendered = []
        image_index = 0
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data or "image/svg+xml" in data:
                mime = "image/png" if "image/png" in data else "image/svg+xml"
                ext = "png" if mime == "image/png" else "svg"
                image_index += 1
                filename = f"figure-{index + 1}-{image_index}.{ext}"
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
    """Reject source drift or data mutation before publication."""
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


def retain_execution_metadata(
    manifest, original, config, science_root, device, revision, elapsed
):
    """Identify the rendering separately from the original simulation."""
    if science_root is not None:
        manifest["historical_run"] = config["historical_run"]
        manifest["simulation_fingerprint"] = original["simulation_fingerprint"]
    for key in ("plot_source_revision", "execution_data_sha256"):
        if key in original:
            manifest[key] = original[key]
    manifest["render_versions"] = manifest["versions"]
    manifest["render_device"] = device
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


def verify_execution_checkout(
    config, science_root, *, recorded_package_hash=None
):
    """Require the recorded scientific revision and exact package contents."""
    pin = config.get("historical_run", {})
    revision = subprocess.check_output(  # noqa: S603 -- fixed Git arguments
        [GIT, "rev-parse", "HEAD"], cwd=science_root, text=True
    ).strip()
    if revision != pin.get("revision") or (
        package_source_hash(science_root)
        != pin.get("package_source_sha256", recorded_package_hash)
    ):
        raise ValueError(
            "Scientific checkout differs from the pinned execution"
        )


def record_historical_execution(manifest, config, historical):
    """Record both source revisions of a fresh run with an archived model."""
    if historical:
        manifest["historical_run"] = config["historical_run"]
        manifest["plot_source_revision"] = manifest["revision"]
        if "data.json" in manifest["files"]:
            manifest["execution_data_sha256"] = manifest["files"]["data.json"]
        manifest["revision"] = config["historical_run"]["revision"]


def refresh(
    example_id,
    config,
    root=ROOT,
    results=RESULTS,
    *,
    replot=False,
    science_root=None,
):
    """Publish a complete result generation only after successful execution."""
    historical_execution = science_root is not None and not replot
    execution_root = root
    options = {}
    if historical_execution:
        execution_root = Path(science_root).resolve()
        verify_execution_checkout(config, execution_root)
        options["science_root"] = execution_root
    source = (root / config["source"]).read_bytes()
    initial_fingerprint = fingerprint(config, root)
    initial_package_hash = package_source_hash(execution_root)
    initial_science_hash = package_source_hash(
        science_root or execution_root, science_only=True
    )
    initial_versions = package_versions(execution_root)
    original = None
    directory = None
    data = None
    if replot:
        directory, original, data = load_replot_data(
            example_id, config, root, results, science_root=science_root
        )
        options["saved_data"] = data
    started = time.monotonic()
    cells, device = execute(config, root, **options)
    elapsed = time.monotonic() - started
    if package_source_hash(execution_root) != initial_package_hash:
        raise RuntimeError(
            "Library changed during execution; refusing to save results"
        )
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
                cwd=execution_root,
                text=True,
            )
        )
        manifest = {
            "schema": 1,
            "figure_dpi": FIGURE_DPI,
            "elapsed_seconds": elapsed,
            "package_source_sha256": initial_package_hash,
            "scientific_source_sha256": initial_science_hash,
            "input_sha256": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in scientific_inputs(config)
            },
            "example": example_id,
            "fingerprint": initial_fingerprint,
            "source": config["source"],
            "revision": revision,
            "package_dirty": dirty,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "python": sys.version.split()[0],
            "versions": initial_versions,
            "outputs": sanitize_execution_output(outputs, root),
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in pending.iterdir()
            },
        }
        record_historical_execution(manifest, config, historical_execution)
        if config.get("data_expression"):
            manifest["simulation_fingerprint"] = simulation_fingerprint(
                config, root
            )
        if replot:
            retain_execution_metadata(
                manifest,
                original,
                config,
                science_root,
                device,
                revision,
                elapsed,
            )
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


def science_matches(manifest, root):
    """Verify data against science alone, or the full tree for legacy runs."""
    if "scientific_source_sha256" in manifest:
        return manifest["scientific_source_sha256"] == package_source_hash(
            root, science_only=True
        )
    return manifest["package_source_sha256"] == package_source_hash(root)


def package_source_hash(root=ROOT, *, science_only=False):
    """Hash Python sources, optionally omitting the plotting package."""
    return hashlib.sha256(
        b"".join(
            str(p.relative_to(root)).encode() + p.read_bytes()
            for p in sorted((root / "particula").rglob("*.py"))
            if not (
                science_only and p.is_relative_to(root / "particula/plotting")
            )
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
            for name in scientific_inputs(config)
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
        dependencies = set(scientific_inputs(config))
        if config.get("full_canvas"):
            dependencies.update(STYLE_INPUTS)
        if all_changed or changed.intersection(dependencies):
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


def validate_example(key, config):
    """Reuse render-only data when science matches; otherwise run validation."""
    if config["policy"] != "cpu":
        print(
            f"SKIP {key}: {config['policy']} execution "
            "requires explicit refresh"
        )
        return
    _, manifest = bundle(key)
    if (
        manifest.get("rendered_at")
        and config.get("data_expression")
        and science_matches(manifest, ROOT)
        and manifest.get("simulation_fingerprint")
        == simulation_fingerprint(config, ROOT)
    ):
        _, _, data = load_replot_data(key, config, ROOT, RESULTS)
        execute(config, saved_data=data)
        print(f"PASS {key}: saved-data rendering; simulation not rerun")
    else:
        execute(config)
        print(f"PASS {key}: validation only; publication results unchanged")


def main():
    """Run explicit refreshes or read-only validation of changed examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("refresh")
    run.add_argument("example")
    run.add_argument("--science-checkout", type=Path)
    replot = sub.add_parser("replot")
    replot.add_argument("example")
    replot.add_argument(
        "--science-checkout",
        type=Path,
        help="Verify archived data against its original scientific checkout.",
    )
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
            args.example,
            configs[args.example],
            replot=args.command == "replot",
            science_root=getattr(args, "science_checkout", None),
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
        validate_example(key, configs[key])
    if not selected:
        print("No executable changes selected")


if __name__ == "__main__":
    main()
