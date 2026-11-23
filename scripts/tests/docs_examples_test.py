"""Check provenance, source selection, and publication transaction boundaries."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docs_examples as examples  # noqa: E402


@pytest.fixture
def example(tmp_path):
    """Make an isolated percent-format source with one declared input."""
    (tmp_path / "example.py").write_text(
        "# %% [markdown]\n# Explanation\n# %%\nx = 2\nprint(x)\n"
    )
    (tmp_path / "input.txt").write_text("input")
    config = {
        "source": "example.py",
        "device": "cpu",
        "policy": "cpu",
        "timeout": 2,
        "inputs": ["input.txt"],
    }
    return tmp_path, config


def test_prose_and_formatting_preserve_results_but_code_and_inputs_invalidate(
    example,
):
    """Prose and formatting preserve results but code and inputs invalidate."""
    root, config = example
    before = examples.fingerprint(config, root)
    source = root / config["source"]
    source.write_text(
        source.read_text()
        .replace("Explanation", "New explanation")
        .replace("x = 2", "x=2 # comment")
    )
    assert examples.fingerprint(config, root) == before
    source.write_text(source.read_text().replace("x=2", "x=3"))
    assert (
        examples.result_state(config, {"fingerprint": before}, root) == "stale"
    )
    after = examples.fingerprint(config, root)
    (root / "input.txt").write_text("new input")
    assert examples.fingerprint(config, root) != after
    assert examples.result_state(config, None, root) == "missing"


def test_failed_execution_never_replaces_saved_results(example, monkeypatch):
    """Failed execution never replaces saved results."""
    root, config = example
    parent = root / "results/demo"
    parent.mkdir(parents=True)
    pointer = parent / "current.json"
    pointer.write_text('{"generation": "previous"}')

    def fail(*args):
        raise TimeoutError("CPU execution timed out")

    monkeypatch.setattr(examples, "execute", fail)
    with pytest.raises(TimeoutError):
        examples.refresh("demo", config, root, root / "results")
    assert pointer.read_text() == '{"generation": "previous"}'


def test_library_change_during_execution_preserves_previous_bundle(
    example, monkeypatch
):
    """A successful calculation cannot be attributed to different source bytes."""
    root, config = example
    library = root / "particula"
    library.mkdir()
    source = library / "physics.py"
    source.write_text("coefficient = 1\n")
    published = root / "results/demo"
    published.mkdir(parents=True)
    manifest = published / "manifest.json"
    manifest.write_text("previous verified result")

    def changed(*args):
        source.write_text("coefficient = 2\n")
        return [], "CPU"

    monkeypatch.setattr(examples, "execute", changed)
    with pytest.raises(RuntimeError, match="Library changed during execution"):
        examples.refresh("demo", config, root, root / "results")
    assert manifest.read_text() == "previous verified result"
    assert not list((root / "results").glob(".pending-*"))


def test_checkout_version_is_distinct_from_installed_metadata(
    tmp_path, monkeypatch
):
    """Editable source identity wins without disguising stale wheel metadata."""
    package = tmp_path / "particula"
    package.mkdir()
    (package / "__init__.py").write_text('__version__ = "9.8.7"\n')
    monkeypatch.setattr(examples.importlib.metadata, "version", lambda _: "old")
    versions = examples.package_versions(tmp_path)
    assert versions["particula"] == "9.8.7"
    assert versions["particula-distribution"] == "old"
    assert versions["numpy"] == "old"


def test_code_section_changes_invalidate_output_placement(example):
    """Splitting code must not reuse output attached to the wrong section."""
    root, config = example
    before = examples.fingerprint(config, root)
    source = root / config["source"]
    source.write_text(source.read_text().replace("print(x)", "# %%\nprint(x)"))
    assert examples.fingerprint(config, root) != before


def test_manifest_detects_missing_or_modified_figures(tmp_path):
    """Manifest detects missing or modified figures."""
    path = tmp_path / "demo/run"
    path.mkdir(parents=True)
    (path.parent / "current.json").write_text('{"generation": "run"}')
    (path / "figure.png").write_bytes(b"original")
    (path / "manifest.json").write_text(
        json.dumps(
            {"files": {"figure.png": hashlib.sha256(b"original").hexdigest()}}
        )
    )
    assert examples.bundle("demo", tmp_path)[0] == path
    (path / "figure.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        examples.bundle("demo", tmp_path)


def test_changed_selection_excludes_prose_only_and_includes_inputs(example):
    """Changed selection excludes prose only and includes inputs."""
    root, config = example

    def git(*args):
        return subprocess.run(  # noqa: S603 -- fixed test Git argument array
            [examples.GIT, *args], cwd=root, check=True, capture_output=True
        )

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Tests",
        "-c",
        "user.email=tests@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    source = root / "example.py"
    source.write_text(
        source.read_text().replace("Explanation", "Revised prose")
    )
    assert examples.selected_changed({"demo": config}, "HEAD", root) == []
    (root / "input.txt").write_text("changed")
    assert examples.selected_changed({"demo": config}, "HEAD", root) == ["demo"]


def test_policy_skips_cuda_without_invoking_runner(monkeypatch, capsys):
    """Policy skips cuda without invoking runner."""
    monkeypatch.setattr(
        examples, "registry", lambda: {"gpu": {"policy": "cuda"}}
    )
    monkeypatch.setattr(examples, "fingerprint", lambda *args: "hash")
    monkeypatch.setattr(
        examples, "bundle", lambda _: (None, {"fingerprint": "hash"})
    )
    monkeypatch.setattr(
        examples,
        "execute",
        lambda _: pytest.fail("CUDA must not execute in check"),
    )
    monkeypatch.setattr(sys, "argv", ["docs_examples", "check"])
    examples.main()
    assert "SKIP gpu" in capsys.readouterr().out


def test_output_redaction_preserves_scientific_warning_details(tmp_path):
    """Checkout/account paths disappear but warnings retain location and text."""
    value = [
        {
            "text": (
                f"{tmp_path}/particula/kernel.py:203: RuntimeWarning: divide by zero\n"
                "/home/alice/.cache/warp/module.py\n"
                "/Users/bob/project/data.csv\n"
                "C:\\Users\\charlie\\AppData\\cache\n"
                "/tmp/particula-example-abcd/data.csv\n"  # noqa: S108 -- test text
                "Mass balance: 1.0"
            )
        }
    ]
    result = examples.sanitize_execution_output(value, tmp_path)
    text = result[0]["text"]
    assert "particula/kernel.py:203: RuntimeWarning: divide by zero" in text
    assert "~/.cache/warp/module.py" in text
    assert "~/project/data.csv" in text
    assert "~\\AppData\\cache" in text
    assert "<execution-temp>/data.csv" in text
    assert "Mass balance: 1.0" in text
    assert all(name not in text for name in ("alice", "bob", "charlie"))
    assert result != value


def test_migration_preserves_files_metadata_and_removes_run_folders(tmp_path):
    """Flatten only the verified selected generation; Git owns old history."""
    parent = tmp_path / "demo"
    for name in ("run-old", "run-new"):
        directory = parent / name
        directory.mkdir(parents=True)
        (directory / "source.py").write_text(name)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "example": "demo",
                    "revision": "original",
                    "files": {
                        "source.py": hashlib.sha256(name.encode()).hexdigest()
                    },
                }
            )
        )
    (parent / "current.json").write_text('{"generation": "run-new"}')
    examples.migrate_results("demo", tmp_path)
    directory, manifest = examples.bundle("demo", tmp_path)
    assert directory == parent
    assert (parent / "source.py").read_text() == "run-new"
    assert manifest["revision"] == "original"
    assert sorted(p.name for p in parent.iterdir()) == [
        "manifest.json",
        "source.py",
    ]
    examples.migrate_results("demo", tmp_path)  # idempotent


def test_publication_failure_restores_previous_folder(tmp_path, monkeypatch):
    """A failed replacement must not discard the last successful output."""
    old, new = tmp_path / "demo", tmp_path / ".pending-demo"
    old.mkdir()
    new.mkdir()
    (old / "figure.png").write_bytes(b"old")
    original = Path.rename

    def fail(source, target):
        if source == new:
            raise OSError("publication failure")
        return original(source, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="publication failure"):
        examples.publish_bundle(new, old)
    assert (old / "figure.png").read_bytes() == b"old"
    assert not (tmp_path / ".backup-demo").exists()


def test_plot_data_size_cap_rejects_without_partial_file(tmp_path, monkeypatch):
    """Enforce the serialized byte limit, not just a count of data points."""
    monkeypatch.setattr(examples, "MAX_PLOT_DATA_BYTES", 32)
    path = tmp_path / "data.json"
    examples.write_plot_data(path, {"x": [1, 2]})
    assert json.loads(path.read_text()) == {"x": [1, 2]}
    with pytest.raises(ValueError, match="10 MiB"):
        examples.write_plot_data(path, {"x": "a" * 33})
    assert not path.exists()
    with pytest.raises(ValueError):
        examples.write_plot_data(path, {"x": float("nan")})
    assert not path.exists()


def test_simulation_fingerprint_excludes_only_rendering(example):
    """Plot styling can change; settings, scientific helpers and drivers cannot."""
    root, config = example
    source = root / "example.py"
    source.write_text(
        "def simulate():\n    return 4\ndef plot_results(x):\n    return x\n"
    )
    config.update(render_functions=["plot_results"], plot_inputs=["style.py"])
    config["inputs"].append("style.py")
    (root / "style.py").write_text("color = 'blue'")
    before = examples.simulation_fingerprint(config, root)
    source.write_text(source.read_text().replace("return x", "return x + 0"))
    (root / "style.py").write_text("color = 'red'")
    assert examples.simulation_fingerprint(config, root) == before
    source.write_text(source.read_text().replace("return 4", "return 5"))
    assert examples.simulation_fingerprint(config, root) != before
    source.write_text(source.read_text().replace("return 5", "return 4"))
    (root / "input.txt").write_text("scientific change")
    assert examples.simulation_fingerprint(config, root) != before


def test_installed_science_dependency_invalidates_saved_result(example):
    """The runtime package can be audited without copying it into a study ZIP."""
    root, config = example
    config["science_inputs"] = ["optional-backend.py"]
    backend = root / "optional-backend.py"
    backend.write_text("value = 0")
    before = (
        examples.fingerprint(config, root),
        examples.simulation_fingerprint(config, root),
    )
    backend.write_text("value = 1")
    after = (
        examples.fingerprint(config, root),
        examples.simulation_fingerprint(config, root),
    )
    assert all(a != b for a, b in zip(before, after, strict=True))
    assert "optional-backend.py" not in config["inputs"]


def test_replot_requires_data_and_valid_scientific_provenance(example):
    """Missing data fails before executing any renderer or simulation."""
    root, config = example
    with pytest.raises(ValueError, match="No saved numerical data"):
        examples.refresh("demo", config, root, root / "results", replot=True)


@pytest.mark.filterwarnings(
    "ignore:importing 'Const' from 'astroid' is deprecated:DeprecationWarning"
)
def test_refresh_then_replot_does_not_repeat_simulation(tmp_path, monkeypatch):
    """A real kernel reuses data while preserving original execution provenance."""
    source = tmp_path / "example.py"
    counter = tmp_path / "simulation-count"
    source.write_text(
        "# %%\nfrom pathlib import Path\n"
        "def simulate():\n"
        f"    path = Path({str(counter)!r})\n"
        "    path.write_text(str(int(path.read_text()) + 1) if path.exists() else '1')\n"
        "    return {'time': [0, 1], 'count': [0, 3]}\n"
        "def plot_results(data):\n"
        "    print('blue', data['count'])\n"
        "    return data\n"
        "def run_example():\n"
        "    return plot_results(simulate())\n"
    )
    config = {
        "source": str(source),
        "device": "cpu",
        "policy": "cpu",
        "timeout": 60,
        "inputs": [],
        "invoke": "result = run_example()",
        "data_expression": "result",
        "replot_invoke": "result = plot_results(saved_data)",
        "render_functions": ["plot_results"],
    }
    results = tmp_path / "results"
    examples.refresh("demo", config, results=results)
    directory, before = examples.bundle("demo", results)
    original_data = (directory / "data.json").read_bytes()
    source.write_text(source.read_text().replace("'blue'", "'red'"))
    examples.refresh("demo", config, results=results, replot=True)
    directory, after = examples.bundle("demo", results)
    assert counter.read_text() == "1"
    assert (directory / "data.json").read_bytes() == original_data
    assert after["executed_at"] == before["executed_at"]
    assert after["revision"] == before["revision"]
    assert after["device"] == before["device"]
    assert "rendered_at" in after
    assert "blue" in (directory / "executed_source.py").read_text()
    assert "red" in (directory / "source.py").read_text()
    assert examples.result_state(config, after) == "current"
    # A replot must not probe CUDA, even for a CUDA-only scientific workload.
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(
            is_cuda_available=lambda: pytest.fail("replot must not probe CUDA")
        ),
    )
    examples.execute(
        {**config, "device": "cuda"}, saved_data=json.loads(original_data)
    )
    assert counter.read_text() == "1"
    source.write_text(source.read_text().replace("[0, 3]", "[0, 4]"))
    with pytest.raises(ValueError, match="Simulation inputs changed"):
        examples.refresh("demo", config, results=results, replot=True)
    assert counter.read_text() == "1"


def test_oversized_refresh_preserves_existing_bundle(example, monkeypatch):
    """A size-limit failure during staging leaves published output intact."""
    root, config = example
    config["data_expression"] = "result"
    published = root / "results/demo"
    published.mkdir(parents=True)
    (published / "manifest.json").write_text("previous successful output")
    monkeypatch.setattr(examples, "MAX_PLOT_DATA_BYTES", 32)
    monkeypatch.setattr(
        examples,
        "execute",
        lambda *args: (
            [{"metadata": {"plot_data": {"values": list(range(100))}}}],
            "CPU",
        ),
    )
    with pytest.raises(ValueError, match="10 MiB"):
        examples.refresh("demo", config, root, root / "results")
    assert (
        published / "manifest.json"
    ).read_text() == "previous successful output"
    assert not list((root / "results").glob(".pending-*"))


def test_renderer_cannot_replace_saved_observations(example):
    """Rendering may change appearance, never the preserved numerical values."""
    root, config = example
    source = (root / config["source"]).read_bytes()
    cells = [{"metadata": {"plot_data": {"count": [0, 5]}}}]
    with pytest.raises(ValueError, match="changed saved numerical data"):
        examples.validate_render(
            config,
            root,
            source,
            examples.fingerprint(config, root),
            cells,
            {"count": [0, 4]},
        )


def test_plot_data_requires_an_object_or_array(tmp_path):
    """Null cannot accidentally select the simulation execution path on replot."""
    with pytest.raises(ValueError, match="object or array"):
        examples.write_plot_data(tmp_path / "data.json", None)


@pytest.mark.parametrize("changed", [None, "library", "simulation"])
def test_validation_reuses_data_after_a_render_only_publication(
    monkeypatch, changed
):
    """Reuse matching data; validate changed science through fresh execution."""
    config = {"policy": "cpu", "data_expression": "result"}
    data = {"time": [0, 1], "count": [0, 4]}
    calls = []
    monkeypatch.setattr(
        examples,
        "bundle",
        lambda key: (
            None,
            {
                "rendered_at": "now",
                "package_source_sha256": "old"
                if changed == "library"
                else "library",
                "simulation_fingerprint": "old"
                if changed == "simulation"
                else "simulation",
            },
        ),
    )
    monkeypatch.setattr(examples, "package_source_hash", lambda root: "library")
    monkeypatch.setattr(
        examples, "simulation_fingerprint", lambda config, root: "simulation"
    )
    monkeypatch.setattr(
        examples, "load_replot_data", lambda *args: (None, {}, data)
    )
    monkeypatch.setattr(
        examples, "execute", lambda c, **kwargs: calls.append(kwargs)
    )
    examples.validate_example("demo", config)
    assert calls == ([{"saved_data": data}] if changed is None else [{}])


def test_figure_names_do_not_depend_on_stdout_chunking(tmp_path):
    """Additional logs must not rename identical scientific figures."""
    import base64

    figure = {"data": {"image/png": base64.b64encode(b"figure").decode()}}
    quiet = tmp_path / "quiet"
    noisy = tmp_path / "noisy"
    quiet.mkdir()
    noisy.mkdir()
    first = examples.save_cell_outputs([{"outputs": [figure]}], quiet)
    second = examples.save_cell_outputs(
        [
            {
                "outputs": [
                    {"output_type": "stream", "text": "Kernel loaded"},
                    figure,
                ]
            }
        ],
        noisy,
    )
    assert first[0][0]["image"] == second[0][1]["image"] == "figure-1-1.png"
    assert (quiet / "figure-1-1.png").read_bytes() == (
        noisy / "figure-1-1.png"
    ).read_bytes()


def test_real_execution_timeout_stops_kernel(example, monkeypatch):
    """A busy cell must time out and release its kernel, not block the deadline."""
    import jupyter_client
    from nbclient.exceptions import CellTimeoutError

    root, config = example
    config["timeout"] = 5
    (root / config["source"]).write_text(
        "import time\ntime.sleep(7)\nprint('should not complete')\n"
    )
    managers = []
    manager_type = jupyter_client.AsyncKernelManager

    def tracked_manager(**kwargs):
        manager = manager_type(**kwargs)
        managers.append(manager)
        return manager

    monkeypatch.setattr(jupyter_client, "AsyncKernelManager", tracked_manager)
    with pytest.raises(CellTimeoutError, match="time.sleep"):
        examples.execute(config, root)
    assert len(managers) == 1
    assert not managers[0].has_kernel


def test_full_canvas_fingerprints_style_without_invalidating_legacy(example):
    """Only full-canvas publication records depend on the style inputs."""
    root, config = example
    legacy = examples.fingerprint(config, root)
    for name in examples.STYLE_INPUTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    styled = {**config, "full_canvas": True}
    initial = examples.fingerprint(styled, root)
    assert initial != legacy
    for name in examples.STYLE_INPUTS:
        (root / name).write_text("updated")
        assert examples.fingerprint(styled, root) != initial
        (root / name).write_text("original")
    assert examples.fingerprint(config, root) == legacy


def test_replot_distinguishes_scientific_and_rendering_sources(example):
    """A renderer edit invalidates the picture while preserving usable data."""
    root, config = example
    config["full_canvas"] = True
    package = root / "particula"
    renderer = root / examples.STYLE_INPUTS[1]
    renderer.parent.mkdir(parents=True)
    renderer.write_text("color = 'blue'\n")
    (root / examples.STYLE_INPUTS[0]).write_text("lines.linewidth: 2\n")
    model = package / "physics.py"
    model.write_text("rate = 3\n")
    results = root / "results"
    folder = results / "demo"
    folder.mkdir(parents=True)
    data = b'{"value":3}'
    (folder / "data.json").write_bytes(data)
    manifest = {
        "fingerprint": examples.fingerprint(config, root),
        "simulation_fingerprint": examples.simulation_fingerprint(config, root),
        "package_source_sha256": examples.package_source_hash(root),
        "scientific_source_sha256": examples.package_source_hash(
            root, science_only=True
        ),
        "files": {"data.json": hashlib.sha256(data).hexdigest()},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    renderer.write_text("color = 'red'\n")
    assert examples.result_state(config, manifest, root) == "stale"
    assert examples.load_replot_data("demo", config, root, results)[2] == {
        "value": 3
    }
    model.write_text("rate = 4\n")
    with pytest.raises(ValueError, match="Library source changed"):
        examples.load_replot_data("demo", config, root, results)


def test_historical_replot_checks_original_inputs_and_pinned_data(tmp_path):
    """An archive must match the original source, library and observations."""
    science = tmp_path / "original"
    science.mkdir()
    (science / "website").mkdir()
    config = {"source": "study.py", "device": "cuda"}
    (science / "study.py").write_text("value = 3\n")
    (science / "website/examples.json").write_text(json.dumps({"demo": config}))

    def git(*args):
        return subprocess.check_output(  # noqa: S603 -- local test repository
            [examples.GIT, *args], cwd=science, text=True
        ).strip()

    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "-qm",
        "Original study",
    )
    revision = git("rev-parse", "HEAD")
    results = tmp_path / "results"
    directory = results / "demo"
    directory.mkdir(parents=True)
    data = b'{"observation": 3}'
    (directory / "data.json").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    manifest = {
        "revision": revision,
        "files": {"data.json": digest},
        "simulation_fingerprint": examples.simulation_fingerprint(
            config, science
        ),
        "package_source_sha256": examples.package_source_hash(science),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    pin = {"revision": revision, "data_sha256": digest}
    pin["package_source_sha256"] = examples.package_source_hash(science)
    current = {**config, "historical_run": pin}
    examples.verify_execution_checkout(current, science)
    (science / "particula").mkdir()
    extra = science / "particula/modified.py"
    extra.write_text("changed = True")
    with pytest.raises(ValueError, match="differs from the pinned execution"):
        examples.verify_execution_checkout(current, science)
    extra.unlink()

    def load_archive(config):
        return examples.load_replot_data(
            "demo", config, tmp_path, results, science_root=science
        )

    assert load_archive(current)[2] == {"observation": 3}
    # A run on dirty sources can be archived at their exact next commit.
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.org",
        "commit",
        "--allow-empty",
        "-qm",
        "Archive exact executed inputs",
    )
    pin["revision"] = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="pinned historical run"):
        load_archive(current)
    manifest["package_dirty"] = True
    (directory / "manifest.json").write_text(json.dumps(manifest))
    assert load_archive(current)[2] == {"observation": 3}
    with pytest.raises(ValueError, match="no pinned"):
        load_archive(config)
    pin["data_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned historical run"):
        load_archive(current)
    pin["data_sha256"] = digest
    (science / "study.py").write_text("value = 4\n")
    with pytest.raises(ValueError, match="Simulation inputs changed"):
        load_archive(current)


def test_archived_science_uses_only_current_plotting_module(tmp_path):
    """The isolated kernel can use old science with the current plot style."""
    package = tmp_path / "particula"
    package.mkdir()
    (package / "__init__.py").write_text("archive_marker = True\n")
    code = (
        "from scripts.docs_examples import ROOT, use_historical_package\n"
        "import sys\n"
        f"use_historical_package({str(tmp_path)!r}, str(ROOT))\n"
        "import particula\n"
        "import matplotlib.pyplot as plt\n"
        "from pathlib import Path\n"
        "assert particula.archive_marker\n"
        f"assert {str(tmp_path)!r} not in sys.path\n"
        "assert Path(particula.plotting.__file__) == "
        "ROOT / 'particula/plotting/__init__.py'\n"
        "with plt.style.context('particula.default'):\n"
        "    fig, ax = particula.plotting.subplots()\n"
        "    assert not ax.spines['top'].get_visible()\n"
    )
    subprocess.run(  # noqa: S603 -- isolated fixed test program
        [sys.executable, "-c", code], cwd=examples.ROOT, check=True
    )


def test_fresh_archived_library_run_replots_from_executed_study(
    tmp_path, monkeypatch
):
    """Retain the current study, its inputs and its own observed data on redraw."""
    science = tmp_path / "archive"
    author = tmp_path / "author"

    def git(repo, *args):
        return subprocess.check_output(  # noqa: S603 -- local Git fixtures
            [examples.GIT, *args], cwd=repo, text=True
        ).strip()

    for repo in (science, author):
        repo.mkdir()
        git(repo, "init", "-q")
        git(
            repo,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            repo.name,
            "--allow-empty",
        )
    revision = git(science, "rev-parse", "HEAD")
    config = {
        "source": "study.py",
        "device": "cpu",
        "inputs": ["input.txt"],
        "data_expression": "data",
        "historical_run": {
            "revision": revision,
            "package_source_sha256": examples.package_source_hash(science),
            "data_sha256": hashlib.sha256(b"original observations").hexdigest(),
        },
    }
    source = author / "study.py"
    source.write_text("data = {'time': 7}\n")
    (author / "input.txt").write_text("material A")
    (science / "study.py").write_text("data = {'time': 3}\n")
    calls = []

    def execute(config, root, *, saved_data=None, science_root=None):
        calls.append((science_root, saved_data))
        return [{"metadata": {"plot_data": {"time": 7}}, "outputs": []}], "CPU"

    monkeypatch.setattr(examples, "execute", execute)
    results = tmp_path / "results"

    def publish(replot=False):
        examples.refresh(
            "demo", config, author, results, replot=replot, science_root=science
        )

    publish()
    folder, before = examples.bundle("demo", results)
    with pytest.raises(ValueError, match="pinned science checkout"):
        examples.load_replot_data("demo", config, author, results)
    for _ in range(2):
        publish(replot=True)
        _, after = examples.bundle("demo", results)
        for key in (
            "execution_data_sha256",
            "plot_source_revision",
            "simulation_fingerprint",
            "executed_at",
            "revision",
        ):
            assert before[key] == after[key]
        assert after["files"]["data.json"] == before["execution_data_sha256"]
    assert calls == [(science, None), (None, {"time": 7}), (None, {"time": 7})]
    (author / "input.txt").write_text("material B")
    with pytest.raises(ValueError, match="Simulation inputs changed"):
        publish(replot=True)
    (author / "input.txt").write_text("material A")
    new = b'{"time":9}'
    (folder / "data.json").write_bytes(new)
    after["files"]["data.json"] = hashlib.sha256(new).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(after))
    with pytest.raises(ValueError, match="pinned historical run"):
        publish(replot=True)
