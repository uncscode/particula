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


def test_successful_publication_prunes_only_its_superseded_bundles(tmp_path):
    """Keep the selected result and unrelated content while pruning old runs."""
    for name, example_id in [
        ("run-old", "demo"),
        ("run-new", "demo"),
        ("run-unrelated", "other"),
    ]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "manifest.json").write_text(
            json.dumps({"example": example_id})
        )
    (tmp_path / "run-notes").mkdir()
    (tmp_path / "run-notes/note.txt").write_text("Keep this note")
    examples.prune_superseded_results("demo", tmp_path, "run-new")
    assert not (tmp_path / "run-old").exists()
    assert (tmp_path / "run-new/manifest.json").is_file()
    assert (tmp_path / "run-unrelated/manifest.json").is_file()
    assert (tmp_path / "run-notes/note.txt").is_file()
