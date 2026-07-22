"""Hardware-free publication tests for the CPU dilution example and guide."""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = ROOT / "docs/Examples/cpu_dilution.py"
GUIDE_PATH = ROOT / "docs/Features/dilution_strategy_system.md"
FEATURES_INDEX_PATH = ROOT / "docs/Features/index.md"
EXAMPLES_INDEX_PATH = ROOT / "docs/Examples/index.md"
DOCS_INDEX_PATH = ROOT / "docs/index.md"
SOURCE_URL = (
    "https://github.com/Gorkowski/particula/blob/main/"
    "docs/Examples/cpu_dilution.py"
)


def _load_example():
    """Load the excluded example source without package-importing docs."""
    spec = importlib.util.spec_from_file_location(
        "cpu_dilution_example", EXAMPLE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalized(content: str) -> str:
    """Normalize Markdown whitespace for narrow wording checks."""
    return " ".join(content.split())


def _local_destinations(content: str) -> list[str]:
    """Return local Markdown link destinations without anchors or remote URLs."""
    destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", content)
    return [
        destination.split("#", maxsplit=1)[0]
        for destination in destinations
        if not destination.startswith(("http://", "https://", "#"))
    ]


def _assert_local_links_resolve(path: Path) -> None:
    """Assert every local Markdown destination from one document resolves."""
    content = path.read_text(encoding="utf-8")
    for destination in _local_destinations(content):
        assert (path.parent / destination).resolve().exists(), destination


def test_cpu_dilution_example_executes_exact_public_api_decay() -> None:
    """Example decays every documented concentration domain exactly."""
    example = _load_example()
    result = example.run_example()
    factor = np.exp(-result.coefficient * result.time_step)

    assert np.isfinite(factor)
    assert 0.0 < factor < 1.0
    for initial, final in (
        (result.particle_initial, result.particle_final),
        (result.partitioning_initial, result.partitioning_final),
        (result.gas_only_initial, result.gas_only_final),
    ):
        assert initial.shape == final.shape
        assert np.all(initial > 0.0)
        assert np.all(final > 0.0)
        assert np.all(final < initial)
        npt.assert_allclose(final, initial * factor, rtol=1e-12, atol=0.0)


def test_cpu_dilution_example_uses_public_runnable_call_chain() -> None:
    """Example constructs and executes the documented public runnable API."""
    tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))
    run_example = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_example"
    )
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(run_example)
        if isinstance(node, ast.Call)
    ]

    assert "par.dynamics.DilutionStrategy" in calls
    assert "par.dynamics.Dilution" in calls
    assert "dilution.execute" in calls


def test_cpu_dilution_example_results_are_isolated_snapshots() -> None:
    """Fresh calls and all initial/final snapshots own independent arrays."""
    example = _load_example()
    first = example.run_example()
    second = example.run_example()

    for initial, final, other in (
        (first.particle_initial, first.particle_final, second.particle_initial),
        (
            first.partitioning_initial,
            first.partitioning_final,
            second.partitioning_initial,
        ),
        (first.gas_only_initial, first.gas_only_final, second.gas_only_initial),
    ):
        assert not np.shares_memory(initial, final)
        assert not np.shares_memory(initial, other)
        snapshot = other.copy()
        initial[...] = -1.0
        npt.assert_array_equal(other, snapshot)


def test_cpu_dilution_example_result_rejects_metadata_reassignment() -> None:
    """Example result keeps its execution metadata immutable."""
    example = _load_example()
    result = example.run_example()

    with pytest.raises(AttributeError, match="ExampleResult is immutable"):
        result.coefficient = 1.0


def test_cpu_dilution_example_main_reports_all_domains(capsys) -> None:
    """Example command output reports metadata and before/after snapshots."""
    example = _load_example()

    example.main()

    output = capsys.readouterr().out.lower()
    for label in (
        "coefficient",
        "duration",
        "decay factor",
        "particle before",
        "particle after",
        "partitioning before",
        "partitioning after",
        "gas-only before",
        "gas-only after",
    ):
        assert label in output


def test_cpu_dilution_example_imports_only_public_dependencies() -> None:
    """Example keeps imports limited to future, NumPy, and public particula."""
    tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    from_imports = [
        node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]

    assert [
        (alias.name, alias.asname) for node in imports for alias in node.names
    ] == [
        ("numpy", "np"),
        ("particula", "par"),
    ]
    assert [
        (node.module, [alias.name for alias in node.names])
        for node in from_imports
    ] == [
        ("__future__", ["annotations"]),
    ]


def test_cpu_dilution_guide_publishes_bounded_cpu_contract() -> None:
    """Guide retains public API, validation, invariants, and scope wording."""
    guide = _normalized(GUIDE_PATH.read_text(encoding="utf-8"))
    required = (
        "par.dynamics.DilutionStrategy(coefficient=0.1)",
        "par.dynamics.Dilution(strategy)",
        "`c_new = c * exp(-alpha * time_step)`",
        "`V` [m³]",
        "`Q` [m³/s]",
        "`alpha = Q / V` [s⁻¹]",
        "particle number concentration is in [1/m³]",
        "gas mass concentration is in [kg/m³]",
        "elementwise particle concentration",
        "scalar partitioning gas concentration",
        "multi-species gas-only concentration",
        "positive, non-boolean Python or NumPy integer",
        "finite, nonnegative scalars",
        "booleans and non-scalars are rejected",
        "exact no-op",
        "still preflighted even for a no-op",
        "finite and nonnegative",
        "rolled back",
        "retry safely",
        "particle distribution/mass, charge, density, representation volume",
        "gas names, molar masses, partitioning metadata",
        "atmospheric temperature and pressure",
        "parent **E6**",
        "**E6-F2** is a downstream consumer",
        "inlet composition or source terms",
        "multi-box transport",
        "GPU, Warp, CUDA, alternate-backend implementation or parity",
        "backend selection, and performance claims",
        "dilute_aerosol` and `get_dilution_step` are concrete-module-only",
        "python docs/Examples/cpu_dilution.py",
        SOURCE_URL,
    )
    for snippet in required:
        assert _normalized(snippet) in guide


def test_cpu_dilution_discoverability_links_resolve() -> None:
    """New dilution entries link the guide locally and source remotely."""
    features = FEATURES_INDEX_PATH.read_text(encoding="utf-8")
    examples = EXAMPLES_INDEX_PATH.read_text(encoding="utf-8")
    docs_index = DOCS_INDEX_PATH.read_text(encoding="utf-8")

    assert "[Dilution Strategy System](dilution_strategy_system.md)" in features
    assert (
        "[Dilution Strategy System](../Features/dilution_strategy_system.md)"
        in examples
    )
    assert SOURCE_URL in examples
    assert "python docs/Examples/cpu_dilution.py" in examples
    assert (
        "[CPU dilution strategy system](Features/dilution_strategy_system.md)"
        in docs_index
    )
    assert SOURCE_URL in docs_index
    for path in (
        GUIDE_PATH,
        FEATURES_INDEX_PATH,
        EXAMPLES_INDEX_PATH,
        DOCS_INDEX_PATH,
    ):
        _assert_local_links_resolve(path)
