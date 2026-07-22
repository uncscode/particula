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
FOUNDATION_PATH = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
AGENTS_PATH = ROOT / "AGENTS.md"
KERNEL_EXPORTS_PATH = ROOT / "particula/gpu/kernels/__init__.py"
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


def _local_destinations(content: str) -> list[tuple[str, str | None]]:
    """Return local Markdown destinations and optional anchor fragments."""
    destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", content)
    local_destinations = []
    for destination in destinations:
        if destination.startswith(("http://", "https://")):
            continue
        target, separator, fragment = destination.partition("#")
        local_destinations.append((target, fragment if separator else None))
    return local_destinations


def _markdown_anchor(heading: str) -> str:
    """Return the normalized anchor corresponding to a Markdown heading."""
    normalized = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"[\s-]+", "-", normalized).strip("-")


def _section(content: str, heading: str) -> str:
    """Return a Markdown heading section through its next peer or parent."""
    match = re.search(
        rf"^(?P<marks>#+) {re.escape(heading)}\s*$",
        content,
        flags=re.MULTILINE,
    )
    assert match is not None, heading
    level = len(match.group("marks"))
    next_heading = re.search(
        rf"^#{{1,{level}}} ", content[match.end() :], flags=re.MULTILINE
    )
    if next_heading is None:
        return content[match.start() :]
    return content[match.start() : match.end() + next_heading.start()]


def _assert_local_links_resolve(path: Path) -> None:
    """Assert local Markdown files and optional anchors resolve."""
    content = path.read_text(encoding="utf-8")
    for destination, fragment in _local_destinations(content):
        target = (
            path if destination == "" else (path.parent / destination).resolve()
        )
        assert target.exists(), destination
        if fragment is not None:
            target_content = target.read_text(encoding="utf-8")
            headings = re.findall(r"^#+ (.+)$", target_content, re.MULTILINE)
            assert fragment in {
                _markdown_anchor(heading) for heading in headings
            }


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
        "one total-duration operation",
        "Custom compatible strategies retain equal",
        "finite, nonnegative real scalars",
        "complex values",
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
        "multi-box particle data and transport",
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


@pytest.mark.parametrize(
    ("path", "heading", "required"),
    [
        pytest.param(
            FOUNDATION_PATH,
            "Direct GPU dilution",
            (
                "from particula.gpu.kernels import dilution_step_gpu",
                "Callers own conversion, device placement, synchronization",
                "fixed-shape, caller-owned in-place operation",
                "changes only their `concentration` fields",
                "same-device `wp.float64` Warp array with shape `(n_boxes,)`",
                "preflight is ordered and read-only",
                "zero scalar coefficient or zero time step",
                "Warp CPU with float64 `rtol=1e-12, atol=0`",
                "CUDA is optional and skips cleanly",
            ),
            id="foundation",
        ),
        pytest.param(
            ROADMAP_PATH,
            "Warp GPU Backend",
            (
                "from particula.gpu.kernels import dilution_step_gpu",
                "Callers own CPU↔Warp transfer, device placement",
                "fixed-shape, caller-owned kernel",
                "mutates only concentration fields in place",
                "same-device `wp.float64` `(n_boxes,)` coefficient array",
                "Complete ordered preflight",
                "scalar-zero coefficient or zero time step",
                "Warp CPU use `rtol=1e-12, atol=0`",
                "CUDA rows are optional and skip cleanly",
            ),
            id="roadmap",
        ),
        pytest.param(
            AGENTS_PATH,
            "GPU dilution P1–P4 contract",
            (
                "from particula.gpu.kernels import dilution_step_gpu",
                "Callers own CPU↔Warp transfer, device placement",
                "fixed-shape particle and gas concentrations",
                "returns the identical containers",
                "same-device `wp.float64` arrays shaped `(n_boxes,)`",
                "Entry-point preflight is deterministic and read-only",
                "Zero scalar coefficients and zero time steps",
                "Warp CPU float64 particle and gas comparisons use",
                "CUDA is optional and skips cleanly",
            ),
            id="maintainer-reference",
        ),
    ],
)
def test_direct_gpu_dilution_contract_is_published(
    path: Path,
    heading: str,
    required: tuple[str, ...],
) -> None:
    """Each published direct-kernel contract retains its scoped guarantees."""
    section = _normalized(_section(path.read_text(encoding="utf-8"), heading))

    for feature in ("E6-F1", "E6-F9"):
        assert re.search(rf"\b{feature}\b", section)
    for snippet in required:
        assert _normalized(snippet) in section


def test_roadmap_replaces_stale_direct_gpu_dilution_status() -> None:
    """Roadmap distinguishes delivered kernels from deferred orchestration."""
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    shipped = _section(roadmap, "Warp GPU Backend")
    epic_f = _section(roadmap, "Epic F: GPU Process Completeness")
    status = _normalized(shipped + epic_f)

    assert "dilution has no process-level CPU reference" not in status
    assert (
        "GPU dilution kernel with parity tests against the CPU reference"
        not in status
    )
    assert "Direct GPU dilution is shipped as a fixed-shape" in status
    assert "GPU runnable and process orchestration" in status


@pytest.mark.parametrize(
    ("path", "heading"),
    [
        (FOUNDATION_PATH, "Direct GPU dilution"),
        (ROADMAP_PATH, "Warp GPU Backend"),
        (AGENTS_PATH, "GPU dilution P1–P4 contract"),
    ],
)
def test_direct_gpu_dilution_sections_defer_unsupported_promises(
    path: Path, heading: str
) -> None:
    """Direct-kernel documentation avoids unsupported capability guarantees."""
    section = _normalized(_section(path.read_text(encoding="utf-8"), heading))

    assert "bitwise parity" not in section.replace("not bitwise parity", "")
    for capability in (
        "hidden transfer",
        "fallback",
        "GPU runnable",
        "resizing",
        "graph capture",
        "autodiff",
        "performance",
    ):
        assert capability in section
    assert "deferred" in section or "does not provide" in section


def test_dilution_kernel_public_surface_is_lazy_export_metadata() -> None:
    """Dilution remains declared without loading its Warp-dependent module."""
    import particula.gpu.kernels as kernels

    assert KERNEL_EXPORTS_PATH.is_file()
    assert "dilution_step_gpu" in kernels.__all__
    assert (
        kernels._SYMBOL_TO_MODULE["dilution_step_gpu"]
        == "particula.gpu.kernels.dilution"
    )


def test_local_link_resolver_supports_same_and_relative_anchors(
    tmp_path: Path,
) -> None:
    """Local Markdown anchors resolve for same-document and relative links."""
    source = tmp_path / "source.md"
    target = tmp_path / "target.md"
    source.write_text(
        "# Source Heading\n[Same](source.md#source-heading) "
        "[Other](target.md#other-heading)\n",
        encoding="utf-8",
    )
    target.write_text("## Other Heading\n", encoding="utf-8")

    _assert_local_links_resolve(source)


@pytest.mark.parametrize(
    "content",
    [
        "[Missing file](missing.md)",
        "# Present\n[Missing anchor](target.md#absent)",
    ],
)
def test_local_link_resolver_rejects_missing_files_and_anchors(
    tmp_path: Path, content: str
) -> None:
    """Local Markdown resolver reports missing targets and fragments."""
    source = tmp_path / "source.md"
    source.write_text(content, encoding="utf-8")
    (tmp_path / "target.md").write_text("# Present\n", encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_local_links_resolve(source)


def test_direct_gpu_dilution_markdown_links_resolve() -> None:
    """Edited Markdown guides retain valid local files and anchors."""
    for path in (FOUNDATION_PATH, ROADMAP_PATH):
        _assert_local_links_resolve(path)
