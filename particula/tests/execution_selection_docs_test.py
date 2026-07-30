"""Regression tests for published execution-selection documentation."""

import ast
import re
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
BACKEND_SELECTION_PATH = ROOT / "docs/Features/backend_selection.md"
FEATURE_INDEX_PATH = ROOT / "docs/Features/index.md"
CONDENSATION_FEATURE_PATH = (
    ROOT / "docs/Features/condensation_strategy_system.md"
)
ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
ARCHITECTURE_PATHS = (
    ROOT / ".opencode/guides/architecture/architecture_guide.md",
    ROOT / ".opencode/guides/architecture/architecture_outline.md",
)
PUBLIC_NAMES = (
    "Backend",
    "Device",
    "Process",
    "Capability",
    "CapabilityRequirements",
    "CapabilityDeclaration",
    "CapabilityMatrix",
    "ExecutionRequest",
    "ExecutionAdapter",
    "ExecutionContext",
    "ExecutionCapabilityReason",
    "ExecutionCapabilityError",
    "UnknownExecutionTargetError",
    "UnavailableExecutionTargetError",
    "UnsupportedExecutionRequestError",
    "UnknownBackendError",
    "UnknownDeviceError",
    "UnavailableRuntimeError",
    "UnavailableDeviceError",
    "UnsupportedProcessError",
    "UnsupportedCapabilityError",
    "InvalidExecutionStateError",
    "FallbackDisallowedError",
    "FallbackPolicy",
    "FallbackBoundary",
    "CPUStateAuthority",
)
SELECTION_NAMES = PUBLIC_NAMES[:10]
ORDERING = "E7-F1 -> E7-F6 -> {E7-F2, E7-F3, E7-F4} -> E7-F5"


def _section(path: Path, heading: str) -> str:
    """Read a second-level Markdown section through its next peer heading.

    Args:
        path: Markdown file containing the section.
        heading: Exact level-two heading to extract.

    Returns:
        Content after the heading through the next level-two heading.
    """
    content = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)",
        content,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"Missing {heading!r} section in {path}."
    return match.group(1)


def _normalize(content: str) -> str:
    """Normalize Markdown whitespace for resilient phrase assertions.

    Args:
        content: Markdown content to normalize.

    Returns:
        Content with consecutive whitespace replaced by single spaces.
    """
    return " ".join(content.split())


def _python_fence(content: str, marker: str) -> str:
    """Return the unique Python fence containing a required marker.

    Args:
        content: Markdown content containing Python fences.
        marker: Required unique text within the desired fence.

    Returns:
        The Python source from the matching fence.
    """
    fences = re.findall(r"```python\n(.*?)```", content, flags=re.DOTALL)
    matches = [fence for fence in fences if marker in fence]
    assert len(matches) == 1, (
        f"Expected one Python fence containing {marker!r}."
    )
    return matches[0]


def _markdown_anchor(heading: str) -> str:
    """Return the MkDocs-compatible anchor for a Markdown heading."""
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def _assert_markdown_link_resolves(source: Path, target: str) -> None:
    """Assert a repository-relative Markdown target and optional anchor exist."""
    path_text, separator, anchor = target.partition("#")
    destination = (source.parent / path_text).resolve()
    assert destination.is_file(), (
        f"Missing link target {target!r} from {source}."
    )
    if separator:
        headings = re.findall(
            r"^#{1,6}\s+(.+?)\s*$",
            destination.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        assert anchor in {_markdown_anchor(heading) for heading in headings}


def _subsection(path: Path, heading: str) -> str:
    """Read a third-level Markdown subsection through its next peer or parent.

    Args:
        path: Markdown file containing the subsection.
        heading: Exact level-three heading to extract.

    Returns:
        Content after the heading through the next level-two or level-three
        heading.
    """
    content = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^### {re.escape(heading)}\n(.*?)(?=^## |^### |\Z)",
        content,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"Missing {heading!r} subsection in {path}."
    return match.group(1)


def test_execution_selection_example_has_public_selection_only_contract() -> (
    None
):
    """Test the published CPU-only example has only public selection imports."""
    section = _section(FEATURE_PATH, "Execution selection")
    fences = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    public_import = next(fence for fence in fences if "Capability," in fence)
    example = next(fence for fence in fences if "class LocalAdapter:" in fence)

    import_tree = ast.parse(public_import)
    imports = [
        node
        for node in ast.walk(import_tree)
        if isinstance(node, ast.ImportFrom)
    ]
    assert all(
        node.module in {"particula", "particula.execution"} for node in imports
    )
    public_names = {alias.name for node in imports for alias in node.names}
    assert public_names <= set(SELECTION_NAMES)

    example_tree = ast.parse(example)
    example_imports = [
        node
        for node in ast.walk(example_tree)
        if isinstance(node, ast.ImportFrom)
    ]
    assert all(
        node.module in {"particula", "particula.execution"}
        for node in example_imports
    )


def test_execution_selection_documentation_states_selection_and_failure_bounds() -> (
    None
):
    """Test public selection wording retains its bounded contract."""
    section = _section(FEATURE_PATH, "Execution selection")
    normalized = _normalize(section)

    for name in SELECTION_NAMES:
        assert name in section
    for phrase in (
        'Device(Backend.CPU, "cpu")',
        "Nonempty requirements must match a complete declaration exactly",
        "Empty requirements are accepted when the matrix contains a declaration "
        "for the same `Device` and `Process`",
        "one exact context-local `(Process, Backend)` lookup",
        "No adapter `execute` call occurs before the final assertion",
        "local to this example, not a generic `ExecutionAdapter`",
        "unsupported declaration fails before lookup",
        "raises `LookupError`",
        "no alternate backend is selected",
        "does not catch, retry, or fallback",
    ):
        assert phrase in normalized
    for private_name in ("CPUExecutionState", "CPUExecutionAdapter"):
        assert private_name not in section


def test_backend_selection_guide_documents_exact_public_and_private_surface() -> (
    None
):
    """Test the new guide preserves the frozen ordered execution value surface."""
    import particula.execution as execution

    content = BACKEND_SELECTION_PATH.read_text(encoding="utf-8")
    stable_values = _section(BACKEND_SELECTION_PATH, "Stable public values")

    assert tuple(execution.__all__) == PUBLIC_NAMES
    positions = [stable_values.index(f"`{name}`") for name in PUBLIC_NAMES]
    assert positions == sorted(positions)
    normalized = _normalize(content)
    for concrete_name in (
        "Availability and fallback mechanics",
        "fallback mechanics",
        "adapters",
        "resident session and checkpoint seams",
        "registries",
        "GPU sidecars",
    ):
        assert concrete_name in normalized


def test_backend_selection_guide_closes_reason_and_retry_outcomes() -> None:
    """Test only the documented five reasons permit explicit CPU fallback."""
    content = BACKEND_SELECTION_PATH.read_text(encoding="utf-8")
    normalized = _normalize(content)
    eligible = (
        "UNKNOWN_DEVICE",
        "RUNTIME_UNAVAILABLE",
        "DEVICE_UNAVAILABLE",
        "PROCESS_UNSUPPORTED",
        "CAPABILITY_UNSUPPORTED",
    )
    rejected = ("UNKNOWN_BACKEND", "INVALID_STATE", "FALLBACK_DISALLOWED")

    rows = re.findall(r"^\| `([A-Z_]+)` \| (.+) \|$", content, re.MULTILINE)
    actions = dict(rows)
    assert tuple(actions) == (
        "UNKNOWN_BACKEND",
        *eligible,
        "INVALID_STATE",
        "FALLBACK_DISALLOWED",
    )
    assert all(
        "explicit CPU policy is eligible" in actions[reason]
        for reason in eligible
    )
    assert all(actions[reason].startswith("Reject;") for reason in rejected)
    assert (
        "Adapter, kernel, and runtime errors after invocation propagate"
        in content
    )
    assert "without CPU retry" in content
    assert "`FallbackPolicy.RAISE` is default-deny" in normalized
    assert "`CPUStateAuthority.CPU_AUTHORITATIVE`" in normalized
    assert "`FallbackBoundary.PRE_UPLOAD`" in normalized
    assert "`FallbackBoundary.RESTORED`" in normalized
    assert "does not change native result metadata" in normalized


def test_backend_selection_guide_states_exact_resolver_and_no_movement() -> (
    None
):
    """Test the concrete resolver order and immutable decision are documented."""
    content = _normalize(BACKEND_SELECTION_PATH.read_text(encoding="utf-8"))

    assert (
        "complete provider registry, target recognition, process declaration, "
        "capability declaration, lazy runtime availability, device availability, "
        "then request-associated state" in content
    )
    assert (
        "neither selects an adapter nor executes, transfers, synchronizes, "
        "allocates execution resources, or mutates state" in content
    )


def test_backend_selection_selection_fence_is_public_only_and_noninvoking() -> (
    None
):
    """Test the Markdown fence is static, public-only, and non-invoking."""
    content = BACKEND_SELECTION_PATH.read_text(encoding="utf-8")
    fence = _python_fence(content, "Selection must not invoke")
    tree = ast.parse(fence)
    imports = [
        node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]

    assert all(node.module == "particula.execution" for node in imports)
    imported = {alias.name for node in imports for alias in node.names}
    assert imported <= set(SELECTION_NAMES)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Import)
        or (
            isinstance(node, ast.ImportFrom)
            and node.module not in {"particula.execution"}
        )
        for node in ast.walk(tree)
    )
    allowed_named_calls = {
        "CapabilityDeclaration",
        "CapabilityMatrix",
        "CapabilityRequirements",
        "Device",
        "ExecutionContext",
        "ExecutionRequest",
        "LocalAdapter",
        "Process",
        "RuntimeError",
        "frozenset",
    }
    allowed_attribute_calls = {"register_adapter", "resolve"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            assert node.func.id in allowed_named_calls
        else:
            assert isinstance(node.func, ast.Attribute)
            assert node.func.attr in allowed_attribute_calls

    namespace: dict[str, Any] = {}
    exec(fence, namespace)  # noqa: S102 -- executes the published CPU-only fence
    adapter = cast(Any, namespace["adapter"])
    context = cast(Any, namespace["context"])
    request = namespace["request"]
    assert context.resolve(request) is adapter
    assert adapter.calls == 0


def test_backend_selection_resident_fence_distinguishes_restart_from_cpu_restore() -> (
    None
):
    """Test resident pseudocode does not present restart as CPU restoration."""
    content = BACKEND_SELECTION_PATH.read_text(encoding="utf-8")
    fence = _python_fence(content, "WARP_AVAILABLE = False")
    tree = ast.parse(fence)
    guarded_imports: list[ast.ImportFrom] = []

    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "WARP_AVAILABLE":
                guarded_imports.extend(
                    child
                    for child in ast.walk(node)
                    if isinstance(child, ast.ImportFrom)
                )
    assert all(
        node in guarded_imports
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert "restart_resident_session" not in fence
    normalized = _normalize(content)
    for phrase in (
        "caller-owned checkpoint/finalize snapshot",
        "separate explicit restoration to CPU-authoritative state plus a "
        "CPU-authority declaration",
        "Resident restart is an exact-device GPU lifecycle operation, not CPU "
        "restoration or fallback",
        "No component restores, transfers, synchronizes, migrates, or retries "
        "silently",
        "from particula.execution.checkpoint import ResidentCheckpointController",
        "from particula.execution.gpu_session import ResidentSession",
        "checkpoint = resident_session.checkpoint(registry, guard)",
        "finalized = resident_session.finalize(registry, guard)",
    ):
        assert phrase in normalized


def test_backend_selection_index_and_foundation_links_resolve() -> None:
    """Test all expected index and foundation handoff links resolve."""
    expected_index_links = (
        "activity_system.md",
        "coagulation_strategy_system.md",
        "condensation_strategy_system.md",
        "wall_loss_strategy_system.md",
        "dilution_strategy_system.md",
        "nucleation_strategy_system.md",
        "data-containers-and-gpu-foundations.md",
        "backend_selection.md",
        "gpu_resident_checkpoints.md",
        "slot_exhaustion_policies.md",
        "particle-data-migration/index.md",
        "Roadmap/index.md",
    )
    for source, expected_links in (
        (FEATURE_INDEX_PATH, expected_index_links),
        (FEATURE_PATH, ("backend_selection.md",)),
        (ROADMAP_PATH, ("../backend_selection.md",)),
    ):
        content = source.read_text(encoding="utf-8")
        links = re.findall(r"\[[^]]+\]\(([^)]+)\)", content)
        for target in expected_links:
            assert target in links
            _assert_markdown_link_resolves(source, target)

    content = BACKEND_SELECTION_PATH.read_text(encoding="utf-8")
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", content)
    for target in (
        "data-containers-and-gpu-foundations.md",
        "Roadmap/data-oriented-gpu.md#epic-g-backend-selection-and-gpu-resident-simulation",
        "gpu_resident_checkpoints.md",
    ):
        assert target in links
        _assert_markdown_link_resolves(BACKEND_SELECTION_PATH, target)


def test_execution_selection_docs_cover_empty_requirements_base_declaration() -> (
    None
):
    """Test empty requirements select from a matching device/process base."""
    from particula import (
        Backend,
        Capability,
        CapabilityDeclaration,
        CapabilityMatrix,
        CapabilityRequirements,
        Device,
        ExecutionContext,
        ExecutionRequest,
        Process,
    )

    class SelectionAdapter:
        """Provide a minimal adapter for a selection-only regression test."""

        def execute(self, *args: object, **kwargs: object) -> object:
            """Provide the required callable attribute without invoking it."""
            return None

    device = Device(Backend.CPU, "cpu")
    process = Process("condensation")
    declared_requirements = CapabilityRequirements(
        frozenset({Capability("isothermal")})
    )
    matrix = CapabilityMatrix(
        frozenset(
            {
                CapabilityDeclaration(
                    device,
                    process,
                    declared_requirements,
                )
            }
        )
    )
    context = ExecutionContext(matrix)
    adapter = SelectionAdapter()
    context.register_adapter(process, Backend.CPU, adapter)
    empty_request = ExecutionRequest(
        Backend.CPU,
        device,
        process,
        CapabilityRequirements(frozenset()),
    )

    assert context.resolve(empty_request) is adapter


def test_selected_condensation_ownership_documents_private_failure_bounds() -> (
    None
):
    """Test ownership reference preserves private API and failure boundaries."""
    section = _section(FEATURE_PATH, "Execution selection")
    normalized = _normalize(section)

    for phrase in (
        "`particula.execution` provides this frozen ordered 26-name public "
        "surface, also re-exported by top-level `particula` alongside its "
        "unrelated APIs",
        "intentionally concrete-only at "
        "`particula.execution.adapters.condensation`",
        "does not bind or validate a supplied CPU runnable or Warp sidecars "
        "against that profile",
        "Adapter dispatch does no hidden upload, restore, retry, or silent CPU "
        "fallback",
        "Direct-kernel validation may perform permitted device scans or status "
        "readbacks",
        "omitted optional scratch or output buffers may use direct-kernel "
        "fallback allocation",
    ):
        assert phrase in normalized


def test_selected_condensation_ownership_subsection_is_private_and_nonrunnable() -> (
    None
):
    """Test the private ownership reference does not publish a workflow."""
    subsection = _subsection(
        FEATURE_PATH,
        "Selected-condensation ownership and API boundary",
    )

    assert "```python" not in subsection
    assert "from particula.execution import" not in subsection
    assert "../Examples/gpu_direct_kernels_quick_start.py" in subsection
    assert "particula.execution.adapters.condensation" in subsection


def test_backend_selected_condensation_documents_bounded_contract() -> None:
    """Test selected-condensation documentation preserves its bounded contract."""
    section = _section(
        CONDENSATION_FEATURE_PATH,
        "Backend-selected condensation",
    )
    normalized = _normalize(section)

    for phrase in (
        "Exactly 8 declared profiles",
        "does not bind it to the supplied runnable",
        "Profiles are capability/catalogue selection and do not validate "
        "supplied sidecars against their semantics",
        "Omitted optional scratch or output buffers may use direct-kernel "
        "fallback allocation",
        "validation may perform permitted device scans or status readbacks",
        "Adapter dispatch itself performs no hidden transfer, restore, retry, "
        "or CPU fallback",
    ):
        assert phrase in normalized


def test_selected_condensation_docs_publish_required_cross_references() -> None:
    """Test selected-condensation text retains all required internal links."""
    content = "\n".join(
        (
            FEATURE_PATH.read_text(encoding="utf-8"),
            CONDENSATION_FEATURE_PATH.read_text(encoding="utf-8"),
        )
    )

    for target in (
        "data-containers-and-gpu-foundations.md#execution-selection",
        "data-containers-and-gpu-foundations.md#gpu-thermodynamics-and-condensation-refresh",
        "Roadmap/data-oriented-gpu.md#epic-g-backend-selection-and-gpu-"
        "resident-simulation",
        "../Examples/gpu_direct_kernels_quick_start.py",
    ):
        assert target in content


def test_execution_roadmap_preserves_shipped_and_deferred_boundaries() -> None:
    """Test the Epic G roadmap preserves shipped and deferred boundaries."""
    section = _section(
        ROADMAP_PATH, "Epic G: Backend Selection and GPU-Resident Simulation"
    )
    normalized = _normalize(section)

    assert "E7-F1 is shipped" in normalized
    assert ORDERING in normalized
    for phrase in (
        "E7-F2 supplies condensation adapters",
        "E7-F3 supplies coagulation adapters",
        "E7-F4 supplies resident session/container/sidecar lifecycle",
        "E7-F5 is their later scheduling consumer",
        "E7-F6 now supplies the shipped availability resolver, typed errors, explicit CPU fallback, frozen stable API, and documentation handoff",
        "E7-F6/Track T6 is shipped through P6",
        "Selection, availability, and fallback never upload, restore, synchronize, migrate, retry, or silently switch",
    ):
        assert phrase in normalized


@pytest.mark.parametrize("path", ARCHITECTURE_PATHS)
def test_execution_architecture_guides_agree_on_public_and_private_contract(
    path: Path,
) -> None:
    """Test each architecture guide states the same selection boundary."""
    content = _normalize(path.read_text(encoding="utf-8"))

    for phrase in (
        "dependency-neutral",
        "standard-library-only",
        "package-level public APIs",
        "context-local",
        "exact matrix validation",
        "Nonempty requirements must match a complete declaration exactly",
        "empty requirements are accepted when the matrix contains a declaration "
        "for the same `Device` and `Process`",
        "retains identity",
        "not executed",
        'Device(Backend.CPU, "cpu")',
        "direct-module-only",
        "E7-F6 owns availability, fallback, error taxonomy, API stability, and export policy",
        ORDERING,
    ):
        assert phrase in content
