"""Regression tests for published execution-selection documentation."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
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
)
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


def test_execution_selection_example_executes_public_selection_only_contract() -> (
    None
):
    """Test the published CPU-only selection example remains executable."""
    section = _section(FEATURE_PATH, "Execution selection")
    fences = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    public_import = next(fence for fence in fences if "Capability," in fence)
    example = next(fence for fence in fences if "class LocalAdapter:" in fence)

    ast.parse(public_import)
    public_namespace: dict[str, object] = {}
    exec(public_import, public_namespace)  # noqa: S102 -- published import fence
    public_names = {
        name for name in public_namespace if not name.startswith("__")
    }
    assert public_names == set(PUBLIC_NAMES)

    ast.parse(example)
    namespace: dict[str, object] = {}
    exec(example, namespace)  # noqa: S102 -- executes the published local example

    assert namespace["selected"] is namespace["adapter"]
    assert namespace["selected"].execute("example-local result") == (  # type: ignore[attr-defined]
        "example-local result"
    )


def test_execution_selection_documentation_states_selection_and_failure_bounds() -> (
    None
):
    """Test public selection wording retains its bounded contract."""
    section = _section(FEATURE_PATH, "Execution selection")
    normalized = _normalize(section)

    for name in PUBLIC_NAMES:
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
        "top-level `particula` export only the ten selection primitives",
        "intentionally concrete-only at "
        "`particula.execution.adapters.condensation`",
        "no provisional public selected-condensation import",
        "caller-owned legacy `Aerosol` and calls a caller-owned "
        "`MassCondensation`",
        "caller-owned resident `WarpParticleData`, `WarpGasData`, and "
        "`WarpEnvironmentData`",
        "same-device fixed-shape sidecars",
        "explicitly use `to_warp_particle_data`, `to_warp_gas_data`, and "
        "`to_warp_environment_data`",
        "synchronize before host observation and restore only at their "
        "checkpoint",
        "does no upload, restore, synchronization, allocation, retry, or "
        "silent CPU fallback",
        "`energy_transfer` is a caller-owned write-only output, never a third "
        "return item",
        "finalized total transfer, eligible scratch/work buffers, and optional "
        "energy output in place",
        "preflight preserve caller primary and output state before a writer "
        "launch",
        "raw-proposal failure occurs before P2 commit in its failing substep",
        "completed earlier substeps remain committed",
        "own snapshot/restore",
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
        "All 36 equal-step/staggered, latent, activity, and surface semantic "
        "combinations are declared",
        "Exactly 8 declared profiles",
        "Staggered/Gauss-Seidel and every `NONREPRESENTABLE`/BAT activity or "
        "surface mapping reject",
        "capability-profile preflight before lazy kernel resolution, native "
        "dispatch, or adapter-driven writes",
        "no conversion, approximation, or fallback occurs",
        "exactly four equal `time_step / 4.0` substeps",
        "temperature-driven vapor pressure",
        "mutates particle masses, gas concentration, and derived GPU vapor "
        "pressure in place",
        "independent fixed-four-substep P2 oracle",
        "particle mass and gas concentration separately",
        "Inventory is checked separately",
        "Warp CPU is the installed-Warp baseline",
        "CUDA rows are optional guarded evidence",
        "defers E7-F4 resident-resource lifecycle and E7-F5 deterministic "
        "scheduling",
        "full scheduler support",
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
        "E7-F6 owns availability, fallback, error taxonomy, API stability, and export policy",
        "GPU adapters, resident loops, scheduling, implicit transfer",
        "fallback, retry, or replacement of direct GPU APIs",
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
