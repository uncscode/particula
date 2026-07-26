"""Hardware-free publication tests for the E6 direct-process documentation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROADMAPS = (
    ROOT / "docs/Features/Roadmap/data-oriented-gpu.md",
    ROOT / "docs/Features/Roadmap/index.md",
)
FOUNDATIONS = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
EXAMPLE = ROOT / "docs/Examples/gpu_complete_process_sequence.py"
EXAMPLE_TEST = (
    ROOT / "particula/gpu/tests/gpu_complete_process_sequence_example_test.py"
)
P2_TEST = ROOT / "particula/gpu/tests/process_sequence_test.py"
TESTING_GUIDE = ROOT / ".opencode/guides/testing_guide.md"
AGENTS_GUIDE = ROOT / "AGENTS.md"
DOCS_INDEX = ROOT / "docs/index.md"
DOCS_TEST_COMMAND = "pytest particula/tests/gpu_complete_process_sequence_docs_test.py -q -Werror"
P3_SOURCE = (
    "https://github.com/Gorkowski/particula/blob/main/docs/Examples/"
    "gpu_complete_process_sequence.py"
)
P4_SECTIONS = {
    name: ROOT / f".opencode/plans/sections/features/E6-F9/{name}.md"
    for name in (
        "phase_details",
        "dependencies",
        "documentation_updates",
        "success_criteria",
        "change_log",
    )
}
E6_SECTIONS = {
    name: ROOT / f".opencode/plans/sections/epics/E6/{name}.md"
    for name in (
        "child_plans",
        "dependency_map",
        "success_metrics",
        "change_log",
    )
}
COMPONENT_GUIDES = tuple(
    ROOT / f"docs/Features/{name}.md"
    for name in (
        "dilution_strategy_system",
        "wall_loss_strategy_system",
        "nucleation_strategy_system",
        "slot_exhaustion_policies",
        "condensation_strategy_system",
        "coagulation_strategy_system",
    )
)


def _section(content: str, heading: str) -> str:
    """Return a Markdown section through its next same-or-higher heading."""
    start = content.index(heading)
    section = content[start:]
    level = len(heading) - len(heading.lstrip("#"))
    for index, line in enumerate(section.splitlines()[1:], start=1):
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            return "\n".join(section.splitlines()[:index])
    return section


def test_direct_process_documentation_preserves_explicit_ownership() -> None:
    """Published ownership guidance names the direct boundary and exclusions."""
    content = _section(
        FOUNDATIONS.read_text(encoding="utf-8"),
        "### Complete direct-process illustration",
    )
    content = " ".join(content.split())
    for name in (
        "condensation_step_gpu",
        "coagulation_step_gpu",
        "dilution_step_gpu",
        "wall_loss_step_gpu",
        "nucleation_step_gpu",
        "to_warp_particle_data",
        "to_warp_gas_data",
        "to_warp_environment_data",
    ):
        assert name in content
    assert content.index("condensation_step_gpu") < content.index(
        "coagulation_step_gpu"
    )
    assert content.index("coagulation_step_gpu") < content.index(
        "dilution_step_gpu"
    )
    assert content.index("dilution_step_gpu") < content.index(
        "wall_loss_step_gpu"
    )
    assert content.index("wall_loss_step_gpu") < content.index(
        "nucleation_step_gpu"
    )
    for excluded in (
        "no hidden transfer",
        "CPU fallback",
        "high-level `Runnable`",
        "backend selector",
        "scheduler",
        "resident production loop",
        "transport API",
    ):
        assert excluded in content
    assert "one conversion per CPU container" in content
    assert "synchronize and restore once at the final checkpoint" in content


def test_evidence_and_commands_are_real_and_hardware_free() -> None:
    """Published P2/P3 evidence points at checked-in artifacts only."""
    assert EXAMPLE.exists()
    assert EXAMPLE_TEST.exists()
    assert P2_TEST.exists()
    for path in ROADMAPS:
        content = path.read_text(encoding="utf-8")
        assert P3_SOURCE in content
        assert "process_sequence_test.py" in content
        assert "illustrative, not a production coordinator" in content
    assert DOCS_TEST_COMMAND in TESTING_GUIDE.read_text(encoding="utf-8")
    assert DOCS_TEST_COMMAND in AGENTS_GUIDE.read_text(encoding="utf-8")
    assert P3_SOURCE in DOCS_INDEX.read_text(encoding="utf-8")
    for path in (
        *P4_SECTIONS.values(),
        *E6_SECTIONS.values(),
        *COMPONENT_GUIDES,
    ):
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        if path in COMPONENT_GUIDES:
            assert P3_SOURCE in content
            assert "process_sequence_test.py" in content
