"""Hardware-free publication tests for the E6 direct-process documentation."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONS = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"


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
