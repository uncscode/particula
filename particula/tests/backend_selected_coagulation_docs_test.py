"""Hardware-free publication tests for selected Brownian coagulation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STRATEGY_PATH = ROOT / "docs/Features/coagulation_strategy_system.md"
EXAMPLE_PATH = ROOT / "docs/Examples/gpu_coagulation_direct.py"
ROADMAP_PATH = ROOT / "docs/Features/Roadmap/data-oriented-gpu.md"
PLAN_PATH = ROOT / ".opencode/plans/features/E7-F3.json"
PHASE_DETAILS_PATH = (
    ROOT / ".opencode/plans/sections/features/E7-F3/phase_details.md"
)
SECTION_HEADING = "### E7-F3 concrete selected-Brownian adapter"
DIRECT_HEADING = "### GPU direct-kernel foundations and limitations"
COMMANDS = (
    "pytest particula/tests/backend_selected_coagulation_docs_test.py -q -Werror",
    "pytest particula/gpu/tests/gpu_coagulation_direct_example_test.py -q -Werror",
    "pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror",
)


def _section(content: str, heading: str) -> str:
    """Return a Markdown section up to its next same-or-higher heading."""
    start = content.index(heading)
    section = content[start:]
    level = len(heading) - len(heading.lstrip("#"))
    for index, line in enumerate(section.splitlines()[1:], start=1):
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            return "\n".join(section.splitlines()[:index])
    return section


def _local_destinations(content: str) -> list[str]:
    """Return local Markdown link destinations without fragments."""
    return [
        destination.split("#", maxsplit=1)[0]
        for destination in re.findall(r"\[[^]]+\]\(([^)]+)\)", content)
        if not destination.startswith(("http://", "https://", "#"))
    ]


def _normalized(content: str) -> str:
    """Normalize Markdown whitespace for stable prose assertions."""
    return " ".join(content.split())


def test_selected_adapter_guide_is_separate_and_bounded() -> None:
    """Test the selected route does not narrow direct-kernel support."""
    strategy = STRATEGY_PATH.read_text(encoding="utf-8")
    direct = _section(strategy, DIRECT_HEADING)
    selected = _section(strategy, SECTION_HEADING)

    direct = _normalized(direct)
    selected = _normalized(selected)
    assert "two-term masks" in direct
    assert "mask `15`" in direct
    for snippet in (
        "caller-owned `Aerosol` and exact `Coagulation` runnable",
        "caller-owned `WarpParticleData`",
        "Brownian `particle_resolved` dispatch only",
        "no seed-by-seed parity claim",
        "from particula.execution.adapters.coagulation import",
        "Neither `particula.execution` nor the top-level `particula` package exports",
        "both `temperature` and `pressure` with `environment=None`",
        "`temperature=None` and `pressure=None` with a `WarpEnvironmentData`",
        "Optional `volume` is valid in either form",
        "Mixing an environment with either direct thermo input fails",
        "required persistent `(n_boxes,)` `wp.uint32` RNG sidecar",
        "charged, sedimentation, turbulent shear, combined, or unknown mechanisms",
        "marker subclasses",
        "Supplied diagnostic outputs have an adapter identity guarantee",
        "omitted direct-kernel diagnostics remain native call-local convenience results",
        "`initialize_rng=True`",
        "`initialize_rng=False`",
        "Callers control conversion, restoration, device placement",
        "does not upload, restore, synchronize, select a fallback, reseed per step",
        "or roll back state after launch",
        "E7-F4 resident-session lifecycle, E7-F5 scheduling, and E7-F8 checkpoint/restart",
        "remain explicitly deferred",
    ):
        assert _normalized(snippet) in selected, snippet


def test_example_uses_adapter_and_forced_no_warp_path() -> None:
    """Test published example has no direct dispatch and runs without Warp."""
    example = EXAMPLE_PATH.read_text(encoding="utf-8")
    for snippet in (
        "WarpBrownianCoagulationState",
        "WarpBrownianCoagulationExecutionState",
        "WarpBrownianCoagulationExecutionAdapter",
        "result.backend_result.value",
        "wp.synchronize()",
        "max(1, particle_data.n_particles // 2)",
    ):
        assert snippet in example
    for forbidden in (
        "coagulation_step_gpu",
        "CoagulationMechanismConfig",
        "max_collisions",
    ):
        assert forbidden not in example
    assert example.index("wp.synchronize()") < example.rindex(
        "from_warp_particle_data("
    )

    process = subprocess.run(  # noqa: S603
        [sys.executable, str(EXAMPLE_PATH)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PARTICULA_EXAMPLE_FORCE_NO_WARP": "1"},
        timeout=10,
    )
    assert "Warp is unavailable or disabled; no kernel ran." in process.stdout


def test_publication_commands_links_and_records_are_resolvable() -> None:
    """Test focused commands, local links, and E7-F3 planning records exist."""
    strategy = STRATEGY_PATH.read_text(encoding="utf-8")
    selected = _section(strategy, SECTION_HEADING)
    roadmap = ROADMAP_PATH.read_text(encoding="utf-8")
    phase_details = PHASE_DETAILS_PATH.read_text(encoding="utf-8")
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))

    assert plan["id"] == "E7-F3"
    phase = next(item for item in plan["phases"] if item["id"] == "E7-F3-P6")
    assert phase["status"] == "Shipped"
    assert phase["issue_number"] == 1482
    assert phase["completion_date"] == "2026-07-28"
    assert "E7-F3" in roadmap
    assert "E7-F3 P6 has shipped" in roadmap
    assert "E7-F3-P6" in phase_details
    assert "Status: Shipped 2026-07-28" in phase_details
    for command in COMMANDS:
        assert "-q" in command and "-Werror" in command
        assert (ROOT / command.split()[1]).exists()
    for destination in _local_destinations(selected):
        assert (STRATEGY_PATH.parent / destination).resolve().exists()
