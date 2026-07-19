"""Hardware-free publication tests for the GPU coagulation contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOUNDATIONS_PATH = ROOT / "docs/Features/data-containers-and-gpu-foundations.md"
STRATEGY_PATH = ROOT / "docs/Features/coagulation_strategy_system.md"
VALIDATION_PATH = ROOT / "docs/Features/Roadmap/coagulation-validation.md"
FOUNDATIONS_HEADING = "### GPU coagulation configuration and sidecar ownership"
STRATEGY_HEADING = "### GPU direct-kernel foundations and limitations"
BASELINE_COMMANDS = (
    "pytest particula/gpu/kernels/tests/coagulation_validation_test.py -q "
    '-m "warp and gpu_parity" -Werror',
    "pytest particula/gpu/kernels/tests/"
    "coagulation_stochastic_validation_test.py -q "
    '-m "warp and stochastic and not cuda" -Werror',
    "pytest particula/tests/gpu_coagulation_docs_test.py -q -Werror",
)
OPTIONAL_CUDA_COMMAND = (
    "pytest particula/gpu/kernels/"
    "tests/coagulation_stochastic_validation_test.py -q "
    '-m "warp and cuda" -Werror'
)
DIRECT_COAGULATION_COMMAND = (
    "pytest particula/gpu/kernels/tests/coagulation_test.py -q -Werror"
)


def _normalized(content: str) -> str:
    """Return content with consecutive whitespace normalized."""
    return " ".join(content.split())


def _section(content: str, heading: str) -> str:
    """Return a Markdown section through its next same-or-higher heading."""
    start = content.index(heading)
    section = content[start:]
    level = len(heading) - len(heading.lstrip("#"))
    for line_index, line in enumerate(section.splitlines()[1:], start=1):
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            return "\n".join(section.splitlines()[:line_index])
    return section


def _local_destinations(content: str) -> list[str]:
    """Return local Markdown link destinations without anchors or remote URLs."""
    destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", content)
    return [
        destination.split("#", maxsplit=1)[0]
        for destination in destinations
        if not destination.startswith(("http://", "https://", "#"))
    ]


def _command_target(command: str) -> Path:
    """Return the repository target named by a published pytest command."""
    match = re.search(r"\b(particula/[^\s]+)", command)
    assert match, f"Command has no particula target: {command}"
    return ROOT / match.group(1)


def test_foundations_guide_publishes_canonical_direct_coagulation_contract() -> (
    None
):
    """Foundations guide keeps the direct coagulation boundary complete."""
    section = _normalized(
        _section(
            FOUNDATIONS_PATH.read_text(encoding="utf-8"), FOUNDATIONS_HEADING
        )
    )

    for snippet in (
        "from particula.gpu.kernels import coagulation_step_gpu",
        "from particula.gpu.kernels.coagulation import CoagulationMechanismConfig",
        "immutable `mechanism_config=CoagulationMechanismConfig(...)` as keyword-only host metadata",
        "not re-exported by `particula.gpu.kernels`",
        '`distribution_type="particle_resolved"',
        "`brownian`",
        "`charged_hard_sphere`",
        "`sedimentation_sp2016`",
        "`turbulent_shear_st1956`",
        "canonical Brownian, charged, sedimentation, then turbulent order",
        "singleton masks are `1`, `2`, `4`, and `8`",
        "two-way mask",
        "`3`",
        "`5`",
        "`6`",
        "`9`",
        "`10`",
        "`12`",
        "four-term mask is `15`",
        "Rejected three-way mask",
        "`7`",
        "`11`",
        "`13`",
        "`14`",
        "`turbulent_dissipation`",
        "`fluid_density`",
        "active-device `wp.float64` Warp arrays shaped `(n_boxes,)`",
        "ignored by non-turbulent masks",
        "The return tuple is exactly `(particles, collision_pairs, n_collisions)`",
        "Accepted collisions mutate caller-owned particle mass, concentration, and charge in place",
        "Supplied collision buffers are returned by identity",
        "reset only when `initialize_rng=True`",
        "Malformed configuration and an unsupported distribution fail before particle access",
        "Mask `7` fails at capability preflight before particle metadata",
        "Masks `11`, `13`, and `14` validate particle metadata",
        "no rollback is guaranteed after such a failure",
        "Transfers are explicit",
        "high-level `Runnable` integration",
        "CPU fallback",
        "hidden transfer",
        "performance or broad-accuracy claims",
    ):
        assert _normalized(snippet) in section, snippet


def test_strategy_guide_matches_the_shipped_direct_mask_boundary() -> None:
    """Strategy guide keeps the shipped direct-kernel boundary in sync."""
    section = _normalized(
        _section(STRATEGY_PATH.read_text(encoding="utf-8"), STRATEGY_HEADING)
    )

    for snippet in (
        "direct-kernel-only path is separate from CPU strategies, builders, factories, and `Runnable` APIs",
        "from particula.gpu.kernels import coagulation_step_gpu",
        "from particula.gpu.kernels.coagulation import CoagulationMechanismConfig",
        '`distribution_type="particle_resolved"',
        "`brownian`",
        "`charged_hard_sphere`",
        "`sedimentation_sp2016`",
        "`turbulent_shear_st1956`",
        "Brownian, charged, sedimentation, then turbulent order",
        "singleton masks `1`, `2`, `4`, `8`",
        "two-term masks `3`, `5`, `6`, `9`, `10`, `12`",
        "four-term mask `15`",
        "Three-term masks `7`, `11`, `13`, and `14` are deferred and fail closed",
        "`turbulent_dissipation` and `fluid_density`",
        "same-device `wp.float64` arrays with shape `(n_boxes,)`",
        "Non-turbulent masks ignore these inputs",
        "caller-owned Warp resources",
        "exact return tuple is `(particles, collision_pairs, n_collisions)`",
        "RNG state is not returned",
        "Malformed configuration and unsupported distributions fail before particle access",
        "later runtime failure has no rollback guarantee",
        "Transfers are explicit",
    ):
        assert _normalized(snippet) in section, snippet
    for command in BASELINE_COMMANDS:
        assert _normalized(command) in section
    assert _normalized(OPTIONAL_CUDA_COMMAND) in section
    assert "Brownian-plus-sedimentation combinations and other" not in section


def test_gpu_coagulation_commands_and_optional_device_policy_are_resolvable() -> (
    None
):
    """Published direct-coagulation commands target checked-in test files."""
    foundations = FOUNDATIONS_PATH.read_text(encoding="utf-8")
    strategy = STRATEGY_PATH.read_text(encoding="utf-8")
    validation = VALIDATION_PATH.read_text(encoding="utf-8")

    for command in BASELINE_COMMANDS:
        assert command in foundations
        assert command in strategy
        assert _command_target(command).exists()
        assert "-q" in command
        assert "-Werror" in command
    assert OPTIONAL_CUDA_COMMAND in foundations
    assert OPTIONAL_CUDA_COMMAND in strategy
    assert OPTIONAL_CUDA_COMMAND in validation
    assert _command_target(OPTIONAL_CUDA_COMMAND).exists()
    assert "-q" in OPTIONAL_CUDA_COMMAND
    assert "-Werror" in OPTIONAL_CUDA_COMMAND
    assert DIRECT_COAGULATION_COMMAND in foundations
    assert DIRECT_COAGULATION_COMMAND in validation
    assert _command_target(DIRECT_COAGULATION_COMMAND).exists()

    normalized_foundations = _normalized(foundations)
    normalized_strategy = _normalized(strategy)
    normalized_validation = _normalized(validation)
    assert (
        'The required baseline is Warp `device="cpu"` when Warp is installed.'
        in normalized_foundations
    )
    assert (
        "CUDA is optional additive evidence and guarded suites skip cleanly"
        in normalized_foundations
    )
    assert "Warp CPU is the installed-Warp baseline." in normalized_strategy
    assert (
        "CUDA is optional/additive, and guarded suites skip cleanly"
        in normalized_strategy
    )
    assert (
        "Warp CPU is the required baseline when Warp is installed."
        in normalized_validation
    )
    assert (
        "CUDA is optional, local/manual additive evidence and skips cleanly"
        in normalized_validation
    )


def test_gpu_coagulation_guide_cross_links_resolve_once() -> None:
    """Required local ownership and evidence links resolve with fixed cardinality."""
    foundations = FOUNDATIONS_PATH.read_text(encoding="utf-8")
    strategy_section = _section(
        STRATEGY_PATH.read_text(encoding="utf-8"), STRATEGY_HEADING
    )
    strategy_destinations = _local_destinations(strategy_section)
    foundations_destinations = _local_destinations(
        _section(foundations, FOUNDATIONS_HEADING)
    )

    assert (
        strategy_destinations.count("./data-containers-and-gpu-foundations.md")
        == 1
    )
    assert (
        strategy_destinations.count("./Roadmap/coagulation-validation.md") == 1
    )
    assert (
        foundations_destinations.count("Roadmap/coagulation-validation.md") == 1
    )
    for path, destination in (
        (STRATEGY_PATH, "./data-containers-and-gpu-foundations.md"),
        (STRATEGY_PATH, "./Roadmap/coagulation-validation.md"),
        (FOUNDATIONS_PATH, "Roadmap/coagulation-validation.md"),
    ):
        assert (path.parent / destination).resolve().exists()


def test_guides_do_not_promote_deferred_runtime_capabilities() -> None:
    """Guides retain bounded negative claims without rejecting accurate negations."""
    content = _normalized(
        FOUNDATIONS_PATH.read_text(encoding="utf-8")
        + "\n"
        + _section(STRATEGY_PATH.read_text(encoding="utf-8"), STRATEGY_HEADING)
    ).lower()

    for affirmative_claim in (
        "high-level `aerosol`/`runnable` integration is supported",
        "automatic transfer is supported",
        "cpu fallback is supported",
        "cuda is mandatory",
        "performance is guaranteed",
        "dns/general turbulence is supported",
        "unsupported mechanisms are supported",
        "unsupported distributions are supported",
    ):
        assert affirmative_claim not in content
